# E1M1 Transformer Backend Loop (VizDoom)

This project runs **the original Doom backend and graphics** through VizDoom using your local `DOOM.WAD`, and adds a deterministic **attention-based Transformer** in the tick loop:

- Input: current backend state (game variables + pooled frame features)
- Output: predicted next backend state vector
- Training: none (weights are hardcoded deterministically)
- Hardcoded initialization keeps LayerNorm gains near `1.0` so attention/control remain input-responsive at runtime.

## Run

From `/Users/kimonfountoulakis/Documents/VizDOOM_second_attempt`:

```bash
python3 e1m1_transformer_backend.py --wad DOOM.WAD --map E1M1
```

Useful flags:

```bash
# Larger window
python3 e1m1_transformer_backend.py --resolution 1280x960

# Headless test
python3 e1m1_transformer_backend.py --headless --max-ticks 200

# Force CPU
python3 e1m1_transformer_backend.py --device cpu

# Force macOS global keyboard sampling
python3 e1m1_transformer_backend.py --keyboard-source macos_global

# Permission-free keyboard capture via dedicated control window
python3 e1m1_transformer_backend.py --keyboard-source pygame_window

# Faster movement/turning
python3 e1m1_transformer_backend.py --action-repeat 5

# Disable Transformer-side movement/collision resolver (fallback to native Doom movement)
python3 e1m1_transformer_backend.py --disable-nn-movement-resolution

# Fine-tune movement and firing cadence
python3 e1m1_transformer_backend.py --move-delta 3.6 --strafe-delta 3.6 --turn-delta 1.35 --fire-cooldown-tics 8

# Tighten NN outputs (more stable control)
python3 e1m1_transformer_backend.py --nn-weight-scale 0.55 --nn-control-gain 1.00 --nn-enemy-gain 0.75 --nn-low-level-gain 0.45

# Slow enemies/world pace
python3 e1m1_transformer_backend.py --doom-ticrate 16 --doom-skill 1

# Build and enable experimental Transformer enemy-backend override
python3 build_enemy_nn_mod.py
python3 e1m1_transformer_backend.py --enemy-backend-transformer --enemy-backend-mod enemy_nn_backend_mod.pk3 --enemy-slots 16
```

## Transformer Backend Details

### 1. How the Transformer is used in the backend

- The Transformer runs every tick on a temporal state window (`context=32`) and outputs:
  - `state_out` (predicted next backend state vector).
  - `control_logits` (decoded into Doom control actions).
  - `enemy_logits` (per-slot enemy backend command logits when enemy backend mode is enabled).
  - `enemy_intent_logits` (per-slot intent logits + timer signal for NN-space enemy state machine).
  - `low_level_logits` (non-standard backend knobs for player dynamics, plus enemy firecd proxy diagnostics).
  - `memory_update_logits` (per-slot memory update channels: gate + delta for persistent enemy latent state).
- The player action sent to Doom is computed from Transformer logits each tick (keyboard-gated, NN-modulated strength/conflict resolution).
- In `--enemy-backend-transformer` mode, the loop also sends per-slot enemy commands to a custom mod (`enemy_nn_backend_mod.pk3`) each tick:
  - Transformer behavior channels are decoded into backend actuation:
    - behavior channels: 37 core channels (`speed_cmd`, `advance_cmd`, `strafe_cmd`, `turn_cmd`, `aim_cmd_logit`, `fire_cmd_logit`, `advance_conf`, `strafe_conf`, `turn_conf`, `aim_conf`, `fire_conf`, `move_mix_cmd`, `strafe_mix_cmd`, `turn_mix_cmd`, `aim_mix_cmd`, `fire_mix_cmd`, `retreat_mix_cmd`, `health_cmd_norm`, `target_index_cmd_norm`, `fwd_final_cmd`, `side_final_cmd`, `turn_final_cmd`, `aim_final_logit`, `fire_final_logit`, `target_blend_logit`, `fire_enable_logit`, `burst_len_norm`, `inter_shot_delay_norm`, `reaction_delay_norm`, `coord_focus_target_index_norm`, `coord_assist_gate_logit`, `coord_spacing_cmd`, `coord_avoidance_cmd`, `nav_desired_heading_cmd`, `nav_desired_speed_norm`, `nav_cover_seek_cmd`, `nav_retreat_seek_cmd`) plus target-selection logits (`target_player_logit`, `target_slot_00_logit` ... `target_slot_15_logit`)
    - per-slot target decode is model-side (`target_index_cmd_norm` + target logits + focus/assist channels), with persistence/state transitions authored by memory channels (`lock_strength`, `retarget_cooldown`, `threat_age_decay`, `retarget_gate` from memory[4:8]); Python only applies validity clamps
    - final movement/pathing/aim are decoded from explicit NN navigation intent channels (`nav_desired_heading_cmd`, `nav_desired_speed_norm`, `nav_cover_seek_cmd`, `nav_retreat_seek_cmd`) plus memory tactical modulation
    - actuation channels sent to Doom/mod: `speed`, `fwd`, `side`, `turn`, `aim`, `fire`
  - aiming/firing are driven by final actuator + explicit fire-policy logits:
    - `aim_final_logit` -> backend `aim`
    - `fire_final_logit` + `fire_enable_logit` + (`burst_len_norm`, `inter_shot_delay_norm`, `reaction_delay_norm`) -> backend `fire` timing
    - `inter_shot_delay_norm` -> backend `nn_enemy_cmd_*_firecd`
    - no rule-based action-state cadence shaping is applied
  - enemy health (`nn_enemy_cmd_*_healthpct`) is now driven from `health_cmd_norm`
  - separate enemy intent head (`chase/flank/retreat/hold` + timer) is currently exposed for diagnostics/compatibility, while tactical state authority is carried by memory channels
  - low-level channels keep a firecd proxy for diagnostics only
- Per-enemy memory state is maintained in-loop as a persistent latent (`8` values/slot), updated each tick only by Transformer memory-update outputs (gate + delta), and fed back into `state_in`.
- Doom remains authoritative for rendering and core simulation (physics, collisions, damage, doors/triggers, pickups, map logic).

### 2. Exact architecture and its parameters

- Model: `HardcodedStateTransformer`.
- Input projection: `Linear(state_dim -> 256)`.
- Transformer core: 4 x `TransformerBlockWithAttention`, each block has:
  - `MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.0, batch_first=True)`.
  - Feed-forward: `Linear(256 -> 512)`, `GELU`, `Linear(512 -> 256)`.
  - Residual + `LayerNorm(256)` after attention and FFN.
- Output heads:
  - `state_out_proj`: `Linear(256 -> state_dim)`.
  - `control_out_proj`: `Linear(256 -> 6)`.
  - `enemy_out_proj`: `Linear(256 -> enemy_slots * enemy_cmd_dim)` (default `16 * 54`; 37 core channels + 17 target logits per slot).
  - `enemy_intent_out_proj`: `Linear(256 -> enemy_slots * enemy_intent_dim)` (default `16 * 5`; 4 intent logits + 1 timer signal per slot).
  - `low_level_out_proj`: `Linear(256 -> low_level_dim)` where `low_level_dim = 4 + enemy_slots * 1` (default `20`).
  - `memory_out_proj`: `Linear(256 -> enemy_slots * memory_update_dim)` where `memory_update_dim = 1 + memory_dim` (default `9` = gate + 8-delta).
- Context length: `32`.
- Training: none. Weights are deterministic hardcoded at startup and frozen (`requires_grad=False`).

### 3. Number of parameters

- Default runtime (`1024x768`, `frame_pool=16`):
  - `state_dim = 3871`.
  - Total parameters: `4,380,793`.
  - Trainable parameters: `0`.
- At `1280x960` with `frame_pool=16`:
  - `state_dim = 5599`.
  - Total parameters: `5,267,257`.

### 4. Input features

Per tick, one `state_in` vector is built and then stacked over time (`context=32`):

- Doom game variables: `132` values (`GameVariable` enum entries).
- Pooled frame features: grayscale average-pooled image, flattened.
  - For `1024x768` with `frame_pool=16`: `3072` values (`64 x 48`).
  - For `1280x960` with `frame_pool=16`: `4800` values (`80 x 60`).
- Keyboard features: `10` values:
  - forward, backward, strafe-left, strafe-right, turn-left, turn-right, look-up, look-down, attack, use.
- Enemy slot features: `enemy_slots * 40` values (default `16 * 40 = 640`), with stable ID->slot tracking:
  - Base features (`21`/slot): alive flag, relative x/y, velocity x/y, facing angle, observed health proxy, distance to player, bearing to player, line-of-sight flag, cooldown proxy, 4 wall-ray probes (forward/left/right/back), incoming-threat direction/intensity, recent-damage proxy, nearest-cover lateral/retreat hints.
  - Feedback features (`11`/slot): last command (`speed/fwd/side/turn/aim/fire`) and observed response (`moved_dist`, `turn_delta`, `LOS_changed`, `blocked`, `shot_fired`).
- Memory features (`8`/slot): persistent NN-updated latent channels (no fixed hand-authored semantics), carried across ticks and fed back into `state_in`.
- Target-valid mask features: `1 + enemy_slots` values (default `17`) for model-facing target validity (`player + slot occupancy`).

### 5. Input feature description

- Game-variable features carry backend state (examples: health, armor, position, ammo, kills, etc.).
- Frame features carry compact visual context from Doom screen buffer:
  - RGB frame -> grayscale mean.
  - Spatial average pooling with `frame_pool`.
  - Flattened and normalized to `[0,1]`.
- Keyboard features carry immediate control intent sampled each tick.
- Final feature vector is stabilized by `tanh(vector / 100.0)` before entering the Transformer.
- Tensor shape fed into the model each step: `[batch=1, context=32, state_dim]`.

## Important behavior

- Gameplay and visuals are rendered by **VizDoom + your `DOOM.WAD`**, which preserves original E1M1 content and interactions.
- The game runs in `PLAYER` mode and the Transformer is authoritative for player controls each tick (`movement`, `aim`, `fire`, `use`) via `make_action`.
- Keyboard input is read each tick and fed into Transformer control decoding (`W/S/A/D`, arrows, fire/use).
- Startup enforces key binds: `W/S/A/D`, `Left/Right/Up/Down`, `Space` (attack), `E` (use). (`Z/Q` are also accepted for AZERTY layouts).
- Startup writes and loads `transformer_controls.cfg` to force this keymap every run.
- Movement/collision resolver can run inside the Transformer loop (`--nn-movement-resolution`, enabled by default):
  - Loads blocking linedefs from `DOOM.WAD` (current map).
  - Resolves player movement with wall/entity collision checks in Python.
  - Applies resolved position via `warp` each tic.
  - Enemy movement commands are also pre-resolved against the same collision map before being sent to the backend mod.
- On macOS visible runs, the script merges Doom button states with global key-state sampling (ApplicationServices) for more reliable key detection.
- On macOS, letter-key capture requires OS permissions. If `keys=[]` while pressing letters, enable:
  - `System Settings -> Privacy & Security -> Accessibility`
  - `System Settings -> Privacy & Security -> Input Monitoring`
- In `auto` mode (visible run), the script now prefers `pygame_window` first for strict key-up/key-down behavior and less sticky-input risk.
- `pygame_window` mode is now a single focused game window (`Transformer Doom (focus this window)`): it renders Doom frames and captures keyboard input in the same window.
- Strict keyboard gating for sampled keys: when no key is sampled, the loop avoids injecting movement/fire/use actions.
- Control actions are now re-sampled every Doom tic (even when `--action-repeat > 1`) to reduce sticky movement on key release.
- `--action-repeat` controls movement/turn speed by applying each decoded action for multiple VizDoom ticks (default `5`).
- Movement/turn speed is independently tunable with `--move-delta`, `--strafe-delta`, `--turn-delta`.
- Holding fire is rate-limited by `--fire-cooldown-tics`, and attack is pulsed for one tick so high action-repeat does not overfire.
- Enemy/world pace is tunable with `--doom-ticrate` (lower = slower) and `--doom-skill`.
- Classic Doom HUD/status bar is forced (`hud_althud 0`, `screenblocks 10`, `set_render_hud(True)`).
- The Transformer predicts both:
  - `state_out` (next state vector)
  - control logits (decoded to Doom buttons)
  - enemy logits (decoded to per-slot enemy backend movement/aim/fire + cadence commands when enabled)
  - enemy intent logits (decoded per slot: chase/flank/retreat/hold + timer channels)
  - low-level logits (decoded to non-standard player dynamics + enemy firecd proxy diagnostics)
- Player control is keyboard-gated but now NN-modulated each tick: control logits scale movement/turn strength, and still resolve opposing-key conflicts.
- Experimental enemy-backend mode:
  - Build mod with `python3 build_enemy_nn_mod.py`.
  - Enable with `--enemy-backend-transformer --enemy-backend-mod enemy_nn_backend_mod.pk3`.
  - Current implementation applies per-slot monster movement/pathing/aiming/firing control commands from Transformer outputs, with direct NN target decode + memory-authored target persistence (lock/cooldown/threat-age), explicit NN fire-policy decode, memory-authoritative tactical state (`aggressive/suppressed/recovering/flanking`), explicit NN coordination decode (`focus-target`, `assist gate`, `spacing`, `avoidance`), and explicit NN navigation-intent decode (`desired heading/speed`, `cover seek`, `retreat seek`) in the backend loop.
  - Target mapping helper is strict index->coordinate lookup with invalid-index clamp only (no confidence shaping).
