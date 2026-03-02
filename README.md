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

# Deterministic world RNG
python3 e1m1_transformer_backend.py --doom-seed 1

# Build and enable experimental Transformer enemy-backend override
python3 build_enemy_nn_mod.py
python3 e1m1_transformer_backend.py --enemy-backend-transformer --enemy-backend-mod enemy_nn_backend_mod.pk3 --enemy-slots 16

# Experimental: move low-level movement/collision/combat into NN world simulator
python3 e1m1_transformer_backend.py --enemy-backend-transformer --enemy-backend-mod enemy_nn_backend_mod.pk3 --nn-world-sim --nn-world-damage-scale 1.0

# Headless regression runner (parses and checks enemy metrics)
python3 run_enemy_regression.py --wad DOOM.WAD --enemy-backend-mod enemy_nn_backend_mod.pk3 --max-ticks 5000
# quick one-off (no mandatory release-suite expansion)
python3 run_enemy_regression.py --quick --wad DOOM.WAD --enemy-backend-mod enemy_nn_backend_mod.pk3 --maps E1M1 --seeds 1 --tick-suites 5000

# Long-run multi-map/multi-seed gate (5k ticks each)
python3 run_enemy_regression.py --wad DOOM.WAD --enemy-backend-mod enemy_nn_backend_mod.pk3 --maps E1M1,E1M2 --seeds 1,2,3 --max-ticks 5000

# Release criteria suite (multi-map, multi-seed, both 5k and 20k suites)
python3 run_enemy_regression.py --wad DOOM.WAD --enemy-backend-mod enemy_nn_backend_mod.pk3
# defaults: maps=E1M1,E1M2,E1M3 seeds=1,2,3 tick_suites=5000,20000
# release-suite expansion is now mandatory by default (unless --quick is used)
```

## Transformer Backend Details

### 1. How the Transformer is used in the backend

- The Transformer runs every tick on a temporal state window (`context=32`) and outputs:
  - `state_out` (predicted next backend state vector).
  - `control_logits` (decoded into Doom control actions).
  - `enemy_logits` (per-slot behavior head; authoritative direct `target_actor_id_raw` command).
  - `enemy_actuator_logits` (per-slot final actuator commands: `speed/fwd/side/turn/aim/fire/firecd/health`).
  - `low_level_logits` (non-standard backend knobs for player dynamics).
  - `memory_update_logits` (per-slot memory update channels: gate + delta for persistent enemy latent state).
- The player action sent to Doom is computed from Transformer logits each tick (keyboard-gated, NN-modulated strength/conflict resolution).
- In `--enemy-backend-transformer` mode, the loop also sends per-slot enemy commands to a custom mod (`enemy_nn_backend_mod.pk3`) each tick:
  - Transformer behavior channels are decoded into backend actuation:
    - behavior channels: minimal authoritative direct `target_actor_id_raw` command (per slot).
    - per-slot target selection is decoded as direct actor-id command with range safety clamp only.
    - final backend actuation is fully direct from final actuator channels with hard safety bounds only
    - actuation channels sent to Doom/mod are normalized float cvars: `speed_norm`, `fwd_norm`, `side_norm`, `turn_norm`, `aim_norm`, `fire_norm`, `firecd_norm`, `health_norm`, `present_norm`, plus direct actor-id cvars `actor_id_raw` and `target_actor_id_raw`
  - aiming/firing/timing are decoded directly from actuator channels (`act_aim_norm`, `act_fire_norm`, `act_firecd_norm`) and passed to the mod as normalized floats.
  - mod-side fire timing uses deterministic continuous decode (`fire_norm` drive + `firecd_norm` cadence) from model outputs with no per-slot cooldown counter/integrator state.
  - mod-side aim uses direct stateless trigger decode (`aim_norm`) without quantized pulse counters.
  - target policy execution in the mod resolves direct model-decoded actor ids (`target_actor_id_raw`) to live actors.
  - enemy health control is driven directly from `act_health_norm` (normalized float path).
- Per-enemy memory latent channels are enabled (`memory_dim=4`) and updated by Transformer memory outputs (gate + 4-delta).
  - Slot identity persistence is reorder-robust via model-memory slot binding, instead of raw observation order.
  - Command ownership at execution remains actor-id keyed in the mod.
- By default, Doom remains authoritative for rendering and core simulation (physics, collisions, damage, doors/triggers, pickups, map logic).
- With `--nn-world-sim` (experimental), low-level movement/collision/combat are stepped in Transformer-side Python state and bridged back into Doom each tick (`warp`/`setangle` + enemy health sync).

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
  - `enemy_out_proj`: `Linear(256 -> enemy_slots * enemy_cmd_dim)` (default `16 * 1`; per-slot direct `target_actor_id_raw`).
  - `enemy_actuator_out_proj`: `Linear(256 -> enemy_slots * enemy_actuator_dim)` (default `16 * 8`; per-slot `speed/fwd/side/turn/aim/fire/firecd/health`).
  - `low_level_out_proj`: `Linear(256 -> low_level_dim)` where `low_level_dim = 4 + enemy_slots * 1` (default `20`).
  - `memory_out_proj`: `Linear(256 -> enemy_slots * memory_update_dim)` where `memory_update_dim = 1 + memory_dim` (default `5` = gate + 4-delta).
- Context length: `32`.
- Training: none. Weights are deterministic hardcoded at startup and frozen (`requires_grad=False`).

### 3. Number of parameters

- Default runtime (`1024x768`, `frame_pool=16`):
  - `state_dim = 3406`.
  - Total parameters: `3,920,200`.
  - Trainable parameters: `0`.
- At `1280x960` with `frame_pool=16`:
  - `state_dim = 5134`.
  - Total parameters: `4,806,664`.

### 4. Input features

Per tick, one `state_in` vector is built and then stacked over time (`context=32`):

- Doom game variables: `132` values (`GameVariable` enum entries).
- Pooled frame features: grayscale average-pooled image, flattened.
  - For `1024x768` with `frame_pool=16`: `3072` values (`64 x 48`).
  - For `1280x960` with `frame_pool=16`: `4800` values (`80 x 60`).
- Keyboard features: `10` values:
  - forward, backward, strafe-left, strafe-right, turn-left, turn-right, look-up, look-down, attack, use.
- Enemy slot features: `enemy_slots * 12` values (default `16 * 12 = 192`):
  - Base features (`8`/slot): reduced raw object state (`id`, `x/y/z`, `vx/vy/vz`, `health`).
- Memory features: `4`/slot model-owned slot-binding latent channels.
- Target-valid mask features: none (direct `target_actor_id_raw` decode path).

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
  - enemy logits (behavior head; authoritative `target_actor_id_raw`)
  - enemy actuator logits (decoded per slot: final movement/turn/aim/fire/firecd/health)
  - low-level logits (decoded to non-standard player dynamics)
- Player control is keyboard-gated but now NN-modulated each tick: control logits scale movement/turn strength, and still resolve opposing-key conflicts.
- Experimental enemy-backend mode:
  - Build mod with `python3 build_enemy_nn_mod.py`.
  - Enable with `--enemy-backend-transformer --enemy-backend-mod enemy_nn_backend_mod.pk3`.
  - Current implementation applies per-slot monster movement/pathing/aiming/firing commands from Transformer outputs, with model-memory slot binding (reorder-robust), direct actuator decode, and normalized float command transfer into the backend mod.
  - Python-side decode shaping is reduced to crash-safe bounds + command transport only (no Python target keep/blend logic and no Python fire-policy gating).
  - Mod-side decode uses deterministic continuous fire timing directly from model channels (`fire_norm`, `firecd_norm`) and no RNG gating.
  - Target selection is decoded from model direct `target_actor_id_raw` and resolved to actual actors by actor id in the mod.
  - Headless runs print regression metrics summary (`shots_per_tick`, `target_switches`, `close_pairs_per_tick`, `identity_churn_rate`, `player_max_stuck_ticks`, `mse_mean`, `mse_drift`) at session end.
  - Release criteria are enforced by default in `run_enemy_regression.py` (maps `E1M1,E1M2,E1M3`, seeds `1,2,3`, tick suites `5000,20000`); use `--quick` only for local iteration.
  - CI release-gate workflow: [`.github/workflows/enemy-regression-gate.yml`](.github/workflows/enemy-regression-gate.yml).
  - CI expects `DOOM.WAD` in repo; if not present, provide secret `DOOM_WAD_B64` (base64-encoded WAD) to provision it at runtime.
