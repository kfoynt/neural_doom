# E1M1 Transformer Backend Loop (VizDoom)

This project runs **the original Doom backend and graphics** through VizDoom using your local `DOOM.WAD`, and adds a deterministic **attention-based Transformer** in the tick loop:

- Input: current backend state (game variables + pooled frame features)
- Output: predicted next backend state vector
- Training: none (weights are hardcoded deterministically)
- Hardcoded initialization keeps LayerNorm gains near `1.0` so attention/control remain input-responsive at runtime.

## Run

From `/Users/kimonfountoulakis/Documents/VizDOOM_second_attempt`:

```bash
./run_pure.sh
```

`run_pure.sh` is the recommended one-command launcher for the current pure authoritative profile.
It rebuilds the mod and starts the game with pure world-sim defaults.
You can pass extra CLI args through it, e.g. `./run_pure.sh --max-ticks 5000`.

Equivalent manual launch:

```bash
python3 build_enemy_nn_mod.py
python3 e1m1_transformer_backend.py --wad DOOM.WAD --map E1M1
```

Default profile is now "most-authoritative Transformer":
- `--resolution 1280x960`
- `--keyboard-source pygame_window`
- `--enemy-backend-transformer --enemy-kinematics-transformer --enemy-combat-transformer`
- `--nn-world-sim --nn-world-sim-strict --nn-world-sim-pure`

Use `--disable-enemy-backend-transformer`, `--disable-enemy-kinematics-transformer`,
`--disable-enemy-combat-transformer`, `--disable-nn-world-sim`,
`--disable-nn-world-sim-strict`, or `--disable-nn-world-sim-pure` to opt out.

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

# Strict world-execution experiment: player movement/fire are Transformer-world-sim authoritative
python3 e1m1_transformer_backend.py --enemy-backend-transformer --enemy-backend-mod enemy_nn_backend_mod.pk3 --nn-world-sim --nn-world-sim-strict --nn-world-damage-scale 1.0

# Pure strict world-execution: visible player world execution also stays in Transformer
python3 e1m1_transformer_backend.py --enemy-backend-transformer --enemy-backend-mod enemy_nn_backend_mod.pk3 --nn-world-sim --nn-world-sim-strict --nn-world-sim-pure --nn-world-damage-scale 1.0

# One-command pure launcher (recommended)
./run_pure.sh

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

### 0. Current Transformer Authority (What It Handles Right Now)

This section is the current authority map for the runtime backend.

- Always Transformer-handled (all modes):
  - Builds `state_in` every tick from game variables, pooled frame features, keyboard features, enemy observation tokens, and enemy memory features.
  - Runs the hardcoded Transformer forward pass every tick (`state_out`, control, enemy heads, low-level head, memory-update head).
  - Decodes player controls from keyboard + `control_logits` (keyboard decides intent; Transformer modulates strength and resolves key conflicts).
  - Applies player fire pulse/cooldown logic in the loop (`_apply_fire_cooldown` + `_make_action_with_fire_pulse`).
  - Updates low-level runtime knobs from `low_level_logits` (move scale, turn scale, fire cooldown scale).
  - Updates per-slot enemy latent memory from `memory_update_logits` (gate + delta).

- Additional Transformer authority with `--enemy-backend-transformer`:
  - Uses `enemy_bind_logits` to bind slot memory to observed enemy tokens (one-to-one assignment with safe Hungarian solve).
  - Uses `enemy_target_logits` to decode per-slot target selection (player token or observed actor token), then transports actor-id targets to the mod.
  - Uses `enemy_actuator_logits` as final per-slot actuation channels (`speed/fwd/side/turn/aim/fire/firecd/health`), bounded only for crash safety.
  - Decodes enemy fire cadence in Transformer-side state (`_decode_enemy_fire_pulse` phase/threshold state), then sends pulse command to mod.

- Additional Transformer authority with `--enemy-kinematics-transformer`:
  - Integrates enemy x/y/angle in Transformer-side state each tick.
  - Resolves enemy movement collisions against map geometry + dynamic circles (player + other enemies).
  - Sends absolute enemy kinematic commands (`x_raw`, `y_raw`, `angle_raw`) to execution mod.
  - Note: current CLI wiring auto-enables `enemy_combat_transformer` when `enemy_kinematics_transformer` is enabled.

- Additional Transformer authority with `--enemy-combat-transformer` (no world-sim):
  - Resolves player -> enemy hit logic in Transformer state (LOS, aim tolerance, distance, cooldown).
  - Resolves enemy -> player hit logic in Transformer state (fire pulse, target-to-player gate, LOS, aim tolerance, distance).
  - Owns enemy health/alive transitions in Transformer state and sends `health_raw`/`dead_raw` to mod.
  - Owns player health/death transitions in Transformer state and syncs to Doom via `sethealth` / `kill`.

- Additional Transformer authority with `--nn-world-sim`:
  - Steps a Transformer-side world state loop for player/enemy kinematics and simplified combat.
  - Bridges world state back to Doom for rendering (`warp`/angle bridge + command sync).
  - In this mode, no-world-sim combat path (`_transformer_enemy_combat_step`) is bypassed and world-sim combat path is used instead.

- Still handled by Doom/mod in normal (non-world-sim) gameplay:
  - Doom remains renderer and core map runtime (sectors/doors/triggers/pickups/hud rendering and native tick execution).
  - Execution mod applies transported enemy commands to live actors (movement/angle/target/fire state call + health/death write).
  - Full native actor/world internals (projectiles, full physics, animation/state machine details outside overridden channels) remain engine-side.

### 1. How the Transformer is used in the backend

- The Transformer runs every tick on a temporal state window (`context=32`) and outputs:
  - `state_out` (predicted next backend state vector).
  - `control_logits` (decoded into Doom control actions).
  - `enemy_logits` (legacy behavior/diagnostic head; not the final target authority path).
  - `enemy_bind_logits` (per-slot binding logits over observed enemy tokens + empty token).
  - `enemy_target_logits` (per-slot target logits over player token + observed enemy tokens).
  - `enemy_actuator_logits` (per-slot final actuator commands: `speed/fwd/side/turn/aim/fire/firecd/health`).
  - `low_level_logits` (non-standard backend knobs for player dynamics).
  - `memory_update_logits` (per-slot memory update channels: gate + delta for persistent enemy latent state).
- The player action sent to Doom is computed from Transformer logits each tick (keyboard-gated, NN-modulated strength/conflict resolution).
- In `--enemy-backend-transformer` mode, the loop also sends per-slot enemy commands to a custom mod (`enemy_nn_backend_mod.pk3`) each tick:
  - Transformer behavior channels are decoded into backend actuation:
    - behavior head is legacy/compatibility-only in current path.
    - per-slot target selection comes from `enemy_target_logits` and is transported as direct actor-id command (`target_actor_id_raw`) with range-safe transport only.
    - per-slot observation-to-memory binding comes from `enemy_bind_logits`.
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
  - In no-world-sim mode, slot assignment uses canonical observation keys + global memory-key matching (order-invariant, no frame-order dependence).
- By default, Doom remains authoritative for rendering and core simulation (physics, collisions, damage, doors/triggers, pickups, map logic).
- With `--nn-world-sim` (experimental), low-level movement/collision/combat are stepped in Transformer-side Python state and bridged back into Doom each tick.
- With `--nn-world-sim-strict`, player movement/fire are not sent to native Doom; Transformer world state drives player bridge (more model-authoritative, less stable).
  - In visible strict mode, keyboard sampling must come from `macos_global+doom_buttons` or `pygame_window` (plain `doom_buttons` is auto-upgraded/fails-fast).
  - Strict bridge uses `warp x y` (not `warp x y z`) for reliable VizDoom execution.
  - Native Doom movement/fire binds are disabled in strict mode to avoid double-driving and visual/collision glitches.
  - Visible strict mode now applies Transformer-decoded movement/turn deltas through Doom (no visible player warp), while keeping firing world-sim-side.
- With `--nn-world-sim-pure` (implies strict), visible player execution is also Transformer-side (Doom player movement solver bypassed).
  - In visible pure mode, player x/y is bridged with `warp`; turn/look/fire/use are sent as native action deltas from Transformer-decoded controls.
  - Pure bridge applies recovery ticks to keep camera/view-height stable across warp and lower-floor transitions.
  - Manual doors are guarded in Transformer collision: closed doors block movement until `Use` (`E`) opens a short pass window.

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
  - `enemy_out_proj`: `Linear(256 -> enemy_slots * enemy_cmd_dim)` (default `16 * 1`; legacy behavior/diagnostic path).
  - `enemy_bind_out_proj`: `Linear(256 -> enemy_slots * enemy_bind_dim)` (default `16 * 17`; bind over 16 obs tokens + empty).
  - `enemy_target_out_proj`: `Linear(256 -> enemy_slots * enemy_target_dim)` (default `16 * 17`; target over player + 16 obs tokens).
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
- Recommended launcher is `./run_pure.sh` for the pure authoritative profile.
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
- In pure strict mode, closed manual doors block movement in Transformer collision until `E` (`Use`) opens a short pass window.
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
