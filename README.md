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

# Attention diagnostics panel on the right (toggle with F3)
python3 e1m1_transformer_backend.py --keyboard-source pygame_window --diagnostic-panel-width 460

# Faster movement/turning
python3 e1m1_transformer_backend.py --action-repeat 5

# Fine-tune movement and firing cadence
python3 e1m1_transformer_backend.py --move-delta 3.6 --strafe-delta 3.6 --turn-delta 1.35 --fire-cooldown-tics 8

# Slow enemies/world pace
python3 e1m1_transformer_backend.py --doom-ticrate 16 --doom-skill 1
```

## Transformer Backend Details

### 1. How the Transformer is used in the backend

- The Transformer runs every tick on a temporal state window (`context=32`) and outputs:
  - `state_out` (predicted next backend state vector).
  - `control_logits` (decoded into Doom control actions).
- The player action sent to Doom is computed from Transformer logits each tick (keyboard-gated, NN-modulated strength/conflict resolution).
- Doom remains authoritative for world simulation and rendering (physics, collisions, enemy AI, damage, doors/triggers, pickups, map logic, graphics).

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
- Context length: `32`.
- Training: none. Weights are deterministic hardcoded at startup and frozen (`requires_grad=False`).

### 3. Number of parameters

- Default runtime (`1024x768`, `frame_pool=16`):
  - `state_dim = 3214`.
  - Total parameters: `3,758,996`.
  - Trainable parameters: `0`.
- At `1280x960` with `frame_pool=16`:
  - `state_dim = 4942`.
  - Total parameters: `4,645,460`.

### 4. Input features

Per tick, one `state_in` vector is built and then stacked over time (`context=32`):

- Doom game variables: `132` values (`GameVariable` enum entries).
- Pooled frame features: grayscale average-pooled image, flattened.
  - For `1024x768` with `frame_pool=16`: `3072` values (`64 x 48`).
  - For `1280x960` with `frame_pool=16`: `4800` values (`80 x 60`).
- Keyboard features: `10` values:
  - forward, backward, strafe-left, strafe-right, turn-left, turn-right, look-up, look-down, attack, use.

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
- On macOS visible runs, the script merges Doom button states with global key-state sampling (ApplicationServices) for more reliable key detection.
- On macOS, letter-key capture requires OS permissions. If `keys=[]` while pressing letters, enable:
  - `System Settings -> Privacy & Security -> Accessibility`
  - `System Settings -> Privacy & Security -> Input Monitoring`
- In `auto` mode (visible run), the script now prefers `pygame_window` first for strict key-up/key-down behavior and less sticky-input risk.
- `pygame_window` mode is now a single focused game window (`Transformer Doom (focus this window)`): it renders Doom frames and captures keyboard input in the same window.
- In `pygame_window` mode, a right-side diagnostics panel visualizes Transformer internals in real time.
- Diagnostics controls: `F3` hide/show panel, `F4` toggle `mean` vs `single-head` view, `F5/F6` change head index.
- Diagnostics also show `dL0..dL3` (mean absolute attention change per layer). Values above `0` confirm live matrix updates.
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
- The diagnostics panel shows per-layer attention matrices. In `single-head` mode you see an actual head matrix (`Lx head h`), and in `mean` mode you see head-averaged attention. Each matrix cell encodes query-time (row) attending to key-time (column) within the context window; blue means lower attention and red means higher attention.
- Player control is keyboard-gated but now NN-modulated each tick: control logits scale movement/turn strength, and still resolve opposing-key conflicts.
- Doom remains authoritative for world logic (physics/collisions/doors/triggers/AI/damage/pickups), preserving full E1M1 functionality and original graphics.
