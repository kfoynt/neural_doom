#!/usr/bin/env python3
"""Run Doom E1M1 with a deterministic Transformer state emulator in the game loop.

The Doom engine (VizDoom + DOOM.WAD) stays authoritative for gameplay/graphics.
The Transformer receives backend state every tick and emits a predicted next state.
"""

from __future__ import annotations

import argparse
import ctypes
import signal
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque

import numpy as np
import torch
from torch import nn
from vizdoom import Button, DoomGame, GameVariable, Mode, ScreenFormat, ScreenResolution


class MacOSKeyboardSampler:
    """Global keyboard/mouse state sampler via macOS ApplicationServices."""

    # macOS virtual keycodes.
    KEY_W = 13
    KEY_Z = 6
    KEY_S = 1
    KEY_A = 0
    KEY_Q = 12
    KEY_D = 2
    KEY_E = 14
    KEY_SPACE = 49
    KEY_LEFT = 123
    KEY_RIGHT = 124
    KEY_DOWN = 125
    KEY_UP = 126

    def __init__(self) -> None:
        framework = "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        cg = ctypes.CDLL(framework)
        self._ax_is_trusted = getattr(cg, "AXIsProcessTrusted", None)
        if self._ax_is_trusted is not None:
            self._ax_is_trusted.restype = ctypes.c_bool
        self._key_state = cg.CGEventSourceKeyState
        self._key_state.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        self._key_state.restype = ctypes.c_bool
        self._state_ids = (0, 1)

    def is_accessibility_trusted(self) -> bool:
        if self._ax_is_trusted is None:
            return True
        return bool(self._ax_is_trusted())

    def _is_key_down(self, keycode: int) -> bool:
        for state_id in self._state_ids:
            if bool(self._key_state(state_id, keycode)):
                return True
        return False

    def read(self) -> np.ndarray:
        forward = (
            self._is_key_down(self.KEY_W)
            or self._is_key_down(self.KEY_Z)
            or self._is_key_down(self.KEY_UP)
        )
        backward = self._is_key_down(self.KEY_S) or self._is_key_down(self.KEY_DOWN)
        move_left = self._is_key_down(self.KEY_A) or self._is_key_down(self.KEY_Q)
        move_right = self._is_key_down(self.KEY_D)
        turn_left = self._is_key_down(self.KEY_LEFT)
        turn_right = self._is_key_down(self.KEY_RIGHT)
        look_up = False
        look_down = False
        attack = self._is_key_down(self.KEY_SPACE)
        use = self._is_key_down(self.KEY_E)

        return np.asarray(
            [
                float(forward),
                float(backward),
                float(move_left),
                float(move_right),
                float(turn_left),
                float(turn_right),
                float(look_up),
                float(look_down),
                float(attack),
                float(use),
            ],
            dtype=np.float32,
        )


class PygameKeyboardSampler:
    """In-process keyboard capture and display via a single pygame game window."""

    def __init__(self, width: int, height: int, diagnostic_panel_width: int) -> None:
        import pygame  # Imported lazily so headless usage keeps working.

        self._pg = pygame
        self._pg.init()
        self._game_width = width
        self._height = height
        self._panel_width = max(0, diagnostic_panel_width)
        window_w = self._game_width + self._panel_width
        self._window = self._pg.display.set_mode((window_w, height))
        self._pg.display.set_caption("Transformer Doom (focus this window)")
        self._font = self._pg.font.SysFont("Courier", 16)
        self._small_font = self._pg.font.SysFont("Courier", 13)
        self._diag_toggle_key = self._pg.K_F3
        self._diag_mode_key = self._pg.K_F4
        self._diag_prev_head_key = self._pg.K_F5
        self._diag_next_head_key = self._pg.K_F6
        self.show_diagnostics = True
        self._diag_mode = "head"
        self._diag_head_index = 0
        self._pressed_keys: set[int] = set()
        self.closed = False

    def read(self) -> np.ndarray:
        for event in self._pg.event.get():
            if event.type == self._pg.QUIT:
                self.closed = True
                self._pressed_keys.clear()
            elif event.type == self._pg.KEYDOWN:
                if int(event.key) == self._diag_toggle_key:
                    self.show_diagnostics = not self.show_diagnostics
                    continue
                if int(event.key) == self._diag_mode_key:
                    self._diag_mode = "mean" if self._diag_mode == "head" else "head"
                    continue
                if int(event.key) == self._diag_prev_head_key:
                    self._diag_head_index = max(0, self._diag_head_index - 1)
                    continue
                if int(event.key) == self._diag_next_head_key:
                    self._diag_head_index += 1
                    continue
                self._pressed_keys.add(int(event.key))
            elif event.type == self._pg.KEYUP:
                self._pressed_keys.discard(int(event.key))
            else:
                focus_lost_type = getattr(self._pg, "WINDOWFOCUSLOST", None)
                if focus_lost_type is not None and event.type == focus_lost_type:
                    self._pressed_keys.clear()

        if not self._pg.key.get_focused():
            self._pressed_keys.clear()

        forward = bool(
            self._pg.K_w in self._pressed_keys
            or self._pg.K_z in self._pressed_keys
            or self._pg.K_UP in self._pressed_keys
        )
        backward = bool(self._pg.K_s in self._pressed_keys or self._pg.K_DOWN in self._pressed_keys)
        move_left = bool(self._pg.K_a in self._pressed_keys or self._pg.K_q in self._pressed_keys)
        move_right = bool(self._pg.K_d in self._pressed_keys)
        turn_left = bool(self._pg.K_LEFT in self._pressed_keys)
        turn_right = bool(self._pg.K_RIGHT in self._pressed_keys)
        look_up = False
        look_down = False
        attack = bool(
            self._pg.K_SPACE in self._pressed_keys
            or self._pg.K_LCTRL in self._pressed_keys
            or self._pg.K_RCTRL in self._pressed_keys
        )
        use = bool(self._pg.K_e in self._pressed_keys)

        return np.asarray(
            [
                float(forward),
                float(backward),
                float(move_left),
                float(move_right),
                float(turn_left),
                float(turn_right),
                float(look_up),
                float(look_down),
                float(attack),
                float(use),
            ],
            dtype=np.float32,
        )

    def _draw_text(
        self,
        text: str,
        x: int,
        y: int,
        *,
        color: tuple[int, int, int] = (220, 220, 220),
        small: bool = False,
    ) -> None:
        font = self._small_font if small else self._font
        self._window.blit(font.render(text, True, color), (x, y))

    def _matrix_surface(self, matrix: np.ndarray) -> Any:
        m = np.asarray(matrix, dtype=np.float32)
        if m.ndim != 2 or m.size == 0:
            m = np.zeros((1, 1), dtype=np.float32)
        m_min = float(np.min(m))
        m_max = float(np.max(m))
        span = max(1e-6, m_max - m_min)
        normalized = (m - m_min) / span
        rgb = np.stack(
            [
                normalized,
                np.clip(1.5 - np.abs(normalized - 0.5) * 3.0, 0.0, 1.0),
                1.0 - normalized,
            ],
            axis=-1,
        )
        rgb = np.ascontiguousarray((rgb * 255.0).astype(np.uint8))
        surface = self._pg.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
        return surface.copy()

    def _render_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        panel_x = self._game_width
        pad = 14
        self._window.fill((20, 20, 24), (panel_x, 0, self._panel_width, self._height))
        self._draw_text("NN Diagnostics (F3)", panel_x + pad, 10, color=(245, 245, 245))
        self._draw_text(
            f"view={self._diag_mode}  head={self._diag_head_index}  (F4/F5/F6)",
            panel_x + pad,
            24,
            small=True,
        )

        tick = diagnostics.get("tick", 0)
        sub_tick = diagnostics.get("sub_tick", -1)
        mse = diagnostics.get("mse", 0.0)
        reward = diagnostics.get("reward", 0.0)
        health = diagnostics.get("health", 0.0)
        armor = diagnostics.get("armor", 0.0)
        kills = diagnostics.get("kills", 0.0)
        if sub_tick >= 0:
            self._draw_text(f"tick {tick}.{sub_tick}", panel_x + pad, 40, small=True)
        else:
            self._draw_text(f"tick {tick}", panel_x + pad, 40, small=True)
        self._draw_text(f"mse {mse:.6f}  reward {reward:.2f}", panel_x + pad, 56, small=True)
        self._draw_text(f"hp {health:.0f}  armor {armor:.0f}  kills {kills:.0f}", panel_x + pad, 72, small=True)

        keys = diagnostics.get("active_keys", [])
        controls = diagnostics.get("active_controls", [])
        attention_deltas = diagnostics.get("attention_deltas", [])
        self._draw_text(
            f"keys: {', '.join(keys[:3]) if keys else 'none'}",
            panel_x + pad,
            92,
            small=True,
        )
        self._draw_text(
            f"act: {', '.join(controls[:2]) if controls else 'none'}",
            panel_x + pad,
            108,
            small=True,
        )
        if attention_deltas:
            delta_text = " ".join([f"dL{i}={v:.2e}" for i, v in enumerate(attention_deltas[:4])])
            self._draw_text(delta_text, panel_x + pad, 124, small=True)
            self._draw_text("Color: blue=low attention, red=high", panel_x + pad, 140, small=True)
            matrix_start_y = 158
        else:
            self._draw_text("Color: blue=low attention, red=high", panel_x + pad, 124, small=True)
            matrix_start_y = 142

        attention_maps = diagnostics.get("attention_maps", [])
        if not attention_maps:
            self._draw_text("No attention maps yet.", panel_x + pad, matrix_start_y, small=True)
            return

        available_h = self._height - 150
        layer_count = min(4, len(attention_maps))
        matrix_size = max(56, min(150, available_h // layer_count - 20, self._panel_width - 2 * pad))
        y = matrix_start_y
        for layer_idx in range(layer_count):
            layer_map = np.asarray(attention_maps[layer_idx], dtype=np.float32)
            if layer_map.ndim == 3:
                heads = layer_map.shape[0]
                head_idx = self._diag_head_index % max(1, heads)
                if self._diag_mode == "head":
                    matrix = layer_map[head_idx]
                    label = f"L{layer_idx} head {head_idx}/{heads - 1}"
                else:
                    matrix = layer_map.mean(axis=0)
                    label = f"L{layer_idx} head-mean ({heads}h)"
            else:
                matrix = layer_map
                heads = 1
                head_idx = 0
                label = f"L{layer_idx} head {head_idx}/{heads - 1}"
            surf = self._matrix_surface(matrix)
            surf = self._pg.transform.scale(surf, (matrix_size, matrix_size))
            self._window.blit(surf, (panel_x + pad, y))
            self._pg.draw.rect(
                self._window,
                (120, 120, 130),
                (panel_x + pad, y, matrix_size, matrix_size),
                width=1,
            )
            self._draw_text(
                label,
                panel_x + pad + matrix_size + 8,
                y + 6,
                small=True,
            )
            self._draw_text(
                "Rows=query time, Cols=key time",
                panel_x + pad + matrix_size + 8,
                y + 22,
                small=True,
            )
            y += matrix_size + 14

    def render_rgb_frame(self, frame: np.ndarray, diagnostics: dict[str, Any] | None = None) -> None:
        if self.closed:
            return
        # frame is HxWx3 RGB from VizDoom.
        surface = self._pg.image.frombuffer(frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")
        if frame.shape[1] != self._game_width or frame.shape[0] != self._height:
            surface = self._pg.transform.scale(surface, (self._game_width, self._height))
        self._window.blit(surface, (0, 0))
        if self._panel_width > 0:
            if self.show_diagnostics:
                self._render_diagnostics(diagnostics or {})
            else:
                panel_x = self._game_width
                self._window.fill((18, 18, 22), (panel_x, 0, self._panel_width, self._height))
                self._draw_text("Diagnostics hidden", panel_x + 16, 14, small=True)
                self._draw_text("Press F3 to show", panel_x + 16, 30, small=True)
        self._pg.display.flip()

    def close(self) -> None:
        self._pg.display.quit()
        self._pg.quit()


def _build_positional_encoding(max_len: int, dim: int) -> torch.Tensor:
    positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-np.log(10000.0) / dim))
    pe = torch.zeros(max_len, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(positions * div)
    pe[:, 1::2] = torch.cos(positions * div)
    return pe


class TransformerBlockWithAttention(nn.Module):
    """Transformer block that returns attention weights for diagnostics."""

    def __init__(self, d_model: int, nhead: int, ff_dim: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=0.0,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.self_attn(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.norm1(x + attn_out)
        ff = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + ff)
        return x, attn_weights


class HardcodedStateTransformer(nn.Module):
    """Attention-based Transformer encoder with deterministic hardcoded weights."""

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        context: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(state_dim, d_model)
        self.state_out_proj = nn.Linear(d_model, state_dim)
        self.control_out_proj = nn.Linear(d_model, control_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlockWithAttention(d_model=d_model, nhead=nhead, ff_dim=ff_dim) for _ in range(num_layers)]
        )
        self.register_buffer("positional_encoding", _build_positional_encoding(context, d_model))
        self._hardcode_parameters()
        self.eval()

    def _hardcode_parameters(self) -> None:
        with torch.no_grad():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(1337)
            for i, (name, parameter) in enumerate(self.named_parameters()):
                index = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
                std = 0.02

                if "in_proj.weight" in name:
                    std = 0.05
                elif "self_attn.in_proj_weight" in name:
                    std = 0.10
                elif "self_attn.out_proj.weight" in name:
                    std = 0.07
                elif "linear1.weight" in name:
                    std = 0.05
                elif "linear2.weight" in name:
                    std = 0.05
                elif "state_out_proj.weight" in name:
                    std = 0.04
                elif "control_out_proj.weight" in name:
                    std = 0.06
                elif "bias" in name:
                    std = 0.015

                if "norm" in name and name.endswith("weight"):
                    # LayerNorm gains must stay near 1.0 to avoid collapsing dynamics.
                    values = 1.0 + 0.05 * torch.sin(index * 0.021 + (i + 1) * 0.13)
                elif "norm" in name and name.endswith("bias"):
                    values = 0.01 * torch.sin(index * 0.017 + (i + 1) * 0.11)
                else:
                    values = torch.randn(
                        parameter.shape,
                        generator=generator,
                        dtype=torch.float32,
                    ) * std
                    values += 0.25 * std * torch.sin(index * 0.013 + (i + 1) * 0.19)

                parameter.copy_(values.to(parameter.dtype))
                parameter.requires_grad_(False)

    def forward(self, state_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        # state_history: [batch, context, state_dim]
        seq_len = state_history.shape[1]
        x = self.in_proj(state_history)
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        attention_maps: list[torch.Tensor] = []
        for block in self.blocks:
            x, attn = block(x)
            attention_maps.append(attn)
        head = x[:, -1, :]
        return self.state_out_proj(head), self.control_out_proj(head), attention_maps


@dataclass(frozen=True)
class EmulationConfig:
    wad_path: Path
    map_name: str
    resolution: ScreenResolution
    frame_width: int
    frame_height: int
    context: int
    frame_pool: int
    visible: bool
    keyboard_source: str
    device: str
    log_interval: int
    max_ticks: int | None
    action_repeat: int
    move_delta: float
    strafe_delta: float
    turn_delta: float
    look_delta: float
    fire_cooldown_tics: int
    doom_ticrate: int
    doom_skill: int
    diagnostic_panel_width: int


class DoomTransformerLoop:
    def __init__(self, config: EmulationConfig) -> None:
        self.config = config
        self.running = True
        self.keyboard_source = "doom_buttons"
        self.keyboard_permission_ok = True
        self.macos_keyboard_sampler: MacOSKeyboardSampler | None = None
        self.pygame_keyboard_sampler: PygameKeyboardSampler | None = None
        self.keyboard_buttons = [
            Button.MOVE_FORWARD,
            Button.MOVE_BACKWARD,
            Button.MOVE_LEFT,
            Button.MOVE_RIGHT,
            Button.TURN_LEFT,
            Button.TURN_RIGHT,
            Button.LOOK_UP,
            Button.LOOK_DOWN,
            Button.ATTACK,
            Button.USE,
        ]
        self.control_buttons = [
            Button.MOVE_FORWARD_BACKWARD_DELTA,
            Button.MOVE_LEFT_RIGHT_DELTA,
            Button.TURN_LEFT_RIGHT_DELTA,
            Button.LOOK_UP_DOWN_DELTA,
            Button.ATTACK,
            Button.USE,
        ]
        self.available_buttons = self.control_buttons + [
            button for button in self.keyboard_buttons if button not in self.control_buttons
        ]
        self.control_dim = len(self.control_buttons)
        self._init_keyboard_source()
        self.game = self._init_game()
        self.state_dim = len(GameVariable.__members__) + self._pooled_pixel_count() + len(
            self.keyboard_buttons
        )
        self.model = HardcodedStateTransformer(
            state_dim=self.state_dim,
            control_dim=self.control_dim,
            context=self.config.context,
        ).to(self.config.device)
        self.history: Deque[np.ndarray] = deque(maxlen=self.config.context)
        self.last_position: tuple[float, float] | None = None
        self.stuck_ticks = 0
        self._last_sampled_keys = np.zeros(len(self.keyboard_buttons), dtype=np.float32)
        self._stale_key_ticks = 0
        self._attack_cooldown_left = 0
        self._last_mse = 0.0
        self.latest_diagnostics: dict[str, Any] = {}
        self._previous_attention_maps: list[np.ndarray] | None = None

    def _keyboard_bindings(self) -> list[tuple[str, str]]:
        return [
            ("w", "+forward"),
            ("z", "+forward"),
            ("s", "+back"),
            ("a", "+moveleft"),
            ("q", "+moveleft"),
            ("d", "+moveright"),
            ("leftarrow", "+left"),
            ("rightarrow", "+right"),
            ("uparrow", "+forward"),
            ("downarrow", "+back"),
            ("e", "+use"),
            ("space", "+attack"),
            ("spacebar", "+attack"),
        ]

    def _init_game(self) -> DoomGame:
        game = DoomGame()
        game.set_doom_game_path(str(self.config.wad_path))
        game.set_doom_map(self.config.map_name)
        game.set_doom_config_path(str(self._ensure_keyboard_config_path()))
        game.set_screen_resolution(self.config.resolution)
        game.set_screen_format(ScreenFormat.RGB24)
        game.set_render_hud(True)
        game.set_render_minimal_hud(False)
        game.set_render_weapon(True)
        window_visible = self.config.visible and self.keyboard_source != "pygame_window"
        game.set_window_visible(window_visible)
        game.set_mode(Mode.PLAYER)
        game.set_ticrate(self.config.doom_ticrate)
        game.set_doom_skill(self.config.doom_skill)
        game.set_sound_enabled(True)
        game.set_episode_timeout(2_147_000_000)
        game.set_available_buttons(self.available_buttons)
        for variable in GameVariable.__members__.values():
            game.add_available_game_variable(variable)
        game.init()
        game.send_game_command("hud_althud 0")
        game.send_game_command("screenblocks 10")
        game.send_game_command("crosshair 0")
        game.send_game_command("cl_run 1")
        game.send_game_command("fastmonsters 0")
        game.send_game_command("sv_fastmonsters 0")
        game.new_episode()
        self._bind_keyboard_controls(game)
        return game

    def _ensure_keyboard_config_path(self) -> Path:
        config_path = Path.cwd() / "transformer_controls.cfg"
        lines = [
            "unbindall",
            "bind w +forward",
            "bind z +forward",
            "bind s +back",
            "bind a +moveleft",
            "bind q +moveleft",
            "bind d +moveright",
            "bind leftarrow +left",
            "bind rightarrow +right",
            "bind uparrow +forward",
            "bind downarrow +back",
            "bind space +attack",
            "bind e +use",
        ]
        config_path.write_text("\n".join(lines) + "\n")
        return config_path

    def _init_keyboard_source(self) -> None:
        preferred = self.config.keyboard_source

        if preferred == "doom_buttons":
            self.keyboard_source = "doom_buttons"
            return

        if preferred == "auto" and self.config.visible:
            try:
                self.pygame_keyboard_sampler = PygameKeyboardSampler(
                    self.config.frame_width,
                    self.config.frame_height,
                    self.config.diagnostic_panel_width,
                )
                self.keyboard_source = "pygame_window"
                return
            except Exception:
                pass

        if preferred == "pygame_window":
            if not self.config.visible:
                raise RuntimeError("keyboard_source=pygame_window requires a visible run (not --headless).")
            try:
                self.pygame_keyboard_sampler = PygameKeyboardSampler(
                    self.config.frame_width,
                    self.config.frame_height,
                    self.config.diagnostic_panel_width,
                )
                self.keyboard_source = "pygame_window"
                return
            except Exception as exc:
                raise RuntimeError("Failed to initialize pygame keyboard window.") from exc

        if self.config.visible:
            try:
                sampler = MacOSKeyboardSampler()
                trusted = sampler.is_accessibility_trusted()
                if trusted:
                    self.macos_keyboard_sampler = sampler
                    self.keyboard_permission_ok = True
                    self.keyboard_source = "macos_global+doom_buttons"
                    return

                self.keyboard_permission_ok = False
                if preferred == "macos_global":
                    raise RuntimeError(
                        "keyboard_source=macos_global requires Accessibility permission."
                    )
                # In auto mode, prefer pygame window over unreliable partial global key reads.
                try:
                    self.pygame_keyboard_sampler = PygameKeyboardSampler(
                        self.config.frame_width,
                        self.config.frame_height,
                        self.config.diagnostic_panel_width,
                    )
                    self.keyboard_source = "pygame_window"
                    return
                except Exception:
                    self.macos_keyboard_sampler = None
                    self.keyboard_source = "doom_buttons"
                    return
            except Exception:
                if preferred == "macos_global":
                    raise RuntimeError(
                        "Requested keyboard_source=macos_global, but initialization failed."
                    )
                if preferred == "auto":
                    try:
                        self.pygame_keyboard_sampler = PygameKeyboardSampler(
                            self.config.frame_width,
                            self.config.frame_height,
                            self.config.diagnostic_panel_width,
                        )
                        self.keyboard_source = "pygame_window"
                        return
                    except Exception:
                        pass
        self.keyboard_source = "doom_buttons"

    def _bind_keyboard_controls(self, game: DoomGame) -> None:
        for key, action in self._keyboard_bindings():
            game.send_game_command(f"bind {key} {action}")
            game.send_game_command(f"bind {key.upper()} {action}")

    def _pooled_pixel_count(self) -> int:
        pooled_h = self.config.frame_height // self.config.frame_pool
        pooled_w = self.config.frame_width // self.config.frame_pool
        return pooled_h * pooled_w

    def _read_keyboard_state(self) -> np.ndarray:
        if self.pygame_keyboard_sampler is not None:
            keys = self.pygame_keyboard_sampler.read()
            if self.pygame_keyboard_sampler.closed:
                self.running = False
            return keys

        values: list[float] = []
        for button in self.keyboard_buttons:
            try:
                values.append(float(self.game.get_button(button)))
            except Exception:
                values.append(0.0)
        doom_keys = np.asarray(values, dtype=np.float32)
        if self.macos_keyboard_sampler is not None:
            mac_keys = self.macos_keyboard_sampler.read()
            return np.maximum(doom_keys, mac_keys)
        return doom_keys

    def _sanitize_keyboard_state(self, keyboard_state: np.ndarray) -> np.ndarray:
        if self.keyboard_source == "pygame_window":
            # Pygame path is event-based (KEYDOWN/KEYUP), so return directly without stale heuristics.
            return keyboard_state

        if np.allclose(keyboard_state, self._last_sampled_keys, atol=1e-6):
            if np.any(keyboard_state > 0.1):
                self._stale_key_ticks += 1
            else:
                self._stale_key_ticks = 0
        else:
            self._stale_key_ticks = 0

        self._last_sampled_keys = keyboard_state.copy()
        if self._stale_key_ticks > 20:
            return np.zeros_like(keyboard_state)
        return keyboard_state

    def _extract_state_vector(self, keyboard_state: np.ndarray | None = None) -> np.ndarray | None:
        state = self.game.get_state()
        if state is None:
            return None
        if keyboard_state is None:
            keyboard_state = np.zeros(len(self.keyboard_buttons), dtype=np.float32)

        variables = np.asarray(state.game_variables, dtype=np.float32)
        frame = np.asarray(state.screen_buffer, dtype=np.float32)
        gray = frame.mean(axis=2)
        p = self.config.frame_pool
        h, w = gray.shape
        h_trim = (h // p) * p
        w_trim = (w // p) * p
        gray = gray[:h_trim, :w_trim]
        pooled = gray.reshape(h_trim // p, p, w_trim // p, p).mean(axis=(1, 3)) / 255.0
        keyboard_features = keyboard_state * 100.0
        vector = np.concatenate([variables, pooled.ravel(), keyboard_features], dtype=np.float32)

        # Keep values in a stable range for deterministic inference.
        vector = np.tanh(vector / 100.0)
        return vector

    def _render_current_frame(self) -> None:
        if self.pygame_keyboard_sampler is None:
            return
        state = self.game.get_state()
        if state is None or state.screen_buffer is None:
            return
        frame = np.asarray(state.screen_buffer)
        self.pygame_keyboard_sampler.render_rgb_frame(frame, diagnostics=self.latest_diagnostics)

    def _attention_maps_from_history(self, history_vectors: list[np.ndarray]) -> list[np.ndarray]:
        if not history_vectors:
            return []
        stacked = np.stack(history_vectors, axis=0).astype(np.float32)
        inputs = torch.from_numpy(stacked).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            _, _, attention_maps = self.model(inputs)
        return [attn.detach().cpu().numpy()[0] for attn in attention_maps]

    def _refresh_live_diagnostics(
        self,
        step: int,
        sub_tick: int,
        reward: float,
        keyboard_state: np.ndarray,
        control_action: list[float],
    ) -> None:
        if self.pygame_keyboard_sampler is None:
            return

        attention_maps_np = self.latest_diagnostics.get("attention_maps", [])
        if self.pygame_keyboard_sampler.show_diagnostics:
            observed = self._extract_state_vector(keyboard_state)
            if observed is not None:
                history_candidate = list(self.history)
                history_candidate.append(observed.copy())
                if len(history_candidate) > self.config.context:
                    history_candidate = history_candidate[-self.config.context :]
                attention_maps_np = self._attention_maps_from_history(history_candidate)
        attention_deltas = self._attention_deltas(attention_maps_np)

        health = self.game.get_game_variable(GameVariable.HEALTH)
        armor = self.game.get_game_variable(GameVariable.ARMOR)
        kills = self.game.get_game_variable(GameVariable.KILLCOUNT)
        active_controls = [
            f"{button.name}={value:.2f}"
            for button, value in zip(self.control_buttons, control_action)
            if abs(value) > 0.05
        ]
        active_keyboard = [
            button.name
            for button, value in zip(self.keyboard_buttons, keyboard_state)
            if value > 0.1
        ]
        self.latest_diagnostics = {
            "tick": step,
            "sub_tick": sub_tick,
            "mse": self._last_mse,
            "reward": reward,
            "health": float(health),
            "armor": float(armor),
            "kills": float(kills),
            "attention_maps": attention_maps_np,
            "attention_deltas": attention_deltas,
            "active_keys": active_keyboard,
            "active_controls": active_controls,
        }

    def _history_tensor(self) -> torch.Tensor:
        stacked = np.stack(list(self.history), axis=0)
        return torch.from_numpy(stacked).unsqueeze(0).to(self.config.device)

    def _prime_history(self) -> None:
        initial_keyboard = self._read_keyboard_state()
        initial = self._extract_state_vector(initial_keyboard)
        if initial is None:
            raise RuntimeError("Could not read initial game state.")
        self._render_current_frame()
        self.history.clear()
        while len(self.history) < self.config.context:
            self.history.append(initial.copy())

    def _current_position(self) -> tuple[float, float]:
        return (
            float(self.game.get_game_variable(GameVariable.POSITION_X)),
            float(self.game.get_game_variable(GameVariable.POSITION_Y)),
        )

    def _update_stuck_counter(self, action: list[float]) -> None:
        position = self._current_position()
        if self.last_position is None:
            self.last_position = position
            self.stuck_ticks = 0
            return

        dx = position[0] - self.last_position[0]
        dy = position[1] - self.last_position[1]
        moved_distance = float(np.hypot(dx, dy))
        movement_intent = abs(action[0]) > 0.05 or abs(action[1]) > 0.05

        if movement_intent and moved_distance < 0.15:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0

        self.last_position = position

    def _decode_controls(
        self, control_logits: np.ndarray, step: int, keyboard_state: np.ndarray
    ) -> list[float]:
        _ = step
        scores = np.tanh(control_logits * 2.0)
        action = [0.0] * self.control_dim

        keys = keyboard_state > 0.1
        movement_keys = bool(keys[0] or keys[1] or keys[2] or keys[3] or keys[4] or keys[5])
        if not np.any(keys):
            return action

        def resolve_delta(
            pos_key_idx: int,
            neg_key_idx: int,
            logit_idx: int,
            axis_strength: float = 1.0,
        ) -> float:
            pos = bool(keys[pos_key_idx])
            neg = bool(keys[neg_key_idx])
            # NN logit magnitude modulates movement/turn strength while keyboard gates intent.
            nn_gain = 0.45 + 0.55 * float(abs(scores[logit_idx]))
            if pos and not neg:
                return axis_strength * nn_gain
            if neg and not pos:
                return -axis_strength * nn_gain
            if pos and neg:
                direction = 1.0 if scores[logit_idx] >= 0.0 else -1.0
                return direction * axis_strength * nn_gain
            return 0.0

        if movement_keys:
            # Keyboard determines whether movement is active; Transformer resolves opposing keys.
            action[0] = resolve_delta(0, 1, 0, axis_strength=self.config.move_delta)  # forward/back
            action[1] = resolve_delta(3, 2, 1, axis_strength=self.config.strafe_delta)  # strafe
            action[2] = resolve_delta(5, 4, 2, axis_strength=self.config.turn_delta)  # turn

        # Look keys are optional.
        action[3] = resolve_delta(7, 6, 3, axis_strength=self.config.look_delta)  # look down/up

        if keys[8]:
            action[4] = 1.0
        if keys[9]:
            action[5] = 1.0

        return action

    def _attention_deltas(self, attention_maps_np: list[np.ndarray]) -> list[float]:
        if self._previous_attention_maps is None or len(self._previous_attention_maps) != len(
            attention_maps_np
        ):
            self._previous_attention_maps = [m.copy() for m in attention_maps_np]
            return [0.0 for _ in attention_maps_np]

        deltas: list[float] = []
        for previous, current in zip(self._previous_attention_maps, attention_maps_np):
            if previous.shape != current.shape:
                deltas.append(0.0)
            else:
                deltas.append(float(np.mean(np.abs(current - previous))))
        self._previous_attention_maps = [m.copy() for m in attention_maps_np]
        return deltas

    def _apply_fire_cooldown(self, action: list[float], keyboard_state: np.ndarray) -> None:
        attack_pressed = bool(keyboard_state[8] > 0.1)
        if not attack_pressed:
            action[4] = 0.0
            return
        if action[4] <= 0.5:
            action[4] = 0.0
            return
        if self._attack_cooldown_left > 0:
            action[4] = 0.0
            return
        self._attack_cooldown_left = self.config.fire_cooldown_tics

    def _make_action_with_fire_pulse(self, action: list[float], repeats: int = 1) -> float:
        if repeats <= 1 or action[4] <= 0.5:
            return self.game.make_action(action, repeats)

        # Emit a single attack pulse, then continue movement/turn for remaining ticks.
        fire_action = action.copy()
        fire_action[4] = 1.0
        reward = self.game.make_action(fire_action, 1)

        if repeats > 1:
            move_only_action = action.copy()
            move_only_action[4] = 0.0
            reward += self.game.make_action(move_only_action, repeats - 1)
        return reward

    def _step_controls_responsive(
        self,
        control_logits: np.ndarray,
        step: int,
        initial_keyboard_state: np.ndarray,
    ) -> tuple[float, list[float], np.ndarray]:
        total_reward = 0.0
        last_action = [0.0] * self.control_dim
        keyboard_state = initial_keyboard_state

        # Re-sample keys every single Doom tic so releasing a key stops immediately.
        for sub_tick in range(max(1, self.config.action_repeat)):
            if sub_tick > 0:
                keyboard_state = self._sanitize_keyboard_state(self._read_keyboard_state())
            self._attack_cooldown_left = max(0, self._attack_cooldown_left - 1)

            if np.any(keyboard_state > 0.1):
                action = self._decode_controls(control_logits, step, keyboard_state)
                self._apply_fire_cooldown(action, keyboard_state)
                total_reward += self._make_action_with_fire_pulse(action, repeats=1)
                last_action = action
            else:
                # Explicit neutral action prevents previous movement/turn from lingering.
                neutral_action = [0.0] * self.control_dim
                total_reward += self.game.make_action(neutral_action, 1)
                last_action = [0.0] * self.control_dim

            self._refresh_live_diagnostics(
                step=step,
                sub_tick=sub_tick,
                reward=total_reward,
                keyboard_state=keyboard_state,
                control_action=last_action,
            )
            self._render_current_frame()

        return total_reward, last_action, keyboard_state

    def run(self) -> None:
        def _handle_signal(_signum: int, _frame: object) -> None:
            self.running = False

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        print("E1M1 live. Close the game window or Ctrl+C in terminal to stop.")
        print("Transformer is authoritative for movement/aim/fire controls.")
        print(
            "Keyboard input is routed through Transformer control decoding "
            f"(source: {self.keyboard_source})."
        )
        print("No sampled key => loop sends explicit neutral action (hard stop).")
        print("Sticky-key protection is enabled.")
        if self.config.visible:
            if self.keyboard_source == "pygame_window":
                print("Focus the 'Transformer Doom (focus this window)' game window for keyboard input.")
            else:
                print("Click the game window once so key states are captured.")
            print("Controls: WASD + arrows for movement/turn, Space for attack, E for use.")
            if self.keyboard_source == "pygame_window":
                print("Press F3 to toggle diagnostics, F4 mean/head view, F5/F6 to change head.")
            if not self.keyboard_permission_ok:
                print(
                    "Accessibility permission is OFF. "
                    "Using doom_buttons fallback to avoid sticky partial keys. "
                    "Enable Accessibility + Input Monitoring for full transformer keyboard handling."
                )
            if self.keyboard_source == "doom_buttons":
                print("Keyboard source is doom_buttons.")
        print(f"Doom ticrate: {self.config.doom_ticrate} | Doom skill: {self.config.doom_skill}")
        print(f"Action repeat: {self.config.action_repeat}")
        print(
            f"Deltas move={self.config.move_delta:.2f} strafe={self.config.strafe_delta:.2f} "
            f"turn={self.config.turn_delta:.2f} look={self.config.look_delta:.2f}"
        )
        print(f"Fire cooldown: {self.config.fire_cooldown_tics} tics")
        print(f"State dim: {self.state_dim} | Transformer context: {self.config.context}")

        self._prime_history()
        self.last_position = self._current_position()

        step = 0
        try:
            while self.running and not self.game.is_episode_finished():
                if self.config.max_ticks is not None and step >= self.config.max_ticks:
                    break

                inputs = self._history_tensor()
                with torch.no_grad():
                    predicted_state, control_logits, attention_maps = self.model(inputs)
                predicted = predicted_state.cpu().numpy()[0]
                control = control_logits.cpu().numpy()[0]
                attention_maps_np = [attn.detach().cpu().numpy()[0] for attn in attention_maps]

                self.latest_diagnostics = {
                    "tick": step,
                    "sub_tick": -1,
                    "mse": self._last_mse,
                    "reward": 0.0,
                    "health": float(self.game.get_game_variable(GameVariable.HEALTH)),
                    "armor": float(self.game.get_game_variable(GameVariable.ARMOR)),
                    "kills": float(self.game.get_game_variable(GameVariable.KILLCOUNT)),
                    "attention_maps": attention_maps_np,
                    "attention_deltas": self._attention_deltas(attention_maps_np),
                    "active_keys": [],
                    "active_controls": [],
                }

                keyboard_state = self._sanitize_keyboard_state(self._read_keyboard_state())
                reward, control_action, next_keyboard = self._step_controls_responsive(
                    control,
                    step,
                    keyboard_state,
                )
                observed_next = self._extract_state_vector(next_keyboard)
                if observed_next is None:
                    continue
                self.history.append(observed_next)
                self._update_stuck_counter(control_action)

                # This is the emulation signal (state-in -> predicted state-out).
                mse = float(np.mean((predicted - observed_next) ** 2))
                self._last_mse = mse
                health = self.game.get_game_variable(GameVariable.HEALTH)
                armor = self.game.get_game_variable(GameVariable.ARMOR)
                kills = self.game.get_game_variable(GameVariable.KILLCOUNT)
                active_controls = [
                    f"{button.name}={value:.2f}"
                    for button, value in zip(self.control_buttons, control_action)
                    if abs(value) > 0.05
                ]
                active_keyboard = [
                    button.name
                    for button, value in zip(self.keyboard_buttons, next_keyboard)
                    if value > 0.1
                ]
                latest_attention = self.latest_diagnostics.get("attention_maps", attention_maps_np)
                latest_deltas = self.latest_diagnostics.get(
                    "attention_deltas", self._attention_deltas(latest_attention)
                )
                self.latest_diagnostics = {
                    "tick": step,
                    "sub_tick": self.config.action_repeat - 1,
                    "mse": mse,
                    "reward": reward,
                    "health": float(health),
                    "armor": float(armor),
                    "kills": float(kills),
                    "attention_maps": latest_attention,
                    "attention_deltas": latest_deltas,
                    "active_keys": active_keyboard,
                    "active_controls": active_controls,
                }
                if step % self.config.log_interval == 0:
                    print(
                        f"tick={step:06d} mse={mse:.6f} "
                        f"health={health:.0f} armor={armor:.0f} kills={kills:.0f} "
                        f"reward={reward:.2f} stuck={self.stuck_ticks} "
                        f"keys={active_keyboard if active_keyboard else '[]'} "
                        f"active={active_controls if active_controls else '[]'}"
                    )
                step += 1
        finally:
            if self.pygame_keyboard_sampler is not None:
                self.pygame_keyboard_sampler.close()
            self.game.close()
            print("Session closed.")


def parse_args() -> EmulationConfig:
    parser = argparse.ArgumentParser(
        description="Run E1M1 with a hardcoded Transformer backend emulator loop."
    )
    parser.add_argument("--wad", type=Path, default=Path("DOOM.WAD"), help="Path to DOOM.WAD")
    parser.add_argument("--map", type=str, default="E1M1", help="Doom map name")
    parser.add_argument(
        "--resolution",
        type=str,
        default="1024x768",
        help="Render resolution as WIDTHxHEIGHT, e.g. 1280x960",
    )
    parser.add_argument("--context", type=int, default=32, help="Transformer context length")
    parser.add_argument(
        "--frame-pool",
        type=int,
        default=16,
        choices=[8, 10, 16, 20],
        help="Spatial pooling factor for frame features",
    )
    parser.add_argument("--headless", action="store_true", help="Disable game window")
    parser.add_argument(
        "--keyboard-source",
        type=str,
        default="auto",
        choices=["auto", "macos_global", "doom_buttons", "pygame_window"],
        help="Keyboard source: auto, macos_global, doom_buttons, or pygame_window.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Torch device",
    )
    parser.add_argument("--log-interval", type=int, default=35, help="Ticks between metric logs")
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Optional maximum ticks before clean exit",
    )
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=5,
        help="VizDoom ticks per control action (higher = faster movement/turn).",
    )
    parser.add_argument(
        "--move-delta",
        type=float,
        default=3.60,
        help="Forward/backward movement delta magnitude.",
    )
    parser.add_argument(
        "--strafe-delta",
        type=float,
        default=3.60,
        help="Left/right strafe delta magnitude.",
    )
    parser.add_argument(
        "--turn-delta",
        type=float,
        default=1.35,
        help="Left/right turn delta magnitude.",
    )
    parser.add_argument(
        "--look-delta",
        type=float,
        default=0.65,
        help="Look up/down delta magnitude.",
    )
    parser.add_argument(
        "--fire-cooldown-tics",
        type=int,
        default=8,
        help="Minimum world tics between attack pulses when holding fire.",
    )
    parser.add_argument(
        "--doom-ticrate",
        type=int,
        default=16,
        help="Global Doom tics per second (lower slows enemies/world).",
    )
    parser.add_argument(
        "--doom-skill",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Doom skill level (1 easiest, 5 hardest).",
    )
    parser.add_argument(
        "--diagnostic-panel-width",
        type=int,
        default=460,
        help="Width in pixels for the right-side NN diagnostics panel in pygame_window mode.",
    )
    args = parser.parse_args()

    wad_path = args.wad.resolve()
    if not wad_path.exists():
        raise FileNotFoundError(f"DOOM WAD not found at: {wad_path}")

    try:
        width_str, height_str = args.resolution.lower().split("x")
        frame_width = int(width_str)
        frame_height = int(height_str)
    except ValueError as exc:
        raise ValueError("Resolution must be in WIDTHxHEIGHT format, e.g. 1024x768") from exc

    resolution_name = f"RES_{frame_width}X{frame_height}"
    if resolution_name not in ScreenResolution.__members__:
        supported = ", ".join(
            name.removeprefix("RES_").replace("X", "x")
            for name in ScreenResolution.__members__.keys()
        )
        raise ValueError(
            f"Unsupported resolution '{args.resolution}'. Supported values: {supported}"
        )

    return EmulationConfig(
        wad_path=wad_path,
        map_name=args.map,
        resolution=ScreenResolution.__members__[resolution_name],
        frame_width=frame_width,
        frame_height=frame_height,
        context=args.context,
        frame_pool=args.frame_pool,
        visible=not args.headless,
        keyboard_source=args.keyboard_source,
        device=args.device,
        log_interval=args.log_interval,
        max_ticks=args.max_ticks,
        action_repeat=max(1, args.action_repeat),
        move_delta=max(0.1, args.move_delta),
        strafe_delta=max(0.1, args.strafe_delta),
        turn_delta=max(0.1, args.turn_delta),
        look_delta=max(0.05, args.look_delta),
        fire_cooldown_tics=max(1, args.fire_cooldown_tics),
        doom_ticrate=max(1, args.doom_ticrate),
        doom_skill=args.doom_skill,
        diagnostic_panel_width=max(0, args.diagnostic_panel_width),
    )


def main() -> None:
    config = parse_args()
    loop = DoomTransformerLoop(config)
    loop.run()


if __name__ == "__main__":
    main()
