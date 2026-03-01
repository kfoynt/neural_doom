#!/usr/bin/env python3
"""Run Doom E1M1 with a deterministic Transformer state emulator in the game loop.

The Doom engine (VizDoom + DOOM.WAD) stays authoritative for gameplay/graphics.
The Transformer receives backend state every tick and emits a predicted next state.
"""

from __future__ import annotations

import argparse
import ctypes
import signal
import struct
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque

import numpy as np
import torch
from torch import nn
from vizdoom import Button, DoomGame, GameVariable, Mode, ScreenFormat, ScreenResolution


class WADCollisionMap:
    """Minimal Doom map collision extractor (blocking linedefs) + 2D motion resolver."""

    def __init__(self, wad_path: Path, map_name: str) -> None:
        self.blocking_segments = self._load_blocking_segments(wad_path, map_name)
        if not self.blocking_segments:
            raise RuntimeError(f"No blocking linedefs found for map {map_name}.")

    @staticmethod
    def _read_lump_directory(raw: bytes) -> list[tuple[str, int, int]]:
        ident, lump_count, dir_offset = struct.unpack_from("<4sii", raw, 0)
        if ident not in (b"IWAD", b"PWAD"):
            raise RuntimeError("Invalid WAD header.")
        lumps: list[tuple[str, int, int]] = []
        for i in range(lump_count):
            offset, size, name_raw = struct.unpack_from("<ii8s", raw, dir_offset + i * 16)
            name = name_raw.split(b"\0", 1)[0].decode("ascii", errors="ignore").upper()
            lumps.append((name, offset, size))
        return lumps

    @staticmethod
    def _load_blocking_segments(wad_path: Path, map_name: str) -> list[tuple[float, float, float, float]]:
        raw = wad_path.read_bytes()
        lumps = WADCollisionMap._read_lump_directory(raw)
        map_name_u = map_name.upper()

        map_index = -1
        for idx, (name, _off, _size) in enumerate(lumps):
            if name == map_name_u:
                map_index = idx
                break
        if map_index < 0:
            raise RuntimeError(f"Map {map_name} not found in {wad_path.name}.")

        local = {name: (off, size) for name, off, size in lumps[map_index + 1 : map_index + 11]}
        if "VERTEXES" not in local or "LINEDEFS" not in local:
            raise RuntimeError(f"Map {map_name} missing VERTEXES/LINEDEFS lumps.")

        v_off, v_size = local["VERTEXES"]
        vertices: list[tuple[float, float]] = []
        for i in range(v_size // 4):
            x, y = struct.unpack_from("<hh", raw, v_off + i * 4)
            vertices.append((float(x), float(y)))

        l_off, l_size = local["LINEDEFS"]
        segments: list[tuple[float, float, float, float]] = []
        for i in range(l_size // 14):
            v1, v2, flags, _special, _tag, right, left = struct.unpack_from("<hhhhhhh", raw, l_off + i * 14)
            if v1 < 0 or v2 < 0 or v1 >= len(vertices) or v2 >= len(vertices):
                continue
            one_sided = right == -1 or left == -1
            impassable = bool(flags & 0x0001)  # ML_BLOCKING
            if not one_sided and not impassable:
                continue
            x1, y1 = vertices[v1]
            x2, y2 = vertices[v2]
            segments.append((x1, y1, x2, y2))
        return segments

    @staticmethod
    def _point_segment_distance_sq(
        px: float,
        py: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float:
        vx = x2 - x1
        vy = y2 - y1
        wx = px - x1
        wy = py - y1
        c1 = vx * wx + vy * wy
        if c1 <= 0.0:
            dx = px - x1
            dy = py - y1
            return dx * dx + dy * dy
        c2 = vx * vx + vy * vy
        if c2 <= 1e-6:
            dx = px - x1
            dy = py - y1
            return dx * dx + dy * dy
        if c2 <= c1:
            dx = px - x2
            dy = py - y2
            return dx * dx + dy * dy
        b = c1 / c2
        bx = x1 + b * vx
        by = y1 + b * vy
        dx = px - bx
        dy = py - by
        return dx * dx + dy * dy

    def _collides(
        self,
        x: float,
        y: float,
        radius: float,
        dynamic_circles: list[tuple[float, float, float]] | None = None,
    ) -> bool:
        r2 = radius * radius
        for x1, y1, x2, y2 in self.blocking_segments:
            if self._point_segment_distance_sq(x, y, x1, y1, x2, y2) < r2:
                return True
        if dynamic_circles is not None:
            for cx, cy, cr in dynamic_circles:
                rr = radius + cr
                if (x - cx) * (x - cx) + (y - cy) * (y - cy) < rr * rr:
                    return True
        return False

    def resolve_motion(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        radius: float,
        dynamic_circles: list[tuple[float, float, float]] | None = None,
    ) -> tuple[float, float]:
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return x, y

        nx = x + dx
        ny = y + dy
        if not self._collides(nx, ny, radius, dynamic_circles):
            return nx, ny

        x_only = (x + dx, y)
        y_only = (x, y + dy)
        x_ok = not self._collides(x_only[0], x_only[1], radius, dynamic_circles)
        y_ok = not self._collides(y_only[0], y_only[1], radius, dynamic_circles)
        if x_ok and y_ok:
            if abs(dx) >= abs(dy):
                return x_only
            return y_only
        if x_ok:
            return x_only
        if y_ok:
            return y_only

        for scale in (0.5, 0.33, 0.25, 0.125):
            sx = x + dx * scale
            sy = y + dy * scale
            if not self._collides(sx, sy, radius, dynamic_circles):
                return sx, sy
            sx_only = x + dx * scale
            sy_only = y + dy * scale
            if not self._collides(sx_only, y, radius, dynamic_circles):
                return sx_only, y
            if not self._collides(x, sy_only, radius, dynamic_circles):
                return x, sy_only

        return x, y


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

    def __init__(self, width: int, height: int) -> None:
        import pygame  # Imported lazily so headless usage keeps working.

        self._pg = pygame
        self._pg.init()
        self._width = width
        self._height = height
        self._window = self._pg.display.set_mode((width, height))
        self._pg.display.set_caption("Transformer Doom (focus this window)")
        self._pressed_keys: set[int] = set()
        self.closed = False

    def read(self) -> np.ndarray:
        for event in self._pg.event.get():
            if event.type == self._pg.QUIT:
                self.closed = True
                self._pressed_keys.clear()
            elif event.type == self._pg.KEYDOWN:
                self._pressed_keys.add(int(event.key))
            elif event.type == self._pg.KEYUP:
                self._pressed_keys.discard(int(event.key))
            else:
                focus_lost_type = getattr(self._pg, "WINDOWFOCUSLOST", None)
                if focus_lost_type is not None and event.type == focus_lost_type:
                    self._pressed_keys.clear()

        if not self._pg.key.get_focused():
            self._pressed_keys.clear()

        key_state = self._pg.key.get_pressed()

        def _down(keycode: int) -> bool:
            # Prefer live keyboard snapshot; fallback to event-tracked set.
            if 0 <= int(keycode) < len(key_state):
                return bool(key_state[int(keycode)])
            return bool(keycode in self._pressed_keys)

        forward = bool(
            _down(self._pg.K_w)
            or _down(self._pg.K_z)
            or _down(self._pg.K_UP)
        )
        backward = bool(_down(self._pg.K_s) or _down(self._pg.K_DOWN))
        move_left = bool(_down(self._pg.K_a) or _down(self._pg.K_q))
        move_right = bool(_down(self._pg.K_d))
        turn_left = bool(_down(self._pg.K_LEFT))
        turn_right = bool(_down(self._pg.K_RIGHT))
        look_up = False
        look_down = False
        attack = bool(
            _down(self._pg.K_SPACE)
            or _down(self._pg.K_LCTRL)
            or _down(self._pg.K_RCTRL)
        )
        use = bool(_down(self._pg.K_e))

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

    def render_rgb_frame(self, frame: np.ndarray) -> None:
        if self.closed:
            return
        # frame is HxWx3 RGB from VizDoom.
        surface = self._pg.image.frombuffer(frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")
        if frame.shape[1] != self._width or frame.shape[0] != self._height:
            surface = self._pg.transform.scale(surface, (self._width, self._height))
        self._window.blit(surface, (0, 0))
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
    """Transformer block that returns attention weights."""

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
        enemy_slots: int = 16,
        enemy_cmd_dim: int = 1,
        enemy_intent_dim: int = 1,
        low_level_dim: int | None = None,
        memory_update_dim: int = 1,
        weight_scale: float = 0.55,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
    ) -> None:
        super().__init__()
        self.enemy_slots = enemy_slots
        self.enemy_cmd_dim = enemy_cmd_dim
        self.enemy_intent_dim = max(1, int(enemy_intent_dim))
        self.low_level_dim = low_level_dim if low_level_dim is not None else (4 + enemy_slots)
        self.memory_update_dim = max(1, int(memory_update_dim))
        self.weight_scale = float(np.clip(weight_scale, 0.05, 2.0))
        self.in_proj = nn.Linear(state_dim, d_model)
        self.state_out_proj = nn.Linear(d_model, state_dim)
        self.control_out_proj = nn.Linear(d_model, control_dim)
        self.enemy_out_proj = nn.Linear(d_model, enemy_slots * enemy_cmd_dim)
        self.enemy_intent_out_proj = nn.Linear(d_model, enemy_slots * self.enemy_intent_dim)
        self.low_level_out_proj = nn.Linear(d_model, self.low_level_dim)
        self.memory_out_proj = nn.Linear(d_model, enemy_slots * self.memory_update_dim)
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
                elif "enemy_out_proj.weight" in name:
                    std = 0.03
                elif "enemy_intent_out_proj.weight" in name:
                    std = 0.03
                elif "low_level_out_proj.weight" in name:
                    std = 0.02
                elif "memory_out_proj.weight" in name:
                    std = 0.03
                elif "bias" in name:
                    std = 0.015
                std *= self.weight_scale

                if "norm" in name and name.endswith("weight"):
                    # LayerNorm gains must stay near 1.0 to avoid collapsing dynamics.
                    values = 1.0 + (0.03 * self.weight_scale) * torch.sin(index * 0.021 + (i + 1) * 0.13)
                elif "norm" in name and name.endswith("bias"):
                    values = (0.006 * self.weight_scale) * torch.sin(index * 0.017 + (i + 1) * 0.11)
                else:
                    values = torch.randn(
                        parameter.shape,
                        generator=generator,
                        dtype=torch.float32,
                    ) * std
                    values += 0.12 * std * torch.sin(index * 0.013 + (i + 1) * 0.19)

                parameter.copy_(values.to(parameter.dtype))
                parameter.requires_grad_(False)

    def forward(
        self, state_history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        # state_history: [batch, context, state_dim]
        seq_len = state_history.shape[1]
        x = self.in_proj(state_history)
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        attention_maps: list[torch.Tensor] = []
        for block in self.blocks:
            x, attn = block(x)
            attention_maps.append(attn)
        head = x[:, -1, :]
        enemy_logits = self.enemy_out_proj(head).view(-1, self.enemy_slots, self.enemy_cmd_dim)
        enemy_intent_logits = self.enemy_intent_out_proj(head).view(-1, self.enemy_slots, self.enemy_intent_dim)
        low_level_logits = self.low_level_out_proj(head)
        memory_update_logits = self.memory_out_proj(head).view(-1, self.enemy_slots, self.memory_update_dim)
        return (
            self.state_out_proj(head),
            self.control_out_proj(head),
            enemy_logits,
            enemy_intent_logits,
            low_level_logits,
            memory_update_logits,
            attention_maps,
        )


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
    enemy_backend_transformer: bool
    enemy_backend_mod: Path
    enemy_slots: int
    nn_movement_resolution: bool
    nn_move_units: float
    nn_player_radius: float
    nn_weight_scale: float
    nn_control_gain: float
    nn_enemy_gain: float
    nn_low_level_gain: float


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
        self.enemy_behavior_core_channels = (
            "speed_cmd",
            "advance_cmd",
            "strafe_cmd",
            "turn_cmd",
            "aim_cmd_logit",
            "fire_cmd_logit",
            "advance_conf",
            "strafe_conf",
            "turn_conf",
            "aim_conf",
            "fire_conf",
            "firecd_cmd_norm",
            "health_cmd_norm",
            "target_hold_gate",
            "retarget_gate",
        )
        self.enemy_target_channels = ("target_player_logit",) + tuple(
            f"target_slot_{slot:02d}_logit" for slot in range(self.config.enemy_slots)
        )
        self.enemy_behavior_channels = self.enemy_behavior_core_channels + self.enemy_target_channels
        self.enemy_behavior_core_dim = len(self.enemy_behavior_core_channels)
        self.enemy_target_dim = len(self.enemy_target_channels)
        self.enemy_cmd_dim = len(self.enemy_behavior_channels)
        self.enemy_intent_channels = (
            "intent_chase_logit",
            "intent_flank_logit",
            "intent_retreat_logit",
            "intent_hold_logit",
            "intent_timer_norm",
        )
        self.enemy_intent_dim = len(self.enemy_intent_channels)
        self.enemy_intent_names = ("chase", "flank", "retreat", "hold")
        self.enemy_base_feature_dim = 11
        self.enemy_feedback_feature_dim = 11
        self.enemy_memory_feature_dim = 8
        self.enemy_memory_update_dim = 1 + self.enemy_memory_feature_dim
        self.enemy_feature_dim = (
            self.enemy_base_feature_dim + self.enemy_feedback_feature_dim + self.enemy_memory_feature_dim
        )
        self.enemy_feature_dim_total = self.config.enemy_slots * self.enemy_feature_dim
        self.low_level_player_dim = 4
        self.low_level_enemy_dim = 1
        self.low_level_dim = self.low_level_player_dim + self.config.enemy_slots * self.low_level_enemy_dim
        self.enemy_name_tokens = (
            "zombie",
            "shotgun",
            "imp",
            "demon",
            "spectre",
            "cacodemon",
            "baron",
            "hell knight",
            "lost soul",
            "pain elemental",
            "chaingun",
            "revenant",
            "mancubus",
            "arachnotron",
            "archvile",
            "spider mastermind",
            "cyberdemon",
        )
        self._last_enemy_cmds: list[dict[str, int]] = [
            {
                "speed": -1,
                "fwd": -999,
                "side": -999,
                "turn": -999,
                "aim": -1,
                "fire": -1,
                "firecd": -1,
                "healthpct": -1,
            }
            for _ in range(self.config.enemy_slots)
        ]
        self._enemy_slot_to_id: list[int | None] = [None for _ in range(self.config.enemy_slots)]
        self._enemy_id_to_slot: dict[int, int] = {}
        self._enemy_prev_x = np.full(self.config.enemy_slots, np.nan, dtype=np.float32)
        self._enemy_prev_y = np.full(self.config.enemy_slots, np.nan, dtype=np.float32)
        self._enemy_prev_angle = np.full(self.config.enemy_slots, np.nan, dtype=np.float32)
        self._enemy_prev_los = np.zeros(self.config.enemy_slots, dtype=np.float32)
        self._enemy_last_cmd = np.zeros((self.config.enemy_slots, 6), dtype=np.float32)
        self._enemy_intent_state = np.zeros(self.config.enemy_slots, dtype=np.int32)
        self._enemy_intent_timer = np.zeros(self.config.enemy_slots, dtype=np.float32)
        self._enemy_memory = np.zeros(
            (self.config.enemy_slots, self.enemy_memory_feature_dim),
            dtype=np.float32,
        )
        self._nn_movement_resolution_active = bool(self.config.nn_movement_resolution)
        self._init_keyboard_source()
        if self._nn_movement_resolution_active and self.keyboard_source == "doom_buttons":
            # doom_buttons requires native binds, which conflicts with Transformer-side movement warping.
            self._nn_movement_resolution_active = False
        self.game = self._init_game()
        self.state_dim = (
            len(GameVariable.__members__)
            + self._pooled_pixel_count()
            + len(self.keyboard_buttons)
            + self.enemy_feature_dim_total
        )
        self.model = HardcodedStateTransformer(
            state_dim=self.state_dim,
            control_dim=self.control_dim,
            context=self.config.context,
            enemy_slots=self.config.enemy_slots,
            enemy_cmd_dim=self.enemy_cmd_dim,
            enemy_intent_dim=self.enemy_intent_dim,
            low_level_dim=self.low_level_dim,
            memory_update_dim=self.enemy_memory_update_dim,
            weight_scale=self.config.nn_weight_scale,
        ).to(self.config.device)
        self.history: Deque[np.ndarray] = deque(maxlen=self.config.context)
        self.last_position: tuple[float, float] | None = None
        self.stuck_ticks = 0
        self._last_sampled_keys = np.zeros(len(self.keyboard_buttons), dtype=np.float32)
        self._stale_key_ticks = 0
        self._attack_cooldown_left = 0
        self._control_logit_gain = self.config.nn_control_gain
        self._enemy_logit_gain = self.config.nn_enemy_gain
        self._low_level_logit_gain = self.config.nn_low_level_gain
        self._runtime_move_scale = 1.0
        self._runtime_turn_scale = 1.0
        self._runtime_fire_cooldown_tics = self.config.fire_cooldown_tics
        self._enemy_target_choice = np.zeros(self.config.enemy_slots, dtype=np.int32)
        self._map_geometry: WADCollisionMap | None = None
        try:
            self._map_geometry = WADCollisionMap(self.config.wad_path, self.config.map_name)
        except Exception as exc:
            if self._nn_movement_resolution_active:
                print(f"Warning: NN movement resolution disabled ({exc}).")
            self._map_geometry = None
        self._collision_map: WADCollisionMap | None = (
            self._map_geometry if self._nn_movement_resolution_active else None
        )

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
        if self.config.enemy_backend_transformer:
            game.set_doom_scenario_path(str(self.config.enemy_backend_mod))
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
        if self.config.enemy_backend_transformer or self.enemy_feature_dim_total > 0:
            game.set_objects_info_enabled(True)
        if self.config.enemy_backend_transformer:
            game.set_labels_buffer_enabled(False)
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
        if self.config.enemy_backend_transformer:
            game.send_game_command("set nn_enemy_override true")
            game.send_game_command(f"set nn_enemy_slot_count {self.config.enemy_slots}")
        game.new_episode()
        self._bind_keyboard_controls(game)
        return game

    def _ensure_keyboard_config_path(self) -> Path:
        config_path = Path.cwd() / "transformer_controls.cfg"
        if self._nn_movement_resolution_active and self.keyboard_source != "doom_buttons":
            # Keep gameplay input authoritative in Transformer loop (no direct Doom movement binds).
            lines = ["unbindall"]
        else:
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
                        )
                        self.keyboard_source = "pygame_window"
                        return
                    except Exception:
                        pass
        self.keyboard_source = "doom_buttons"

    def _bind_keyboard_controls(self, game: DoomGame) -> None:
        if self._nn_movement_resolution_active and self.keyboard_source != "doom_buttons":
            return
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
        if self.keyboard_source in {"pygame_window", "doom_buttons"}:
            # Pygame is event-based and doom_buttons already reflects engine-side key state.
            # Do not apply stale-key heuristics here because it can suppress valid held input.
            return keyboard_state

        if np.allclose(keyboard_state, self._last_sampled_keys, atol=1e-6):
            if np.any(keyboard_state > 0.1):
                self._stale_key_ticks += 1
            else:
                self._stale_key_ticks = 0
        else:
            self._stale_key_ticks = 0

        self._last_sampled_keys = keyboard_state.copy()
        # macos_global fallback can occasionally miss key-up events; keep a long timeout
        # to clear pathological stuck states without breaking normal holds.
        if self._stale_key_ticks > 240:
            return np.zeros_like(keyboard_state)
        return keyboard_state

    @staticmethod
    def _normalize_angle_deg(angle: float) -> float:
        return ((angle + 180.0) % 360.0) - 180.0

    @staticmethod
    def _orientation(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    def _segments_intersect(
        self,
        ax: float,
        ay: float,
        bx: float,
        by: float,
        cx: float,
        cy: float,
        dx: float,
        dy: float,
    ) -> bool:
        o1 = self._orientation(ax, ay, bx, by, cx, cy)
        o2 = self._orientation(ax, ay, bx, by, dx, dy)
        o3 = self._orientation(cx, cy, dx, dy, ax, ay)
        o4 = self._orientation(cx, cy, dx, dy, bx, by)
        return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)

    def _has_line_of_sight_2d(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        if self._map_geometry is None:
            return False
        for sx1, sy1, sx2, sy2 in self._map_geometry.blocking_segments:
            if self._segments_intersect(x1, y1, x2, y2, sx1, sy1, sx2, sy2):
                return False
        return True

    def _enemy_objects_from_state(self, state: object | None) -> list[object]:
        if state is None:
            return []
        objects = getattr(state, "objects", None)
        if objects is None:
            return []
        enemies = [obj for obj in objects if self._is_enemy_object_name(str(obj.name))]
        enemies.sort(key=lambda obj: int(obj.id))
        return enemies

    def _refresh_enemy_slot_assignments(self, state: object | None = None) -> list[object | None]:
        enemies = self._enemy_objects_from_state(state)
        by_id: dict[int, object] = {int(enemy.id): enemy for enemy in enemies}

        for enemy_id, slot in list(self._enemy_id_to_slot.items()):
            if enemy_id in by_id:
                continue
            self._enemy_id_to_slot.pop(enemy_id, None)
            if 0 <= slot < self.config.enemy_slots and self._enemy_slot_to_id[slot] == enemy_id:
                self._enemy_slot_to_id[slot] = None
                self._enemy_prev_x[slot] = np.nan
                self._enemy_prev_y[slot] = np.nan
                self._enemy_prev_angle[slot] = np.nan
                self._enemy_prev_los[slot] = 0.0
                self._enemy_last_cmd[slot, :] = 0.0
                self._enemy_intent_state[slot] = 0
                self._enemy_intent_timer[slot] = 0.0
                self._enemy_target_choice[slot] = 0
                self._enemy_memory[slot, :] = 0.0

        for enemy in enemies:
            enemy_id = int(enemy.id)
            if enemy_id in self._enemy_id_to_slot:
                continue
            free_slot = next((i for i, slot_id in enumerate(self._enemy_slot_to_id) if slot_id is None), None)
            if free_slot is None:
                break
            self._enemy_slot_to_id[free_slot] = enemy_id
            self._enemy_id_to_slot[enemy_id] = free_slot

        slot_enemies: list[object | None] = [None for _ in range(self.config.enemy_slots)]
        for slot, enemy_id in enumerate(self._enemy_slot_to_id):
            if enemy_id is None:
                continue
            enemy = by_id.get(enemy_id)
            if enemy is None:
                self._enemy_slot_to_id[slot] = None
                self._enemy_id_to_slot.pop(enemy_id, None)
                self._enemy_prev_x[slot] = np.nan
                self._enemy_prev_y[slot] = np.nan
                self._enemy_prev_angle[slot] = np.nan
                self._enemy_prev_los[slot] = 0.0
                self._enemy_last_cmd[slot, :] = 0.0
                self._enemy_intent_state[slot] = 0
                self._enemy_intent_timer[slot] = 0.0
                self._enemy_target_choice[slot] = 0
                self._enemy_memory[slot, :] = 0.0
                continue
            slot_enemies[slot] = enemy

        return slot_enemies

    def _enemy_feature_block(self, state: object | None) -> np.ndarray:
        block = np.zeros(self.enemy_feature_dim_total, dtype=np.float32)
        slot_enemies = self._refresh_enemy_slot_assignments(state)
        player_x, player_y = self._current_position()

        for slot, enemy in enumerate(slot_enemies):
            base = slot * self.enemy_feature_dim
            if enemy is None:
                continue

            ex = float(enemy.position_x)
            ey = float(enemy.position_y)
            vx = float(enemy.velocity_x)
            vy = float(enemy.velocity_y)
            angle = self._normalize_angle_deg(float(enemy.angle))
            dx = ex - player_x
            dy = ey - player_y
            dist = float(np.hypot(dx, dy))
            player_bearing = self._normalize_angle_deg(float(np.degrees(np.arctan2(player_y - ey, player_x - ex))))
            los = 100.0 if self._has_line_of_sight_2d(ex, ey, player_x, player_y) else 0.0

            health_proxy = float(
                self._last_enemy_cmds[slot]["healthpct"] if self._last_enemy_cmds[slot]["healthpct"] >= 0 else 100
            )
            cooldown_proxy = float(
                self._last_enemy_cmds[slot]["firecd"] if self._last_enemy_cmds[slot]["firecd"] >= 0 else 12
            )

            # Per-slot feature layout:
            # Base: alive, x, y, vx, vy, angle, health_proxy, dist_to_player, bearing_to_player, line_of_sight, cooldown_proxy
            # Feedback: last command + observed response
            # Memory: persistent latent updated by Transformer memory gate+delta outputs.
            block[base + 0] = 100.0
            block[base + 1] = float(np.clip(dx, -4096.0, 4096.0))
            block[base + 2] = float(np.clip(dy, -4096.0, 4096.0))
            block[base + 3] = float(np.clip(vx * 64.0, -512.0, 512.0))
            block[base + 4] = float(np.clip(vy * 64.0, -512.0, 512.0))
            block[base + 5] = angle
            block[base + 6] = float(np.clip(health_proxy, 0.0, 400.0))
            block[base + 7] = float(np.clip(dist, 0.0, 4096.0))
            block[base + 8] = player_bearing
            block[base + 9] = los
            block[base + 10] = float(np.clip(cooldown_proxy * 8.0, 0.0, 256.0))

            prev_x = float(self._enemy_prev_x[slot])
            prev_y = float(self._enemy_prev_y[slot])
            prev_angle = float(self._enemy_prev_angle[slot])
            prev_los = float(self._enemy_prev_los[slot])
            if np.isfinite(prev_x) and np.isfinite(prev_y):
                moved_dist = float(np.hypot(ex - prev_x, ey - prev_y))
            else:
                moved_dist = 0.0
            if np.isfinite(prev_angle):
                turn_delta = self._normalize_angle_deg(angle - prev_angle)
            else:
                turn_delta = 0.0
            los_changed = 100.0 if abs(los - prev_los) > 1e-3 else 0.0

            cmd_speed = float(self._enemy_last_cmd[slot, 0])
            cmd_fwd = float(self._enemy_last_cmd[slot, 1])
            cmd_side = float(self._enemy_last_cmd[slot, 2])
            cmd_turn = float(self._enemy_last_cmd[slot, 3])
            cmd_aim = float(self._enemy_last_cmd[slot, 4])
            cmd_fire = float(self._enemy_last_cmd[slot, 5])
            move_intent = abs(cmd_fwd) + abs(cmd_side)
            blocked = 100.0 if (move_intent > 20.0 and moved_dist < 1.25) else 0.0
            shot_fired = 100.0 if cmd_fire > 0.5 else 0.0

            fb = base + self.enemy_base_feature_dim
            # Feedback: last_cmd(speed,fwd,side,turn,aim,fire), moved_dist, turn_delta, los_changed, blocked, shot_fired
            block[fb + 0] = float(np.clip(cmd_speed - 100.0, -100.0, 150.0))
            block[fb + 1] = float(np.clip(cmd_fwd, -100.0, 100.0))
            block[fb + 2] = float(np.clip(cmd_side, -100.0, 100.0))
            block[fb + 3] = float(np.clip(cmd_turn, -120.0, 120.0))
            block[fb + 4] = 100.0 if cmd_aim > 0.5 else 0.0
            block[fb + 5] = 100.0 if cmd_fire > 0.5 else 0.0
            block[fb + 6] = float(np.clip(moved_dist * 8.0, 0.0, 256.0))
            block[fb + 7] = float(np.clip(turn_delta, -180.0, 180.0))
            block[fb + 8] = los_changed
            block[fb + 9] = blocked
            block[fb + 10] = shot_fired

            mb = fb + self.enemy_feedback_feature_dim
            # Memory latent channels (NN state with intent/timer diagnostics overlaid).
            memory = np.clip(self._enemy_memory[slot], -1.0, 1.0)
            block[mb : mb + self.enemy_memory_feature_dim] = memory * 100.0

            self._enemy_prev_x[slot] = ex
            self._enemy_prev_y[slot] = ey
            self._enemy_prev_angle[slot] = angle
            self._enemy_prev_los[slot] = los

        return block

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
        enemy_features = self._enemy_feature_block(state)
        vector = np.concatenate([variables, pooled.ravel(), keyboard_features, enemy_features], dtype=np.float32)

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
        self.pygame_keyboard_sampler.render_rgb_frame(frame)

    def _history_tensor(self) -> torch.Tensor:
        stacked = np.stack(list(self.history), axis=0)
        return torch.from_numpy(stacked).unsqueeze(0).to(self.config.device)

    def _is_enemy_object_name(self, name: str) -> bool:
        lowered = name.lower()
        return any(token in lowered for token in self.enemy_name_tokens)

    def _enemy_objects(self) -> list[object]:
        state = self.game.get_state()
        return self._enemy_objects_from_state(state)

    def _enemy_collision_circles(self) -> list[tuple[float, float, float]]:
        circles: list[tuple[float, float, float]] = []
        for enemy in self._enemy_objects():
            circles.append((float(enemy.position_x), float(enemy.position_y), 20.0))
        return circles

    def _select_enemy_target_point(
        self,
        slot: int,
        slot_enemies: list[object | None],
        target_logits: np.ndarray,
        player_x: float,
        player_y: float,
        hold_gate: float,
        retarget_gate: float,
    ) -> tuple[float, float, int, float]:
        candidate_points: list[tuple[float, float]] = [(player_x, player_y)]
        valid: list[bool] = [True]
        for target_slot in range(self.config.enemy_slots):
            target_enemy = slot_enemies[target_slot] if target_slot < len(slot_enemies) else None
            if target_enemy is None or target_slot == slot:
                candidate_points.append((player_x, player_y))
                valid.append(False)
            else:
                candidate_points.append((float(target_enemy.position_x), float(target_enemy.position_y)))
                valid.append(True)

        logits = np.asarray(target_logits, dtype=np.float32).reshape(-1)
        if logits.size < self.enemy_target_dim:
            padded = np.zeros(self.enemy_target_dim, dtype=np.float32)
            padded[: logits.size] = logits
            logits = padded
        else:
            logits = logits[: self.enemy_target_dim]

        masked = logits.copy()
        for idx, is_valid in enumerate(valid):
            if not is_valid:
                masked[idx] = -1e9

        selected_raw = int(np.argmax(masked))
        if not np.isfinite(float(masked[selected_raw])):
            selected_raw = 0
        previous = int(self._enemy_target_choice[slot])
        selected = selected_raw
        prev_valid = 0 <= previous < len(valid) and valid[previous]
        hold = float(np.clip(hold_gate, 0.0, 1.0))
        retarget = float(np.clip(retarget_gate, 0.0, 1.0))
        # Direct NN gate decode: hold keeps previous target, retarget takes fresh argmax.
        if prev_valid and hold >= 0.5 and retarget < 0.5:
            selected = previous
        self._enemy_target_choice[slot] = selected

        confidence = 0.0
        valid_indices = [idx for idx, is_valid in enumerate(valid) if is_valid]
        if valid_indices and 0 <= selected < len(valid):
            scores = np.asarray([masked[idx] for idx in valid_indices], dtype=np.float32)
            scores = scores - float(np.max(scores))
            probs = np.exp(scores)
            probs_sum = float(np.sum(probs))
            if probs_sum > 1e-8:
                probs /= probs_sum
                try:
                    sel_pos = valid_indices.index(selected)
                    confidence = float(probs[sel_pos])
                except ValueError:
                    confidence = 0.0

        target_x, target_y = candidate_points[selected]
        return float(target_x), float(target_y), selected, confidence

    def _update_enemy_memory_state(
        self,
        slot: int,
        memory_update_logits: np.ndarray,
    ) -> None:
        if slot < 0 or slot >= self.config.enemy_slots:
            return
        memory = self._enemy_memory[slot]
        if memory.shape[0] != self.enemy_memory_feature_dim:
            return

        update = np.asarray(memory_update_logits, dtype=np.float32).reshape(-1)
        if update.size < self.enemy_memory_update_dim:
            padded = np.zeros(self.enemy_memory_update_dim, dtype=np.float32)
            padded[: update.size] = update
            update = padded
        else:
            update = update[: self.enemy_memory_update_dim]

        gate = float(np.clip(0.5 + 0.5 * np.tanh(float(update[0])), 0.02, 0.98))
        delta = np.tanh(update[1 : 1 + self.enemy_memory_feature_dim])
        # Fully NN-driven memory evolution: gate + delta are produced by Transformer.
        memory[:] = np.tanh((1.0 - gate) * memory + gate * delta)

    def _decode_enemy_intent_state(
        self,
        slot: int,
        intent_logits: np.ndarray,
    ) -> tuple[int, float, np.ndarray]:
        values = np.asarray(intent_logits, dtype=np.float32).reshape(-1)
        if values.size < self.enemy_intent_dim:
            padded = np.zeros(self.enemy_intent_dim, dtype=np.float32)
            padded[: values.size] = values
            values = padded
        else:
            values = values[: self.enemy_intent_dim]

        intent_scores = values[: len(self.enemy_intent_names)] * self._enemy_logit_gain
        timer_norm = float(np.clip(0.5 + 0.5 * np.tanh(float(values[-1])), 0.0, 1.0))
        scores = intent_scores.astype(np.float32)
        scores = scores - float(np.max(scores))
        probs = np.exp(scores)
        denom = float(np.sum(probs))
        if denom > 1e-8:
            probs /= denom
        else:
            probs[:] = 0.0
            probs[0] = 1.0
        current = int(np.argmax(probs))
        # Timer is NN-authoritative each tick (no Python countdown/hysteresis).
        timer_left = float(np.clip(round(2.0 + 30.0 * timer_norm), 2, 32))
        self._enemy_intent_state[slot] = current
        self._enemy_intent_timer[slot] = timer_left
        return current, timer_norm, probs

    def _apply_player_motion_resolution(self, action: list[float]) -> bool:
        if self._collision_map is None:
            return False
        fwd = float(action[0])
        side = float(action[1])
        if abs(fwd) < 1e-5 and abs(side) < 1e-5:
            return False

        x, y = self._current_position()
        angle_deg = float(self.game.get_game_variable(GameVariable.ANGLE))
        angle_rad = np.deg2rad(angle_deg)
        cos_a = float(np.cos(angle_rad))
        sin_a = float(np.sin(angle_rad))

        step_scale = self.config.nn_move_units
        dx = (cos_a * fwd - sin_a * side) * step_scale
        dy = (sin_a * fwd + cos_a * side) * step_scale
        tx = x + dx
        ty = y + dy

        dynamic = self._enemy_collision_circles()
        nx, ny = self._collision_map.resolve_motion(
            x,
            y,
            dx,
            dy,
            radius=self.config.nn_player_radius,
            dynamic_circles=dynamic,
        )
        # Only apply warp when collision resolution meaningfully differs from
        # intended free-space motion. Native Doom movement remains active.
        corrected = (abs(nx - tx) > 1e-3) or (abs(ny - ty) > 1e-3)
        moved = (abs(nx - x) > 1e-4) or (abs(ny - y) > 1e-4)
        if corrected and moved:
            self.game.send_game_command(f"warp {nx:.3f} {ny:.3f}")
            return True
        return False

    def _apply_enemy_backend_commands(
        self,
        enemy_logits: np.ndarray,
        enemy_intent_logits: np.ndarray | None = None,
        memory_update_logits: np.ndarray | None = None,
    ) -> tuple[int, list[tuple[int, int, int, int, int, int]]]:
        if not self.config.enemy_backend_transformer:
            return 0, []

        state = self.game.get_state()
        slot_enemies = self._refresh_enemy_slot_assignments(state)
        player_x, player_y = self._current_position()
        enemy_circles_all = [
            (float(enemy.position_x), float(enemy.position_y), float(enemy.id))
            for enemy in slot_enemies
            if enemy is not None
        ]
        commands: list[tuple[int, int, int, int, int, int]] = []
        if memory_update_logits is None:
            memory_updates = np.zeros(
                (self.config.enemy_slots, self.enemy_memory_update_dim),
                dtype=np.float32,
            )
        else:
            mem = np.asarray(memory_update_logits, dtype=np.float32)
            if mem.ndim == 1:
                mem = mem.reshape(1, -1)
            if mem.shape[0] < self.config.enemy_slots:
                padded = np.zeros((self.config.enemy_slots, mem.shape[1]), dtype=np.float32)
                padded[: mem.shape[0], :] = mem
                mem = padded
            memory_updates = mem
        if enemy_intent_logits is None:
            intent_updates = np.zeros(
                (self.config.enemy_slots, self.enemy_intent_dim),
                dtype=np.float32,
            )
        else:
            intent = np.asarray(enemy_intent_logits, dtype=np.float32)
            if intent.ndim == 1:
                intent = intent.reshape(1, -1)
            if intent.shape[0] < self.config.enemy_slots:
                padded = np.zeros((self.config.enemy_slots, intent.shape[1]), dtype=np.float32)
                padded[: intent.shape[0], :] = intent
                intent = padded
            intent_updates = intent
        for slot in range(self.config.enemy_slots):
            enemy = slot_enemies[slot] if slot < len(slot_enemies) else None
            if enemy is not None:
                slot_logits = np.asarray(enemy_logits[slot], dtype=np.float32).reshape(-1)
                if slot_logits.size < self.enemy_cmd_dim:
                    padded = np.zeros(self.enemy_cmd_dim, dtype=np.float32)
                    padded[: slot_logits.size] = slot_logits
                    slot_logits = padded
                else:
                    slot_logits = slot_logits[: self.enemy_cmd_dim]
                logits = slot_logits * self._enemy_logit_gain
                drive = np.tanh(logits)

                speed_cmd = float(drive[0])
                advance_cmd = float(drive[1])
                strafe_cmd = float(drive[2])
                turn_cmd = float(drive[3])
                aim_cmd_logit = float(drive[4])
                fire_cmd_logit = float(drive[5])
                advance_conf = float(0.5 + 0.5 * drive[6])
                strafe_conf = float(0.5 + 0.5 * drive[7])
                turn_conf = float(0.5 + 0.5 * drive[8])
                aim_conf = float(0.5 + 0.5 * drive[9])
                fire_conf = float(0.5 + 0.5 * drive[10])
                firecd_cmd_norm = float(0.5 + 0.5 * drive[11])
                health_cmd_norm = float(0.5 + 0.5 * drive[12])
                hold_gate = float(0.5 + 0.5 * drive[13])
                retarget_gate = float(0.5 + 0.5 * drive[14])
                target_logits = logits[
                    self.enemy_behavior_core_dim : self.enemy_behavior_core_dim + self.enemy_target_dim
                ]

                # Target selection remains NN-driven (logits + hold/retarget gates), but
                # backend actuation below is decoded directly from NN channels.
                _target_x, _target_y, _target_index, target_confidence = self._select_enemy_target_point(
                    slot,
                    slot_enemies,
                    target_logits,
                    player_x,
                    player_y,
                    hold_gate,
                    retarget_gate,
                )

                intent_id, intent_timer_norm, intent_probs = self._decode_enemy_intent_state(slot, intent_updates[slot])
                chase_p, flank_p, retreat_p, hold_p = [float(v) for v in intent_probs]
                move_gain = float(np.clip(0.35 + 0.65 * (chase_p + flank_p), 0.2, 1.0))
                strafe_gain = float(np.clip(0.35 + 0.65 * (flank_p + retreat_p), 0.2, 1.0))
                turn_gain = float(np.clip(0.35 + 0.65 * (flank_p + hold_p), 0.2, 1.0))
                aim_gate = float(np.clip(chase_p + hold_p, 0.0, 1.0))
                fire_gate = float(np.clip(chase_p + 0.5 * hold_p, 0.0, 1.0))
                retreat_sign = float(np.clip(1.0 - 2.0 * retreat_p, -1.0, 1.0))

                # Step 1: direct NN decode with explicit confidence channels.
                speed = int(np.clip(round(35.0 + 195.0 * (0.5 + 0.5 * speed_cmd)), 35, 230))
                fwd_drive = float(np.clip(advance_cmd * advance_conf * move_gain * retreat_sign, -1.0, 1.0))
                side_drive = float(np.clip(strafe_cmd * strafe_conf * strafe_gain, -1.0, 1.0))
                turn_drive = float(np.clip(turn_cmd * turn_conf * turn_gain, -1.0, 1.0))
                fwd = int(np.clip(round(100.0 * fwd_drive), -100, 100))
                side = int(np.clip(round(100.0 * side_drive), -100, 100))
                turn = int(np.clip(round(120.0 * turn_drive), -120, 120))
                aim_strength = float(aim_cmd_logit * aim_conf * aim_gate)
                fire_strength = float(fire_cmd_logit * fire_conf * fire_gate)
                aim = 1 if aim_strength > 0.05 else 0
                fire = 1 if fire_strength > 0.05 else 0
                cadence_firecd = int(np.clip(round(2.0 + 26.0 * firecd_cmd_norm), 2, 28))
                if cadence_firecd != self._last_enemy_cmds[slot]["firecd"]:
                    self.game.send_game_command(f"set nn_enemy_cmd_{slot:02d}_firecd {cadence_firecd}")
                    self._last_enemy_cmds[slot]["firecd"] = cadence_firecd

                healthpct = int(np.clip(round(40.0 + 160.0 * health_cmd_norm), 40, 200))
                if healthpct != self._last_enemy_cmds[slot]["healthpct"]:
                    self.game.send_game_command(f"set nn_enemy_cmd_{slot:02d}_healthpct {healthpct}")
                    self._last_enemy_cmds[slot]["healthpct"] = healthpct

                if self._collision_map is not None:
                    dynamic: list[tuple[float, float, float]] = [(player_x, player_y, 16.0)]
                    current_id = float(enemy.id)
                    for ex, ey, eid in enemy_circles_all:
                        if eid == current_id:
                            continue
                        dynamic.append((ex, ey, 20.0))
                    fwd, side = self._resolve_enemy_motion_command(enemy, fwd, side, speed, dynamic)

                self._update_enemy_memory_state(
                    slot=slot,
                    memory_update_logits=memory_updates[slot],
                )
                # Reserve memory channels for explicit intent state + timer diagnostics.
                memory = self._enemy_memory[slot]
                intent_code = -np.ones(len(self.enemy_intent_names), dtype=np.float32)
                intent_code[intent_id] = 1.0
                memory[0:4] = 0.5 * memory[0:4] + 0.5 * intent_code
                memory[4] = float(np.clip(2.0 * intent_timer_norm - 1.0, -1.0, 1.0))
                memory[5] = float(np.clip(2.0 * target_confidence - 1.0, -1.0, 1.0))
                memory[6] = float(
                    np.clip(
                        (float(self._enemy_target_choice[slot]) / float(max(1, self.enemy_target_dim - 1))) * 2.0 - 1.0,
                        -1.0,
                        1.0,
                    )
                )
                memory[7] = float(np.clip(2.0 * firecd_cmd_norm - 1.0, -1.0, 1.0))
            else:
                speed = 100
                fwd = 0
                side = 0
                turn = 0
                aim = 0
                fire = 0
                self._enemy_intent_state[slot] = 0
                self._enemy_intent_timer[slot] = 0.0
                self._enemy_target_choice[slot] = 0
                self._enemy_memory[slot, :] = 0.0
                healthpct = 100
            commands.append((speed, fwd, side, turn, aim, fire))
            self._enemy_last_cmd[slot, 0] = float(speed)
            self._enemy_last_cmd[slot, 1] = float(fwd)
            self._enemy_last_cmd[slot, 2] = float(side)
            self._enemy_last_cmd[slot, 3] = float(turn)
            self._enemy_last_cmd[slot, 4] = float(aim)
            self._enemy_last_cmd[slot, 5] = float(fire)

            if speed != self._last_enemy_cmds[slot]["speed"]:
                self.game.send_game_command(f"set nn_enemy_cmd_{slot:02d}_speed {speed}")
                self._last_enemy_cmds[slot]["speed"] = speed
            if fwd != self._last_enemy_cmds[slot]["fwd"]:
                self.game.send_game_command(f"set nn_enemy_cmd_{slot:02d}_fwd {fwd}")
                self._last_enemy_cmds[slot]["fwd"] = fwd
            if side != self._last_enemy_cmds[slot]["side"]:
                self.game.send_game_command(f"set nn_enemy_cmd_{slot:02d}_side {side}")
                self._last_enemy_cmds[slot]["side"] = side
            if turn != self._last_enemy_cmds[slot]["turn"]:
                self.game.send_game_command(f"set nn_enemy_cmd_{slot:02d}_turn {turn}")
                self._last_enemy_cmds[slot]["turn"] = turn
            if aim != self._last_enemy_cmds[slot]["aim"]:
                self.game.send_game_command(f"set nn_enemy_cmd_{slot:02d}_aim {aim}")
                self._last_enemy_cmds[slot]["aim"] = aim
            if fire != self._last_enemy_cmds[slot]["fire"]:
                self.game.send_game_command(f"set nn_enemy_cmd_{slot:02d}_fire {fire}")
                self._last_enemy_cmds[slot]["fire"] = fire

        enemy_count = sum(1 for enemy in slot_enemies if enemy is not None)
        return enemy_count, commands

    def _resolve_enemy_motion_command(
        self,
        enemy: object,
        fwd_pct: int,
        side_pct: int,
        speed_pct: int,
        dynamic_circles: list[tuple[float, float, float]],
    ) -> tuple[int, int]:
        if self._collision_map is None:
            return fwd_pct, side_pct

        local_fwd = float(np.clip(fwd_pct, -100, 100)) / 100.0
        local_side = float(np.clip(side_pct, -100, 100)) / 100.0
        if abs(local_fwd) < 1e-5 and abs(local_side) < 1e-5:
            return fwd_pct, side_pct

        speed_scale = float(np.clip(speed_pct / 100.0, 0.35, 2.5))
        step_units = 7.5 * speed_scale
        angle_rad = np.deg2rad(float(enemy.angle))
        cos_a = float(np.cos(angle_rad))
        sin_a = float(np.sin(angle_rad))

        dx = (cos_a * local_fwd - sin_a * local_side) * step_units
        dy = (sin_a * local_fwd + cos_a * local_side) * step_units
        x = float(enemy.position_x)
        y = float(enemy.position_y)
        nx, ny = self._collision_map.resolve_motion(
            x,
            y,
            dx,
            dy,
            radius=20.0,
            dynamic_circles=dynamic_circles,
        )
        resolved_dx = nx - x
        resolved_dy = ny - y
        if abs(resolved_dx) < 1e-6 and abs(resolved_dy) < 1e-6:
            return 0, 0

        resolved_fwd = (cos_a * resolved_dx + sin_a * resolved_dy) / max(step_units, 1e-6)
        resolved_side = (-sin_a * resolved_dx + cos_a * resolved_dy) / max(step_units, 1e-6)
        out_fwd = int(np.clip(round(resolved_fwd * 100.0), -100, 100))
        out_side = int(np.clip(round(resolved_side * 100.0), -100, 100))
        return out_fwd, out_side

    def _apply_low_level_backend_controls(
        self, low_level_logits: np.ndarray
    ) -> tuple[float, float, int, list[tuple[int, int]]]:
        values = np.tanh(np.asarray(low_level_logits, dtype=np.float32) * self._low_level_logit_gain)
        if values.size < self.low_level_dim:
            padded = np.zeros(self.low_level_dim, dtype=np.float32)
            padded[: values.size] = values
            values = padded
        else:
            values = values[: self.low_level_dim]

        move_scale = float(np.clip(1.0 + 0.18 * float(values[0]), 0.82, 1.22))
        turn_scale = float(np.clip(1.0 + 0.22 * float(values[1]), 0.78, 1.28))
        cooldown_scale = float(np.clip(1.0 + 0.55 * float(values[2]), 0.55, 1.85))
        fire_cooldown = int(np.clip(round(self.config.fire_cooldown_tics * cooldown_scale), 2, 24))

        self._runtime_move_scale = move_scale
        self._runtime_turn_scale = turn_scale
        self._runtime_fire_cooldown_tics = fire_cooldown

        enemy_low_level: list[tuple[int, int]] = []
        if not self.config.enemy_backend_transformer:
            return move_scale, turn_scale, fire_cooldown, enemy_low_level

        base = self.low_level_player_dim
        for slot in range(self.config.enemy_slots):
            idx = base + slot * self.low_level_enemy_dim
            firecd_proxy = int(np.clip(round(10.0 + 7.0 * float(values[idx])), 4, 24))
            healthpct = self._last_enemy_cmds[slot]["healthpct"] if self._last_enemy_cmds[slot]["healthpct"] >= 0 else 100
            enemy_low_level.append((firecd_proxy, int(healthpct)))

        return move_scale, turn_scale, fire_cooldown, enemy_low_level

    def _prime_history(self) -> None:
        self._enemy_slot_to_id = [None for _ in range(self.config.enemy_slots)]
        self._enemy_id_to_slot.clear()
        self._enemy_prev_x.fill(np.nan)
        self._enemy_prev_y.fill(np.nan)
        self._enemy_prev_angle.fill(np.nan)
        self._enemy_prev_los.fill(0.0)
        self._enemy_last_cmd.fill(0.0)
        self._enemy_target_choice.fill(0)
        self._enemy_memory.fill(0.0)
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
        scores = np.tanh(control_logits * self._control_logit_gain)
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
            # Keep keyboard intent dominant; NN only fine-tunes strength in a tight range.
            nn_gain = 0.82 + 0.18 * float(abs(scores[logit_idx]))
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
            action[0] = resolve_delta(
                0,
                1,
                0,
                axis_strength=self.config.move_delta * self._runtime_move_scale,
            )  # forward/back
            action[1] = resolve_delta(
                3,
                2,
                1,
                axis_strength=self.config.strafe_delta * self._runtime_move_scale,
            )  # strafe
            action[2] = resolve_delta(
                5,
                4,
                2,
                axis_strength=self.config.turn_delta * self._runtime_turn_scale,
            )  # turn

        # Look keys are optional.
        action[3] = resolve_delta(
            7,
            6,
            3,
            axis_strength=self.config.look_delta * self._runtime_turn_scale,
        )  # look down/up

        if keys[8]:
            action[4] = 1.0
        if keys[9]:
            action[5] = 1.0

        return action

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
        self._attack_cooldown_left = self._runtime_fire_cooldown_tics

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
                decoded_action = self._decode_controls(control_logits, step, keyboard_state)
                action = decoded_action.copy()
                if self._nn_movement_resolution_active and self._collision_map is not None:
                    # Transformer-side collision correction overlays native movement.
                    # This avoids dead controls if warp commands are ignored/limited.
                    self._apply_player_motion_resolution(decoded_action)
                self._apply_fire_cooldown(action, keyboard_state)
                total_reward += self._make_action_with_fire_pulse(action, repeats=1)
                last_action = decoded_action
            else:
                # Explicit neutral action prevents previous movement/turn from lingering.
                neutral_action = [0.0] * self.control_dim
                total_reward += self.game.make_action(neutral_action, 1)
                last_action = [0.0] * self.control_dim

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
            "Transformer non-standard low-level head is active "
            "(player dynamics knobs)."
        )
        print(
            "Keyboard input is routed through Transformer control decoding "
            f"(source: {self.keyboard_source})."
        )
        print("No sampled key => loop sends explicit neutral action (hard stop).")
        if self.keyboard_source == "macos_global+doom_buttons":
            print("Sticky-key protection is enabled (macOS global fallback).")
        else:
            print("Sticky-key protection is disabled for this keyboard source.")
        if self._nn_movement_resolution_active and self._collision_map is not None:
            print(
                "NN movement resolution is ON: player movement/collision is resolved in Transformer loop "
                f"(segments={len(self._collision_map.blocking_segments)}, move_units={self.config.nn_move_units:.2f})."
            )
        elif self.config.nn_movement_resolution:
            print("NN movement resolution requested but unavailable; using native Doom movement.")
        if self.config.enemy_backend_transformer:
            print(
                "Enemy backend override is ON: Transformer writes per-slot enemy movement/aim/fire commands "
                f"to {self.config.enemy_backend_mod.name}."
            )
        if self.config.visible:
            if self.keyboard_source == "pygame_window":
                print("Focus the 'Transformer Doom (focus this window)' game window for keyboard input.")
            else:
                print("Click the game window once so key states are captured.")
            print("Controls: WASD + arrows for movement/turn, Space for attack, E for use.")
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
        print(
            "NN gains: "
            f"weight_scale={self.config.nn_weight_scale:.2f} "
            f"control_gain={self.config.nn_control_gain:.2f} "
            f"enemy_gain={self.config.nn_enemy_gain:.2f} "
            f"low_level_gain={self.config.nn_low_level_gain:.2f}"
        )
        print(
            f"Enemy state features: slots={self.config.enemy_slots} per_slot={self.enemy_feature_dim} "
            f"(base={self.enemy_base_feature_dim}+feedback={self.enemy_feedback_feature_dim}+memory={self.enemy_memory_feature_dim}) "
            f"total={self.enemy_feature_dim_total} (stable ID->slot tracking ON)"
        )
        print(
            f"Enemy behavior head: channels={self.enemy_cmd_dim} "
            f"{'/'.join(self.enemy_behavior_channels)}"
        )
        print(
            f"Enemy intent head: channels={self.enemy_intent_dim} "
            f"{'/'.join(self.enemy_intent_channels)}"
        )
        print(
            f"Enemy memory-update head: channels={self.enemy_memory_update_dim} "
            "(gate + 8-dim delta per slot)"
        )
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
                    (
                        predicted_state,
                        control_logits,
                        enemy_logits,
                        enemy_intent_logits,
                        low_level_logits,
                        memory_update_logits,
                        _,
                    ) = self.model(inputs)
                predicted = predicted_state.cpu().numpy()[0]
                control = control_logits.cpu().numpy()[0]
                enemy = enemy_logits.cpu().numpy()[0]
                enemy_intent = enemy_intent_logits.cpu().numpy()[0]
                low_level = low_level_logits.cpu().numpy()[0]
                memory_update = memory_update_logits.cpu().numpy()[0]
                move_scale, turn_scale, fire_cd, enemy_low = self._apply_low_level_backend_controls(low_level)
                enemy_count, enemy_cmd = self._apply_enemy_backend_commands(enemy, enemy_intent, memory_update)

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
                if step % self.config.log_interval == 0:
                    enemy_info = ""
                    if self.config.enemy_backend_transformer:
                        sample_cmd = enemy_cmd[: min(2, len(enemy_cmd))]
                        sample_low = enemy_low[: min(2, len(enemy_low))]
                        enemy_info = (
                            f" enemies={enemy_count} enemy_cmd={sample_cmd} enemy_low={sample_low}"
                        )
                    print(
                        f"tick={step:06d} mse={mse:.6f} "
                        f"health={health:.0f} armor={armor:.0f} kills={kills:.0f} "
                        f"reward={reward:.2f} stuck={self.stuck_ticks}{enemy_info} "
                        f" ll_move={move_scale:.2f} ll_turn={turn_scale:.2f} ll_firecd={fire_cd} "
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
        "--enemy-backend-transformer",
        action="store_true",
        help=(
            "Enable experimental enemy backend override via custom scenario mod and "
            "Transformer enemy command head."
        ),
    )
    parser.add_argument(
        "--enemy-backend-mod",
        type=Path,
        default=Path("enemy_nn_backend_mod.pk3"),
        help="Path to the custom enemy backend mod PK3.",
    )
    parser.add_argument(
        "--enemy-slots",
        type=int,
        default=16,
        help="Max enemy slots controlled by Transformer backend mod.",
    )
    parser.add_argument(
        "--nn-movement-resolution",
        dest="nn_movement_resolution",
        action="store_true",
        help=(
            "Resolve player movement/collision in Transformer loop (WAD linedef collision + warp), "
            "instead of direct Doom key movement."
        ),
    )
    parser.add_argument(
        "--disable-nn-movement-resolution",
        dest="nn_movement_resolution",
        action="store_false",
        help="Disable Transformer-side movement/collision resolution and use native Doom movement.",
    )
    parser.set_defaults(nn_movement_resolution=True)
    parser.add_argument(
        "--nn-move-units",
        type=float,
        default=5.2,
        help="World units per resolved movement step multiplier for Transformer movement.",
    )
    parser.add_argument(
        "--nn-player-radius",
        type=float,
        default=16.0,
        help="Collision radius used by Transformer-side player resolver.",
    )
    parser.add_argument(
        "--nn-weight-scale",
        type=float,
        default=0.55,
        help="Scale for hardcoded parameter initialization std (lower = tighter/stabler).",
    )
    parser.add_argument(
        "--nn-control-gain",
        type=float,
        default=1.0,
        help="Control head logit gain before tanh decoding.",
    )
    parser.add_argument(
        "--nn-enemy-gain",
        type=float,
        default=0.75,
        help="Enemy head logit gain before tanh decoding.",
    )
    parser.add_argument(
        "--nn-low-level-gain",
        type=float,
        default=0.45,
        help="Low-level head logit gain before tanh decoding.",
    )
    args = parser.parse_args()

    wad_path = args.wad.resolve()
    if not wad_path.exists():
        raise FileNotFoundError(f"DOOM WAD not found at: {wad_path}")

    enemy_backend_mod = args.enemy_backend_mod.resolve()
    if args.enemy_backend_transformer and not enemy_backend_mod.exists():
        raise FileNotFoundError(
            f"Enemy backend mod not found at: {enemy_backend_mod}. "
            "Build it first (see build_enemy_nn_mod.py)."
        )

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
        enemy_backend_transformer=args.enemy_backend_transformer,
        enemy_backend_mod=enemy_backend_mod,
        enemy_slots=max(1, min(64, args.enemy_slots)),
        nn_movement_resolution=bool(args.nn_movement_resolution),
        nn_move_units=max(0.1, float(args.nn_move_units)),
        nn_player_radius=max(4.0, float(args.nn_player_radius)),
        nn_weight_scale=float(np.clip(args.nn_weight_scale, 0.05, 2.0)),
        nn_control_gain=float(np.clip(args.nn_control_gain, 0.1, 4.0)),
        nn_enemy_gain=float(np.clip(args.nn_enemy_gain, 0.1, 4.0)),
        nn_low_level_gain=float(np.clip(args.nn_low_level_gain, 0.05, 4.0)),
    )


def main() -> None:
    config = parse_args()
    loop = DoomTransformerLoop(config)
    loop.run()


if __name__ == "__main__":
    main()
