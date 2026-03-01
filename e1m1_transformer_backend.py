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
            # Prefer scancode-resolved live keyboard snapshot.
            # On pygame2, many keys (arrows, function keys) use large SDLK values
            # that are not direct indices into get_pressed().
            try:
                sc = int(self._pg.key.get_scancode_from_key(int(keycode)))
                if 0 <= sc < len(key_state) and bool(key_state[sc]):
                    return True
            except Exception:
                pass
            # Fallback for layouts where keycode is directly indexable.
            if 0 <= int(keycode) < len(key_state):
                return bool(key_state[int(keycode)])
            # Event-tracked fallback for any non-indexable keys.
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
        enemy_actuator_dim: int = 21,
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
        self.enemy_actuator_dim = max(1, int(enemy_actuator_dim))
        self.low_level_dim = low_level_dim if low_level_dim is not None else (4 + enemy_slots)
        self.memory_update_dim = max(1, int(memory_update_dim))
        self.weight_scale = float(np.clip(weight_scale, 0.05, 2.0))
        self.in_proj = nn.Linear(state_dim, d_model)
        self.state_out_proj = nn.Linear(d_model, state_dim)
        self.control_out_proj = nn.Linear(d_model, control_dim)
        self.enemy_out_proj = nn.Linear(d_model, enemy_slots * enemy_cmd_dim)
        self.enemy_intent_out_proj = nn.Linear(d_model, enemy_slots * self.enemy_intent_dim)
        self.enemy_actuator_out_proj = nn.Linear(d_model, enemy_slots * self.enemy_actuator_dim)
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
                elif "enemy_actuator_out_proj.weight" in name:
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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
    ]:
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
        enemy_actuator_logits = self.enemy_actuator_out_proj(head).view(-1, self.enemy_slots, self.enemy_actuator_dim)
        low_level_logits = self.low_level_out_proj(head)
        memory_update_logits = self.memory_out_proj(head).view(-1, self.enemy_slots, self.memory_update_dim)
        return (
            self.state_out_proj(head),
            self.control_out_proj(head),
            enemy_logits,
            enemy_intent_logits,
            enemy_actuator_logits,
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
    doom_seed: int | None
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
    nn_world_sim: bool
    nn_world_damage_scale: float


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
            "move_mix_cmd",
            "strafe_mix_cmd",
            "turn_mix_cmd",
            "aim_mix_cmd",
            "fire_mix_cmd",
            "retreat_mix_cmd",
            "health_cmd_norm",
            "target_index_cmd_norm",
            "fwd_final_cmd",
            "side_final_cmd",
            "turn_final_cmd",
            "aim_final_logit",
            "fire_final_logit",
            "target_blend_logit",
            "fire_enable_logit",
            "burst_len_norm",
            "inter_shot_delay_norm",
            "reaction_delay_norm",
            "coord_focus_target_index_norm",
            "coord_assist_gate_logit",
            "coord_spacing_cmd",
            "coord_avoidance_cmd",
            "nav_desired_heading_cmd",
            "nav_desired_speed_norm",
            "nav_cover_seek_cmd",
            "nav_retreat_seek_cmd",
            "firecd_cmd_norm",
        )
        self.enemy_target_channels = ("target_player_logit",) + tuple(
            f"target_slot_{slot:02d}_logit" for slot in range(self.config.enemy_slots)
        )
        self.enemy_behavior_channels = self.enemy_behavior_core_channels + self.enemy_target_channels
        self.enemy_behavior_core_dim = len(self.enemy_behavior_core_channels)
        self.enemy_target_dim = len(self.enemy_target_channels)
        self.enemy_target_mask_dim = self.enemy_target_dim
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
        self.enemy_actuator_channels = (
            "act_speed_norm",
            "act_fwd_cmd",
            "act_side_cmd",
            "act_turn_cmd",
            "act_aim_logit",
            "act_fire_logit",
            "act_firecd_norm",
            "act_health_norm",
            "act_aim_gate_logit",
            "act_fire_gate_logit",
            "act_move_dx_cmd",
            "act_move_dy_cmd",
            "act_slide_bias_norm",
            "act_separation_gain_norm",
            "act_target_index_norm",
            "act_target_keep_gate_logit",
            "act_target_switch_gate_logit",
            "act_fire_enable_logit",
            "act_burst_len_norm",
            "act_inter_shot_delay_norm",
            "act_reaction_delay_norm",
        )
        self.enemy_actuator_dim = len(self.enemy_actuator_channels)
        self.enemy_base_feature_dim = 24
        self.enemy_feedback_feature_dim = 11
        self.enemy_memory_feature_dim = 10
        self.enemy_memory_channels = (
            "intent_chase_latent",
            "intent_flank_latent",
            "intent_retreat_latent",
            "intent_hold_latent",
            "identity_lock_strength",
            "identity_write_gate",
            "target_index_norm",
            "target_keep_gate",
            "target_identity_norm",
            "slot_presence_norm",
        )
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
        self._enemy_prev_x = np.full(self.config.enemy_slots, np.nan, dtype=np.float32)
        self._enemy_prev_y = np.full(self.config.enemy_slots, np.nan, dtype=np.float32)
        self._enemy_prev_angle = np.full(self.config.enemy_slots, np.nan, dtype=np.float32)
        self._enemy_prev_health_obs = np.full(self.config.enemy_slots, np.nan, dtype=np.float32)
        self._enemy_last_cmd = np.zeros((self.config.enemy_slots, 6), dtype=np.float32)
        self._enemy_last_target_index = np.full(self.config.enemy_slots, -1, dtype=np.int32)
        self._enemy_memory = np.zeros(
            (self.config.enemy_slots, self.enemy_memory_feature_dim),
            dtype=np.float32,
        )
        self._enemy_slot_obs_identity = np.zeros(self.config.enemy_slots, dtype=np.float32)
        self._enemy_slot_obs_present = np.zeros(self.config.enemy_slots, dtype=np.float32)
        self._enemy_metric_ticks = 0
        self._enemy_metric_shots = 0
        self._enemy_metric_target_switches = 0
        self._enemy_metric_active_enemy_ticks = 0
        self._enemy_metric_close_pairs = 0
        self._enemy_metric_identity_churn = 0
        self._enemy_metric_identity_samples = 0
        self._enemy_prev_obs_identity = np.full(self.config.enemy_slots, np.nan, dtype=np.float32)
        self._enemy_fire_cooldown_left = np.zeros(self.config.enemy_slots, dtype=np.int32)
        self._mse_total_sum = 0.0
        self._mse_total_count = 0
        self._mse_head_window = 256
        self._mse_head_sum = 0.0
        self._mse_head_count = 0
        self._mse_tail_window = 256
        self._mse_tail_values: Deque[float] = deque(maxlen=self._mse_tail_window)
        self._world_sim_initialized = False
        self._world_sim_tick = 0
        self._world_player = np.zeros(4, dtype=np.float32)  # x, y, angle_deg, health
        self._world_enemies = np.zeros((self.config.enemy_slots, 5), dtype=np.float32)  # x, y, angle_deg, health, alive
        self._world_kills = 0
        self._world_last_bridge_x = np.nan
        self._world_last_bridge_y = np.nan
        self._world_last_bridge_angle = np.nan
        self._latest_target_mask = np.zeros(self.enemy_target_mask_dim, dtype=np.float32)
        if self.enemy_target_mask_dim > 0:
            self._latest_target_mask[0] = 100.0
        self._nn_movement_resolution_active = bool(self.config.nn_movement_resolution)
        self._init_keyboard_source()
        if self._nn_movement_resolution_active and self._uses_doom_buttons_source():
            # doom_buttons requires native binds, which conflicts with Transformer-side movement warping.
            self._nn_movement_resolution_active = False
        if self.config.nn_world_sim:
            # World sim owns player movement/collision; disable native NN movement resolver path.
            self._nn_movement_resolution_active = False
        self.game = self._init_game()
        self.state_dim = (
            len(GameVariable.__members__)
            + self._pooled_pixel_count()
            + len(self.keyboard_buttons)
            + self.enemy_feature_dim_total
            + self.enemy_target_mask_dim
        )
        self.model = HardcodedStateTransformer(
            state_dim=self.state_dim,
            control_dim=self.control_dim,
            context=self.config.context,
            enemy_slots=self.config.enemy_slots,
            enemy_cmd_dim=self.enemy_cmd_dim,
            enemy_intent_dim=self.enemy_intent_dim,
            enemy_actuator_dim=self.enemy_actuator_dim,
            low_level_dim=self.low_level_dim,
            memory_update_dim=self.enemy_memory_update_dim,
            weight_scale=self.config.nn_weight_scale,
        ).to(self.config.device)
        self.history: Deque[np.ndarray] = deque(maxlen=self.config.context)
        self.last_position: tuple[float, float] | None = None
        self.stuck_ticks = 0
        self._player_max_stuck_ticks = 0
        self._last_sampled_keys = np.zeros(len(self.keyboard_buttons), dtype=np.float32)
        self._stale_key_ticks = 0
        self._attack_cooldown_left = 0
        self._control_logit_gain = self.config.nn_control_gain
        self._enemy_logit_gain = self.config.nn_enemy_gain
        self._low_level_logit_gain = self.config.nn_low_level_gain
        self._runtime_move_scale = 1.0
        self._runtime_turn_scale = 1.0
        self._runtime_fire_cooldown_tics = self.config.fire_cooldown_tics
        self._runtime_fire_cooldown_float = float(self.config.fire_cooldown_tics)
        self._map_geometry: WADCollisionMap | None = None
        try:
            self._map_geometry = WADCollisionMap(self.config.wad_path, self.config.map_name)
        except Exception as exc:
            if self._nn_movement_resolution_active:
                print(f"Warning: NN movement resolution disabled ({exc}).")
            self._map_geometry = None
        self._collision_map: WADCollisionMap | None = (
            self._map_geometry if (self._nn_movement_resolution_active or self.config.nn_world_sim) else None
        )

    def _uses_doom_buttons_source(self) -> bool:
        return "doom_buttons" in self.keyboard_source

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
        if self.config.doom_seed is not None:
            try:
                game.set_seed(int(self.config.doom_seed))
            except Exception:
                pass
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
        if self._nn_movement_resolution_active and not self._uses_doom_buttons_source():
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
            if self.config.nn_world_sim and self.config.visible:
                try:
                    self.pygame_keyboard_sampler = PygameKeyboardSampler(
                        self.config.frame_width,
                        self.config.frame_height,
                    )
                    self.keyboard_source = "pygame_window"
                except Exception:
                    pass
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
        if self.config.nn_world_sim and self.config.visible:
            try:
                self.pygame_keyboard_sampler = PygameKeyboardSampler(
                    self.config.frame_width,
                    self.config.frame_height,
                )
                self.keyboard_source = "pygame_window"
            except Exception:
                pass

    def _bind_keyboard_controls(self, game: DoomGame) -> None:
        if self._nn_movement_resolution_active and not self._uses_doom_buttons_source():
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
        if self.keyboard_source == "pygame_window" or self._uses_doom_buttons_source():
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

    def _raycast_distance_2d(
        self,
        origin_x: float,
        origin_y: float,
        dir_x: float,
        dir_y: float,
        max_dist: float = 512.0,
    ) -> float:
        if self._map_geometry is None:
            return float(max_dist)
        norm = float(np.hypot(dir_x, dir_y))
        if norm < 1e-6:
            return float(max_dist)
        rx = dir_x / norm
        ry = dir_y / norm
        best = float(max_dist)
        for sx1, sy1, sx2, sy2 in self._map_geometry.blocking_segments:
            qpx = sx1 - origin_x
            qpy = sy1 - origin_y
            sx = sx2 - sx1
            sy = sy2 - sy1
            denom = rx * sy - ry * sx
            if abs(denom) < 1e-8:
                continue
            t = (qpx * sy - qpy * sx) / denom
            u = (qpx * ry - qpy * rx) / denom
            if t < 0.0 or u < 0.0 or u > 1.0:
                continue
            if t < best:
                best = float(t)
        return best

    def _enemy_objects_from_state(self, state: object | None) -> list[object]:
        if state is None:
            return []
        objects = getattr(state, "objects", None)
        if objects is None:
            return []
        return [obj for obj in objects if self._is_enemy_object_name(str(obj.name))]

    def _enemy_attr_float(self, enemy: object, attr: str, fallback: float = 0.0) -> float:
        value = getattr(enemy, attr, fallback)
        if value is None:
            return float(fallback)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)

    def _enemy_observed_identity_norm(
        self,
        enemy: object,
        player_x: float,
        player_y: float,
    ) -> float:
        ex = self._enemy_attr_float(enemy, "position_x")
        ey = self._enemy_attr_float(enemy, "position_y")
        vx = self._enemy_attr_float(enemy, "velocity_x")
        vy = self._enemy_attr_float(enemy, "velocity_y")
        angle = self._normalize_angle_deg(self._enemy_attr_float(enemy, "angle"))
        pitch = self._normalize_angle_deg(self._enemy_attr_float(enemy, "pitch"))
        health = self._enemy_attr_float(enemy, "health")
        dx = ex - player_x
        dy = ey - player_y
        name = str(getattr(enemy, "name", "")).lower()
        name_hash = float(((sum((i + 1) * ord(ch) for i, ch in enumerate(name[:24])) % 997) / 498.5) - 1.0)
        ident = (
            0.00067 * ex
            + 0.00059 * ey
            + 0.15 * vx
            + 0.13 * vy
            + 0.0085 * angle
            + 0.0040 * pitch
            + 0.0013 * dx
            + 0.0011 * dy
            + 0.0022 * health
            + 0.21 * name_hash
        )
        return float(np.tanh(ident))

    def _reset_enemy_slot_runtime(self, slot: int) -> None:
        if slot < 0 or slot >= self.config.enemy_slots:
            return
        self._enemy_prev_x[slot] = np.nan
        self._enemy_prev_y[slot] = np.nan
        self._enemy_prev_angle[slot] = np.nan
        self._enemy_prev_health_obs[slot] = np.nan
        self._enemy_slot_obs_identity[slot] = 0.0
        self._enemy_slot_obs_present[slot] = 0.0
        self._enemy_prev_obs_identity[slot] = np.nan
        self._enemy_fire_cooldown_left[slot] = 0
        self._enemy_last_cmd[slot, :] = 0.0
        self._enemy_last_target_index[slot] = -1
        self._enemy_memory[slot, :] = 0.0
        self._last_enemy_cmds[slot]["speed"] = -1
        self._last_enemy_cmds[slot]["fwd"] = -999
        self._last_enemy_cmds[slot]["side"] = -999
        self._last_enemy_cmds[slot]["turn"] = -999
        self._last_enemy_cmds[slot]["aim"] = -1
        self._last_enemy_cmds[slot]["fire"] = -1
        self._last_enemy_cmds[slot]["firecd"] = -1
        self._last_enemy_cmds[slot]["healthpct"] = -1

    def _refresh_enemy_slot_assignments(self, state: object | None = None) -> list[object | None]:
        # Slot lifecycle is model-owned: assignment uses model memory identity channel.
        enemies = self._enemy_objects_from_state(state)
        slot_enemies: list[object | None] = [None for _ in range(self.config.enemy_slots)]
        if not enemies:
            return slot_enemies

        player_x, player_y = self._current_position()
        obs_identity = np.asarray(
            [
                self._enemy_observed_identity_norm(enemy, player_x, player_y)
                for enemy in enemies
            ],
            dtype=np.float32,
        )
        slot_identity = np.asarray(self._enemy_memory[:, 8], dtype=np.float32)
        slot_presence = 0.5 + 0.5 * np.asarray(self._enemy_memory[:, 9], dtype=np.float32)
        slot_lock = 0.5 + 0.5 * np.asarray(self._enemy_memory[:, 4], dtype=np.float32)

        # Cold start: deterministic bootstrap using observed identity order.
        if float(np.max(slot_presence)) < 0.05:
            sorted_enemy_indices = sorted(range(len(enemies)), key=lambda idx: float(obs_identity[idx]))
            for slot, enemy_idx in enumerate(sorted_enemy_indices[: self.config.enemy_slots]):
                slot_enemies[slot] = enemies[enemy_idx]
            return slot_enemies

        free_slots = list(range(self.config.enemy_slots))
        free_enemies = list(range(len(enemies)))
        while free_slots and free_enemies:
            best_score = None
            best_slot = -1
            best_enemy = -1
            for slot in free_slots:
                sid = float(slot_identity[slot])
                presence = float(np.clip(slot_presence[slot], 0.0, 1.0))
                lock = float(np.clip(slot_lock[slot], 0.0, 1.0))
                for enemy_idx in free_enemies:
                    score = abs(float(obs_identity[enemy_idx]) - sid)
                    score *= (0.35 + 0.65 * presence)
                    score *= (1.0 - 0.50 * lock)
                    score += 0.15 * (1.0 - presence)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_slot = slot
                        best_enemy = enemy_idx
            if best_slot < 0 or best_enemy < 0:
                break
            slot_enemies[best_slot] = enemies[best_enemy]
            free_slots.remove(best_slot)
            free_enemies.remove(best_enemy)
        return slot_enemies

    def _enemy_feature_block(self, state: object | None) -> tuple[np.ndarray, np.ndarray]:
        block = np.zeros(self.enemy_feature_dim_total, dtype=np.float32)
        target_mask = np.zeros(self.enemy_target_mask_dim, dtype=np.float32)
        target_mask[0] = 100.0  # Player target is always valid.
        slot_enemies = self._refresh_enemy_slot_assignments(state)
        player_x, player_y = self._current_position()
        player_angle = float(self.game.get_game_variable(GameVariable.ANGLE))
        player_health = float(self.game.get_game_variable(GameVariable.HEALTH))
        self._enemy_slot_obs_identity.fill(0.0)
        self._enemy_slot_obs_present.fill(0.0)

        for slot, enemy in enumerate(slot_enemies):
            base = slot * self.enemy_feature_dim
            if enemy is None:
                continue
            target_mask[slot + 1] = 100.0

            ex = self._enemy_attr_float(enemy, "position_x")
            ey = self._enemy_attr_float(enemy, "position_y")
            ez = self._enemy_attr_float(enemy, "position_z")
            vx = self._enemy_attr_float(enemy, "velocity_x")
            vy = self._enemy_attr_float(enemy, "velocity_y")
            vz = self._enemy_attr_float(enemy, "velocity_z")
            angle = self._normalize_angle_deg(self._enemy_attr_float(enemy, "angle"))
            pitch = self._normalize_angle_deg(self._enemy_attr_float(enemy, "pitch"))
            radius = self._enemy_attr_float(enemy, "radius", 20.0)
            height = self._enemy_attr_float(enemy, "height", 56.0)
            mass = self._enemy_attr_float(enemy, "mass", 100.0)
            reaction_time = self._enemy_attr_float(enemy, "reactiontime", 0.0)
            threshold = self._enemy_attr_float(enemy, "threshold", 0.0)
            dx = ex - player_x
            dy = ey - player_y

            health_proxy = float(
                self._last_enemy_cmds[slot]["healthpct"] if self._last_enemy_cmds[slot]["healthpct"] >= 0 else 100
            )
            health_obs_raw = getattr(enemy, "health", health_proxy)
            health_obs = float(health_obs_raw) if health_obs_raw is not None else float(health_proxy)
            prev_health_obs = float(self._enemy_prev_health_obs[slot])
            self._enemy_prev_health_obs[slot] = health_obs
            obs_identity = self._enemy_observed_identity_norm(enemy, player_x, player_y)
            self._enemy_slot_obs_identity[slot] = obs_identity
            self._enemy_slot_obs_present[slot] = 1.0

            # Per-slot feature layout:
            # Base (24): raw object/player state with minimal derived geometry.
            # Feedback (11): last command + raw frame-to-frame deltas.
            # Memory (10): persistent latent updated by Transformer memory gate+delta outputs.
            block[base + 0] = 100.0
            block[base + 1] = float(np.clip(ex, -8192.0, 8192.0))
            block[base + 2] = float(np.clip(ey, -8192.0, 8192.0))
            block[base + 3] = float(np.clip(ez, -2048.0, 2048.0))
            block[base + 4] = float(np.clip(vx * 64.0, -1024.0, 1024.0))
            block[base + 5] = float(np.clip(vy * 64.0, -1024.0, 1024.0))
            block[base + 6] = float(np.clip(vz * 64.0, -1024.0, 1024.0))
            block[base + 7] = angle
            block[base + 8] = pitch
            block[base + 9] = float(np.clip(radius, 0.0, 256.0))
            block[base + 10] = float(np.clip(height, 0.0, 512.0))
            block[base + 11] = float(np.clip(health_obs, 0.0, 400.0))
            block[base + 12] = float(np.clip(mass, 0.0, 4096.0))
            block[base + 13] = float(np.clip(reaction_time, -256.0, 256.0))
            block[base + 14] = float(np.clip(threshold, -1024.0, 1024.0))
            block[base + 15] = float(np.clip(player_x, -8192.0, 8192.0))
            block[base + 16] = float(np.clip(player_y, -8192.0, 8192.0))
            block[base + 17] = float(np.clip(player_angle, -180.0, 180.0))
            block[base + 18] = float(np.clip(player_health, 0.0, 400.0))
            block[base + 19] = float(np.clip(dx, -4096.0, 4096.0))
            block[base + 20] = float(np.clip(dy, -4096.0, 4096.0))
            block[base + 21] = float(np.clip(obs_identity * 100.0, -100.0, 100.0))
            block[base + 22] = 100.0 if np.isfinite(self._enemy_prev_x[slot]) else 0.0
            block[base + 23] = 0.0

            prev_x = float(self._enemy_prev_x[slot])
            prev_y = float(self._enemy_prev_y[slot])
            prev_angle = float(self._enemy_prev_angle[slot])
            delta_x = ex - prev_x if np.isfinite(prev_x) else 0.0
            delta_y = ey - prev_y if np.isfinite(prev_y) else 0.0
            if np.isfinite(prev_angle):
                delta_angle = self._normalize_angle_deg(angle - prev_angle)
            else:
                delta_angle = 0.0
            delta_health = health_obs - prev_health_obs if np.isfinite(prev_health_obs) else 0.0
            has_prev = 100.0 if (np.isfinite(prev_x) and np.isfinite(prev_y)) else 0.0

            cmd_speed = float(self._enemy_last_cmd[slot, 0])
            cmd_fwd = float(self._enemy_last_cmd[slot, 1])
            cmd_side = float(self._enemy_last_cmd[slot, 2])
            cmd_turn = float(self._enemy_last_cmd[slot, 3])
            cmd_aim = float(self._enemy_last_cmd[slot, 4])
            cmd_fire = float(self._enemy_last_cmd[slot, 5])

            fb = base + self.enemy_base_feature_dim
            # Feedback: last_cmd(speed,fwd,side,turn,aim,fire), raw deltas, and prev-observed flag.
            block[fb + 0] = float(np.clip(cmd_speed - 100.0, -100.0, 150.0))
            block[fb + 1] = float(np.clip(cmd_fwd, -100.0, 100.0))
            block[fb + 2] = float(np.clip(cmd_side, -100.0, 100.0))
            block[fb + 3] = float(np.clip(cmd_turn, -120.0, 120.0))
            block[fb + 4] = float(np.clip(cmd_aim, -100.0, 100.0))
            block[fb + 5] = float(np.clip(cmd_fire, -100.0, 100.0))
            block[fb + 6] = float(np.clip(delta_x, -512.0, 512.0))
            block[fb + 7] = float(np.clip(delta_y, -512.0, 512.0))
            block[fb + 8] = float(np.clip(delta_angle, -180.0, 180.0))
            block[fb + 9] = float(np.clip(delta_health, -256.0, 256.0))
            block[fb + 10] = has_prev

            mb = fb + self.enemy_feedback_feature_dim
            # Memory latent channels (updated only by Transformer memory-update head).
            memory = np.clip(self._enemy_memory[slot], -1.0, 1.0)
            block[mb : mb + self.enemy_memory_feature_dim] = memory * 100.0

            self._enemy_prev_x[slot] = ex
            self._enemy_prev_y[slot] = ey
            self._enemy_prev_angle[slot] = angle

        return block, target_mask

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
        enemy_features, target_mask = self._enemy_feature_block(state)
        self._latest_target_mask = target_mask.copy()
        vector = np.concatenate([variables, pooled.ravel(), keyboard_features, enemy_features, target_mask], dtype=np.float32)

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
        slot_enemies: list[object | None],
        player_x: float,
        player_y: float,
        selected_index: int,
    ) -> tuple[float, float, int]:
        candidate_points: list[tuple[float, float]] = [(player_x, player_y)]
        valid: list[bool] = [True]
        for target_slot in range(self.config.enemy_slots):
            target_enemy = slot_enemies[target_slot] if target_slot < len(slot_enemies) else None
            if target_enemy is None:
                candidate_points.append((player_x, player_y))
                valid.append(False)
            else:
                candidate_points.append((float(target_enemy.position_x), float(target_enemy.position_y)))
                valid.append(True)
        selected = int(np.clip(selected_index, 0, len(candidate_points) - 1))
        if not valid[selected]:
            selected = 0
        target_x, target_y = candidate_points[selected]
        return float(target_x), float(target_y), selected

    def _update_enemy_memory_state(
        self,
        slot: int,
        memory_update_logits: np.ndarray,
        observed_identity_norm: float = 0.0,
        observed_present: float = 0.0,
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

        # Identity/presence persistence stays model-owned via memory-controlled write gates.
        obs_identity = float(np.clip(observed_identity_norm, -1.0, 1.0))
        obs_present = float(np.clip(observed_present, 0.0, 1.0))
        identity_write = float(np.clip(0.5 + 0.5 * memory[5], 0.0, 1.0)) * obs_present
        presence_write = float(np.clip(0.5 + 0.5 * memory[4], 0.0, 1.0))
        target_presence = 2.0 * obs_present - 1.0
        memory[8] = float(np.tanh((1.0 - identity_write) * memory[8] + identity_write * obs_identity))
        memory[9] = float(np.tanh((1.0 - presence_write) * memory[9] + presence_write * target_presence))

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
        enemy_actuator_logits: np.ndarray,
        memory_update_logits: np.ndarray | None = None,
    ) -> tuple[int, list[tuple[int, int, int, int, int, int]]]:
        if not self.config.enemy_backend_transformer:
            return 0, []

        state = self.game.get_state()
        slot_enemies = self._refresh_enemy_slot_assignments(state)
        if self.config.nn_world_sim:
            if not self._world_sim_initialized:
                self._world_sim_bootstrap(slot_enemies)
            else:
                self._world_sim_sync_slots(slot_enemies)
        if self.config.nn_world_sim and self._world_sim_initialized:
            player_x = float(self._world_player[0])
            player_y = float(self._world_player[1])
        else:
            player_x, player_y = self._current_position()
        commands: list[tuple[int, int, int, int, int, int]] = []
        shots_tick = 0
        target_switches_tick = 0
        if self.config.nn_world_sim and self._world_sim_initialized:
            active_positions = [
                (float(self._world_enemies[slot, 0]), float(self._world_enemies[slot, 1]))
                for slot in range(self.config.enemy_slots)
                if self._world_enemies[slot, 4] > 0.5
            ]
        else:
            active_positions = [
                (float(enemy.position_x), float(enemy.position_y))
                for enemy in slot_enemies
                if enemy is not None
            ]
        close_pairs_tick = 0
        for i in range(len(active_positions)):
            x1, y1 = active_positions[i]
            for j in range(i + 1, len(active_positions)):
                x2, y2 = active_positions[j]
                if float(np.hypot(x1 - x2, y1 - y2)) < 64.0:
                    close_pairs_tick += 1
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
        actuator = np.asarray(enemy_actuator_logits, dtype=np.float32)
        if actuator.ndim == 1:
            actuator = actuator.reshape(1, -1)
        if actuator.shape[0] < self.config.enemy_slots:
            padded_act = np.zeros((self.config.enemy_slots, actuator.shape[1]), dtype=np.float32)
            padded_act[: actuator.shape[0], :] = actuator
            actuator = padded_act
        enemy_actuators = actuator
        for slot in range(self.config.enemy_slots):
            if self._enemy_fire_cooldown_left[slot] > 0:
                self._enemy_fire_cooldown_left[slot] -= 1
            enemy = slot_enemies[slot] if slot < len(slot_enemies) else None
            if enemy is not None:
                obs_identity = float(self._enemy_slot_obs_identity[slot])
                prev_obs_identity = float(self._enemy_prev_obs_identity[slot])
                if np.isfinite(prev_obs_identity):
                    self._enemy_metric_identity_samples += 1
                    if abs(obs_identity - prev_obs_identity) > 0.35:
                        self._enemy_metric_identity_churn += 1
                self._enemy_prev_obs_identity[slot] = obs_identity
                slot_actuator = np.asarray(enemy_actuators[slot], dtype=np.float32).reshape(-1)
                if slot_actuator.size < self.enemy_actuator_dim:
                    padded_actuator = np.zeros(self.enemy_actuator_dim, dtype=np.float32)
                    padded_actuator[: slot_actuator.size] = slot_actuator
                    slot_actuator = padded_actuator
                else:
                    slot_actuator = slot_actuator[: self.enemy_actuator_dim]
                actuator_drive = np.nan_to_num(
                    slot_actuator * self._enemy_logit_gain,
                    nan=0.0,
                    posinf=4.0,
                    neginf=-4.0,
                )
                actuator_drive = np.clip(actuator_drive, -1.0, 1.0)
                act_speed_norm = float(0.5 + 0.5 * actuator_drive[0])
                act_fwd_cmd = float(actuator_drive[1])
                act_side_cmd = float(actuator_drive[2])
                act_turn_cmd = float(actuator_drive[3])
                act_aim_cmd = float(actuator_drive[4])
                act_fire_cmd = float(actuator_drive[5])
                act_firecd_norm = float(0.5 + 0.5 * actuator_drive[6])
                act_health_norm = float(0.5 + 0.5 * actuator_drive[7])

                # Target lifecycle is model-owned. Python only keeps index in legal command range.
                last_target = int(self._enemy_last_target_index[slot])
                target_mem_norm = float(0.5 + 0.5 * float(np.clip(self._enemy_memory[slot, 6], -1.0, 1.0)))
                proposed_index = int(
                    np.clip(
                        round(target_mem_norm * float(max(1, self.enemy_target_dim - 1))),
                        0,
                        self.enemy_target_dim - 1,
                    )
                )
                if last_target >= 0:
                    keep_gate = float(np.clip(0.5 + 0.5 * float(self._enemy_memory[slot, 7]), 0.0, 1.0))
                    selected_index = int(
                        np.clip(
                            round(keep_gate * float(last_target) + (1.0 - keep_gate) * float(proposed_index)),
                            0,
                            self.enemy_target_dim - 1,
                        )
                    )
                else:
                    selected_index = proposed_index
                if last_target >= 0 and selected_index != last_target:
                    target_switches_tick += 1
                self._enemy_last_target_index[slot] = selected_index

                _ = (player_x, player_y, selected_index)

                # Model-authoritative decode with only absolute command safety bounds.
                speed = int(np.clip(np.rint(30.0 + 220.0 * act_speed_norm), 30, 250))
                fwd = int(np.clip(np.rint(100.0 * act_fwd_cmd), -100, 100))
                side = int(np.clip(np.rint(100.0 * act_side_cmd), -100, 100))
                turn = int(np.clip(np.rint(120.0 * act_turn_cmd), -120, 120))
                aim = int(np.clip(np.rint(100.0 * act_aim_cmd), -100, 100))
                fire = int(np.clip(np.rint(100.0 * act_fire_cmd), -100, 100))

                cadence_firecd = int(np.clip(np.rint(1.0 + 34.0 * act_firecd_norm), 1, 35))
                if fire > 0 and aim > 0 and self._enemy_fire_cooldown_left[slot] <= 0:
                    shots_tick += 1
                    self._enemy_fire_cooldown_left[slot] = cadence_firecd
                if cadence_firecd != self._last_enemy_cmds[slot]["firecd"]:
                    self.game.send_game_command(f"set nn_enemy_cmd_{slot:02d}_firecd {cadence_firecd}")
                    self._last_enemy_cmds[slot]["firecd"] = cadence_firecd

                healthpct = int(np.clip(np.rint(20.0 + 280.0 * act_health_norm), 20, 300))
                if healthpct != self._last_enemy_cmds[slot]["healthpct"]:
                    self.game.send_game_command(f"set nn_enemy_cmd_{slot:02d}_healthpct {healthpct}")
                    self._last_enemy_cmds[slot]["healthpct"] = healthpct

                self._update_enemy_memory_state(
                    slot=slot,
                    memory_update_logits=memory_updates[slot],
                    observed_identity_norm=float(self._enemy_slot_obs_identity[slot]),
                    observed_present=float(self._enemy_slot_obs_present[slot]),
                )
            else:
                speed = 100
                fwd = 0
                side = 0
                turn = 0
                aim = 0
                fire = 0
                self._enemy_last_target_index[slot] = -1
                self._enemy_prev_obs_identity[slot] = np.nan
                self._enemy_fire_cooldown_left[slot] = 0
                healthpct = 100
                self._update_enemy_memory_state(
                    slot=slot,
                    memory_update_logits=memory_updates[slot],
                    observed_identity_norm=0.0,
                    observed_present=0.0,
                )
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
        self._enemy_metric_ticks += 1
        self._enemy_metric_active_enemy_ticks += enemy_count
        self._enemy_metric_shots += shots_tick
        self._enemy_metric_target_switches += target_switches_tick
        self._enemy_metric_close_pairs += close_pairs_tick
        return enemy_count, commands

    def _wrap_angle_deg(self, angle_deg: float) -> float:
        return float((angle_deg + 180.0) % 360.0 - 180.0)

    def _world_sim_bootstrap(self, slot_enemies: list[object | None] | None = None) -> None:
        if slot_enemies is None:
            state = self.game.get_state()
            slot_enemies = self._refresh_enemy_slot_assignments(state)
        px, py = self._current_position()
        pangle = float(self.game.get_game_variable(GameVariable.ANGLE))
        self._world_player[0] = float(px)
        self._world_player[1] = float(py)
        self._world_player[2] = self._wrap_angle_deg(pangle)
        self._world_player[3] = float(np.clip(self.game.get_game_variable(GameVariable.HEALTH), 0.0, 200.0))
        self._world_enemies[:, :] = 0.0
        for slot in range(self.config.enemy_slots):
            enemy = slot_enemies[slot] if slot < len(slot_enemies) else None
            if enemy is None:
                continue
            self._world_enemies[slot, 0] = float(enemy.position_x)
            self._world_enemies[slot, 1] = float(enemy.position_y)
            self._world_enemies[slot, 2] = self._wrap_angle_deg(float(enemy.angle))
            self._world_enemies[slot, 3] = 100.0
            self._world_enemies[slot, 4] = 1.0
        self._world_kills = 0
        self._world_sim_tick = 0
        self._world_last_bridge_x = np.nan
        self._world_last_bridge_y = np.nan
        self._world_last_bridge_angle = np.nan
        self._world_sim_initialized = True

    def _world_sim_sync_slots(self, slot_enemies: list[object | None]) -> None:
        for slot in range(self.config.enemy_slots):
            enemy = slot_enemies[slot] if slot < len(slot_enemies) else None
            if enemy is None:
                continue
            if self._world_enemies[slot, 4] <= 0.0:
                self._world_enemies[slot, 0] = float(enemy.position_x)
                self._world_enemies[slot, 1] = float(enemy.position_y)
                self._world_enemies[slot, 2] = self._wrap_angle_deg(float(enemy.angle))
                self._world_enemies[slot, 3] = 100.0
                self._world_enemies[slot, 4] = 1.0

    def _world_sim_step(
        self,
        player_action: list[float],
        enemy_commands: list[tuple[int, int, int, int, int, int]],
        sub_steps: int | None = None,
    ) -> None:
        if not self._world_sim_initialized:
            self._world_sim_bootstrap()
        sim_sub_steps = max(1, self.config.action_repeat if sub_steps is None else sub_steps)
        for _ in range(sim_sub_steps):
            # Player kinematics from decoded control action.
            px = float(self._world_player[0])
            py = float(self._world_player[1])
            move_norm = float(
                np.clip(float(player_action[0]) / max(0.1, self.config.move_delta), -1.0, 1.0)
            )
            strafe_norm = float(
                np.clip(float(player_action[1]) / max(0.1, self.config.strafe_delta), -1.0, 1.0)
            )
            turn_norm = float(
                np.clip(float(player_action[2]) / max(0.1, self.config.turn_delta), -1.0, 1.0)
            )
            pangle = self._wrap_angle_deg(float(self._world_player[2] + 5.0 * turn_norm))
            self._world_player[2] = pangle
            prad = np.deg2rad(pangle)
            cos_a = float(np.cos(prad))
            sin_a = float(np.sin(prad))
            pdx = (cos_a * move_norm - sin_a * strafe_norm) * self.config.nn_move_units
            pdy = (sin_a * move_norm + cos_a * strafe_norm) * self.config.nn_move_units
            if self._collision_map is not None:
                dyn = [
                    (float(self._world_enemies[s, 0]), float(self._world_enemies[s, 1]), 20.0)
                    for s in range(self.config.enemy_slots)
                    if self._world_enemies[s, 4] > 0.5
                ]
                npx, npy = self._collision_map.resolve_motion(
                    px,
                    py,
                    pdx,
                    pdy,
                    radius=self.config.nn_player_radius,
                    dynamic_circles=dyn,
                )
                self._world_player[0] = float(npx)
                self._world_player[1] = float(npy)
            else:
                self._world_player[0] = float(px + pdx)
                self._world_player[1] = float(py + pdy)

            # Enemy kinematics from actuator-resolved commands.
            for slot in range(min(len(enemy_commands), self.config.enemy_slots)):
                if self._world_enemies[slot, 4] <= 0.5:
                    continue
                speed, fwd, side, turn, aim, fire = enemy_commands[slot]
                _ = (aim, fire)
                ex = float(self._world_enemies[slot, 0])
                ey = float(self._world_enemies[slot, 1])
                eangle = self._wrap_angle_deg(float(self._world_enemies[slot, 2] + 0.35 * float(turn)))
                self._world_enemies[slot, 2] = eangle
                erad = np.deg2rad(eangle)
                ecos = float(np.cos(erad))
                esin = float(np.sin(erad))
                speed_scale = float(np.clip(speed / 100.0, 0.35, 2.30))
                edx = (ecos * (fwd / 100.0) - esin * (side / 100.0)) * (5.0 * speed_scale)
                edy = (esin * (fwd / 100.0) + ecos * (side / 100.0)) * (5.0 * speed_scale)
                if self._collision_map is not None:
                    dyn: list[tuple[float, float, float]] = [
                        (float(self._world_player[0]), float(self._world_player[1]), 16.0)
                    ]
                    for other in range(self.config.enemy_slots):
                        if other == slot or self._world_enemies[other, 4] <= 0.5:
                            continue
                        dyn.append((float(self._world_enemies[other, 0]), float(self._world_enemies[other, 1]), 20.0))
                    nex, ney = self._collision_map.resolve_motion(
                        ex,
                        ey,
                        edx,
                        edy,
                        radius=20.0,
                        dynamic_circles=dyn,
                    )
                    self._world_enemies[slot, 0] = float(nex)
                    self._world_enemies[slot, 1] = float(ney)
                else:
                    self._world_enemies[slot, 0] = float(ex + edx)
                    self._world_enemies[slot, 1] = float(ey + edy)

            # Combat simulation: player -> enemies.
            player_fire = bool(len(player_action) > 4 and player_action[4] > 0.5 and self._world_player[3] > 0.0)
            if player_fire:
                pangle = float(self._world_player[2])
                px = float(self._world_player[0])
                py = float(self._world_player[1])
                best_slot = None
                best_dist = 1e9
                for slot in range(self.config.enemy_slots):
                    if self._world_enemies[slot, 4] <= 0.5:
                        continue
                    dx = float(self._world_enemies[slot, 0] - px)
                    dy = float(self._world_enemies[slot, 1] - py)
                    dist = float(np.hypot(dx, dy))
                    if dist > 900.0 or dist < 1e-4:
                        continue
                    bearing = float(np.degrees(np.arctan2(dy, dx)))
                    err = abs(self._wrap_angle_deg(bearing - pangle))
                    if err > 9.0:
                        continue
                    if dist < best_dist:
                        best_dist = dist
                        best_slot = slot
                if best_slot is not None:
                    self._world_enemies[best_slot, 3] = float(
                        max(0.0, self._world_enemies[best_slot, 3] - 22.0 * self.config.nn_world_damage_scale)
                    )
                    if self._world_enemies[best_slot, 3] <= 0.0 and self._world_enemies[best_slot, 4] > 0.5:
                        self._world_enemies[best_slot, 4] = 0.0
                        self._world_kills += 1

            # Combat simulation: enemies -> player.
            px = float(self._world_player[0])
            py = float(self._world_player[1])
            for slot in range(min(len(enemy_commands), self.config.enemy_slots)):
                if self._world_enemies[slot, 4] <= 0.5:
                    continue
                _speed, _fwd, _side, _turn, aim, fire = enemy_commands[slot]
                if not (aim > 0 and fire > 0):
                    continue
                ex = float(self._world_enemies[slot, 0])
                ey = float(self._world_enemies[slot, 1])
                eang = float(self._world_enemies[slot, 2])
                dx = px - ex
                dy = py - ey
                dist = float(np.hypot(dx, dy))
                if dist > 800.0 or dist < 1e-4:
                    continue
                bearing = float(np.degrees(np.arctan2(dy, dx)))
                err = abs(self._wrap_angle_deg(bearing - eang))
                if err > 14.0:
                    continue
                self._world_player[3] = float(max(0.0, self._world_player[3] - 5.0 * self.config.nn_world_damage_scale))

            self._world_sim_tick += 1

    def _world_sim_apply_render_bridge(self) -> None:
        if not self._world_sim_initialized:
            return
        px = float(self._world_player[0])
        py = float(self._world_player[1])
        pang = float(self._world_player[2])
        if (
            np.isnan(self._world_last_bridge_x)
            or np.isnan(self._world_last_bridge_y)
            or np.isnan(self._world_last_bridge_angle)
        ):
            do_bridge = True
        else:
            do_bridge = (
                abs(px - float(self._world_last_bridge_x)) > 0.08
                or abs(py - float(self._world_last_bridge_y)) > 0.08
                or abs(self._wrap_angle_deg(pang - float(self._world_last_bridge_angle))) > 0.20
            )
        if do_bridge:
            self.game.send_game_command(f"warp {px:.3f} {py:.3f}")
            self.game.send_game_command(f"setangle {pang:.3f}")
            self._world_last_bridge_x = px
            self._world_last_bridge_y = py
            self._world_last_bridge_angle = pang
        if not self.config.enemy_backend_transformer:
            return
        for slot in range(self.config.enemy_slots):
            if self._world_enemies[slot, 4] > 0.5:
                healthpct = int(np.clip(round(self._world_enemies[slot, 3]), 1, 200))
            else:
                healthpct = 0
            if healthpct != self._last_enemy_cmds[slot]["healthpct"]:
                self.game.send_game_command(f"set nn_enemy_cmd_{slot:02d}_healthpct {healthpct}")
                self._last_enemy_cmds[slot]["healthpct"] = healthpct

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

        # Tight decode: smaller ranges + smoothing to avoid jittery/sticky feel.
        move_scale_target = float(np.clip(1.0 + 0.10 * float(values[0]), 0.90, 1.12))
        turn_scale_target = float(np.clip(1.0 + 0.12 * float(values[1]), 0.88, 1.18))
        cooldown_scale = float(np.clip(1.0 + 0.35 * float(values[2]), 1.00, 1.80))
        fire_cooldown_target = float(
            np.clip(
                round(self.config.fire_cooldown_tics * cooldown_scale),
                self.config.fire_cooldown_tics,
                24,
            )
        )
        smooth = 0.18
        self._runtime_move_scale = float(
            (1.0 - smooth) * self._runtime_move_scale + smooth * move_scale_target
        )
        self._runtime_turn_scale = float(
            (1.0 - smooth) * self._runtime_turn_scale + smooth * turn_scale_target
        )
        self._runtime_fire_cooldown_float = float(
            (1.0 - smooth) * self._runtime_fire_cooldown_float + smooth * fire_cooldown_target
        )
        fire_cooldown = int(
            np.clip(
                round(self._runtime_fire_cooldown_float),
                self.config.fire_cooldown_tics,
                24,
            )
        )
        self._runtime_fire_cooldown_tics = fire_cooldown
        move_scale = self._runtime_move_scale
        turn_scale = self._runtime_turn_scale

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
        self._enemy_metric_ticks = 0
        self._enemy_metric_shots = 0
        self._enemy_metric_target_switches = 0
        self._enemy_metric_active_enemy_ticks = 0
        self._enemy_metric_close_pairs = 0
        self._enemy_metric_identity_churn = 0
        self._enemy_metric_identity_samples = 0
        self._mse_total_sum = 0.0
        self._mse_total_count = 0
        self._mse_head_sum = 0.0
        self._mse_head_count = 0
        self._mse_tail_values.clear()
        self._player_max_stuck_ticks = 0
        self.stuck_ticks = 0
        for slot in range(self.config.enemy_slots):
            self._reset_enemy_slot_runtime(slot)
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

        if self.stuck_ticks > self._player_max_stuck_ticks:
            self._player_max_stuck_ticks = self.stuck_ticks
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
            nn_gain = 0.90 + 0.10 * float(abs(scores[logit_idx]))
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
        sub_total = max(1, self.config.action_repeat)
        for sub_tick in range(sub_total):
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

    def _step_controls_world_sim(
        self,
        control_logits: np.ndarray,
        step: int,
        initial_keyboard_state: np.ndarray,
        enemy_commands: list[tuple[int, int, int, int, int, int]],
    ) -> tuple[float, list[float], np.ndarray]:
        total_reward = 0.0
        last_action = [0.0] * self.control_dim
        keyboard_state = initial_keyboard_state
        neutral_action = [0.0] * self.control_dim

        # Keep Doom ticking for rendering while Transformer world-sim owns movement/combat.
        sub_total = max(1, self.config.action_repeat)
        for sub_tick in range(sub_total):
            if sub_tick > 0:
                keyboard_state = self._sanitize_keyboard_state(self._read_keyboard_state())
            self._attack_cooldown_left = max(0, self._attack_cooldown_left - 1)

            if np.any(keyboard_state > 0.1):
                decoded_action = self._decode_controls(control_logits, step, keyboard_state)
                action = decoded_action.copy()
                self._apply_fire_cooldown(action, keyboard_state)
                self._world_sim_step(action, enemy_commands, sub_steps=1)
                last_action = action
            else:
                self._world_sim_step(neutral_action, enemy_commands, sub_steps=1)
                last_action = neutral_action.copy()

            total_reward += self.game.make_action(neutral_action, 1)
            if sub_tick == sub_total - 1:
                self._world_sim_apply_render_bridge()
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
        if self.config.nn_world_sim:
            print("NN movement resolution path is bypassed in --nn-world-sim mode.")
        elif self._nn_movement_resolution_active and self._collision_map is not None:
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
        if self.config.nn_world_sim:
            print(
                "Experimental NN world-sim mode is ON: Transformer loop owns low-level movement/collision/combat "
                "(Doom remains render/IO bridge)."
            )
            print(
                f"NN world-sim damage scale: {self.config.nn_world_damage_scale:.2f}"
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
        print(
            f"Doom ticrate: {self.config.doom_ticrate} | Doom skill: {self.config.doom_skill} | "
            f"Doom seed: {self.config.doom_seed if self.config.doom_seed is not None else 'auto'}"
        )
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
            f"total={self.enemy_feature_dim_total} + target_mask={self.enemy_target_mask_dim} "
            "(raw object-state features; slot lifecycle/identity persistence in model memory)"
        )
        print(
            f"Enemy behavior head (legacy diagnostic): channels={self.enemy_cmd_dim} "
            f"{'/'.join(self.enemy_behavior_channels)}"
        )
        print(
            f"Enemy intent head (legacy diagnostic): channels={self.enemy_intent_dim} "
            f"{'/'.join(self.enemy_intent_channels)}"
        )
        print(
            f"Enemy actuator head: channels={self.enemy_actuator_dim} "
            f"{'/'.join(self.enemy_actuator_channels)}"
        )
        print(
            f"Enemy memory-update head: channels={self.enemy_memory_update_dim} "
            f"(gate + {self.enemy_memory_feature_dim}-dim delta per slot)"
        )
        print(
            "Enemy intent head source: model context/diagnostics => "
            f"{'/'.join(self.enemy_intent_names)} + intent_timer_norm (not in final actuator path)"
        )
        print(
            "Enemy target source: model memory target_index_norm/target_keep_gate; "
            "Python only applies index-range safety clamp"
        )
        print(
            "Enemy fire timing source: direct actuator firecd command "
            "(integer projection + safety clamp only)"
        )
        print(
            "Enemy fire state machine source: direct actuator fire command"
        )
        print(
            "Enemy fire source: direct actuator channel projection (no Python policy shaping)"
        )
        print(
            "Enemy fire cadence source: direct actuator firecd channel"
        )
        print(
            "Enemy aim source: direct actuator aim channel"
        )
        print(
            "Enemy steering source: direct actuator speed/fwd/side/turn channels"
        )
        print(
            "Enemy command decode source: direct NN actuator channels; "
            "Python applies hard command clamps only"
        )
        print(
            "Enemy collision resolver: Python enemy-side collision resolution is disabled "
            "(no _resolve_enemy_motion_command in the actuation path)"
        )
        print(f"State dim: {self.state_dim} | Transformer context: {self.config.context}")

        self._prime_history()
        self.last_position = self._current_position()
        if self.config.nn_world_sim:
            self._world_sim_bootstrap()

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
                        _legacy_enemy_logits,
                        _legacy_enemy_intent_logits,
                        enemy_actuator_logits,
                        low_level_logits,
                        memory_update_logits,
                        _,
                    ) = self.model(inputs)
                predicted = predicted_state.cpu().numpy()[0]
                control = control_logits.cpu().numpy()[0]
                enemy_actuator = enemy_actuator_logits.cpu().numpy()[0]
                low_level = low_level_logits.cpu().numpy()[0]
                memory_update = memory_update_logits.cpu().numpy()[0]
                move_scale, turn_scale, fire_cd, enemy_low = self._apply_low_level_backend_controls(low_level)
                enemy_count, enemy_cmd = self._apply_enemy_backend_commands(
                    enemy_actuator,
                    memory_update,
                )

                keyboard_state = self._sanitize_keyboard_state(self._read_keyboard_state())
                if self.config.nn_world_sim:
                    reward, control_action, next_keyboard = self._step_controls_world_sim(
                        control,
                        step,
                        keyboard_state,
                        enemy_cmd,
                    )
                else:
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
                self._mse_total_sum += mse
                self._mse_total_count += 1
                if self._mse_head_count < self._mse_head_window:
                    self._mse_head_sum += mse
                    self._mse_head_count += 1
                self._mse_tail_values.append(mse)
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
            if self.config.enemy_backend_transformer and self._enemy_metric_ticks > 0:
                shots_per_tick = self._enemy_metric_shots / float(self._enemy_metric_ticks)
                switches_per_tick = self._enemy_metric_target_switches / float(self._enemy_metric_ticks)
                switches_per_enemy_tick = self._enemy_metric_target_switches / float(max(1, self._enemy_metric_active_enemy_ticks))
                close_pairs_per_tick = self._enemy_metric_close_pairs / float(self._enemy_metric_ticks)
                identity_churn_rate = self._enemy_metric_identity_churn / float(max(1, self._enemy_metric_identity_samples))
                mse_mean = self._mse_total_sum / float(max(1, self._mse_total_count))
                head_mean = self._mse_head_sum / float(max(1, self._mse_head_count))
                tail_mean = (
                    float(np.mean(np.asarray(self._mse_tail_values, dtype=np.float32)))
                    if len(self._mse_tail_values) > 0
                    else 0.0
                )
                mse_drift = tail_mean - head_mean
                print(
                    "Enemy regression metrics: "
                    f"ticks={self._enemy_metric_ticks} "
                    f"shots_per_tick={shots_per_tick:.3f} "
                    f"target_switches_per_tick={switches_per_tick:.3f} "
                    f"target_switches_per_enemy_tick={switches_per_enemy_tick:.4f} "
                    f"close_pairs_per_tick={close_pairs_per_tick:.3f} "
                    f"identity_churn_rate={identity_churn_rate:.4f} "
                    f"player_max_stuck_ticks={self._player_max_stuck_ticks} "
                    f"mse_mean={mse_mean:.6f} "
                    f"mse_drift={mse_drift:.6f}"
                )
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
        default=10,
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
        "--doom-seed",
        type=int,
        default=None,
        help="Optional deterministic Doom RNG seed.",
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
        default=4.4,
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
        default=0.42,
        help="Scale for hardcoded parameter initialization std (lower = tighter/stabler).",
    )
    parser.add_argument(
        "--nn-control-gain",
        type=float,
        default=0.85,
        help="Control head logit gain before tanh decoding.",
    )
    parser.add_argument(
        "--nn-enemy-gain",
        type=float,
        default=0.62,
        help="Enemy head logit gain before tanh decoding.",
    )
    parser.add_argument(
        "--nn-low-level-gain",
        type=float,
        default=0.25,
        help="Low-level head logit gain before tanh decoding.",
    )
    parser.add_argument(
        "--nn-world-sim",
        action="store_true",
        help=(
            "Experimental mode: move low-level movement/collision/combat step into Transformer-side "
            "world simulator, while Doom remains rendering/input bridge."
        ),
    )
    parser.add_argument(
        "--nn-world-damage-scale",
        type=float,
        default=1.0,
        help="Damage multiplier used by experimental NN world simulator.",
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
        doom_seed=args.doom_seed,
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
        nn_world_sim=bool(args.nn_world_sim),
        nn_world_damage_scale=float(np.clip(args.nn_world_damage_scale, 0.1, 4.0)),
    )


def main() -> None:
    config = parse_args()
    loop = DoomTransformerLoop(config)
    loop.run()


if __name__ == "__main__":
    main()
