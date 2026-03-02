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

    # Doom-format manual door linedef specials (use/switch activated).
    # Keep this tight to avoid blocking generic trigger lines.
    DOOR_MANUAL_SPECIALS = {
        1,    # DR Door Open Wait Close
        26,   # Blue key door
        27,   # Yellow key door
        28,   # Red key door
        31,   # D1 Door Open Stay
        32,   # Blue key door (open stay)
        33,   # Red key door (open stay)
        34,   # Yellow key door (open stay)
        63,   # SR Door Open Stay
        103,  # S1 Door Open Stay
        117,  # Blazing door variants
        118,
    }

    def __init__(self, wad_path: Path, map_name: str) -> None:
        self.blocking_segments, self.door_segments = self._load_blocking_segments(wad_path, map_name)
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
    def _load_blocking_segments(
        wad_path: Path,
        map_name: str,
    ) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float, float, float]]]:
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
        door_segments: list[tuple[float, float, float, float]] = []
        for i in range(l_size // 14):
            v1, v2, flags, special, _tag, right, left = struct.unpack_from("<hhhhhhh", raw, l_off + i * 14)
            if v1 < 0 or v2 < 0 or v1 >= len(vertices) or v2 >= len(vertices):
                continue
            x1, y1 = vertices[v1]
            x2, y2 = vertices[v2]
            is_manual_door = int(special) in WADCollisionMap.DOOR_MANUAL_SPECIALS
            if is_manual_door:
                door_segments.append((x1, y1, x2, y2))
            one_sided = right == -1 or left == -1
            impassable = bool(flags & 0x0001)  # ML_BLOCKING
            if is_manual_door:
                # Manual door lines are handled by dynamic door gating in world-sim.
                # Do not keep them in permanent static blockers.
                continue
            if not one_sided and not impassable:
                continue
            segments.append((x1, y1, x2, y2))
        return segments, door_segments

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
        extra_blocking_segments: list[tuple[float, float, float, float]] | None = None,
    ) -> bool:
        r2 = radius * radius
        for x1, y1, x2, y2 in self.blocking_segments:
            if self._point_segment_distance_sq(x, y, x1, y1, x2, y2) < r2:
                return True
        if extra_blocking_segments is not None:
            for x1, y1, x2, y2 in extra_blocking_segments:
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
        extra_blocking_segments: list[tuple[float, float, float, float]] | None = None,
    ) -> tuple[float, float]:
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return x, y

        # Substep long moves to reduce wall tunneling in strict NN world-sim.
        # Keep this deterministic and bounded for stable realtime behavior.
        dist = float(np.hypot(dx, dy))
        step_len = 2.0
        steps = int(np.clip(np.ceil(dist / step_len), 1, 32))
        sx = dx / float(steps)
        sy = dy / float(steps)

        cx = x
        cy = y
        for _ in range(steps):
            nx, ny = self._resolve_motion_single_step(
                cx,
                cy,
                sx,
                sy,
                radius,
                dynamic_circles,
                extra_blocking_segments,
            )
            if abs(nx - cx) < 1e-6 and abs(ny - cy) < 1e-6:
                break
            cx, cy = nx, ny
        return cx, cy

    def _resolve_motion_single_step(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        radius: float,
        dynamic_circles: list[tuple[float, float, float]] | None = None,
        extra_blocking_segments: list[tuple[float, float, float, float]] | None = None,
    ) -> tuple[float, float]:
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return x, y

        nx = x + dx
        ny = y + dy
        if not self._collides(nx, ny, radius, dynamic_circles, extra_blocking_segments):
            return nx, ny

        x_only = (x + dx, y)
        y_only = (x, y + dy)
        x_ok = not self._collides(x_only[0], x_only[1], radius, dynamic_circles, extra_blocking_segments)
        y_ok = not self._collides(y_only[0], y_only[1], radius, dynamic_circles, extra_blocking_segments)
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
            if not self._collides(sx, sy, radius, dynamic_circles, extra_blocking_segments):
                return sx, sy
            sx_only = x + dx * scale
            sy_only = y + dy * scale
            if not self._collides(sx_only, y, radius, dynamic_circles, extra_blocking_segments):
                return sx_only, y
            if not self._collides(x, sy_only, radius, dynamic_circles, extra_blocking_segments):
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
        self._diag_panel_width = 460
        self._show_diagnostics = False
        self._window = self._pg.display.set_mode((width, height))
        self._pg.display.set_caption("Transformer Doom (focus this window)")
        self._pressed_keys: set[int] = set()
        self._font = self._pg.font.SysFont("Menlo", 16)
        self._small_font = self._pg.font.SysFont("Menlo", 13)
        self._diag_toggle_tick = 0
        self.closed = False

    def read(self) -> np.ndarray:
        for event in self._pg.event.get():
            if event.type == self._pg.QUIT:
                self.closed = True
                self._pressed_keys.clear()
            elif event.type == self._pg.KEYDOWN:
                if int(event.key) == int(self._pg.K_F1):
                    now = self._pg.time.get_ticks()
                    if now - self._diag_toggle_tick < 120:
                        continue
                    self._diag_toggle_tick = now
                    self._show_diagnostics = not self._show_diagnostics
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

    def _ensure_window_shape(self, diagnostics_active: bool) -> None:
        target_w = self._width + (self._diag_panel_width if diagnostics_active else 0)
        current_w, current_h = self._window.get_size()
        if current_w != target_w or current_h != self._height:
            self._window = self._pg.display.set_mode((target_w, self._height))

    def _draw_text(self, surface: object, text: str, x: int, y: int, small: bool = False) -> None:
        font = self._small_font if small else self._font
        surface.blit(font.render(text, True, (220, 220, 220)), (x, y))

    def _draw_control_bars(self, panel: object, control: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        labels = ["fwd", "side", "turn", "look", "fire", "use"]
        rows = min(len(labels), int(control.size))
        if rows <= 0:
            return
        row_h = max(10, h // rows)
        for i in range(rows):
            yy = y + i * row_h
            val = float(np.clip(control[i], -1.0, 1.0))
            self._draw_text(panel, f"{labels[i]} {val:+.2f}", x, yy, small=True)
            bar_x = x + 110
            bar_y = yy + 2
            bar_w = w - 120
            bar_h = max(6, row_h - 6)
            self._pg.draw.rect(panel, (65, 65, 65), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
            zero_x = bar_x + bar_w // 2
            self._pg.draw.line(panel, (120, 120, 120), (zero_x, bar_y), (zero_x, bar_y + bar_h), 1)
            fill_w = int(abs(val) * (bar_w // 2))
            if fill_w <= 0:
                continue
            if val >= 0:
                self._pg.draw.rect(
                    panel,
                    (90, 190, 120),
                    (zero_x, bar_y, fill_w, bar_h),
                    border_radius=3,
                )
            else:
                self._pg.draw.rect(
                    panel,
                    (210, 120, 90),
                    (zero_x - fill_w, bar_y, fill_w, bar_h),
                    border_radius=3,
                )

    def _draw_attention_grid(
        self,
        panel: object,
        attention: np.ndarray,
        x: int,
        y: int,
        cell_w: int,
        cell_h: int,
    ) -> None:
        if attention.ndim != 3 or attention.shape[0] <= 0:
            self._draw_text(panel, "No attention tensors", x, y)
            return
        heads_to_draw = min(4, attention.shape[0])
        cols = 2
        rows = 2
        for idx in range(heads_to_draw):
            r = idx // cols
            c = idx % cols
            hx = x + c * (cell_w + 8)
            hy = y + r * (cell_h + 26)
            mat = np.asarray(attention[idx], dtype=np.float32)
            mat = np.nan_to_num(mat, nan=0.0, posinf=1.0, neginf=0.0)
            mn = float(np.min(mat))
            mx = float(np.max(mat))
            if mx - mn < 1e-8:
                norm = np.zeros_like(mat, dtype=np.float32)
            else:
                norm = (mat - mn) / (mx - mn)
            heat = np.clip(np.rint(norm * 255.0), 0, 255).astype(np.uint8)
            rgb = np.stack([heat, np.clip((64 + heat // 2), 0, 255), (255 - heat)], axis=-1)
            surf = self._pg.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
            surf = self._pg.transform.scale(surf, (cell_w, cell_h))
            panel.blit(surf, (hx, hy))
            self._draw_text(panel, f"H{idx} min={mn:.3f} max={mx:.3f}", hx, hy + cell_h + 3, small=True)

    def _render_diagnostics_panel(self, diagnostics: dict[str, object]) -> object:
        panel = self._pg.Surface((self._diag_panel_width, self._height))
        panel.fill((18, 20, 24))
        self._draw_text(panel, "Diagnostics (F1 toggle)", 14, 12)
        tick = int(diagnostics.get("tick", -1))
        mse = float(diagnostics.get("mse", 0.0))
        self._draw_text(panel, f"tick={tick}  mse={mse:.6f}", 14, 36, small=True)

        keys = np.asarray(diagnostics.get("keys", np.zeros(10, dtype=np.float32)), dtype=np.float32)
        pressed = int(np.sum(keys > 0.1))
        self._draw_text(panel, f"keys_pressed={pressed}", 14, 54, small=True)

        control = np.asarray(diagnostics.get("control", np.zeros(6, dtype=np.float32)), dtype=np.float32)
        self._draw_text(panel, "Control Decode", 14, 78)
        self._draw_control_bars(panel, control, x=14, y=100, w=self._diag_panel_width - 24, h=120)

        self._draw_text(panel, "Attention (last layer, live)", 14, 234)
        attention = np.asarray(diagnostics.get("attention", np.zeros((0, 0, 0), dtype=np.float32)), dtype=np.float32)
        grid_w = self._diag_panel_width - 30
        cell_w = (grid_w - 8) // 2
        cell_h = 120
        self._draw_attention_grid(panel, attention, x=14, y=258, cell_w=cell_w, cell_h=cell_h)
        return panel

    def render_rgb_frame(self, frame: np.ndarray, diagnostics: dict[str, object] | None = None) -> None:
        if self.closed:
            return
        diagnostics_active = self._show_diagnostics and (diagnostics is not None)
        self._ensure_window_shape(diagnostics_active)
        # frame is HxWx3 RGB from VizDoom.
        surface = self._pg.image.frombuffer(frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")
        if frame.shape[1] != self._width or frame.shape[0] != self._height:
            surface = self._pg.transform.scale(surface, (self._width, self._height))
        self._window.blit(surface, (0, 0))
        if diagnostics_active and diagnostics is not None:
            panel = self._render_diagnostics_panel(diagnostics)
            self._window.blit(panel, (self._width, 0))
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
        enemy_bind_dim: int = 17,
        enemy_target_dim: int = 17,
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
        self.enemy_bind_dim = max(1, int(enemy_bind_dim))
        self.enemy_target_dim = max(1, int(enemy_target_dim))
        self.enemy_actuator_dim = max(1, int(enemy_actuator_dim))
        self.low_level_dim = low_level_dim if low_level_dim is not None else (4 + enemy_slots)
        self.memory_update_dim = max(1, int(memory_update_dim))
        self.weight_scale = float(np.clip(weight_scale, 0.05, 2.0))
        self.in_proj = nn.Linear(state_dim, d_model)
        self.state_out_proj = nn.Linear(d_model, state_dim)
        self.control_out_proj = nn.Linear(d_model, control_dim)
        self.enemy_out_proj = nn.Linear(d_model, enemy_slots * enemy_cmd_dim)
        self.enemy_bind_out_proj = nn.Linear(d_model, enemy_slots * self.enemy_bind_dim)
        self.enemy_target_out_proj = nn.Linear(d_model, enemy_slots * self.enemy_target_dim)
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
                elif "enemy_bind_out_proj.weight" in name:
                    std = 0.03
                elif "enemy_target_out_proj.weight" in name:
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
        enemy_bind_logits = self.enemy_bind_out_proj(head).view(-1, self.enemy_slots, self.enemy_bind_dim)
        enemy_target_logits = self.enemy_target_out_proj(head).view(-1, self.enemy_slots, self.enemy_target_dim)
        enemy_actuator_logits = self.enemy_actuator_out_proj(head).view(-1, self.enemy_slots, self.enemy_actuator_dim)
        low_level_logits = self.low_level_out_proj(head)
        memory_update_logits = self.memory_out_proj(head).view(-1, self.enemy_slots, self.memory_update_dim)
        return (
            self.state_out_proj(head),
            self.control_out_proj(head),
            enemy_logits,
            enemy_actuator_logits,
            enemy_bind_logits,
            enemy_target_logits,
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
    shuffle_enemy_observations: bool
    enemy_kinematics_transformer: bool
    enemy_combat_transformer: bool
    nn_movement_resolution: bool
    nn_move_units: float
    nn_player_radius: float
    nn_weight_scale: float
    nn_control_gain: float
    nn_enemy_gain: float
    nn_low_level_gain: float
    nn_world_sim: bool
    nn_world_sim_strict: bool
    nn_world_sim_pure: bool
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
        # Behavior head kept for compatibility/diagnostics; authoritative targeting
        # now comes from explicit target-logit head.
        self.enemy_behavior_core_channels = ("legacy_target_actor_id_raw",)
        self.enemy_obs_max = self.config.enemy_slots
        self.enemy_target_channels = ("target_player_logit",) + tuple(
            f"target_obs_{idx:02d}_logit" for idx in range(self.enemy_obs_max)
        )
        self.enemy_bind_channels = tuple(
            f"bind_obs_{idx:02d}_logit" for idx in range(self.enemy_obs_max)
        ) + ("bind_empty_logit",)
        self.enemy_behavior_channels = self.enemy_behavior_core_channels
        self.enemy_behavior_core_dim = len(self.enemy_behavior_core_channels)
        self.enemy_bind_dim = len(self.enemy_bind_channels)
        self.enemy_target_dim = len(self.enemy_target_channels)
        self.enemy_target_mask_dim = 0
        self.enemy_cmd_dim = len(self.enemy_behavior_channels)
        self.enemy_intent_channels: tuple[str, ...] = ()
        self.enemy_intent_dim = 0
        self.enemy_actuator_channels = (
            "act_speed_norm",
            "act_fwd_cmd",
            "act_side_cmd",
            "act_turn_cmd",
            "act_aim_norm",
            "act_fire_norm",
            "act_firecd_norm",
            "act_health_norm",
        )
        self.enemy_actuator_dim = len(self.enemy_actuator_channels)
        self.enemy_obs_feature_dim = 9
        self.enemy_base_feature_dim = self.enemy_obs_feature_dim
        self.enemy_feedback_feature_dim = 0
        self.enemy_memory_feature_dim = 4
        self.enemy_memory_channels: tuple[str, ...] = ()
        self.enemy_binding_dim = self.enemy_memory_feature_dim
        self.enemy_memory_update_dim = 1 + self.enemy_memory_feature_dim
        self.enemy_feature_dim = self.enemy_obs_feature_dim
        self.enemy_observed_feature_dim_total = self.enemy_obs_max * self.enemy_obs_feature_dim
        self.enemy_memory_feature_dim_total = self.config.enemy_slots * self.enemy_memory_feature_dim
        self.enemy_feature_dim_total = self.enemy_observed_feature_dim_total + self.enemy_memory_feature_dim_total
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
        self._last_enemy_cmds: list[dict[str, float]] = [
            {
                "speed_norm": 9.0,
                "fwd_norm": 9.0,
                "side_norm": 9.0,
                "turn_norm": 9.0,
                "aim_norm": 9.0,
                "fire_norm": 9.0,
                "firecd_norm": 9.0,
                "health_norm": 9.0,
                "x_raw": 9_999_999.0,
                "y_raw": 9_999_999.0,
                "angle_raw": 9_999_999.0,
                "target_actor_id_raw": 9999.0,
                "actor_id_raw": -9999.0,
                "present_norm": 9.0,
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
        self._enemy_last_bind_obs = np.full(self.config.enemy_slots, -1, dtype=np.int32)
        self._enemy_kin_actor_id = np.full(self.config.enemy_slots, -1.0, dtype=np.float32)
        self._enemy_combat_player_cooldown = 0
        self._enemy_combat_player_iframe = 0
        self._combat_player_health = 100.0
        self._combat_player_dead = False
        self._combat_player_sync_initialized = False
        self._enemy_fire_phase = np.zeros(self.config.enemy_slots, dtype=np.float32)
        self._enemy_fire_threshold = np.ones(self.config.enemy_slots, dtype=np.float32)
        self._enemy_obs_actor_id = np.full(self.enemy_obs_max, -1.0, dtype=np.float32)
        self._enemy_obs_present = np.zeros(self.enemy_obs_max, dtype=np.float32)
        self._enemy_obs_tokens = np.zeros((self.enemy_obs_max, self.enemy_obs_feature_dim), dtype=np.float32)
        self._enemy_obs_objects: list[object | None] = [None for _ in range(self.enemy_obs_max)]
        shuffle_seed = int(self.config.doom_seed) if self.config.doom_seed is not None else 1337
        self._enemy_obs_rng = np.random.default_rng(shuffle_seed)
        self._enemy_memory = np.zeros(
            (self.config.enemy_slots, self.enemy_memory_feature_dim),
            dtype=np.float32,
        )
        self._enemy_slot_obs_identity = np.zeros(self.config.enemy_slots, dtype=np.float32)
        self._enemy_slot_obs_present = np.zeros(self.config.enemy_slots, dtype=np.float32)
        self._enemy_metric_ticks = 0
        self._enemy_metric_shots = 0.0
        self._enemy_metric_target_switches = 0
        self._enemy_metric_active_enemy_ticks = 0
        self._enemy_metric_close_pairs = 0
        self._enemy_metric_identity_churn = 0
        self._enemy_metric_identity_samples = 0
        self._enemy_prev_obs_identity = np.full(self.config.enemy_slots, np.nan, dtype=np.float32)
        self._mse_total_sum = 0.0
        self._mse_total_count = 0
        self._mse_head_window = 256
        self._mse_head_sum = 0.0
        self._mse_head_count = 0
        self._last_mse = 0.0
        self._diagnostics_payload: dict[str, object] | None = None
        self._mse_tail_window = 256
        self._mse_tail_values: Deque[float] = deque(maxlen=self._mse_tail_window)
        self._world_sim_initialized = False
        self._world_sim_tick = 0
        self._world_player = np.zeros(4, dtype=np.float32)  # x, y, angle_deg, health
        self._world_player_z = 0.0
        self._world_enemies = np.zeros((self.config.enemy_slots, 5), dtype=np.float32)  # x, y, angle_deg, health, alive
        self._world_kills = 0
        self._world_last_bridge_x = np.nan
        self._world_last_bridge_y = np.nan
        self._world_last_bridge_angle = np.nan
        self._last_player_cmd_x = np.nan
        self._last_player_cmd_y = np.nan
        self._last_player_cmd_angle = np.nan
        self._last_player_cmd_fire = np.nan
        self._last_player_cmd_use = np.nan
        self._latest_target_mask = np.zeros(self.enemy_target_mask_dim, dtype=np.float32)
        if self.enemy_target_mask_dim > 0:
            self._latest_target_mask[0] = 100.0
        self._nn_movement_resolution_active = bool(self.config.nn_movement_resolution)
        self._init_keyboard_source()
        if self.config.nn_world_sim_strict and self.config.visible and self.keyboard_source == "doom_buttons":
            # Strict world-sim cannot rely on raw doom_buttons alone because action-loop
            # neutral ticks can mask sampled key state; require global or pygame capture.
            try:
                self.pygame_keyboard_sampler = PygameKeyboardSampler(
                    self.config.frame_width,
                    self.config.frame_height,
                )
                self.keyboard_source = "pygame_window"
            except Exception as exc:
                raise RuntimeError(
                    "--nn-world-sim-strict requires non-doom_buttons keyboard sampling. "
                    "Use --keyboard-source pygame_window or grant macOS Accessibility "
                    "for --keyboard-source auto/macos_global."
                ) from exc
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
            enemy_bind_dim=self.enemy_bind_dim,
            enemy_target_dim=self.enemy_target_dim,
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
        self._doom_button_zero_ticks = 0
        self._stale_key_ticks = 0
        self._attack_cooldown_left = 0
        self._strict_turn_hold = 0
        self._control_logit_gain = self.config.nn_control_gain
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
            self._map_geometry
            if (
                self._nn_movement_resolution_active
                or self.config.nn_world_sim
                or self.config.enemy_kinematics_transformer
            )
            else None
        )
        door_count = len(self._collision_map.door_segments) if self._collision_map is not None else 0
        self._world_door_open_ticks = np.zeros(door_count, dtype=np.int32)

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
            game.send_game_command(
                f"set nn_enemy_kinematics_transformer {1 if self.config.enemy_kinematics_transformer else 0}"
            )
            game.send_game_command(
                f"set nn_enemy_combat_transformer {1 if self.config.enemy_combat_transformer else 0}"
            )
            # Keep player override disabled in runtime for stability in pure path.
            # Player pose is bridged via warp + native turn/fire/use deltas.
            game.send_game_command("set nn_player_override 0")
            game.send_game_command("set nn_player_fire_raw 0")
            game.send_game_command("set nn_player_use_raw 0")
        game.new_episode()
        self._bind_keyboard_controls(game)
        return game

    def _ensure_keyboard_config_path(self) -> Path:
        config_path = Path.cwd() / "transformer_controls.cfg"
        if self.config.nn_world_sim_strict:
            # Strict mode keeps movement/fire authoritative in Transformer world-sim.
            # Prevent native key binds from fighting world-sim warp/collision.
            lines = ["unbindall"]
        elif self._nn_movement_resolution_active and not self._uses_doom_buttons_source():
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
            if self.config.visible:
                try:
                    sampler = MacOSKeyboardSampler()
                    if sampler.is_accessibility_trusted():
                        self.macos_keyboard_sampler = sampler
                        self.keyboard_permission_ok = True
                        self.keyboard_source = "macos_global+doom_buttons"
                except Exception:
                    pass
            return

        if preferred == "auto":
            self.keyboard_source = "doom_buttons"
            if not self.config.visible:
                return
            # Prefer a single-window setup: Doom buttons remain primary and
            # macOS global sampling is merged when Accessibility is available.
            try:
                sampler = MacOSKeyboardSampler()
                trusted = sampler.is_accessibility_trusted()
                if trusted:
                    self.macos_keyboard_sampler = sampler
                    self.keyboard_permission_ok = True
                    self.keyboard_source = "macos_global+doom_buttons"
                    return
                self.keyboard_permission_ok = False
            except Exception:
                pass
            # Fallback to pygame window capture when global key access is unavailable.
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
            return

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
        if self.config.nn_world_sim_strict:
            return
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
        if self._uses_doom_buttons_source():
            # Some VizDoom builds can report sparse/zero states for discrete movement
            # buttons even when native key binds are active. Reconstruct keyboard intent
            # from delta-axis buttons as a reliable fallback.
            move_mag = float(
                abs(doom_keys[0]) + abs(doom_keys[1]) + abs(doom_keys[2]) + abs(doom_keys[3])
            )
            turn_mag = float(abs(doom_keys[4]) + abs(doom_keys[5]))
            if move_mag < 1e-4 and turn_mag < 1e-4:
                try:
                    fb = float(self.game.get_button(Button.MOVE_FORWARD_BACKWARD_DELTA))
                except Exception:
                    fb = 0.0
                try:
                    lr = float(self.game.get_button(Button.MOVE_LEFT_RIGHT_DELTA))
                except Exception:
                    lr = 0.0
                try:
                    tr = float(self.game.get_button(Button.TURN_LEFT_RIGHT_DELTA))
                except Exception:
                    tr = 0.0
                try:
                    lk = float(self.game.get_button(Button.LOOK_UP_DOWN_DELTA))
                except Exception:
                    lk = 0.0
                # Preserve attack/use from discrete buttons (indices 8,9).
                attack_val = float(doom_keys[8]) if doom_keys.shape[0] > 8 else 0.0
                use_val = float(doom_keys[9]) if doom_keys.shape[0] > 9 else 0.0
                eps = 1e-4
                doom_keys = np.asarray(
                    [
                        1.0 if fb > eps else 0.0,   # forward
                        1.0 if fb < -eps else 0.0,  # backward
                        1.0 if lr < -eps else 0.0,  # move left
                        1.0 if lr > eps else 0.0,   # move right
                        1.0 if tr < -eps else 0.0,  # turn left
                        1.0 if tr > eps else 0.0,   # turn right
                        1.0 if lk < -eps else 0.0,  # look up
                        1.0 if lk > eps else 0.0,   # look down
                        1.0 if attack_val > 0.1 else 0.0,
                        1.0 if use_val > 0.1 else 0.0,
                    ],
                    dtype=np.float32,
                )
            if float(np.max(np.abs(doom_keys))) <= 1e-6:
                self._doom_button_zero_ticks += 1
            else:
                self._doom_button_zero_ticks = 0
            if (
                self.config.visible
                and self.pygame_keyboard_sampler is None
                and self._doom_button_zero_ticks >= 180
            ):
                try:
                    self.pygame_keyboard_sampler = PygameKeyboardSampler(
                        self.config.frame_width,
                        self.config.frame_height,
                    )
                    self.keyboard_source = "pygame_window"
                    print(
                        "Keyboard fallback: doom_buttons stayed inactive; switched to pygame_window input. "
                        "Focus the pygame window for WASD."
                    )
                    keys = self.pygame_keyboard_sampler.read()
                    if self.pygame_keyboard_sampler.closed:
                        self.running = False
                    return keys
                except Exception:
                    pass
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

    def _safe_game_variable(self, var_name: str, fallback: float = 0.0) -> float:
        try:
            variable = getattr(GameVariable, var_name)
        except AttributeError:
            return float(fallback)
        try:
            return float(self.game.get_game_variable(variable))
        except (TypeError, ValueError, RuntimeError):
            return float(fallback)

    def _reset_enemy_slot_runtime(self, slot: int) -> None:
        if slot < 0 or slot >= self.config.enemy_slots:
            return
        self._enemy_prev_x[slot] = np.nan
        self._enemy_prev_y[slot] = np.nan
        self._enemy_prev_angle[slot] = np.nan
        self._enemy_prev_health_obs[slot] = np.nan
        self._enemy_last_bind_obs[slot] = -1
        self._enemy_kin_actor_id[slot] = -1.0
        self._enemy_fire_phase[slot] = 0.0
        self._enemy_fire_threshold[slot] = 1.0
        self._enemy_slot_obs_identity[slot] = 0.0
        self._enemy_slot_obs_present[slot] = 0.0
        self._enemy_prev_obs_identity[slot] = np.nan
        self._enemy_last_cmd[slot, :] = 0.0
        self._enemy_last_target_index[slot] = -1
        self._enemy_memory[slot, :] = 0.0
        self._last_enemy_cmds[slot]["speed_norm"] = 9.0
        self._last_enemy_cmds[slot]["fwd_norm"] = 9.0
        self._last_enemy_cmds[slot]["side_norm"] = 9.0
        self._last_enemy_cmds[slot]["turn_norm"] = 9.0
        self._last_enemy_cmds[slot]["aim_norm"] = 9.0
        self._last_enemy_cmds[slot]["fire_norm"] = 9.0
        self._last_enemy_cmds[slot]["firecd_norm"] = 9.0
        self._last_enemy_cmds[slot]["health_norm"] = 9.0
        self._last_enemy_cmds[slot]["x_raw"] = 9_999_999.0
        self._last_enemy_cmds[slot]["y_raw"] = 9_999_999.0
        self._last_enemy_cmds[slot]["angle_raw"] = 9_999_999.0
        self._last_enemy_cmds[slot]["health_raw"] = 9_999_999.0
        self._last_enemy_cmds[slot]["dead_raw"] = 9.0
        self._last_enemy_cmds[slot]["target_actor_id_raw"] = 9999.0
        self._last_enemy_cmds[slot]["actor_id_raw"] = -9999.0
        self._last_enemy_cmds[slot]["present_norm"] = 9.0
        self._last_enemy_cmds[slot]["healthpct"] = -1

    def _enemy_observation_binding_token(self, enemy: object) -> np.ndarray:
        # Raw-ish object token used by model-memory slot binding. No hand-tuned
        # policy rules are applied here; this is a deterministic feature projection.
        actor_id = self._enemy_attr_float(enemy, "id", 0.0)
        actor_type = self._enemy_attr_float(enemy, "type", 0.0)
        ex = self._enemy_attr_float(enemy, "position_x")
        ey = self._enemy_attr_float(enemy, "position_y")
        token = np.asarray(
            [
                np.tanh(actor_id / 32768.0),
                np.tanh(actor_type / 1024.0),
                np.tanh(ex / 4096.0),
                np.tanh(ey / 4096.0),
            ],
            dtype=np.float32,
        )
        return token

    def _enemy_observation_token(self, enemy: object) -> np.ndarray:
        actor_id = self._enemy_attr_float(enemy, "id", 0.0)
        actor_type = self._enemy_attr_float(enemy, "type", 0.0)
        ex = self._enemy_attr_float(enemy, "position_x")
        ey = self._enemy_attr_float(enemy, "position_y")
        ez = self._enemy_attr_float(enemy, "position_z")
        vx = self._enemy_attr_float(enemy, "velocity_x")
        vy = self._enemy_attr_float(enemy, "velocity_y")
        vz = self._enemy_attr_float(enemy, "velocity_z")
        health_obs = self._enemy_attr_float(enemy, "health", 0.0)
        return np.asarray(
            [
                actor_id,
                actor_type,
                ex,
                ey,
                ez,
                vx,
                vy,
                vz,
                health_obs,
            ],
            dtype=np.float32,
        )

    def _refresh_enemy_observations(self, state: object | None) -> None:
        enemies = self._enemy_objects_from_state(state)
        if self.config.shuffle_enemy_observations and len(enemies) > 1:
            order = self._enemy_obs_rng.permutation(len(enemies))
            enemies = [enemies[int(idx)] for idx in order.tolist()]
        enemies = enemies[: self.enemy_obs_max]

        self._enemy_obs_actor_id.fill(-1.0)
        self._enemy_obs_present.fill(0.0)
        self._enemy_obs_tokens.fill(0.0)
        self._enemy_obs_objects = [None for _ in range(self.enemy_obs_max)]

        for obs_idx, enemy in enumerate(enemies):
            token = self._enemy_observation_token(enemy)
            self._enemy_obs_tokens[obs_idx, :] = token
            self._enemy_obs_actor_id[obs_idx] = float(token[0])
            self._enemy_obs_present[obs_idx] = 1.0
            self._enemy_obs_objects[obs_idx] = enemy

    @staticmethod
    def _hungarian_maximize(scores: np.ndarray) -> np.ndarray:
        matrix = np.asarray(scores, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("scores must be a 2D matrix")
        n_rows, n_cols = matrix.shape
        if n_rows == 0:
            return np.zeros(0, dtype=np.int32)
        if n_cols < n_rows:
            raise ValueError("assignment matrix must have at least as many columns as rows")

        # Convert max-score assignment to min-cost assignment.
        safe_scores = np.nan_to_num(matrix, nan=-1e9, posinf=1e9, neginf=-1e9)
        cost = -safe_scores

        u = np.zeros(n_rows + 1, dtype=np.float64)
        v = np.zeros(n_cols + 1, dtype=np.float64)
        p = np.zeros(n_cols + 1, dtype=np.int32)
        way = np.zeros(n_cols + 1, dtype=np.int32)

        for i in range(1, n_rows + 1):
            p[0] = i
            minv = np.full(n_cols + 1, np.inf, dtype=np.float64)
            used = np.zeros(n_cols + 1, dtype=bool)
            j0 = 0
            while True:
                used[j0] = True
                i0 = int(p[j0])
                delta = np.inf
                j1 = 0
                row = cost[i0 - 1, :]
                for j in range(1, n_cols + 1):
                    if used[j]:
                        continue
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
                if not np.isfinite(delta):
                    delta = 0.0
                for j in range(0, n_cols + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        assignment = np.full(n_rows, -1, dtype=np.int32)
        for j in range(1, n_cols + 1):
            i = int(p[j])
            if i > 0:
                assignment[i - 1] = j - 1
        return assignment

    def _decode_enemy_bind_assignments(self, enemy_bind_logits: np.ndarray) -> np.ndarray:
        bind = np.asarray(enemy_bind_logits, dtype=np.float32)
        if bind.ndim == 1:
            bind = bind.reshape(1, -1)
        if bind.shape[0] < self.config.enemy_slots:
            padded = np.zeros((self.config.enemy_slots, bind.shape[1]), dtype=np.float32)
            padded[: bind.shape[0], :] = bind
            bind = padded
        bind = bind[: self.config.enemy_slots, :]

        expected_dim = self.enemy_obs_max + 1
        if bind.shape[1] < expected_dim:
            padded = np.zeros((self.config.enemy_slots, expected_dim), dtype=np.float32)
            padded[:, : bind.shape[1]] = bind
            bind = padded
        else:
            bind = bind[:, :expected_dim]

        obs_logits = bind[:, : self.enemy_obs_max]
        empty_logits = bind[:, self.enemy_obs_max]

        valid_obs = (self._enemy_obs_present > 0.5).astype(np.float32)
        masked_obs_logits = np.where(valid_obs.reshape(1, -1) > 0.5, obs_logits, -1e6)

        # Expand "empty" into per-slot virtual columns so multiple slots can map to empty
        # while preserving one-to-one assignment over real observations.
        empty_cols = np.repeat(empty_logits.reshape(-1, 1), self.config.enemy_slots, axis=1)
        score_matrix = np.concatenate([masked_obs_logits, empty_cols], axis=1)
        assignment = self._hungarian_maximize(score_matrix)

        slot_to_obs = np.full(self.config.enemy_slots, -1, dtype=np.int32)
        for slot in range(self.config.enemy_slots):
            col = int(assignment[slot]) if slot < assignment.size else -1
            if 0 <= col < self.enemy_obs_max and self._enemy_obs_present[col] > 0.5:
                slot_to_obs[slot] = col
        self._enemy_last_bind_obs[:] = slot_to_obs
        return slot_to_obs

    def _slot_enemies_from_bindings(self, slot_to_obs: np.ndarray) -> list[object | None]:
        slot_enemies: list[object | None] = [None for _ in range(self.config.enemy_slots)]
        for slot in range(self.config.enemy_slots):
            obs_idx = int(slot_to_obs[slot]) if slot < slot_to_obs.size else -1
            if 0 <= obs_idx < self.enemy_obs_max:
                slot_enemies[slot] = self._enemy_obs_objects[obs_idx]
        return slot_enemies

    def _decode_enemy_target_actor_ids(
        self,
        enemy_target_logits: np.ndarray,
        slot_to_obs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        target = np.asarray(enemy_target_logits, dtype=np.float32)
        if target.ndim == 1:
            target = target.reshape(1, -1)
        if target.shape[0] < self.config.enemy_slots:
            padded = np.zeros((self.config.enemy_slots, target.shape[1]), dtype=np.float32)
            padded[: target.shape[0], :] = target
            target = padded
        target = target[: self.config.enemy_slots, :]

        expected_dim = 1 + self.enemy_obs_max
        if target.shape[1] < expected_dim:
            padded = np.zeros((self.config.enemy_slots, expected_dim), dtype=np.float32)
            padded[:, : target.shape[1]] = target
            target = padded
        else:
            target = target[:, :expected_dim]

        actor_ids = np.full(self.config.enemy_slots, -1.0, dtype=np.float32)
        target_index = np.full(self.config.enemy_slots, -1, dtype=np.int32)
        for slot in range(self.config.enemy_slots):
            obs_idx = int(slot_to_obs[slot]) if slot < slot_to_obs.size else -1
            if obs_idx < 0:
                continue
            row = np.nan_to_num(target[slot], nan=-1e6, posinf=1e6, neginf=-1e6)
            valid_scores = np.full(expected_dim, -1e6, dtype=np.float32)
            valid_scores[0] = row[0]  # player token
            if self.enemy_obs_max > 0:
                valid_scores[1:] = np.where(self._enemy_obs_present > 0.5, row[1:], -1e6)
            choice = int(np.argmax(valid_scores))
            target_index[slot] = choice
            if choice == 0:
                # Reserved player target id token.
                actor_ids[slot] = 0.0
            else:
                obs_target = choice - 1
                if 0 <= obs_target < self.enemy_obs_max and self._enemy_obs_present[obs_target] > 0.5:
                    actor_ids[slot] = float(self._enemy_obs_actor_id[obs_target])
        return actor_ids, target_index

    def _enemy_feature_block(self, state: object | None) -> tuple[np.ndarray, np.ndarray]:
        block = np.zeros(self.enemy_feature_dim_total, dtype=np.float32)
        target_mask = np.zeros(self.enemy_target_mask_dim, dtype=np.float32)
        self._refresh_enemy_observations(state)

        obs_total = self.enemy_observed_feature_dim_total
        mem_total = self.enemy_memory_feature_dim_total
        if obs_total > 0:
            block[:obs_total] = self._enemy_obs_tokens.reshape(-1)
        if mem_total > 0:
            mem_base = obs_total
            for slot in range(self.config.enemy_slots):
                start = mem_base + slot * self.enemy_memory_feature_dim
                end = start + self.enemy_memory_feature_dim
                memory = np.clip(self._enemy_memory[slot], -1.0, 1.0)
                block[start:end] = memory * 100.0

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
        self.pygame_keyboard_sampler.render_rgb_frame(frame, diagnostics=self._diagnostics_payload)

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

    def _send_enemy_float_command(
        self,
        slot: int,
        suffix: str,
        value: float,
        cache_key: str,
        min_value: float = -1.0,
        max_value: float = 1.0,
    ) -> None:
        bounded = float(np.clip(value, min_value, max_value))
        previous = float(self._last_enemy_cmds[slot][cache_key])
        if abs(bounded - previous) <= 1e-4:
            return
        self.game.send_game_command(f"set nn_enemy_cmd_{slot:02d}_{suffix} {bounded:.6f}")
        self._last_enemy_cmds[slot][cache_key] = bounded

    def _send_enemy_int_command(
        self,
        slot: int,
        suffix: str,
        value: int,
        cache_key: str,
        min_value: int,
        max_value: int,
    ) -> None:
        bounded = int(np.clip(int(value), min_value, max_value))
        previous = int(self._last_enemy_cmds[slot][cache_key])
        if bounded == previous:
            return
        self.game.send_game_command(f"set nn_enemy_cmd_{slot:02d}_{suffix} {bounded}")
        self._last_enemy_cmds[slot][cache_key] = float(bounded)

    def _update_enemy_kinematics_state(
        self,
        slot_enemies: list[object | None],
        enemy_commands: np.ndarray,
    ) -> None:
        if not self.config.enemy_kinematics_transformer or self.config.nn_world_sim:
            return

        for slot in range(self.config.enemy_slots):
            enemy = slot_enemies[slot] if slot < len(slot_enemies) else None
            if enemy is None:
                self._world_enemies[slot, 4] = 0.0
                self._enemy_kin_actor_id[slot] = -1.0
                continue

            actor_id = self._enemy_attr_float(enemy, "id", -1.0)
            if (
                self._world_enemies[slot, 4] <= 0.5
                or abs(float(self._enemy_kin_actor_id[slot]) - float(actor_id)) > 0.5
            ):
                self._world_enemies[slot, 0] = float(self._enemy_attr_float(enemy, "position_x"))
                self._world_enemies[slot, 1] = float(self._enemy_attr_float(enemy, "position_y"))
                self._world_enemies[slot, 2] = self._wrap_angle_deg(
                    float(self._enemy_attr_float(enemy, "angle", 0.0))
                )
                self._world_enemies[slot, 3] = float(max(1.0, self._enemy_attr_float(enemy, "health", 100.0)))
                self._world_enemies[slot, 4] = 1.0
            self._enemy_kin_actor_id[slot] = float(actor_id)

        if enemy_commands.ndim == 1:
            commands = enemy_commands.reshape(1, -1)
        else:
            commands = enemy_commands
        if commands.shape[0] < self.config.enemy_slots:
            padded = np.zeros((self.config.enemy_slots, commands.shape[1]), dtype=np.float32)
            padded[: commands.shape[0], :] = commands
            commands = padded
        commands = commands[: self.config.enemy_slots, :]

        sub_steps = max(1, int(self.config.action_repeat))
        for _ in range(sub_steps):
            for slot in range(self.config.enemy_slots):
                if self._world_enemies[slot, 4] <= 0.5:
                    continue
                speed_norm = float(commands[slot, 0]) if commands.shape[1] > 0 else 0.0
                fwd_norm = float(commands[slot, 1]) if commands.shape[1] > 1 else 0.0
                side_norm = float(commands[slot, 2]) if commands.shape[1] > 2 else 0.0
                turn_norm = float(commands[slot, 3]) if commands.shape[1] > 3 else 0.0

                ex = float(self._world_enemies[slot, 0])
                ey = float(self._world_enemies[slot, 1])
                eangle = self._wrap_angle_deg(float(self._world_enemies[slot, 2] + 42.0 * turn_norm))
                self._world_enemies[slot, 2] = eangle
                erad = np.deg2rad(eangle)
                ecos = float(np.cos(erad))
                esin = float(np.sin(erad))
                speed_scale = float(np.clip(speed_norm, 0.0, 3.0))
                edx = (ecos * fwd_norm - esin * side_norm) * (5.0 * speed_scale)
                edy = (esin * fwd_norm + ecos * side_norm) * (5.0 * speed_scale)

                if self._collision_map is not None:
                    px, py = self._current_position()
                    dyn: list[tuple[float, float, float]] = [(float(px), float(py), 16.0)]
                    for other in range(self.config.enemy_slots):
                        if other == slot or self._world_enemies[other, 4] <= 0.5:
                            continue
                        dyn.append(
                            (
                                float(self._world_enemies[other, 0]),
                                float(self._world_enemies[other, 1]),
                                20.0,
                            )
                        )
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

    def _send_enemy_kinematics_commands(self, slot: int, present_norm: float) -> None:
        if not self.config.enemy_kinematics_transformer or self.config.nn_world_sim:
            return
        if present_norm <= 0.0:
            self._send_enemy_float_command(slot, "x_raw", 0.0, "x_raw", -32768.0, 32768.0)
            self._send_enemy_float_command(slot, "y_raw", 0.0, "y_raw", -32768.0, 32768.0)
            self._send_enemy_float_command(slot, "angle_raw", 0.0, "angle_raw", -2048.0, 2048.0)
            return

        ex = float(self._world_enemies[slot, 0])
        ey = float(self._world_enemies[slot, 1])
        eangle = float(self._world_enemies[slot, 2])
        self._send_enemy_float_command(slot, "x_raw", ex, "x_raw", -32768.0, 32768.0)
        self._send_enemy_float_command(slot, "y_raw", ey, "y_raw", -32768.0, 32768.0)
        self._send_enemy_float_command(slot, "angle_raw", eangle, "angle_raw", -2048.0, 2048.0)

    def _decode_enemy_fire_pulse(
        self,
        slot: int,
        fire_cmd_raw: float,
        firecd_cmd_raw: float,
        present_norm: float,
    ) -> tuple[float, float]:
        if slot < 0 or slot >= self.config.enemy_slots:
            return 0.0, 1.0
        if present_norm <= 0.0:
            self._enemy_fire_phase[slot] = 0.0
            self._enemy_fire_threshold[slot] = 1.0
            return 0.0, 1.0

        drive = float(np.clip(fire_cmd_raw, 0.0, 8.0))
        cadence_ctrl = float(np.clip(np.tanh(abs(float(firecd_cmd_raw))), 0.0, 1.0))
        threshold = float(np.clip(2.0 + 18.0 * cadence_ctrl, 2.0, 20.0))
        phase_inc = float(np.clip(drive * 4.0, 0.0, 8.0))
        phase = float(np.clip(self._enemy_fire_phase[slot] + phase_inc, 0.0, 64.0))

        pulse = 0.0
        if drive > 1e-4 and phase >= threshold:
            pulse = 1.0
            phase = float(max(0.0, phase - threshold))
        elif drive <= 1e-4:
            phase = float(max(0.0, phase - 0.5))

        self._enemy_fire_phase[slot] = phase
        self._enemy_fire_threshold[slot] = threshold
        return pulse, threshold

    def _transformer_enemy_combat_step(
        self,
        slot_enemies: list[object | None],
        commands: np.ndarray,
        fire_pulses: np.ndarray,
        target_actor_ids: np.ndarray,
        player_attack_pressed: bool,
    ) -> None:
        if not self.config.enemy_combat_transformer or self.config.nn_world_sim:
            return
        if not self._combat_player_sync_initialized:
            try:
                self._combat_player_health = float(
                    np.clip(self.game.get_game_variable(GameVariable.HEALTH), 0.0, 200.0)
                )
            except Exception:
                self._combat_player_health = 100.0
            self._combat_player_dead = bool(self._combat_player_health <= 0.0)
            self._combat_player_sync_initialized = True

        self._enemy_combat_player_cooldown = max(0, int(self._enemy_combat_player_cooldown) - 1)
        self._enemy_combat_player_iframe = max(0, int(self._enemy_combat_player_iframe) - 1)
        px, py = self._current_position()
        pangle = float(self.game.get_game_variable(GameVariable.ANGLE))

        # Player->enemy hit resolution (Transformer-side).
        if player_attack_pressed and self._enemy_combat_player_cooldown <= 0:
            best_slot = -1
            best_dist = 1e9
            for slot in range(self.config.enemy_slots):
                if self._world_enemies[slot, 4] <= 0.5:
                    continue
                dx = float(self._world_enemies[slot, 0] - px)
                dy = float(self._world_enemies[slot, 1] - py)
                dist = float(np.hypot(dx, dy))
                if dist < 1e-4 or dist > 1024.0:
                    continue
                bearing = float(np.degrees(np.arctan2(dy, dx)))
                err = abs(self._wrap_angle_deg(bearing - pangle))
                if err > 9.0:
                    continue
                if self._map_geometry is not None and not self._has_line_of_sight_2d(px, py, px + dx, py + dy):
                    continue
                if dist < best_dist:
                    best_dist = dist
                    best_slot = slot
            if best_slot >= 0:
                dmg = 22.0
                self._world_enemies[best_slot, 3] = float(max(0.0, self._world_enemies[best_slot, 3] - dmg))
                if self._world_enemies[best_slot, 3] <= 0.0 and self._world_enemies[best_slot, 4] > 0.5:
                    self._world_enemies[best_slot, 4] = 0.0
                    self._world_kills += 1
            self._enemy_combat_player_cooldown = max(1, int(self._runtime_fire_cooldown_tics))

        # Enemy->player hit resolution (Transformer-side).
        if not self._combat_player_dead and self._combat_player_health > 0.0:
            for slot in range(self.config.enemy_slots):
                if self._world_enemies[slot, 4] <= 0.5:
                    continue
                if slot >= fire_pulses.size or float(fire_pulses[slot]) <= 0.5:
                    continue
                if slot < target_actor_ids.size and abs(float(target_actor_ids[slot])) > 0.5:
                    continue
                if self._enemy_combat_player_iframe > 0:
                    break
                ex = float(self._world_enemies[slot, 0])
                ey = float(self._world_enemies[slot, 1])
                eang = float(self._world_enemies[slot, 2])
                dx = px - ex
                dy = py - ey
                dist = float(np.hypot(dx, dy))
                if dist < 1e-4 or dist > 1024.0:
                    continue
                bearing = float(np.degrees(np.arctan2(dy, dx)))
                err = abs(self._wrap_angle_deg(bearing - eang))
                aim_drive = 0.0
                if commands.ndim == 2 and slot < commands.shape[0] and commands.shape[1] > 4:
                    aim_drive = float(np.clip(abs(float(commands[slot, 4])), 0.0, 1.0))
                aim_tol = float(np.clip(10.0 + 18.0 * aim_drive, 8.0, 28.0))
                if err > aim_tol:
                    continue
                if self._map_geometry is not None and not self._has_line_of_sight_2d(ex, ey, px, py):
                    continue
                speed_drive = 0.0
                if commands.ndim == 2 and slot < commands.shape[0] and commands.shape[1] > 0:
                    speed_drive = float(np.clip(abs(float(commands[slot, 0])), 0.0, 1.0))
                dmg = float(np.clip(5.0 + 12.0 * speed_drive, 5.0, 22.0))
                self._combat_player_health = float(max(0.0, self._combat_player_health - dmg))
                self._enemy_combat_player_iframe = max(2, int(round(2.0 + 4.0 * (1.0 - aim_drive))))
                if self._combat_player_health <= 0.0:
                    self._combat_player_dead = True
                break

        self._sync_transformer_player_health_to_doom()
        _ = slot_enemies

    def _sync_transformer_player_health_to_doom(self) -> None:
        if not self.config.enemy_combat_transformer or self.config.nn_world_sim:
            return
        health = float(np.clip(self._combat_player_health, 0.0, 200.0))
        if health <= 0.0:
            if not self._combat_player_dead:
                self._combat_player_dead = True
                try:
                    self.game.send_game_command("kill")
                except Exception:
                    pass
            return
        self._combat_player_dead = False
        desired = int(np.clip(round(health), 1, 200))
        try:
            current = int(round(float(self.game.get_game_variable(GameVariable.HEALTH))))
        except Exception:
            current = -1
        if current != desired:
            self.game.send_game_command(f"sethealth {desired}")

    def _send_enemy_health_commands(self, slot: int, present_norm: float) -> None:
        health_raw = 0.0
        dead_raw = 1.0
        if self.config.enemy_combat_transformer:
            alive = bool(self._world_enemies[slot, 4] > 0.5)
            health_raw = float(max(0.0, self._world_enemies[slot, 3]))
            dead_raw = 0.0 if alive else 1.0
        else:
            if present_norm > 0.0:
                health_raw = 100.0
                dead_raw = 0.0
        self._send_enemy_float_command(slot, "health_raw", health_raw, "health_raw", 0.0, 4096.0)
        self._send_enemy_float_command(slot, "dead_raw", dead_raw, "dead_raw", 0.0, 1.0)

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
        # Side axis uses positive=right in control decode.
        dx = (cos_a * fwd + sin_a * side) * step_scale
        dy = (sin_a * fwd - cos_a * side) * step_scale
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
        enemy_behavior_logits: np.ndarray,
        enemy_actuator_logits: np.ndarray,
        enemy_bind_logits: np.ndarray,
        enemy_target_logits: np.ndarray,
        player_attack_pressed: bool = False,
        memory_update_logits: np.ndarray | None = None,
    ) -> tuple[int, list[tuple[float, float, float, float, float, float]]]:
        if not self.config.enemy_backend_transformer:
            return 0, []
        _ = enemy_behavior_logits  # legacy diagnostic head
        slot_to_obs = self._decode_enemy_bind_assignments(enemy_bind_logits)
        slot_enemies = self._slot_enemies_from_bindings(slot_to_obs)
        active_enemy_count = sum(1 for enemy in slot_enemies if enemy is not None)
        if self.config.nn_world_sim:
            if not self._world_sim_initialized:
                self._world_sim_bootstrap(slot_enemies)
            else:
                self._world_sim_sync_slots(slot_enemies)
        target_actor_ids, target_indices = self._decode_enemy_target_actor_ids(
            enemy_target_logits,
            slot_to_obs,
        )
        commands: list[tuple[float, float, float, float, float, float]] = []
        shots_tick = 0.0
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
        self._update_enemy_kinematics_state(slot_enemies, enemy_actuators)

        fire_pulse_cache = np.zeros(self.config.enemy_slots, dtype=np.float32)
        for slot in range(self.config.enemy_slots):
            enemy = slot_enemies[slot] if slot < len(slot_enemies) else None
            present_norm = 1.0 if enemy is not None else -1.0
            slot_actuator = np.asarray(enemy_actuators[slot], dtype=np.float32).reshape(-1)
            fire_cmd_raw = float(slot_actuator[5]) if slot_actuator.size > 5 else 0.0
            firecd_cmd_raw = float(slot_actuator[6]) if slot_actuator.size > 6 else 0.0
            pulse, _threshold = self._decode_enemy_fire_pulse(
                slot=slot,
                fire_cmd_raw=fire_cmd_raw,
                firecd_cmd_raw=firecd_cmd_raw,
                present_norm=present_norm,
            )
            fire_pulse_cache[slot] = float(pulse)

        self._transformer_enemy_combat_step(
            slot_enemies=slot_enemies,
            commands=enemy_actuators,
            fire_pulses=fire_pulse_cache,
            target_actor_ids=target_actor_ids,
            player_attack_pressed=bool(player_attack_pressed),
        )
        if self.config.enemy_combat_transformer and not self.config.nn_world_sim:
            active_enemy_count = sum(1 for slot in range(self.config.enemy_slots) if self._world_enemies[slot, 4] > 0.5)

        for slot in range(self.config.enemy_slots):
            enemy = slot_enemies[slot] if slot < len(slot_enemies) else None
            slot_actuator = np.asarray(enemy_actuators[slot], dtype=np.float32).reshape(-1)
            if slot_actuator.size < self.enemy_actuator_dim:
                padded_actuator = np.zeros(self.enemy_actuator_dim, dtype=np.float32)
                padded_actuator[: slot_actuator.size] = slot_actuator
                slot_actuator = padded_actuator
            else:
                slot_actuator = slot_actuator[: self.enemy_actuator_dim]
            actuator_drive = np.nan_to_num(
                slot_actuator,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            speed_norm = float(actuator_drive[0])
            fwd_norm = float(actuator_drive[1])
            side_norm = float(actuator_drive[2])
            turn_norm = float(actuator_drive[3])
            aim_norm = float(actuator_drive[4])
            fire_cmd_raw = float(actuator_drive[5])
            firecd_cmd_raw = float(actuator_drive[6])
            health_norm = float(actuator_drive[7])
            target_actor_id_raw = float(target_actor_ids[slot])

            if enemy is not None:
                present_norm = 1.0
                obs_idx = int(slot_to_obs[slot])
                actor_id_raw = float(self._enemy_obs_actor_id[obs_idx]) if obs_idx >= 0 else -1.0
                obs_identity = actor_id_raw
                prev_identity = self._enemy_prev_obs_identity[slot]
                if np.isfinite(prev_identity):
                    self._enemy_metric_identity_samples += 1
                    if abs(float(prev_identity) - obs_identity) > 0.5:
                        self._enemy_metric_identity_churn += 1
                self._enemy_prev_obs_identity[slot] = obs_identity

                target_idx = int(target_indices[slot])
                prev_target_idx = int(self._enemy_last_target_index[slot])
                if prev_target_idx >= 0 and target_idx >= 0 and target_idx != prev_target_idx:
                    target_switches_tick += 1
                self._enemy_last_target_index[slot] = target_idx

            else:
                present_norm = -1.0
                self._enemy_prev_obs_identity[slot] = np.nan
                self._enemy_last_target_index[slot] = -1
                actor_id_raw = -1.0
                speed_norm = 0.0
                fwd_norm = 0.0
                side_norm = 0.0
                turn_norm = 0.0
                aim_norm = 0.0
                fire_cmd_raw = 0.0
                firecd_cmd_raw = 0.0
                health_norm = 0.0
                target_actor_id_raw = -1.0

            if self.config.enemy_combat_transformer and not self.config.nn_world_sim:
                if self._world_enemies[slot, 4] <= 0.5:
                    present_norm = -1.0
                    speed_norm = 0.0
                    fwd_norm = 0.0
                    side_norm = 0.0
                    turn_norm = 0.0
                    aim_norm = 0.0
                    fire_cmd_raw = 0.0
                    firecd_cmd_raw = 0.0
            fire_norm = float(fire_pulse_cache[slot])
            fire_threshold = float(self._enemy_fire_threshold[slot])
            firecd_norm = float(fire_threshold)
            if enemy is not None:
                # Continuous shot-pressure proxy after cadence decode.
                shots_tick += (
                    float(np.clip(max(0.0, fire_norm), 0.0, 1.0))
                    / float(max(1, active_enemy_count))
                )

            self._send_enemy_float_command(slot, "speed_norm", speed_norm, "speed_norm", -32.0, 32.0)
            self._send_enemy_float_command(slot, "fwd_norm", fwd_norm, "fwd_norm", -64.0, 64.0)
            self._send_enemy_float_command(slot, "side_norm", side_norm, "side_norm", -64.0, 64.0)
            self._send_enemy_float_command(slot, "turn_norm", turn_norm, "turn_norm", -180.0, 180.0)
            self._send_enemy_float_command(slot, "aim_norm", aim_norm, "aim_norm", -180.0, 180.0)
            self._send_enemy_float_command(slot, "fire_norm", fire_norm, "fire_norm", -64.0, 64.0)
            self._send_enemy_float_command(slot, "firecd_norm", firecd_norm, "firecd_norm", -64.0, 64.0)
            self._send_enemy_float_command(slot, "health_norm", health_norm, "health_norm", -256.0, 256.0)
            self._send_enemy_health_commands(slot, present_norm)
            self._send_enemy_kinematics_commands(slot, present_norm)
            self._send_enemy_float_command(
                slot,
                "target_actor_id_raw",
                target_actor_id_raw,
                "target_actor_id_raw",
                -32768.0,
                32768.0,
            )
            self._send_enemy_float_command(
                slot,
                "actor_id_raw",
                actor_id_raw,
                "actor_id_raw",
                -1.0,
                1000000000.0,
            )
            self._send_enemy_float_command(slot, "present_norm", present_norm, "present_norm", -1.0, 1.0)

            # Retain this scalar for diagnostics/legacy low-level logs.
            if self.config.enemy_combat_transformer and not self.config.nn_world_sim:
                healthpct = int(np.clip(np.rint(self._world_enemies[slot, 3]), 0, 300))
            else:
                healthpct = int(np.clip(np.rint(health_norm), 0, 300))
            self._last_enemy_cmds[slot]["healthpct"] = float(healthpct)

            self._update_enemy_memory_state(
                slot=slot,
                memory_update_logits=memory_updates[slot],
            )
            commands.append((speed_norm, fwd_norm, side_norm, turn_norm, aim_norm, fire_norm))
            self._enemy_last_cmd[slot, 0] = float(speed_norm * 100.0)
            self._enemy_last_cmd[slot, 1] = float(fwd_norm * 100.0)
            self._enemy_last_cmd[slot, 2] = float(side_norm * 100.0)
            self._enemy_last_cmd[slot, 3] = float(turn_norm * 100.0)
            self._enemy_last_cmd[slot, 4] = float(aim_norm * 100.0)
            self._enemy_last_cmd[slot, 5] = float(fire_norm * 100.0)

        enemy_count = active_enemy_count
        self._enemy_metric_ticks += 1
        self._enemy_metric_active_enemy_ticks += enemy_count
        self._enemy_metric_shots += shots_tick
        self._enemy_metric_target_switches += target_switches_tick
        self._enemy_metric_close_pairs += close_pairs_tick
        return enemy_count, commands

    def _wrap_angle_deg(self, angle_deg: float) -> float:
        return float((angle_deg + 180.0) % 360.0 - 180.0)

    def _world_update_manual_door_state(self, player_action: list[float]) -> None:
        if self._collision_map is None or self._world_door_open_ticks.size == 0:
            return
        if np.any(self._world_door_open_ticks > 0):
            self._world_door_open_ticks = np.maximum(self._world_door_open_ticks - 1, 0)

        use_pressed = bool(len(player_action) > 5 and player_action[5] > 0.5)
        if not use_pressed:
            return

        px = float(self._world_player[0])
        py = float(self._world_player[1])
        pang = float(self._world_player[2])
        prad = np.deg2rad(pang)
        fx = float(np.cos(prad))
        fy = float(np.sin(prad))

        max_d2 = 64.0 * 64.0
        open_window = max(32, int(round(4.0 * self.config.doom_ticrate)))
        for idx, (x1, y1, x2, y2) in enumerate(self._collision_map.door_segments):
            d2 = WADCollisionMap._point_segment_distance_sq(px, py, x1, y1, x2, y2)
            if d2 > max_d2:
                continue
            mx = 0.5 * (x1 + x2)
            my = 0.5 * (y1 + y2)
            # Keep "use" interaction front-facing.
            facing = (mx - px) * fx + (my - py) * fy
            if facing < -12.0:
                continue
            self._world_door_open_ticks[idx] = max(self._world_door_open_ticks[idx], open_window)

    def _world_active_manual_door_blockers(self) -> list[tuple[float, float, float, float]] | None:
        if self._collision_map is None or self._world_door_open_ticks.size == 0:
            return None
        blocked: list[tuple[float, float, float, float]] = []
        for idx, seg in enumerate(self._collision_map.door_segments):
            if self._world_door_open_ticks[idx] <= 0:
                blocked.append(seg)
        return blocked

    def _world_sim_bootstrap(self, slot_enemies: list[object | None] | None = None) -> None:
        if slot_enemies is None:
            state = self.game.get_state()
            self._refresh_enemy_observations(state)
            if np.any(self._enemy_last_bind_obs >= 0):
                slot_enemies = self._slot_enemies_from_bindings(self._enemy_last_bind_obs)
            else:
                bootstrap_bind = np.full(self.config.enemy_slots, -1, dtype=np.int32)
                for slot in range(min(self.config.enemy_slots, self.enemy_obs_max)):
                    if self._enemy_obs_present[slot] > 0.5:
                        bootstrap_bind[slot] = slot
                slot_enemies = self._slot_enemies_from_bindings(bootstrap_bind)
        px, py = self._current_position()
        pangle = float(self.game.get_game_variable(GameVariable.ANGLE))
        pz = self._safe_game_variable("POSITION_Z", 0.0)
        self._world_player[0] = float(px)
        self._world_player[1] = float(py)
        self._world_player[2] = self._wrap_angle_deg(pangle)
        self._world_player[3] = float(np.clip(self.game.get_game_variable(GameVariable.HEALTH), 0.0, 200.0))
        self._world_player_z = float(pz)
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
        if self._world_door_open_ticks.size > 0:
            self._world_door_open_ticks.fill(0)
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
        enemy_commands: list[tuple[float, float, float, float, float, float]],
        sub_steps: int | None = None,
        use_player_kinematics: bool = True,
    ) -> None:
        if not self._world_sim_initialized:
            self._world_sim_bootstrap()
        sim_sub_steps = max(1, self.config.action_repeat if sub_steps is None else sub_steps)
        for _ in range(sim_sub_steps):
            self._world_update_manual_door_state(player_action)
            # Player world-sim kinematics can conflict with native visible player
            # grounding/physics; keep it optional.
            if use_player_kinematics:
                px = float(self._world_player[0])
                py = float(self._world_player[1])
                strict_move_scale = 0.72 if self.config.nn_world_sim_strict else 1.0
                strict_turn_scale = 0.82 if self.config.nn_world_sim_strict else 1.0
                move_norm = float(
                    np.clip(float(player_action[0]) / max(0.1, self.config.move_delta), -1.0, 1.0)
                )
                strafe_norm = float(
                    np.clip(float(player_action[1]) / max(0.1, self.config.strafe_delta), -1.0, 1.0)
                )
                turn_norm = float(
                    np.clip(float(player_action[2]) / max(0.1, self.config.turn_delta), -1.0, 1.0)
                )
                turn_step_deg = 5.0 * strict_turn_scale * turn_norm
                if (
                    self.config.nn_world_sim_strict
                    and self.config.visible
                    and not self.config.nn_world_sim_pure
                ):
                    # In visible strict mode, Doom view turn is bridged via turn/look
                    # action ticks; avoid double-integrating angle here.
                    turn_step_deg = 0.0
                pangle = self._wrap_angle_deg(
                    float(self._world_player[2] + turn_step_deg)
                )
                self._world_player[2] = pangle
                prad = np.deg2rad(pangle)
                cos_a = float(np.cos(prad))
                sin_a = float(np.sin(prad))
                # Strafe convention: positive strafe_norm means "right".
                pdx = (cos_a * move_norm + sin_a * strafe_norm) * self.config.nn_move_units * strict_move_scale
                pdy = (sin_a * move_norm - cos_a * strafe_norm) * self.config.nn_move_units * strict_move_scale
                if self._collision_map is not None:
                    # Use static-map collision only for player in world-sim mode.
                    # Enemy dynamic blockers can drift from rendered Doom actors until
                    # full actor-state bridging is implemented, causing "invisible wall" feel.
                    dyn: list[tuple[float, float, float]] = []
                    extra_segments = self._world_active_manual_door_blockers()
                    npx, npy = self._collision_map.resolve_motion(
                        px,
                        py,
                        pdx,
                        pdy,
                        radius=self.config.nn_player_radius,
                        dynamic_circles=dyn,
                        extra_blocking_segments=extra_segments,
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
                speed_norm, fwd_norm, side_norm, turn_norm, aim_norm, fire_norm = enemy_commands[slot]
                _ = (aim_norm, fire_norm)
                ex = float(self._world_enemies[slot, 0])
                ey = float(self._world_enemies[slot, 1])
                eangle = self._wrap_angle_deg(float(self._world_enemies[slot, 2] + 42.0 * float(turn_norm)))
                self._world_enemies[slot, 2] = eangle
                erad = np.deg2rad(eangle)
                ecos = float(np.cos(erad))
                esin = float(np.sin(erad))
                speed_scale = float(np.clip(speed_norm, 0.0, 3.0))
                edx = (ecos * fwd_norm - esin * side_norm) * (5.0 * speed_scale)
                edy = (esin * fwd_norm + ecos * side_norm) * (5.0 * speed_scale)
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
                _speed, _fwd, _side, _turn, aim_norm, fire_norm = enemy_commands[slot]
                if aim_norm <= 0.0 or fire_norm <= 0.0:
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

    def _sync_world_player_from_doom(self) -> None:
        px, py = self._current_position()
        self._world_player[0] = float(px)
        self._world_player[1] = float(py)
        self._world_player[2] = self._wrap_angle_deg(float(self.game.get_game_variable(GameVariable.ANGLE)))
        self._world_player[3] = float(np.clip(self.game.get_game_variable(GameVariable.HEALTH), 0.0, 200.0))
        self._world_player_z = float(self._safe_game_variable("POSITION_Z", self._world_player_z))

    def _strict_update_turn_hold(self, action: list[float], keyboard_state: np.ndarray) -> None:
        if not self.config.nn_world_sim_strict:
            return

        # Keep turn authority model-driven but only when turn keys are actually pressed.
        left_pressed = bool(keyboard_state[4] > 0.1)
        right_pressed = bool(keyboard_state[5] > 0.1)
        desired = 0
        if left_pressed and not right_pressed:
            desired = -1
        elif right_pressed and not left_pressed:
            desired = 1
        elif left_pressed and right_pressed:
            turn_val = float(action[2]) if len(action) > 2 else 0.0
            if turn_val < -0.05:
                desired = -1
            elif turn_val > 0.05:
                desired = 1

        if desired == self._strict_turn_hold:
            return
        if self._strict_turn_hold < 0:
            self.game.send_game_command("-left")
        elif self._strict_turn_hold > 0:
            self.game.send_game_command("-right")

        if desired < 0:
            self.game.send_game_command("+left")
        elif desired > 0:
            self.game.send_game_command("+right")
        self._strict_turn_hold = desired

    def _world_sim_apply_render_bridge(self) -> None:
        if not self._world_sim_initialized:
            return
        px = float(self._world_player[0])
        py = float(self._world_player[1])
        pang = float(self._world_player[2])
        first_bridge = (
            np.isnan(self._world_last_bridge_x)
            or np.isnan(self._world_last_bridge_y)
            or np.isnan(self._world_last_bridge_angle)
        )
        if first_bridge:
            pos_changed = True
            ang_changed = True
        else:
            pos_changed = (
                abs(px - float(self._world_last_bridge_x)) > 0.08
                or abs(py - float(self._world_last_bridge_y)) > 0.08
            )
            ang_changed = abs(self._wrap_angle_deg(pang - float(self._world_last_bridge_angle))) > 0.20
        # Keep visible player motion native by default. In strict+pure mode, bridge
        # visible player from world-sim state as well.
        bridge_player = (not self.config.visible) or (
            self.config.nn_world_sim_strict and self.config.nn_world_sim_pure
        )
        if bridge_player:
            if pos_changed:
                # Use warp x y only; warp x y z is ignored on this runtime path.
                self.game.send_game_command(f"warp {px:.3f} {py:.3f}")
                if self.config.nn_world_sim_strict and self.config.nn_world_sim_pure:
                    # `warp x y` can collapse camera view-height in Doom player mode.
                    # Re-center view each pure-bridge warp so the player does not
                    # visually sink into the floor while moving.
                    self.game.send_game_command("centerview")
            if pos_changed or ang_changed:
                self._world_last_bridge_x = px
                self._world_last_bridge_y = py
                self._world_last_bridge_angle = pang
        if not self.config.enemy_backend_transformer:
            return
        for slot in range(self.config.enemy_slots):
            if self._world_enemies[slot, 4] > 0.5:
                healthpct = int(np.clip(round(self._world_enemies[slot, 3]), 1, 200))
                health_norm = float(np.clip((healthpct - 20.0) / 280.0, 0.0, 1.0))
                present_norm = 1.0
            else:
                healthpct = 0
                health_norm = 0.0
                present_norm = -1.0
            self._send_enemy_float_command(slot, "health_norm", health_norm, "health_norm", 0.0, 1.0)
            self._send_enemy_float_command(slot, "present_norm", present_norm, "present_norm", -1.0, 1.0)
            self._last_enemy_cmds[slot]["healthpct"] = float(healthpct)

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
            healthpct = (
                int(round(self._last_enemy_cmds[slot]["healthpct"]))
                if self._last_enemy_cmds[slot]["healthpct"] >= 0.0
                else 100
            )
            enemy_low_level.append((firecd_proxy, int(healthpct)))

        return move_scale, turn_scale, fire_cooldown, enemy_low_level

    def _prime_history(self) -> None:
        self._enemy_metric_ticks = 0
        self._enemy_metric_shots = 0.0
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
        self._enemy_combat_player_cooldown = 0
        self._enemy_combat_player_iframe = 0
        self._combat_player_dead = False
        self._combat_player_sync_initialized = False
        try:
            self._combat_player_health = float(
                np.clip(self.game.get_game_variable(GameVariable.HEALTH), 0.0, 200.0)
            )
        except Exception:
            self._combat_player_health = 100.0
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

    def _advance_native_tick(self) -> float:
        try:
            self.game.advance_action(1, True)
        except TypeError:
            try:
                self.game.advance_action(1)
            except TypeError:
                self.game.advance_action()
        except Exception:
            # Fallback to a neutral action tick if advance_action is unavailable.
            try:
                return float(self.game.make_action([0.0] * self.control_dim, 1))
            except Exception:
                return 0.0
        try:
            return float(self.game.get_last_reward())
        except Exception:
            return 0.0

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

            if self._uses_doom_buttons_source() and not self._nn_movement_resolution_active:
                # Native Doom key binds own movement/turn/fire in doom_buttons mode.
                # Avoid injecting neutral make_action ticks that can suppress input.
                if np.any(keyboard_state > 0.1):
                    last_action = self._decode_controls(control_logits, step, keyboard_state)
                else:
                    last_action = [0.0] * self.control_dim
                total_reward += self._advance_native_tick()
                self._render_current_frame()
                continue

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
        enemy_commands: list[tuple[float, float, float, float, float, float]],
    ) -> tuple[float, list[float], np.ndarray]:
        total_reward = 0.0
        last_action = [0.0] * self.control_dim
        keyboard_state = initial_keyboard_state
        neutral_action = [0.0] * self.control_dim
        strict_visible_native_player = (
            self.config.nn_world_sim_strict
            and self.config.visible
            and not self.config.nn_world_sim_pure
        )
        use_player_kinematics = self.config.nn_world_sim_strict and not strict_visible_native_player

        # Keep Doom ticking for rendering while Transformer world-sim owns movement/combat.
        sub_total = 1 if self.config.nn_world_sim_pure else max(1, self.config.action_repeat)
        for sub_tick in range(sub_total):
            if sub_tick > 0:
                keyboard_state = self._sanitize_keyboard_state(self._read_keyboard_state())
            self._attack_cooldown_left = max(0, self._attack_cooldown_left - 1)

            if np.any(keyboard_state > 0.1):
                decoded_action = self._decode_controls(control_logits, step, keyboard_state)
                action = decoded_action.copy()
                self._apply_fire_cooldown(action, keyboard_state)
            else:
                decoded_action = neutral_action.copy()
                action = neutral_action.copy()

            if self.config.nn_world_sim_strict and self.config.nn_world_sim_pure:
                # Pure strict mode: world-sim owns movement/combat state.
                # Runtime bridge executes XY via warp, and turn/look/fire/use via native deltas.
                self._world_sim_step(
                    action,
                    enemy_commands,
                    sub_steps=1,
                    use_player_kinematics=True,
                )
                self._world_sim_apply_render_bridge()
                pure_bridge_action = [0.0] * self.control_dim
                pure_bridge_action[2] = float(action[2])  # turn
                pure_bridge_action[3] = float(action[3])  # look
                pure_bridge_action[4] = float(action[4])  # fire pulse
                pure_bridge_action[5] = float(action[5])  # use
                total_reward += self.game.make_action(pure_bridge_action, 1)
                if self.config.visible:
                    # `warp x y` transiently collapses view-height for one tick in player mode.
                    # Consume one neutral recovery tick before rendering so camera stays stable.
                    total_reward += self.game.make_action(neutral_action, 1)
                    # Some lower-floor sector transitions need one extra recovery tick
                    # to restore full view-height (otherwise the player appears half-sunk).
                    if float(self._safe_game_variable("VIEW_HEIGHT", 41.0)) < 35.0:
                        total_reward += self.game.make_action(neutral_action, 1)
                self._sync_world_player_from_doom()
                last_action = decoded_action
                self._render_current_frame()
                continue

            if np.any(keyboard_state > 0.1):
                if self.config.nn_world_sim_strict:
                    # Visible strict mode: use Transformer-decoded native movement/turn deltas
                    # to keep Doom floor/collision rendering stable; keep firing world-sim only.
                    if strict_visible_native_player:
                        strict_bridge_action = [0.0] * self.control_dim
                        strict_bridge_action[0] = float(action[0])  # forward/back
                        strict_bridge_action[1] = float(action[1])  # strafe
                        strict_bridge_action[2] = float(action[2])  # turn
                        strict_bridge_action[3] = float(action[3])  # look
                        strict_bridge_action[4] = float(action[4])  # fire pulse
                        strict_bridge_action[5] = float(action[5])  # use
                        total_reward += self.game.make_action(strict_bridge_action, 1)
                    else:
                        # Pure strict world-sim (visible or headless): never inject native
                        # movement/fire. Turn/view comes from Transformer bridge state.
                        if not self.config.nn_world_sim_pure:
                            self._strict_update_turn_hold(action, keyboard_state)
                        total_reward += self.game.make_action(neutral_action, 1)
                else:
                    total_reward += self._make_action_with_fire_pulse(action, repeats=1)
                self._world_player_z = float(self._safe_game_variable("POSITION_Z", self._world_player_z))
                if strict_visible_native_player:
                    self._world_player[2] = self._wrap_angle_deg(
                        float(self.game.get_game_variable(GameVariable.ANGLE))
                    )
                if (
                    strict_visible_native_player
                    or (not use_player_kinematics)
                    or (self.config.nn_world_sim_pure and self.config.visible)
                ):
                    self._sync_world_player_from_doom()
                self._world_sim_step(
                    action,
                    enemy_commands,
                    sub_steps=1,
                    use_player_kinematics=use_player_kinematics,
                )
                last_action = decoded_action
            else:
                if self.config.nn_world_sim_strict and not strict_visible_native_player:
                    if not self.config.nn_world_sim_pure:
                        self._strict_update_turn_hold(neutral_action, keyboard_state)
                total_reward += self.game.make_action(neutral_action, 1)
                self._world_player_z = float(self._safe_game_variable("POSITION_Z", self._world_player_z))
                if (
                    strict_visible_native_player
                    or (not use_player_kinematics)
                    or (self.config.nn_world_sim_pure and self.config.visible)
                ):
                    self._sync_world_player_from_doom()
                self._world_sim_step(
                    neutral_action,
                    enemy_commands,
                    sub_steps=1,
                    use_player_kinematics=use_player_kinematics,
                )
                last_action = neutral_action.copy()

            # Bridge every sub-tick in world-sim mode to avoid jumpy/sticky visual updates.
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
        if self._uses_doom_buttons_source() and not self._nn_movement_resolution_active:
            print("doom_buttons mode: loop advances native Doom ticks (no forced neutral-action override).")
        else:
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
            if self.config.enemy_kinematics_transformer:
                if self.config.nn_world_sim:
                    print(
                        "Enemy kinematics mode requested, but --nn-world-sim already owns enemy kinematics; "
                        "dedicated no-world-sim kinematics path is bypassed."
                    )
                else:
                    print(
                        "Enemy kinematics mode is ON: Transformer updates enemy x/y/angle state each tick "
                        "(mod executes model-provided absolute kinematics)."
                    )
            if self.config.enemy_combat_transformer and not self.config.nn_world_sim:
                print(
                    "Enemy combat mode is ON: Transformer resolves player<->enemy hit resolution "
                    "and owns player/enemy health+death transitions."
                )
        if self.config.nn_world_sim:
            print(
                "Experimental NN world-sim mode is ON: Transformer loop owns low-level movement/collision/combat "
                "(Doom remains render/IO bridge)."
            )
            if self.config.nn_world_sim_strict:
                print(
                    "Strict world-sim execution is ON: player movement is Transformer-side; "
                    "Doom is used as render/IO bridge."
                )
                if self.config.nn_world_sim_pure:
                    print(
                        "Pure strict mode is ON: movement/position resolve in Transformer; "
                        "native Doom receives only turn/look/fire/use deltas."
                    )
                    if self._world_door_open_ticks.size > 0:
                        print(
                            "Pure mode door guard is ON: manual door linedefs block movement "
                            "until Use (E) opens a short pass window."
                        )
            print(
                f"NN world-sim damage scale: {self.config.nn_world_damage_scale:.2f}"
            )
        if self.config.visible:
            if self.keyboard_source == "pygame_window":
                print("Focus the 'Transformer Doom (focus this window)' game window for keyboard input.")
                print("Press F1 to toggle diagnostics panel (live attention/control view).")
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
            f"low_level_gain={self.config.nn_low_level_gain:.2f}"
        )
        if abs(self.config.nn_enemy_gain - 1.0) > 1e-6:
            print(
                "Note: nn_enemy_gain is compatibility-only and no longer used in "
                "enemy backend decode."
            )
        print(
            f"Enemy state features: observed_max={self.enemy_obs_max} obs_token_dim={self.enemy_obs_feature_dim} "
            f"memory_slots={self.config.enemy_slots} memory_dim={self.enemy_memory_feature_dim} "
            f"total={self.enemy_feature_dim_total} + target_mask={self.enemy_target_mask_dim}"
        )
        print(
            f"Enemy observation order shuffle: {'ON' if self.config.shuffle_enemy_observations else 'OFF'}"
        )
        print(
            f"Enemy behavior head (target decode): channels={self.enemy_cmd_dim} "
            f"{'/'.join(self.enemy_behavior_channels)}"
        )
        print(
            f"Enemy bind head: channels={self.enemy_bind_dim} "
            f"{'/'.join(self.enemy_bind_channels)}"
        )
        print(
            f"Enemy target head: channels={self.enemy_target_dim} "
            f"{'/'.join(self.enemy_target_channels)}"
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
            "Enemy target source: direct model actor-id target command (range-safe transport only)"
        )
        print(
            "Enemy fire timing source: Transformer-side per-slot fire-phase state update "
            "(model fire/firecd channels -> pulse decode, no mod-side timing state)."
        )
        print(
            "Enemy fire pulse source: Transformer-side pulse decode (stateful cadence)."
        )
        print(
            "Enemy fire source: direct actuator channel projection (no Python policy shaping)"
        )
        print(
            "Enemy fire cadence source: Transformer fire-phase threshold from actuator firecd channel"
        )
        print(
            "Enemy aim source: direct actuator aim channel"
        )
        print(
            "Enemy steering source: direct actuator speed/fwd/side/turn channels"
        )
        print(
            "Enemy command decode source: direct NN actuator channels; "
            "Python applies hard crash-safe bounds only"
        )
        if self.config.enemy_combat_transformer and not self.config.nn_world_sim:
            print(
                "Enemy health/death source: Transformer combat state writes health_raw/dead_raw "
                "to mod (execution-only)."
            )
        if self.config.enemy_kinematics_transformer and not self.config.nn_world_sim:
            print(
                "Enemy collision resolver: Transformer-side map/dynamic collision resolution is ON "
                "for enemy kinematic integration."
            )
        else:
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
                        enemy_behavior_logits,
                        enemy_actuator_logits,
                        enemy_bind_logits,
                        enemy_target_logits,
                        low_level_logits,
                        memory_update_logits,
                        attention_maps,
                    ) = self.model(inputs)
                predicted = predicted_state.cpu().numpy()[0]
                control = control_logits.cpu().numpy()[0]
                enemy_behavior = enemy_behavior_logits.cpu().numpy()[0]
                enemy_actuator = enemy_actuator_logits.cpu().numpy()[0]
                enemy_bind = enemy_bind_logits.cpu().numpy()[0]
                enemy_target = enemy_target_logits.cpu().numpy()[0]
                low_level = low_level_logits.cpu().numpy()[0]
                memory_update = memory_update_logits.cpu().numpy()[0]
                attention_last = np.zeros((0, 0, 0), dtype=np.float32)
                if len(attention_maps) > 0:
                    try:
                        attention_last = (
                            attention_maps[-1]
                            .detach()
                            .cpu()
                            .numpy()[0]
                            .astype(np.float32, copy=False)
                        )
                    except Exception:
                        attention_last = np.zeros((0, 0, 0), dtype=np.float32)
                move_scale, turn_scale, fire_cd, enemy_low = self._apply_low_level_backend_controls(low_level)
                keyboard_state = self._sanitize_keyboard_state(self._read_keyboard_state())
                self._diagnostics_payload = {
                    "tick": int(step),
                    "mse": float(self._last_mse),
                    "control": np.asarray(control, dtype=np.float32).copy(),
                    "keys": np.asarray(keyboard_state, dtype=np.float32).copy(),
                    "attention": attention_last,
                }
                enemy_count, enemy_cmd = self._apply_enemy_backend_commands(
                    enemy_behavior,
                    enemy_actuator,
                    enemy_bind,
                    enemy_target,
                    player_attack_pressed=bool(keyboard_state[8] > 0.1),
                    memory_update_logits=memory_update,
                )

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
                self._last_mse = mse
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
                        sample_kin = []
                        if self.config.enemy_kinematics_transformer and not self.config.nn_world_sim:
                            for slot in range(min(2, self.config.enemy_slots)):
                                if self._world_enemies[slot, 4] > 0.5:
                                    sample_kin.append(
                                        (
                                            round(float(self._world_enemies[slot, 0]), 1),
                                            round(float(self._world_enemies[slot, 1]), 1),
                                            round(float(self._world_enemies[slot, 2]), 1),
                                        )
                                    )
                        enemy_info = (
                            f" enemies={enemy_count} enemy_cmd={sample_cmd} enemy_low={sample_low}"
                        )
                        if sample_kin:
                            enemy_info += f" enemy_kin={sample_kin}"
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
            if self.config.nn_world_sim_strict:
                if self._strict_turn_hold < 0:
                    self.game.send_game_command("-left")
                elif self._strict_turn_hold > 0:
                    self.game.send_game_command("-right")
                self._strict_turn_hold = 0
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
        default="1280x960",
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
        default="pygame_window",
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
    parser.add_argument("--log-interval", type=int, default=1, help="Ticks between metric logs")
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
        dest="enemy_backend_transformer",
        action="store_true",
        help=(
            "Enable experimental enemy backend override via custom scenario mod and "
            "Transformer enemy command head."
        ),
    )
    parser.add_argument(
        "--disable-enemy-backend-transformer",
        dest="enemy_backend_transformer",
        action="store_false",
        help="Disable Transformer enemy backend override.",
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
        "--shuffle-enemy-observations",
        action="store_true",
        help=(
            "Randomize enemy observation token order every tick before Transformer inference "
            "(used for permutation-stability validation)."
        ),
    )
    parser.add_argument(
        "--enemy-kinematics-transformer",
        dest="enemy_kinematics_transformer",
        action="store_true",
        help=(
            "Move enemy kinematic state update (x/y/angle integration) to Transformer-side backend "
            "before mod execution."
        ),
    )
    parser.add_argument(
        "--disable-enemy-kinematics-transformer",
        dest="enemy_kinematics_transformer",
        action="store_false",
        help="Disable Transformer-side enemy kinematics update path.",
    )
    parser.add_argument(
        "--enemy-combat-transformer",
        dest="enemy_combat_transformer",
        action="store_true",
        help=(
            "Move enemy hit resolution + damage/death transitions to Transformer-side backend "
            "(mod executes only health/death writes)."
        ),
    )
    parser.add_argument(
        "--disable-enemy-combat-transformer",
        dest="enemy_combat_transformer",
        action="store_false",
        help="Disable Transformer-side enemy combat hit/health transition path.",
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
    parser.set_defaults(
        nn_movement_resolution=True,
        enemy_backend_transformer=True,
        enemy_kinematics_transformer=True,
        enemy_combat_transformer=True,
        nn_world_sim=True,
        nn_world_sim_strict=True,
        nn_world_sim_pure=True,
    )
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
        help="Compatibility-only knob (enemy decode path no longer applies this gain).",
    )
    parser.add_argument(
        "--nn-low-level-gain",
        type=float,
        default=0.25,
        help="Low-level head logit gain before tanh decoding.",
    )
    parser.add_argument(
        "--nn-world-sim",
        dest="nn_world_sim",
        action="store_true",
        help=(
            "Experimental mode: move low-level movement/collision/combat step into Transformer-side "
            "world simulator, while Doom remains rendering/input bridge."
        ),
    )
    parser.add_argument(
        "--disable-nn-world-sim",
        dest="nn_world_sim",
        action="store_false",
        help="Disable Transformer world-sim stepping path.",
    )
    parser.add_argument(
        "--nn-world-sim-strict",
        dest="nn_world_sim_strict",
        action="store_true",
        help=(
            "Strict world-sim execution: do not send player movement/fire to native Doom; "
            "Transformer world state is bridged to rendering."
        ),
    )
    parser.add_argument(
        "--disable-nn-world-sim-strict",
        dest="nn_world_sim_strict",
        action="store_false",
        help="Disable strict world-sim execution mode.",
    )
    parser.add_argument(
        "--nn-world-sim-pure",
        dest="nn_world_sim_pure",
        action="store_true",
        help=(
            "With --nn-world-sim-strict, keep visible player world execution in Transformer as well "
            "(no native Doom player movement solver path)."
        ),
    )
    parser.add_argument(
        "--disable-nn-world-sim-pure",
        dest="nn_world_sim_pure",
        action="store_false",
        help="Disable pure world-sim execution mode.",
    )
    parser.add_argument(
        "--nn-world-damage-scale",
        type=float,
        default=1.0,
        help="Damage multiplier used by experimental NN world simulator.",
    )
    args = parser.parse_args()
    if args.nn_world_sim_strict and not args.nn_world_sim:
        args.nn_world_sim = True
    if args.nn_world_sim_pure and not args.nn_world_sim_strict:
        args.nn_world_sim_strict = True
        args.nn_world_sim = True
    if args.headless and args.keyboard_source == "pygame_window":
        # pygame window sampling requires a visible window; fall back automatically.
        args.keyboard_source = "doom_buttons"
    if args.enemy_kinematics_transformer and not args.enemy_backend_transformer:
        raise ValueError("--enemy-kinematics-transformer requires --enemy-backend-transformer.")
    if args.enemy_combat_transformer and not args.enemy_backend_transformer:
        raise ValueError("--enemy-combat-transformer requires --enemy-backend-transformer.")
    if args.enemy_combat_transformer and not args.enemy_kinematics_transformer:
        raise ValueError("--enemy-combat-transformer currently requires --enemy-kinematics-transformer.")

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
        shuffle_enemy_observations=bool(args.shuffle_enemy_observations),
        enemy_kinematics_transformer=bool(args.enemy_kinematics_transformer),
        enemy_combat_transformer=bool(args.enemy_combat_transformer or args.enemy_kinematics_transformer),
        nn_movement_resolution=bool(args.nn_movement_resolution),
        nn_move_units=max(0.1, float(args.nn_move_units)),
        nn_player_radius=max(4.0, float(args.nn_player_radius)),
        nn_weight_scale=float(np.clip(args.nn_weight_scale, 0.05, 2.0)),
        nn_control_gain=float(np.clip(args.nn_control_gain, 0.1, 4.0)),
        nn_enemy_gain=float(np.clip(args.nn_enemy_gain, 0.1, 4.0)),
        nn_low_level_gain=float(np.clip(args.nn_low_level_gain, 0.05, 4.0)),
        nn_world_sim=bool(args.nn_world_sim),
        nn_world_sim_strict=bool(args.nn_world_sim_strict),
        nn_world_sim_pure=bool(args.nn_world_sim_pure),
        nn_world_damage_scale=float(np.clip(args.nn_world_damage_scale, 0.1, 4.0)),
    )


def main() -> None:
    config = parse_args()
    loop = DoomTransformerLoop(config)
    loop.run()


if __name__ == "__main__":
    main()
