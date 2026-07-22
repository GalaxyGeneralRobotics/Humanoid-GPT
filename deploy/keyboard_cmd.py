"""Keyboard command interface for mode switching and velocity control.

Runs a pygame GUI in a subprocess; the main process polls for updates.
Reuses the keyboard_gamepad infrastructure from gx_loco_deploy.
"""

from __future__ import annotations

import os
import time
import threading
from collections.abc import Sequence
from multiprocessing import get_context
from dataclasses import dataclass, field

os.environ["SDL_AUDIODRIVER"] = "dummy"
import pygame  # noqa: E402

from deploy.constants import KEYBOARD_MAX_SPEED


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _char_to_keycode(ch: str) -> int:
    if len(ch) != 1:
        raise ValueError("Key identifier must be a single character.")
    if not pygame.get_init():
        pygame.init()
    return pygame.key.key_code(ch.lower())


@dataclass(slots=True)
class CmdDim:
    name: str
    low_key: str
    high_key: str
    init: float = 0.0
    low: float = -1.0
    high: float = 1.0
    delta: float = 0.1
    damp: bool = True
    low_key_code: int | None = field(init=False)
    high_key_code: int | None = field(init=False)

    def __post_init__(self):
        self.low_key_code = _char_to_keycode(self.low_key)
        self.high_key_code = _char_to_keycode(self.high_key)


@dataclass(slots=True)
class CmdBtn:
    name: str
    key: str
    init: bool = False
    key_code: int | None = field(init=False)

    def __post_init__(self):
        self.key_code = _char_to_keycode(self.key)


# ---------------------------------------------------------------------------
# Background keyboard integrator
# ---------------------------------------------------------------------------

class KeyboardCommander:
    def __init__(self, dims: Sequence[CmdDim | CmdBtn], *, freq: float = 60.0):
        self._dims: dict[str, CmdDim | CmdBtn] = {d.name: d for d in dims}
        self._vals: dict[str, float | bool] = {
            d.name: (d.init if isinstance(d, CmdDim) else d.init) for d in dims
        }
        self._pressed: dict[int, bool] = {}
        self._period = 1.0 / freq
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def handle_event(self, ev: pygame.event.Event):
        if ev.type == pygame.KEYDOWN:
            self._pressed[ev.key] = True
        elif ev.type == pygame.KEYUP:
            self._pressed[ev.key] = False

    def get_value(self, name: str) -> float | bool:
        return self._vals[name]

    def get_all(self) -> dict[str, float | bool]:
        return dict(self._vals)

    def reset(self):
        for n, d in self._dims.items():
            self._vals[n] = d.init if isinstance(d, CmdBtn) else d.init

    def close(self):
        self._running = False

    def _loop(self):
        def _clamp(v, lo, hi):
            return max(lo, min(hi, v))

        while self._running:
            t0 = time.perf_counter()
            for name, d in self._dims.items():
                if isinstance(d, CmdDim):
                    hi = self._pressed.get(d.high_key_code, False)
                    lo = self._pressed.get(d.low_key_code, False)
                    if hi ^ lo:
                        inc = d.delta if hi else -d.delta
                        self._vals[name] = _clamp(float(self._vals[name]) + inc, d.low, d.high)
                    else:
                        if d.damp and not (hi or lo):
                            v = float(self._vals[name])
                            if v > 0.0:
                                v = max(0.0, v - d.delta)
                            elif v < 0.0:
                                v = min(0.0, v + d.delta)
                            self._vals[name] = _clamp(v, d.low, d.high)
                else:
                    self._vals[name] = self._pressed.get(d.key_code, False)
            dt = time.perf_counter() - t0
            if (sleep := self._period - dt) > 0:
                time.sleep(sleep)


# ---------------------------------------------------------------------------
# Pygame HUD window (styled)
# ---------------------------------------------------------------------------

# Palette - Warm dark theme with teal accent
_BG         = (18, 20, 26)
_BG_GRAD    = (24, 26, 34)
_PANEL      = (28, 31, 42)
_PANEL_HI   = (35, 38, 52)
_BORDER     = (48, 52, 68)
_BORDER_LIT = (65, 72, 95)
_TEXT       = (230, 232, 240)
_TEXT_DIM   = (110, 118, 140)
_ACCENT     = (45, 195, 185)   # Teal
_ACCENT_DIM = (35, 90, 95)
_BAR_BG     = (38, 42, 56)
_BAR_POS    = (52, 200, 165)   # Teal-green
_BAR_NEG    = (220, 95, 115)   # Soft coral
_BTN_ON     = (45, 195, 185)
_BTN_OFF    = (55, 58, 75)
_MODE_COLORS = [
    (45, 195, 185),   # Walk (teal)
    (255, 180, 70),   # Online (amber)
    (120, 215, 165),  # Offline 0 (mint)
    (185, 140, 255),  # Offline 1 (lavender)
    (255, 130, 130),  # Offline 2 (coral)
    (90, 200, 230),   # Offline 3 (sky)
    (255, 210, 95),   # Offline 4 (gold)
    (200, 160, 130),  # Offline 5
    (140, 180, 220),  # Offline 6
    (230, 170, 195),  # Offline 7
]
_MODE_LABELS = ["Walk", "Online"]


class CommandWindow:
    WIDTH = 720
    MARGIN = 28
    PANEL_R = 12
    BAR_H = 20
    ROW_H = 52

    def __init__(self, commander: KeyboardCommander, *, width: int = 720, fps: int = 60):
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        self.commander = commander
        self.fps = fps
        self.width = max(560, width)
        self.M = self.MARGIN

        # Fonts (SysFont uses first available from comma-separated list)
        self.font_title = pygame.font.SysFont("Segoe UI,Helvetica,Arial,sans-serif", 28, bold=True)
        self.font_label = pygame.font.SysFont("Helvetica,Arial,sans-serif", 18)
        self.font_value = pygame.font.SysFont("Menlo,Consolas,monospace", 18, bold=True)
        self.font_key = pygame.font.SysFont("Menlo,Consolas,monospace", 15)
        self.font_mode = pygame.font.SysFont("Helvetica,Arial,sans-serif", 15, bold=True)
        self.font_section = pygame.font.SysFont("Helvetica,Arial,sans-serif", 14)

        self._axes = [d for d in commander._dims.values() if isinstance(d, CmdDim)]
        self._mode_btns = [d for d in commander._dims.values()
                           if isinstance(d, CmdBtn) and d.name.startswith("Mode")]
        self._action_btns = [d for d in commander._dims.values()
                             if isinstance(d, CmdBtn) and not d.name.startswith("Mode")]

        # Layout heights
        h = self.M
        h += 48                                    # title
        h += 20                                    # spacing
        h += 44 + len(self._mode_btns) // 5 * 44   # mode selector row(s)
        h += 28                                    # spacing
        h += len(self._axes) * self.ROW_H + 20     # velocity sliders
        h += 24                                    # spacing
        h += max(1, len(self._action_btns)) * 40 + 16  # action buttons
        h += self.M
        self.height = h

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Humanoid-GPT Deploy Controller")
        self.clock = pygame.time.Clock()

    def _draw_rounded_rect(self, rect, color, radius=10):
        pygame.draw.rect(self.screen, color, rect, border_radius=radius)

    def _draw_panel(self, x, y, w, h):
        r = pygame.Rect(x, y, w, h)
        self._draw_rounded_rect(r, _PANEL, self.PANEL_R)
        pygame.draw.rect(self.screen, _BORDER, r, 1, self.PANEL_R)
        return r

    def _draw_section_header(self, text, x, y):
        surf = self.font_section.render(text.upper(), True, _TEXT_DIM)
        self.screen.blit(surf, (x, y))
        # Subtle accent line
        line_w = surf.get_width() + 10
        pygame.draw.line(self.screen, _BORDER_LIT, (x, y + 18), (x + line_w, y + 18), 1)

    def draw_frame(self):
        # Subtle gradient-like background (top darker)
        self.screen.fill(_BG)
        grad_rect = pygame.Rect(0, 0, self.width, self.height // 2)
        pygame.draw.rect(self.screen, _BG_GRAD, grad_rect)

        vals = self.commander.get_all()
        M = self.M
        pw = self.width - 2 * M  # panel width
        y = M

        # -- Title with accent --
        title_surf = self.font_title.render("Humanoid-GPT Deploy", True, _TEXT)
        self.screen.blit(title_surf, (M, y))
        # Accent underline
        pygame.draw.rect(
            self.screen, _ACCENT,
            (M, y + 36, min(100, title_surf.get_width()), 4),
            border_radius=2
        )
        y += 48

        # -- Mode selector --
        self._draw_section_header("Mode", M, y)
        y += 28

        num_modes = len(self._mode_btns)
        cols = min(num_modes, 5)
        btn_gap = 10
        btn_w = (pw - btn_gap * (cols - 1)) // cols
        btn_h = 38

        for i, mb in enumerate(self._mode_btns):
            col = i % cols
            row = i // cols
            bx = M + col * (btn_w + btn_gap)
            by = y + row * (btn_h + btn_gap)
            mode_idx = int(mb.name.replace("Mode", ""))

            is_active = bool(vals.get(mb.name, False))
            color = _MODE_COLORS[mode_idx % len(_MODE_COLORS)]
            bg = color if is_active else _PANEL

            rect = pygame.Rect(bx, by, btn_w, btn_h)
            self._draw_rounded_rect(rect, bg, 10)
            border_color = _BORDER_LIT if is_active else _BORDER
            pygame.draw.rect(self.screen, border_color, rect, 2 if is_active else 1, 10)

            if mode_idx < len(_MODE_LABELS):
                label = f"{mode_idx}: {_MODE_LABELS[mode_idx]}"
            else:
                label = f"{mode_idx}: Track {mode_idx - 2}"
            tc = (255, 255, 255) if is_active else _TEXT_DIM
            lbl_surf = self.font_mode.render(label, True, tc)
            lbl_rect = lbl_surf.get_rect(center=rect.center)
            self.screen.blit(lbl_surf, lbl_rect)

        rows_of_modes = (num_modes + cols - 1) // cols
        y += rows_of_modes * (btn_h + btn_gap) + 24

        # -- Velocity sliders --
        self._draw_section_header("Velocity", M, y)
        y += 28

        panel_rect = self._draw_panel(M, y, pw, len(self._axes) * self.ROW_H + 20)
        iy = y + 12

        label_w = 130
        key_w = 150
        bar_x = M + label_w + 16
        bar_w = pw - label_w - key_w - 36

        for d in self._axes:
            v = float(vals[d.name])

            # Label
            lbl = self.font_label.render(d.name, True, _TEXT)
            self.screen.blit(lbl, (M + 18, iy + 8))

            # Bar background
            bar_rect = pygame.Rect(bar_x, iy + 12, bar_w, self.BAR_H)
            self._draw_rounded_rect(bar_rect, _BAR_BG, 6)

            # Bar fill (centered at midpoint)
            mid_x = bar_x + bar_w // 2
            frac = (v - d.low) / (d.high - d.low) if d.high != d.low else 0.5
            fill_px = int(frac * bar_w)
            if v >= 0:
                fx = mid_x
                fw = fill_px - bar_w // 2
                if fw > 0:
                    fill_rect = pygame.Rect(fx, iy + 12, fw, self.BAR_H)
                    self._draw_rounded_rect(fill_rect, _BAR_POS, 6)
            else:
                fx = bar_x + fill_px
                fw = mid_x - fx
                if fw > 0:
                    fill_rect = pygame.Rect(fx, iy + 12, fw, self.BAR_H)
                    self._draw_rounded_rect(fill_rect, _BAR_NEG, 6)

            # Center tick (subtle)
            pygame.draw.line(
                self.screen, _BORDER_LIT,
                (mid_x, iy + 10), (mid_x, iy + 12 + self.BAR_H + 6), 1
            )

            # Value text
            val_color = _BAR_POS if v > 0.001 else (_BAR_NEG if v < -0.001 else _TEXT_DIM)
            val_surf = self.font_value.render(f"{v:+.2f}", True, val_color)
            self.screen.blit(val_surf, (bar_x + bar_w + 14, iy + 8))

            # Key hints (styled)
            key_surf = self.font_key.render(f"{d.high_key.upper()} / {d.low_key.upper()}", True, _TEXT_DIM)
            self.screen.blit(key_surf, (bar_x + bar_w + 88, iy + 10))

            iy += self.ROW_H

        y = panel_rect.bottom + 20

        # -- Action buttons --
        if self._action_btns:
            self._draw_section_header("Actions", M, y)
            y += 28

            for b in self._action_btns:
                state = bool(vals[b.name])
                dot_color = _BTN_ON if state else _BTN_OFF
                dot_x, dot_y = M + 18, y + 16
                # Outer ring
                pygame.draw.circle(self.screen, _BORDER, (dot_x, dot_y), 10)
                pygame.draw.circle(self.screen, dot_color, (dot_x, dot_y), 8)
                if state:
                    pygame.draw.circle(self.screen, (255, 255, 255), (dot_x, dot_y), 4)

                lbl = self.font_label.render(b.name, True, _TEXT if state else _TEXT_DIM)
                self.screen.blit(lbl, (M + 40, y + 6))
                key_surf = self.font_key.render(f"[{b.key.upper()}]", True, _TEXT_DIM)
                self.screen.blit(key_surf, (M + 40 + lbl.get_width() + 12, y + 8))
                y += 40

        pygame.display.flip()
        self.clock.tick(self.fps)


# ---------------------------------------------------------------------------
# Subprocess GUI worker (module-level so it is picklable on macOS spawn)
# ---------------------------------------------------------------------------

def _gui_worker(dim_specs, freq, win_w, shm_arr, stop_flag):
    """Run the keyboard GUI in a child process.

    Values are written directly to ``shm_arr`` (multiprocessing.Array)
    every frame, so the main process can read them with zero latency.
    """
    import numpy as _np
    import pygame as _pg
    _pg.init()

    dims = []
    for kind, kwargs in dim_specs:
        dims.append(CmdDim(**kwargs) if kind == "dim" else CmdBtn(**kwargs))

    names = [d.name for d in dims]
    commander = KeyboardCommander(dims, freq=freq)
    window = CommandWindow(commander, width=win_w)

    try:
        while stop_flag.value == 0:
            for ev in _pg.event.get():
                if ev.type == _pg.QUIT:
                    stop_flag.value = 1
                elif ev.type in (_pg.KEYDOWN, _pg.KEYUP):
                    commander.handle_event(ev)

            # Push current values into shared memory every frame
            vals = commander.get_all()
            with shm_arr.get_lock():
                buf = _np.frombuffer(shm_arr.get_obj(), dtype=_np.float32)
                for i, n in enumerate(names):
                    buf[i] = float(vals[n])

            window.draw_frame()
    except KeyboardInterrupt:
        pass
    finally:
        commander.close()
        _pg.quit()


# ---------------------------------------------------------------------------
# Cross-process wrapper (shared-memory, zero-latency reads)
# ---------------------------------------------------------------------------

class KeyboardCmdPad:
    """Run the GUI in a subprocess; read values via shared memory."""

    def __init__(self, dims: Sequence[CmdDim | CmdBtn], *, freq: float = 60.0,
                 window_width: int = 720):
        import numpy as np

        self._names = [d.name for d in dims]
        self._name_to_idx = {n: i for i, n in enumerate(self._names)}
        n = len(dims)
        self._ctx = get_context("spawn")

        # Shared float array: subprocess writes, main process reads
        self._shm = self._ctx.Array("f", n, lock=True)
        self._stop = self._ctx.Value("i", 0)

        # Initialize with default values
        with self._shm.get_lock():
            buf = np.frombuffer(self._shm.get_obj(), dtype=np.float32)
            for i, d in enumerate(dims):
                buf[i] = float(d.init)

        # Serialize dim specs as plain dicts (picklable across spawn)
        dim_specs = []
        for d in dims:
            if isinstance(d, CmdDim):
                dim_specs.append(("dim", {
                    "name": d.name, "low_key": d.low_key, "high_key": d.high_key,
                    "init": d.init, "low": d.low, "high": d.high,
                    "delta": d.delta, "damp": d.damp,
                }))
            else:
                dim_specs.append(("btn", {"name": d.name, "key": d.key, "init": d.init}))

        self._proc = self._ctx.Process(
            target=_gui_worker,
            args=(dim_specs, freq, window_width, self._shm, self._stop),
            daemon=True,
        )
        self._proc.start()
        time.sleep(0.3)
        if not self._proc.is_alive():
            raise RuntimeError(
                "Keyboard UI process exited during startup; "
                "check pygame/display errors above."
            )

    def get_command(self, axis: str) -> float:
        import numpy as np
        idx = self._name_to_idx[axis]
        with self._shm.get_lock():
            return float(np.frombuffer(self._shm.get_obj(), dtype=np.float32)[idx])

    def get_all(self) -> dict[str, float]:
        import numpy as np
        with self._shm.get_lock():
            buf = np.frombuffer(self._shm.get_obj(), dtype=np.float32).copy()
        return {n: float(buf[i]) for i, n in enumerate(self._names)}

    def close(self):
        self._stop.value = 1
        if self._proc.is_alive():
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()


# ---------------------------------------------------------------------------
# High-level command object
# ---------------------------------------------------------------------------

@dataclass
class HighCommand:
    vel_lin_x: float = 0.0
    vel_lin_y: float = 0.0
    vel_ang_yaw: float = 0.0
    mode: int = 0
    kill: bool = False


class DeployKeyboardCMD:
    """Keyboard controller for deploy: mode switching + velocity commands."""

    def __init__(self, num_track_ref: int = 2):
        self.num_mode = 2 + num_track_ref
        self._cmd_pad = KeyboardCmdPad(
            dims=[
                CmdDim(
                    "LinVelX", "s", "w",
                    low=-KEYBOARD_MAX_SPEED, high=KEYBOARD_MAX_SPEED, delta=0.05,
                ),
                CmdDim(
                    "LinVelY", "d", "a",
                    low=-KEYBOARD_MAX_SPEED, high=KEYBOARD_MAX_SPEED, delta=0.05,
                ),
                CmdDim(
                    "LinVelYaw", "e", "q",
                    low=-KEYBOARD_MAX_SPEED, high=KEYBOARD_MAX_SPEED, delta=0.05,
                ),
                CmdBtn("Kill", "`"),
                CmdBtn("Reset", "r"),
            ]
            + [CmdBtn(f"Mode{m}", str(m)) for m in range(self.num_mode)]
        )
        self._last_cmd = {f"Mode{m}": 0 for m in range(self.num_mode)}
        self._last_cmd["Reset"] = 0
        self.mode = 0
        self.reset_requested = False

    def step_command(self) -> HighCommand:
        for m in range(self.num_mode):
            key = f"Mode{m}"
            if (self._last_cmd[key], self._cmd_pad.get_command(key)) == (0, 1):
                self.mode = m
            self._last_cmd[key] = self._cmd_pad.get_command(key)

        if (self._last_cmd["Reset"], self._cmd_pad.get_command("Reset")) == (0, 1):
            self.reset_requested = True
        self._last_cmd["Reset"] = self._cmd_pad.get_command("Reset")

        return HighCommand(
            vel_lin_x=self._cmd_pad.get_command("LinVelX"),
            vel_lin_y=self._cmd_pad.get_command("LinVelY"),
            vel_ang_yaw=self._cmd_pad.get_command("LinVelYaw"),
            mode=self.mode,
            kill=bool(self._cmd_pad.get_command("Kill")),
        )

    def check_reset_request(self) -> bool:
        if self.reset_requested:
            self.reset_requested = False
            return True
        return False

    def close(self):
        self._cmd_pad.close()
