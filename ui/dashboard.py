"""
ui/dashboard.py — Professional Surveillance Platform Interface.

Design: Military/Security operations center aesthetic.
- Dark tactical theme with amber/red alert accents
- Live video feed with scan-line overlay effect
- Real-time event log (last 10 events scrolling)
- Three actor panels (Authorized / Unauthorized / Unknown)
- Animated alert banner on ALARM
- Status indicators with blinking lights
- Camera info + FPS + uptime bar
"""

import tkinter as tk
from tkinter import font as tkfont
import cv2
import numpy as np
import os
import time
from datetime import datetime
from PIL import Image, ImageTk
from typing import Optional


# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    "bg":         "#080c10",      # near-black background
    "panel":      "#0d1117",      # panel background
    "panel2":     "#111820",      # slightly lighter panel
    "border":     "#1e2d3d",      # panel borders
    "border2":    "#0f4060",      # active border
    "accent":     "#00d4ff",      # cyan accent
    "amber":      "#ffb300",      # amber / warning
    "green":      "#00e676",      # authorized green
    "orange":     "#ff6d00",      # unauthorized orange
    "red":        "#ff1744",      # alarm red
    "text_hi":    "#e8f4f8",      # primary text
    "text_lo":    "#4a6272",      # muted text
    "text_mid":   "#8faabb",      # secondary text
    "grid":       "#0d1f2d",      # subtle grid lines
}

ALERT_COLOR = {
    "LOG_ONLY": C["green"],
    "ALERT":    C["orange"],
    "ALARM":    C["red"],
}

CAT_COLOR = {
    "AUTHORIZED":   C["green"],
    "UNAUTHORIZED": C["orange"],
    "UNKNOWN":      C["red"],
}


class SurveillanceDashboard(tk.Tk):
    def __init__(self, on_quit=None):
        super().__init__()
        self._on_quit_cb  = on_quit
        self._start_time  = time.time()
        self._event_log   = []       # list of dicts
        self._blink_state = False
        self._alert_active = False
        self._alert_level  = "LOG_ONLY"

        self.title("SURVEILLANCE OPS — CAM-01")
        self.configure(bg=C["bg"])
        self.resizable(True, True)

        self._build_ui()
        self._tick_uptime()
        self._tick_blink()
        self.protocol("WM_DELETE_WINDOW", self._quit)

    # ══════════════════════════════════════════════════════════════════════════
    #  UI Construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # Fonts
        self.fn_mono  = ("Courier New", 9)
        self.fn_mono2 = ("Courier New", 8)
        self.fn_label = ("Courier New", 10, "bold")
        self.fn_big   = ("Courier New", 14, "bold")
        self.fn_title = ("Courier New", 11, "bold")

        # ── Top bar ───────────────────────────────────────────────────────────
        self._build_topbar()

        # ── Main content ──────────────────────────────────────────────────────
        main = tk.Frame(self, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # Left column: video + event log
        left = tk.Frame(main, bg=C["bg"])
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.rowconfigure(0, weight=3)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        self._build_video_panel(left)
        self._build_event_log(left)

        # Right column: status + actors + controls
        right = tk.Frame(main, bg=C["bg"])
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        self._build_status_panel(right)
        self._build_actors_panel(right)
        self._build_controls(right)

    def _build_topbar(self):
        bar = tk.Frame(self, bg=C["panel"], height=44)
        bar.pack(fill="x", padx=8, pady=(8, 6))
        bar.pack_propagate(False)

        # Left: title
        tk.Label(bar, text="[ SURVEILLANCE OPS CENTER ]",
                 font=("Courier New", 13, "bold"),
                 bg=C["panel"], fg=C["accent"]).pack(side="left", padx=16, pady=8)

        # Right: time + status dot
        right = tk.Frame(bar, bg=C["panel"])
        right.pack(side="right", padx=16)

        self._dot = tk.Label(right, text="●", font=("Courier New", 14),
                             bg=C["panel"], fg=C["green"])
        self._dot.pack(side="right", padx=(6, 0))

        tk.Label(right, text="SYSTEM ONLINE",
                 font=self.fn_mono, bg=C["panel"],
                 fg=C["text_mid"]).pack(side="right")

        self._clock_var = tk.StringVar(value="--:--:--")
        tk.Label(bar, textvariable=self._clock_var,
                 font=("Courier New", 12, "bold"),
                 bg=C["panel"], fg=C["amber"]).pack(side="right", padx=20)

        # Update clock
        def _update_clock():
            self._clock_var.set(datetime.now().strftime("%H:%M:%S"))
            self.after(1000, _update_clock)
        _update_clock()

    def _build_video_panel(self, parent):
        frame = tk.Frame(parent, bg=C["border"], padx=1, pady=1)
        frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))

        header = tk.Frame(frame, bg=C["panel2"], height=28)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(header, text="▶  LIVE FEED — CAM-01",
                 font=self.fn_label, bg=C["panel2"],
                 fg=C["accent"]).pack(side="left", padx=10)

        self._fps_var = tk.StringVar(value="FPS: --")
        tk.Label(header, textvariable=self._fps_var,
                 font=self.fn_mono, bg=C["panel2"],
                 fg=C["text_lo"]).pack(side="right", padx=10)

        self.canvas = tk.Canvas(frame, width=640, height=460,
                                bg="#000000", highlightthickness=0)
        self.canvas.pack()

        # Scan-line overlay text
        self._overlay_var = tk.StringVar(value="")
        tk.Label(frame, textvariable=self._overlay_var,
                 font=self.fn_mono2, bg=C["panel"],
                 fg=C["text_lo"]).pack(fill="x", padx=8, pady=2)

    def _build_event_log(self, parent):
        frame = self._panel(parent, "  EVENT LOG", row=1)

        self._log_frame = tk.Frame(frame, bg=C["panel"])
        self._log_frame.pack(fill="both", expand=True, padx=8, pady=6)

        # 8 log rows
        self._log_rows = []
        for _ in range(8):
            row = tk.Frame(self._log_frame, bg=C["panel"])
            row.pack(fill="x", pady=1)
            ts  = tk.Label(row, text="", font=self.fn_mono2,
                           bg=C["panel"], fg=C["text_lo"], width=9, anchor="w")
            ts.pack(side="left")
            cat = tk.Label(row, text="", font=self.fn_mono2,
                           bg=C["panel"], fg=C["text_lo"], width=14, anchor="w")
            cat.pack(side="left", padx=(4, 0))
            name = tk.Label(row, text="", font=self.fn_mono2,
                            bg=C["panel"], fg=C["text_mid"], anchor="w")
            name.pack(side="left", padx=(4, 0))
            self._log_rows.append((ts, cat, name))

    def _build_status_panel(self, parent):
        frame = self._panel(parent, "  DETECTION STATUS", row=0)

        rows = [
            ("PERSON",    "var_person",   C["text_hi"]),
            ("CATEGORY",  "var_category", C["text_hi"]),
            ("ALERT",     "var_alert",    C["amber"]),
            ("CONFIDENCE","var_conf",     C["text_mid"]),
            ("FPS",       "var_fps",      C["accent"]),
            ("UPTIME",    "var_uptime",   C["text_lo"]),
        ]

        for label, varname, color in rows:
            r = tk.Frame(frame, bg=C["panel"])
            r.pack(fill="x", padx=10, pady=3)

            tk.Label(r, text=f"{label:<12}", font=self.fn_mono2,
                     bg=C["panel"], fg=C["text_lo"]).pack(side="left")
            var = tk.StringVar(value="—")
            setattr(self, varname, var)
            tk.Label(r, textvariable=var, font=self.fn_label,
                     bg=C["panel"], fg=color, anchor="w").pack(side="left")

        # Alert banner
        self._alert_banner = tk.Label(frame, text="",
                                       font=("Courier New", 11, "bold"),
                                       bg=C["panel"], fg=C["red"],
                                       pady=4)
        self._alert_banner.pack(fill="x", padx=10, pady=(4, 6))

    def _build_actors_panel(self, parent):
        frame = self._panel(parent, "  ACTOR CLASSIFICATION", row=1)

        actors = [
            ("AUTHORIZED",   "auth_count",   C["green"],  "● ACCESS GRANTED"),
            ("UNAUTHORIZED", "unauth_count",  C["orange"], "▲ ALERT TRIGGERED"),
            ("UNKNOWN",      "unknown_count", C["red"],    "■ ALARM RAISED"),
        ]

        for cat, countvar, color, sublabel in actors:
            card = tk.Frame(frame, bg=C["panel2"],
                            highlightbackground=color,
                            highlightthickness=1)
            card.pack(fill="x", padx=10, pady=4)

            top = tk.Frame(card, bg=C["panel2"])
            top.pack(fill="x", padx=8, pady=(6, 2))

            tk.Label(top, text=cat, font=self.fn_label,
                     bg=C["panel2"], fg=color).pack(side="left")

            var = tk.StringVar(value="0")
            setattr(self, countvar, var)
            tk.Label(top, textvariable=var,
                     font=("Courier New", 18, "bold"),
                     bg=C["panel2"], fg=color).pack(side="right", padx=4)

            tk.Label(card, text=sublabel, font=self.fn_mono2,
                     bg=C["panel2"], fg=C["text_lo"]).pack(
                         anchor="w", padx=8, pady=(0, 6))

        # Counters
        self._counts = {"AUTHORIZED": 0, "UNAUTHORIZED": 0, "UNKNOWN": 0}

        # Last capture
        lc = tk.Frame(frame, bg=C["panel"])
        lc.pack(fill="x", padx=10, pady=(4, 8))
        tk.Label(lc, text="LAST CAPTURE", font=self.fn_mono2,
                 bg=C["panel"], fg=C["text_lo"]).pack(anchor="w")
        self.var_capture = tk.StringVar(value="—")
        tk.Label(lc, textvariable=self.var_capture,
                 font=self.fn_mono2, bg=C["panel"],
                 fg=C["text_mid"], wraplength=220, anchor="w").pack(
                     fill="x", pady=(2, 0))

    def _build_controls(self, parent):
        frame = self._panel(parent, "  CONTROLS", row=2)

        btn_cfg = dict(font=self.fn_label, relief="flat",
                       cursor="hand2", pady=6)

        tk.Button(frame, text="VERIFY CAPTURE",
                  bg=C["border2"], fg=C["accent"],
                  command=self._open_verify,
                  **btn_cfg).pack(fill="x", padx=10, pady=(6, 3))

        tk.Button(frame, text="STOP SURVEILLANCE",
                  bg="#1a0a0a", fg=C["red"],
                  command=self._quit,
                  **btn_cfg).pack(fill="x", padx=10, pady=(3, 8))

    # ── Helper: build a titled panel ─────────────────────────────────────────

    def _panel(self, parent, title: str, row: int) -> tk.Frame:
        outer = tk.Frame(parent, bg=C["border"], padx=1, pady=1)
        outer.grid(row=row, column=0, sticky="nsew", pady=(0, 6))

        header = tk.Frame(outer, bg=C["panel2"], height=26)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text=title, font=self.fn_mono,
                 bg=C["panel2"], fg=C["accent"]).pack(
                     side="left", padx=8, pady=4)

        inner = tk.Frame(outer, bg=C["panel"])
        inner.pack(fill="both", expand=True)
        return inner

    # ══════════════════════════════════════════════════════════════════════════
    #  Public API (called from main.py)
    # ══════════════════════════════════════════════════════════════════════════

    def update_frame(self, frame: np.ndarray):
        """Push a BGR OpenCV frame to the canvas."""
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb)
        # Fit to canvas size
        img   = img.resize((640, 460), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
        # Overlay info
        self._overlay_var.set(
            f"REC ● {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
            f"CAM-01  |  {self.var_fps.get()}"
        )

    def update_status(self, person: str, category: str,
                      alert_level: str, confidence: float = 0.0,
                      fps: Optional[float] = None):
        self.var_person.set(person)

        cat_color = CAT_COLOR.get(category, C["text_hi"])
        self.var_category.set(category)

        al_color = ALERT_COLOR.get(alert_level, C["text_hi"])
        self.var_alert.set(alert_level)

        if confidence > 0:
            self.var_conf.set(f"{confidence:.1f}")

        if fps is not None:
            self.var_fps.set(f"{fps:.1f}")
            self._fps_var.set(f"FPS: {fps:.1f}")

        # Alert banner
        if alert_level == "ALARM":
            self._alert_banner.config(text="⚠  INTRUDER ALARM  ⚠",
                                       fg=C["red"])
            self._alert_active = True
            self._alert_level  = "ALARM"
        elif alert_level == "ALERT":
            self._alert_banner.config(text="▲  UNAUTHORIZED DETECTED  ▲",
                                       fg=C["orange"])
            self._alert_active = True
            self._alert_level  = "ALERT"
        else:
            self._alert_banner.config(text="")
            self._alert_active = False

        # Update actor counts
        if category in self._counts:
            self._counts[category] += 1
            self.auth_count.set(str(self._counts["AUTHORIZED"]))
            self.unauth_count.set(str(self._counts["UNAUTHORIZED"]))
            self.unknown_count.set(str(self._counts["UNKNOWN"]))

    def log_event(self, timestamp: str, category: str,
                  person: str, alert_level: str):
        """Add an entry to the scrolling event log."""
        color = ALERT_COLOR.get(alert_level, C["text_mid"])
        self._event_log.insert(0, {
            "ts":       timestamp[-8:],   # just HH:MM:SS
            "category": category[:12],
            "person":   person,
            "color":    color,
        })
        self._event_log = self._event_log[:8]   # keep last 8

        for i, entry in enumerate(self._event_log):
            ts_lbl, cat_lbl, name_lbl = self._log_rows[i]
            ts_lbl.config(text=entry["ts"],       fg=C["text_lo"])
            cat_lbl.config(text=entry["category"], fg=entry["color"])
            name_lbl.config(text=entry["person"],  fg=C["text_mid"])

    def update_last_capture(self, image_path: str):
        self.var_capture.set(os.path.basename(image_path))

    # ══════════════════════════════════════════════════════════════════════════
    #  Internal tickers
    # ══════════════════════════════════════════════════════════════════════════

    def _tick_uptime(self):
        elapsed = int(time.time() - self._start_time)
        h, rem  = divmod(elapsed, 3600)
        m, s    = divmod(rem, 60)
        self.var_uptime.set(f"{h:02d}:{m:02d}:{s:02d}")
        self.after(1000, self._tick_uptime)

    def _tick_blink(self):
        """Blink the status dot and alert banner."""
        self._blink_state = not self._blink_state
        if self._alert_active:
            color = ALERT_COLOR.get(self._alert_level, C["red"])
            self._dot.config(fg=color if self._blink_state else C["panel"])
        else:
            self._dot.config(fg=C["green"])
        self.after(600, self._tick_blink)

    # ══════════════════════════════════════════════════════════════════════════
    #  Actions
    # ══════════════════════════════════════════════════════════════════════════

    def _open_verify(self):
        """Open a simple dialog to verify a capture."""
        win = tk.Toplevel(self)
        win.title("Verify Capture Integrity")
        win.configure(bg=C["bg"])
        win.geometry("500x200")

        tk.Label(win, text="CAPTURE INTEGRITY VERIFICATION",
                 font=self.fn_title, bg=C["bg"],
                 fg=C["accent"]).pack(pady=(16, 8))

        tk.Label(win, text="Enter image path:",
                 font=self.fn_mono, bg=C["bg"],
                 fg=C["text_mid"]).pack()

        entry = tk.Entry(win, font=self.fn_mono, bg=C["panel"],
                         fg=C["text_hi"], insertbackground=C["accent"],
                         width=55)
        entry.pack(padx=20, pady=8)

        result_var = tk.StringVar(value="")
        result_lbl = tk.Label(win, textvariable=result_var,
                              font=self.fn_label, bg=C["bg"])
        result_lbl.pack(pady=4)

        def do_verify():
            path = entry.get().strip()
            if not path:
                return
            try:
                import sys
                sys.path.insert(0, os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))
                from modules.verification.verify import verify_capture
                ok = verify_capture(path, silent=True)
                if ok:
                    result_var.set("✔  VALID — not tampered")
                    result_lbl.config(fg=C["green"])
                else:
                    result_var.set("✗  MODIFIED or NOT FOUND")
                    result_lbl.config(fg=C["red"])
            except Exception as e:
                result_var.set(f"Error: {e}")
                result_lbl.config(fg=C["orange"])

        tk.Button(win, text="VERIFY", font=self.fn_label,
                  bg=C["border2"], fg=C["accent"],
                  relief="flat", command=do_verify).pack(pady=4)

    def _quit(self):
        if self._on_quit_cb:
            self._on_quit_cb()
        self.destroy()
