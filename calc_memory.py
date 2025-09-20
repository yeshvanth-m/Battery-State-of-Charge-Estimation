#!/usr/bin/env python3
# FCN memory playground for 16-bit fixed-point inference
# Includes per-layer activation buffers and accumulator modeling

import tkinter as tk
from tkinter import ttk

BYTES_PER_PARAM = 2  # int16 = 2 bytes

class FCNMemoryGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FCN Memory Estimator (Fixed-point 16-bit)")
        self.geometry("980x720")
        self.minsize(980, 720)

        # --- Top config frame ---
        top = ttk.LabelFrame(self, text="Configuration")
        top.pack(fill="x", padx=10, pady=8)

        # Fixed-point width (read-only)
        ttk.Label(top, text="Fixed-point width (bits):").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.fp_bits = tk.IntVar(value=16)
        e_fp = ttk.Entry(top, width=6, textvariable=self.fp_bits, state="readonly")
        e_fp.grid(row=0, column=1, sticky="w", padx=2, pady=4)

        # Number of input features
        ttk.Label(top, text="# Input features:").grid(row=0, column=2, sticky="w", padx=16, pady=4)
        self.n_features = tk.StringVar(value="5")
        sb_nf = ttk.Spinbox(top, from_=1, to=4096, increment=1, width=8,
                            textvariable=self.n_features, command=self.update_all)
        sb_nf.grid(row=0, column=3, sticky="w", padx=2, pady=4)
        self._bind_spinbox_typing(sb_nf)

        # Hidden layers count
        ttk.Label(top, text="# Hidden layers:").grid(row=0, column=4, sticky="w", padx=16, pady=4)
        self.n_layers = tk.StringVar(value="3")
        sb_nl = ttk.Spinbox(top, from_=0, to=6, increment=1, width=6,
                            textvariable=self.n_layers, command=self.refresh_layer_rows)
        sb_nl.grid(row=0, column=5, sticky="w", padx=2, pady=4)
        self._bind_spinbox_typing(sb_nl)

        # Alignment & rounding
        self.enforce_align = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Align widths", variable=self.enforce_align,
                        command=self.apply_alignment).grid(row=1, column=0, sticky="w", padx=6)
        ttk.Label(top, text="to multiple of").grid(row=1, column=1, sticky="w")

        self.align_mult = tk.StringVar(value="16")
        sb_am = ttk.Spinbox(top, from_=1, to=256, increment=1, width=6,
                            textvariable=self.align_mult, command=self.apply_alignment)
        sb_am.grid(row=1, column=2, sticky="w")
        self._bind_spinbox_typing(sb_am, callback=self.apply_alignment)

        # Memory cap (KiB)
        ttk.Label(top, text="Memory cap (KiB):").grid(row=1, column=3, sticky="w", padx=16)
        self.mem_cap_kib = tk.StringVar(value="128")  # change if you want a different default
        sb_mem = ttk.Spinbox(top, from_=1, to=65536, increment=1, width=10,
                             textvariable=self.mem_cap_kib, command=self.update_all)
        sb_mem.grid(row=1, column=4, sticky="w")
        self._bind_spinbox_typing(sb_mem)

        # Activation buffers controls
        self.include_act = tk.BooleanVar(value=True)
        self.ping_pong = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Include activation buffers", variable=self.include_act,
                        command=self.update_all).grid(row=2, column=0, sticky="w", padx=6, pady=4, columnspan=2)
        ttk.Checkbutton(top, text="Ping-pong (double buffer)", variable=self.ping_pong,
                        command=self.update_all).grid(row=2, column=2, sticky="w", padx=16, pady=4, columnspan=2)

        # Per-layer activations vs single max-width buffer (recommended: per-layer)
        self.per_layer_act = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Per-layer activation buffers (I/O)", variable=self.per_layer_act,
                        command=self.update_all).grid(row=2, column=4, sticky="w", padx=16, pady=4, columnspan=2)

        # --- Accumulator modeling ---
        self.include_acc = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Include accumulators", variable=self.include_acc,
                        command=self.update_all).grid(row=3, column=0, sticky="w", padx=6, pady=4, columnspan=2)

        ttk.Label(top, text="Acc width (bits):").grid(row=3, column=2, sticky="w", padx=16)
        self.acc_bits = tk.StringVar(value="32")
        sb_ab = ttk.Spinbox(top, from_=16, to=64, increment=8, width=6,
                            textvariable=self.acc_bits, command=self.update_all)
        sb_ab.grid(row=3, column=3, sticky="w")
        self._bind_spinbox_typing(sb_ab)

        ttk.Label(top, text="PEs (parallel neurons):").grid(row=3, column=4, sticky="w", padx=16)
        self.num_pes = tk.StringVar(value="0")  # 0 = serial (1 accumulator)
        sb_pes = ttk.Spinbox(top, from_=0, to=4096, increment=1, width=10,
                             textvariable=self.num_pes, command=self.update_all)
        sb_pes.grid(row=3, column=5, sticky="w")
        self._bind_spinbox_typing(sb_pes)

        # --- Layer editor frame ---
        self.layer_frame = ttk.LabelFrame(self, text="Hidden Layers (neurons per layer)")
        self.layer_frame.pack(fill="x", padx=10, pady=8)

        # Pre-create up to 6 layer rows
        self.layer_vars = []
        self.layer_rows = []
        defaults = [128, 64, 32, 16, 16, 16]
        for i in range(6):
            v = tk.StringVar(value=str(defaults[i]))
            self.layer_vars.append(v)
            row = self._make_layer_row(self.layer_frame, i, v)
            self.layer_rows.append(row)

        # --- Results frame ---
        res = ttk.LabelFrame(self, text="Results")
        res.pack(fill="both", expand=True, padx=10, pady=8)

        cols = ("Layer", "Weights", "Biases", "Params", "Param Bytes", "Act In (B)", "Act Out (B)", "Acc (B)", "Layer Total (B)", "Layer KiB")
        self.tree = ttk.Treeview(res, columns=cols, show="headings", height=14)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor="center", width=120)
        self.tree.pack(fill="both", expand=True, padx=6, pady=6)

        self.total_lbl = ttk.Label(res, text="", font=("TkDefaultFont", 10, "bold"))
        self.total_lbl.pack(anchor="w", padx=8, pady=6)

        self.status_lbl = ttk.Label(res, text="", font=("TkDefaultFont", 10))
        self.status_lbl.pack(anchor="w", padx=8, pady=2)

        self.note_lbl = ttk.Label(
            res,
            text=(
                "Notes: Params = (weights + biases) × 2 bytes (int16). "
                "Per-layer activations: input+output buffers × (#buffers). "
                "Accumulators: count depends on PEs (0→serial=1 acc)."
            ),
            foreground="#555"
        )
        self.note_lbl.pack(anchor="w", padx=8, pady=2)

        # Traces so typing triggers updates
        self.n_features.trace_add("write", lambda *_: self.update_all())
        self.n_layers.trace_add("write", lambda *_: self.refresh_layer_rows())
        self.align_mult.trace_add("write", lambda *_: self.apply_alignment())
        self.mem_cap_kib.trace_add("write", lambda *_: self.update_all())
        for v in self.layer_vars:
            v.trace_add("write", lambda *_: self.update_all())

        # Initial UI setup
        self.refresh_layer_rows()
        self.update_all()

    # ---------- UI helpers ----------
    def _bind_spinbox_typing(self, spinbox, callback=None):
        """Ensure manual typing triggers updates."""
        def on_return(_event):
            (callback or self.update_all)()
        def on_focus_out(_event):
            (callback or self.update_all)()
        spinbox.bind("<Return>", on_return)
        spinbox.bind("<FocusOut>", on_focus_out)

    def _make_layer_row(self, parent, idx, var):
        frame = ttk.Frame(parent)
        ttk.Label(frame, text=f"Hidden L{idx+1} neurons:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        sb = ttk.Spinbox(frame, from_=0, to=4096, increment=8, width=10, textvariable=var, command=self.update_all)
        sb.grid(row=0, column=1, sticky="w", padx=2, pady=4)
        ttk.Label(frame, text="(0 disables this layer)").grid(row=0, column=2, sticky="w", padx=8)
        self._bind_spinbox_typing(sb)
        return frame

    def refresh_layer_rows(self):
        n = self._safe_int(self.n_layers, default=0)
        n = max(0, min(6, n))
        for i, row in enumerate(self.layer_rows):
            if i < n:
                row.pack(fill="x", padx=8, pady=2)
            else:
                row.pack_forget()
                self.layer_vars[i].set("0")
        # Keep alignment invariant after depth changes
        self.apply_alignment()
        self.update_all()

    def apply_alignment(self):
        if not self.enforce_align.get():
            self.update_all()
            return
        m = max(1, self._safe_int(self.align_mult, default=16))
        n = self._safe_int(self.n_layers, default=0)
        for i in range(n):
            v = self._safe_int(self.layer_vars[i], default=0)
            if v == 0:
                continue
            aligned = max(m, round(v / m) * m)
            self.layer_vars[i].set(str(int(aligned)))
        self.update_all()

    # ---------- Core math ----------
    def compute_memory(self):
        """
        Returns:
          details: list of dicts per layer (hidden + output)
          totals: dict with param_bytes, act_bytes, acc_bytes, total_bytes
        """
        features = max(1, self._safe_int(self.n_features, default=5))
        sizes = []
        for i in range(self._safe_int(self.n_layers, default=0)):
            s = max(0, self._safe_int(self.layer_vars[i], default=0))
            if s > 0:
                sizes.append(s)
        out_dim = 1

        # Config
        bytes_per_param = BYTES_PER_PARAM  # 16-bit weights/biases
        include_act = self.include_act.get()
        per_layer_act = self.per_layer_act.get()
        pingpong = self.ping_pong.get()
        include_acc = self.include_acc.get()
        acc_bits = max(16, self._safe_int(self.acc_bits, default=32))
        acc_bytes_per = (acc_bits + 7) // 8
        pes = self._safe_int(self.num_pes, default=0)  # 0 => serial (1 accumulator)

        details = []
        param_bytes_total = 0
        act_bytes_total = 0
        acc_bytes_total = 0

        prev = features

        def layer_accounting(prev_w, cur_w, name):
            nonlocal param_bytes_total, act_bytes_total, acc_bytes_total

            # Parameters
            w = prev_w * cur_w
            b = cur_w
            p = w + b
            param_bytes = p * bytes_per_param
            param_bytes_total += param_bytes

            # Activations
            act_in_bytes = 0
            act_out_bytes = 0
            if include_act:
                if per_layer_act:
                    nbuf = 2 if pingpong else 1
                    act_in_bytes  = prev_w * bytes_per_param * nbuf
                    act_out_bytes = cur_w  * bytes_per_param * nbuf
                else:
                    # Back-compat: single global buffer approach (less accurate)
                    max_width = max(prev_w, cur_w, 1)
                    nbuf = 2 if pingpong else 1
                    act_out_bytes = max_width * bytes_per_param * nbuf
            act_bytes_total += (act_in_bytes + act_out_bytes)

            # Accumulators
            acc_bytes = 0
            if include_acc:
                if pes <= 0:
                    acc_cnt = 1  # fully serial MAC engine
                else:
                    acc_cnt = min(cur_w, pes)  # partially/fully parallel
                acc_bytes = acc_cnt * acc_bytes_per
                acc_bytes_total += acc_bytes

            layer_total = param_bytes + act_in_bytes + act_out_bytes + acc_bytes

            return {
                "name": f"{name} ({prev_w}→{cur_w})",
                "w": w, "b": b, "p": p,
                "param_bytes": param_bytes,
                "act_in_bytes": act_in_bytes,
                "act_out_bytes": act_out_bytes,
                "acc_bytes": acc_bytes,
                "layer_total": layer_total,
            }

        # Hidden layers
        for idx, s in enumerate(sizes, start=1):
            details.append(layer_accounting(prev, s, f"Hidden L{idx}"))
            prev = s

        # Output layer
        details.append(layer_accounting(prev, out_dim, "Output"))

        totals = {
            "param_bytes": param_bytes_total,
            "act_bytes": act_bytes_total,
            "acc_bytes": acc_bytes_total,
            "total_bytes": param_bytes_total + act_bytes_total + acc_bytes_total,
        }
        return details, totals

    def update_all(self):
        # Recompute and refresh table
        for row in self.tree.get_children():
            self.tree.delete(row)

        details, totals = self.compute_memory()

        for d in details:
            kib = d["layer_total"] / 1024.0
            self.tree.insert("", "end", values=(
                d["name"], d["w"], d["b"], d["p"],
                d["param_bytes"],
                d["act_in_bytes"],
                d["act_out_bytes"],
                d["acc_bytes"],
                d["layer_total"],
                f"{kib:.2f}"
            ))

        weights_kib = totals["param_bytes"] / 1024.0
        act_kib = totals["act_bytes"] / 1024.0
        acc_kib = totals["acc_bytes"] / 1024.0
        total_kib = totals["total_bytes"] / 1024.0
        self.total_lbl.config(
            text=(
                f"Weights+Biases: {totals['param_bytes']} B ({weights_kib:.2f} KiB)   "
                f"+ Activations: {totals['act_bytes']} B ({act_kib:.2f} KiB)   "
                f"+ Accumulators: {totals['acc_bytes']} B ({acc_kib:.2f} KiB)   "
                f"= Total: {totals['total_bytes']} B ({total_kib:.2f} KiB)"
            )
        )

        cap_kib = max(1.0, self._safe_float(self.mem_cap_kib, default=128.0))
        cap_bytes = cap_kib * 1024.0
        ok = totals["total_bytes"] <= cap_bytes
        status = "WITHIN cap" if ok else "OVER cap"
        color = "#0a7" if ok else "#c00"
        self.status_lbl.config(text=f"Cap: {cap_kib:.0f} KiB → {status}", foreground=color)

    # ---------- Safe getters ----------
    def _safe_int(self, var: tk.StringVar, default=0):
        try:
            s = var.get().strip()
            if s == "":
                return default
            return int(float(s))
        except Exception:
            return default

    def _safe_float(self, var: tk.StringVar, default=0.0):
        try:
            s = var.get().strip()
            if s == "":
                return default
            return float(s)
        except Exception:
            return default


if __name__ == "__main__":
    app = FCNMemoryGUI()
    app.mainloop()
