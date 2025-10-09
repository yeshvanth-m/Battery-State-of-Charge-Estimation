#!/usr/bin/env python3
"""
analyze_and_plot_fixedpoint.py

Parse a C header produced by export_fcn_weights.py (soc_fcn_weights.h),
analyze weight/bias ranges, suggest fixed-point Q formats, visualize distributions,
and simulate quantized inference to measure error.

Outputs saved to ./fp_analysis_outputs:
 - CSV summary
 - Histograms, scatter plots, MSE-vs-f plots
 - quant_vs_float inference comparison (MSE)
"""

import re
import os
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------- User-editable ----------
HEADER_FILE = "soc_fcn_weights.h"   # point to your header if needed
OUT_DIR = Path("fp_analysis_outputs")
OUT_DIR.mkdir(exist_ok=True)
# Which integer sizes to consider
INT_CANDIDATES = [8, 16, 32]
# When suggesting f, ensure we don't exceed B-1 (bits for magnitude)
# and keep f >= 0
# Reserve nothing else here; you can optionally reduce f to keep headroom.
# ---------------------------------

def read_header_text(path):
    return Path(path).read_text()

def parse_array(text, name):
    # Finds "float NAME[...]= { ... };" capturing contents
    pattern = rf"float\s+{re.escape(name)}\s*\[.*?\]\s*=\s*\{{(.*?)\}};"
    m = re.search(pattern, text, re.S)
    if not m:
        return None
    arr_text = m.group(1)
    arr_text = arr_text.replace("\n", " ").replace("{", " ").replace("}", " ")
    parts = [p.strip() for p in arr_text.split(",") if p.strip()]
    vals = np.array([float(p) for p in parts], dtype=np.float32)
    return vals

def find_defines(text):
    defines = {}
    for m in re.finditer(r"#define\s+([A-Za-z0-9_]+)\s+([0-9]+)", text):
        defines[m.group(1)] = int(m.group(2))
    return defines

def reshape_if_possible(arr, name, defines):
    if arr is None:
        return None
    # known naming patterns
    if name == "FCN_L1_W" and "FCN_L1_OUT" in defines and "FCN_L1_IN" in defines:
        return arr.reshape(defines["FCN_L1_OUT"], defines["FCN_L1_IN"])
    if name == "FCN_L2_W" and "FCN_L2_OUT" in defines and "FCN_L2_IN" in defines:
        return arr.reshape(defines["FCN_L2_OUT"], defines["FCN_L2_IN"])
    if name == "FCN_OUT_W" and "FCN_OUT_OUT" in defines and "FCN_OUT_IN" in defines:
        return arr.reshape(defines["FCN_OUT_OUT"], defines["FCN_OUT_IN"])
    # biases shapes known too
    if name.startswith("FCN_L") and name.endswith("_b"):
        # try to pick matching define
        key_out = name.replace("FCN_", "").replace("_b","").replace("_W","") + "_OUT"
    return arr

def suggest_fractional_bits(vals, total_bits=16, signed=True):
    """Return suggested fractional bits f such that max(|vals|) * 2^f <= max_int, capped to [0, total_bits-1]."""
    if vals.size == 0:
        return 0
    max_abs = float(np.max(np.abs(vals)))
    if max_abs == 0.0:
        return total_bits-1
    if signed:
        max_int = 2**(total_bits-1) - 1
    else:
        max_int = 2**total_bits - 1
    f_max = math.floor(math.log2(max_int / max_abs))
    # cap
    f_max = max(0, f_max)
    f_cap = min(f_max, total_bits-1)
    return int(f_cap)

def quantize_array(vals, total_bits=16, frac_bits=15):
    """Quantize float array to signed integers of specified bits and frac bits, then dequantize to float."""
    if vals.size == 0:
        return vals.copy()
    # compute scale
    scale = 2 ** frac_bits
    # saturation limits for signed
    max_int = 2**(total_bits-1) - 1
    min_int = -2**(total_bits-1)
    ints = np.round(vals * scale).astype(np.int64)
    ints = np.clip(ints, min_int, max_int)
    deq = (ints.astype(np.float32) / float(scale))
    return deq, ints

def detect_fixed_fraction(vals, max_k=28, tol=1e-6):
    """Detect smallest k such that values are multiples of 2^-k within tolerance."""
    if vals.size == 0:
        return None
    for k in range(0, max_k+1):
        scaled = vals * (2**k)
        if np.max(np.abs(np.round(scaled) - scaled)) < tol:
            return k
    return None

# ----------------- Main -----------------
def main():
    text = read_header_text(HEADER_FILE)
    defines = find_defines(text)

    # Identify arrays present
    candidate_names = []
    # find all float arrays
    for m in re.finditer(r"float\s+([A-Za-z0-9_]+)\s*\[", text):
        candidate_names.append(m.group(1))
    candidate_names = list(dict.fromkeys(candidate_names))  # preserve order unique

    arrays = {}
    for name in candidate_names:
        arr = parse_array(text, name)
        if arr is None:
            continue
        arr = reshape_if_possible(arr, name, defines)
        arrays[name] = arr

    # We'll look for FCN_L1_W, FCN_L1_b, FCN_L2_W, FCN_L2_b, FCN_OUT_W, FCN_OUT_b
    # but the script is generic enough to process any float arrays
    summary_rows = []
    plots = []

    # For quantized inference we need layer order. We'll attempt to use FCN_L1, FCN_L2, FCN_OUT
    layer_order = []
    if "FCN_L1_W" in arrays:
        layer_order.append(("L1", "FCN_L1_W", "FCN_L1_b"))
    if "FCN_L2_W" in arrays:
        layer_order.append(("L2", "FCN_L2_W", "FCN_L2_b"))
    # Output
    if "FCN_OUT_W" in arrays:
        layer_order.append(("OUT", "FCN_OUT_W", "FCN_OUT_b"))

    # Per-array analysis
    for name, arr in arrays.items():
        vals = arr.ravel()
        mn, mx = float(np.min(vals)), float(np.max(vals))
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        max_abs = float(np.max(np.abs(vals)))
        detected_k = detect_fixed_fraction(vals, max_k=28, tol=1e-8)
        suggestions = {}
        for B in INT_CANDIDATES:
            f = suggest_fractional_bits(vals, total_bits=B, signed=True)
            # compute mse if we quantize with this f
            deq, _ = quantize_array(vals, total_bits=B, frac_bits=f)
            mse = float(np.mean((vals - deq)**2))
            suggestions[B] = {"f": f, "mse": mse}
        summary_rows.append({
            "array": name,
            "shape": getattr(arr, "shape", (arr.size,)),
            "min": mn, "max": mx, "mean": mean, "std": std, "max_abs": max_abs,
            "detected_fractional_k": detected_k,
            "suggestions": suggestions
        })

        # PLOTS: histogram
        plt.figure(figsize=(6,4))
        plt.hist(vals, bins=120)
        plt.title(f"Histogram {name} shape={arr.shape}")
        plt.xlabel("value")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{name}_hist.png")
        plt.close()

        # If this is a weight matrix with matching bias, we'll do scatter later.

        # MSE vs f (for B=16)
        B = 16
        if max_abs > 0:
            max_int = 2**(B-1)-1
            f_max_possible = math.floor(math.log2(max_int / max_abs))
            f_range = list(range(max(0, f_max_possible-6), f_max_possible+5))
            errs = []
            for f in f_range:
                deq, _ = quantize_array(vals, total_bits=B, frac_bits=f)
                errs.append(float(np.mean((vals - deq)**2)))
            plt.figure(figsize=(6,4))
            plt.plot(f_range, errs, marker='o')
            plt.title(f"MSE vs fractional bits for {name} (B=16)")
            plt.xlabel("fractional bits (f)")
            plt.ylabel("MSE")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"{name}_mse_vs_f.png")
            plt.close()

    # Scatter plots: weights vs biases per layer
    for lname, Wname, bname in layer_order:
        if Wname not in arrays:
            continue
        W = arrays[Wname]
        b = arrays.get(bname, None)
        if b is None:
            continue
        # For scatter: pair each weight with its neuron's bias (repeat bias for each input)
        W_flat = W.ravel()
        b_rep = np.repeat(b, W.shape[1])
        plt.figure(figsize=(6,4))
        plt.scatter(W_flat, b_rep, s=6, alpha=0.6)
        plt.title(f"Weights vs Biases scatter for {lname} (W:{W.shape} b:{b.shape})")
        plt.xlabel("weight")
        plt.ylabel("bias (repeated per input)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{lname}_weights_vs_bias_scatter.png")
        plt.close()

    # Build summary dataframe and write CSV
    rows_for_df = []
    for r in summary_rows:
        row = {
            "array": r["array"],
            "shape": str(r["shape"]),
            "min": r["min"], "max": r["max"], "max_abs": r["max_abs"],
            "detected_fractional_k": r["detected_fractional_k"]
        }
        for B in INT_CANDIDATES:
            row[f"Q{B}_f"] = r["suggestions"][B]["f"]
            row[f"Q{B}_mse"] = r["suggestions"][B]["mse"]
        rows_for_df.append(row)
    df = pd.DataFrame(rows_for_df)
    csv_path = OUT_DIR / "fixed_point_suggestion_summary.csv"
    df.to_csv(csv_path, index=False)

    # ---------------- Simulated inference with quantized params ----------------
    # We'll quantize per layer using suggested int16 by default (or choose the best)
    quant_choice = {}
    for r in summary_rows:
        name = r["array"]
        # prefer int16 if it can represent values (i.e., suggested f > 0), else choose int32
        s16 = r["suggestions"].get(16, None)
        if s16 is not None and s16["f"] >= 0:
            quant_choice[name] = (16, s16["f"])
        else:
            s32 = r["suggestions"].get(32, {"f": 0})
            quant_choice[name] = (32, s32["f"])

    # Create quantized-dequantized arrays
    deq_arrays = {}
    int_arrays = {}
    for name, arr in arrays.items():
        B, f = quant_choice.get(name, (16, 0))
        deq, ints = quantize_array(arr.ravel(), total_bits=B, frac_bits=f)
        deq = deq.reshape(arr.shape).astype(np.float32)
        int_arrays[name] = ints.reshape(arr.shape)
        deq_arrays[name] = deq

    # Simulate N random inputs and compute float vs quantized-dequantized outputs
    rng = np.random.RandomState(0)
    # infer input size from defines or from FCN_L1_W shape
    if "FCN_INPUT_SIZE" in defines:
        input_size = defines["FCN_INPUT_SIZE"]
    elif "FCN_L1_W" in arrays:
        input_size = arrays["FCN_L1_W"].shape[1]
    else:
        input_size = 5

    # Choose random test set (uniform within [-1,1]) and a set scaled by typical weight magnitudes
    N_test = 200
    X1 = rng.uniform(-1.0, 1.0, size=(N_test, input_size)).astype(np.float32)
    # Also try gaussian scaled inputs
    X2 = rng.normal(scale=0.1, size=(N_test, input_size)).astype(np.float32)

    def forward_float(X):
        a = X
        for lname, Wname, bname in layer_order:
            W = arrays[Wname]   # float
            b = arrays[bname]
            a = np.dot(a, W.T) + b  # (batch, out)
            a = np.maximum(a, 0.0)
        # final output:
        return a

    def forward_dequant(X):
        a = X
        for lname, Wname, bname in layer_order:
            W = deq_arrays[Wname]
            b = deq_arrays[bname]
            a = np.dot(a, W.T) + b
            a = np.maximum(a, 0.0)
        return a

    Yf1 = forward_float(X1)
    Yq1 = forward_dequant(X1)
    mse1 = float(np.mean((Yf1 - Yq1)**2))

    Yf2 = forward_float(X2)
    Yq2 = forward_dequant(X2)
    mse2 = float(np.mean((Yf2 - Yq2)**2))

    # Save simple report
    with open(OUT_DIR / "quant_inference_report.txt", "w") as fh:
        fh.write("Quantized inference simulation report\n")
        fh.write("===================================\n\n")
        fh.write(f"Header parsed: {HEADER_FILE}\n")
        fh.write(f"Input size used: {input_size}\n")
        fh.write("\nPer-array chosen quantization (B, f):\n")
        for name, (B,f) in quant_choice.items():
            fh.write(f"  {name}: B={B}, f={f}\n")
        fh.write("\nSimulated inference MSEs (float vs dequantized-fixed):\n")
        fh.write(f"  Uniform[-1,1] inputs (N={N_test}): MSE = {mse1:.6e}\n")
        fh.write(f"  Gaussian(scale=0.1) inputs (N={N_test}): MSE = {mse2:.6e}\n")

    # Also print top-level summary
    print("Saved analysis into:", OUT_DIR)
    print("Summary CSV:", csv_path)
    print("Quantized inference report:", OUT_DIR / "quant_inference_report.txt")
    print("Example plots (histograms + mse-vs-f + scatter) saved in folder.")

if __name__ == "__main__":
    main()
