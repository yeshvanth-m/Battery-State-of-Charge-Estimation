#!/usr/bin/env python3
"""
save_fp_recommendations_with_minmax.py

Parses:
 - /mnt/data/scaler_with_rows.h
 - /mnt/data/soc_fcn_weights.h

Produces a JSON at /mnt/data/fp_register_recommendations.json containing:
 - per-feature min/max/max_abs for selected inputs
 - per-weight and bias min/max/max_abs
 - per-neuron pre-activation min/max
 - recommended fixed-point (conservative) and minimum (tight, no overflow) formats
 - accumulator bit requirements (uses signed ranges)

No string matching: set INPUT_INDICES to the exact 0-based columns used by your model.
"""
import re, math, json
from pathlib import Path
import numpy as np

# -------- CONFIG --------
SCALER_HEADER = Path("scaler_with_rows.h")
WEIGHTS_HEADER = Path("soc_fcn_weights.h")
OUT_JSON = Path("fp_register_recommendations.json")

# Input indices (0-based) — edit if your inputs are in different columns
INPUT_INDICES = [2, 3, 4, 5, 7]
INPUT_LABELS = ["Voltage", "Current", "Temperature", "Capacity", "Cumulative_Capacity"]

# Recommended storage defaults
RECOMMENDED_TOTAL_BITS = 16
RECOMMENDED_MIN_FRAC = 8
RECOMMENDED_WEIGHT_FRAC = 13  # Q0.13 for weights

# Minimum policy: choose smallest fractional bits that still allow representing values without overflow.
MIN_TOTAL_BITS_CANDIDATES = [8, 16, 32]  # consider these containers for minimum suggestion

# -------- helpers --------
def read_text_or_raise(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    return p.read_text()

def parse_normalized_data(text: str):
    m = re.search(r"NORMALIZED_DATA\s*\[.*?\]\s*=\s*\{(.*)\}\s*;", text, re.S)
    if not m:
        raise ValueError("NORMALIZED_DATA not found in scaler header")
    inner = m.group(1)
    py = "[" + inner.replace("{", "[").replace("}", "]") + "]"
    arr = np.array(eval(py), dtype=np.float32)
    return arr

def parse_array_by_name(text: str, name: str):
    m = re.search(rf"float\s+{re.escape(name)}\s*\[.*?\]\s*=\s*\{{(.*?)\}};", text, re.S)
    if not m:
        return None
    inner = m.group(1)
    py = "[" + inner.replace("{", "[").replace("}", "]") + "]"
    arr = np.array(eval(py), dtype=np.float32)
    return arr

def find_defines(text: str, prefix=""):
    d = {}
    for m in re.finditer(r"#define\s+([A-Za-z0-9_]+)\s+([0-9]+)", text):
        k, v = m.group(1), int(m.group(2))
        if prefix == "" or k.startswith(prefix):
            d[k] = v
    return d

def signed_required_integer_bits(min_val, max_val):
    """Return integer_bits needed to represent signed values (magnitude) without fractional part.
    integer_bits = ceil(log2(max(|min|,|max|))) if >=1 else 0.
    """
    max_abs = max(abs(min_val), abs(max_val))
    if max_abs < 1.0:
        return 0
    return int(math.ceil(math.log2(max_abs)))

def fraction_bits_for_total(total_bits, min_val, max_val):
    """Return the maximum fractional bits f such that (value * 2^f) fits into signed (total_bits) container without overflow.
       We compute f = floor(log2(max_int / max_abs)), capped to [0, total_bits-1].
    """
    max_abs = max(abs(min_val), abs(max_val))
    if max_abs == 0:
        return total_bits - 1
    max_int = 2**(total_bits-1) - 1
    f = int(math.floor(math.log2(max_int / max_abs))) if max_abs > 0 else total_bits - 1
    f = max(0, min(f, total_bits - 1))
    return f

def recommend_recommended_format(min_val, max_val, total_bits=16, min_frac=8):
    """Conservative 'recommended' format: reserve integer bits based on magnitude, then use remaining bits for fractional,
       but ensure at least min_frac fractional bits."""
    int_bits = signed_required_integer_bits(min_val, max_val)
    frac = total_bits - 1 - max(0, int_bits)
    frac = max(min_frac, frac)
    # if frac would be negative (not enough total bits), clamp to 0 and increase int_bits accordingly
    if frac < 0:
        frac = 0
    return {"total_bits": total_bits, "sign_bits": 1, "integer_bits": int_bits, "fractional_bits": int(frac), "notation": f"Q{int_bits}.{int(frac)}"}

def recommend_minimum_format(min_val, max_val, candidates=(8,16,32)):
    """Find smallest container and largest possible fractional bits that avoid overflow."""
    best = None
    for B in sorted(candidates):
        f = fraction_bits_for_total(B, min_val, max_val)
        # If fractional bits computed zero or more, this container works (no overflow)
        if f >= 0:
            cand = {"total_bits": int(B), "sign_bits":1, "fractional_bits": int(f), "integer_bits": int(max(0, B-1-f)), "notation": f"Q{int(max(0,B-1-f))}.{int(f)}"}
            best = cand
            # We prefer smallest container, so break at first valid
            break
    if best is None:
        # fallback: use largest container with its max fractional bits
        B = max(candidates)
        f = fraction_bits_for_total(B, min_val, max_val)
        best = {"total_bits": int(B), "sign_bits":1, "fractional_bits": int(f), "integer_bits": int(max(0, B-1-f)), "notation": f"Q{int(max(0,B-1-f))}.{int(f)}"}
    return best

# -------- main --------
def main():
    s_text = read_text_or_raise(SCALER_HEADER)
    w_text = read_text_or_raise(WEIGHTS_HEADER)

    normalized = parse_normalized_data(s_text)  # (rows, features)
    n_rows, n_feats = normalized.shape

    # validate indices
    if any(i < 0 or i >= n_feats for i in INPUT_INDICES):
        raise IndexError(f"INPUT_INDICES out of range (0..{n_feats-1})")

    # pick selected columns
    selected_norm = normalized[:, INPUT_INDICES]  # (rows, 5)
    # compute per-feature min/max/max_abs
    input_min = np.min(selected_norm, axis=0)
    input_max = np.max(selected_norm, axis=0)
    input_maxabs = np.max(np.abs(selected_norm), axis=0)

    # parse weights & biases and reshape if defines present
    arrays = {}
    for name in ["FCN_L1_W","FCN_L1_b","FCN_L2_W","FCN_L2_b","FCN_OUT_W","FCN_OUT_b"]:
        arrays[name] = parse_array_by_name(w_text, name)

    defines = find_defines(w_text, "FCN_")
    try:
        if arrays.get("FCN_L1_W") is not None and "FCN_L1_OUT" in defines and "FCN_L1_IN" in defines:
            arrays["FCN_L1_W"] = arrays["FCN_L1_W"].reshape(defines["FCN_L1_OUT"], defines["FCN_L1_IN"])
        if arrays.get("FCN_L2_W") is not None and "FCN_L2_OUT" in defines and "FCN_L2_IN" in defines:
            arrays["FCN_L2_W"] = arrays["FCN_L2_W"].reshape(defines["FCN_L2_OUT"], defines["FCN_L2_IN"])
        if arrays.get("FCN_OUT_W") is not None and "FCN_OUT_OUT" in defines and "FCN_OUT_IN" in defines:
            arrays["FCN_OUT_W"] = arrays["FCN_OUT_W"].reshape(defines["FCN_OUT_OUT"], defines["FCN_OUT_IN"])
    except Exception:
        pass

    # gather min/max stats for weights and biases
    w_stats = {}
    for k in ["FCN_L1_W","FCN_L2_W","FCN_OUT_W"]:
        arr = arrays.get(k)
        if arr is None:
            w_stats[k] = None
        else:
            w_stats[k] = {
                "shape": tuple(arr.shape),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "max_abs": float(np.max(np.abs(arr)))
            }
    b_stats = {}
    for k in ["FCN_L1_b","FCN_L2_b","FCN_OUT_b"]:
        arr = arrays.get(k)
        if arr is None:
            b_stats[k] = None
        else:
            b_stats[k] = {"shape": tuple(arr.shape), "min": float(np.min(arr)), "max": float(np.max(arr)), "max_abs": float(np.max(np.abs(arr)))}

    # compute per-neuron pre-activation min/max bounds using interval worst-case (inputs in [-max_abs_i, +max_abs_i])
    # pre_min_j = -sum_i |W[j,i]|*max_abs_input[i] + b_j
    # pre_max_j = +sum_i |W[j,i]|*max_abs_input[i] + b_j
    pre_stats = {}
    # L1
    W1 = arrays["FCN_L1_W"]
    b1 = arrays["FCN_L1_b"]
    W1_abs = np.abs(W1)
    sum_abs_mul_inputs = np.dot(W1_abs, input_maxabs)  # (out,)
    pre_min_L1 = -sum_abs_mul_inputs + b1
    pre_max_L1 = sum_abs_mul_inputs + b1
    pre_stats["L1"] = {"per_neuron_min": pre_min_L1.tolist(), "per_neuron_max": pre_max_L1.tolist(),
                       "min": float(np.min(pre_min_L1)), "max": float(np.max(pre_max_L1))}

    # L2 uses activations from L1 (ReLU -> non-negative). Conservative: use per-neuron L1 max as input max for L2.
    act1_max_per_neuron = np.maximum(sum_abs_mul_inputs + b1, 0.0)
    max_act1 = float(np.max(act1_max_per_neuron))
    # use a vector of input maxima for L2 equal to max_act1 repeated
    W2 = arrays["FCN_L2_W"]
    b2 = arrays["FCN_L2_b"]
    W2_abs = np.abs(W2)
    sum_abs_mul_act1 = np.dot(W2_abs, np.full(W2.shape[1], max_act1))
    pre_min_L2 = -sum_abs_mul_act1 + b2
    pre_max_L2 = sum_abs_mul_act1 + b2
    pre_stats["L2"] = {"per_neuron_min": pre_min_L2.tolist(), "per_neuron_max": pre_max_L2.tolist(),
                       "min": float(np.min(pre_min_L2)), "max": float(np.max(pre_max_L2))}

    # OUT similarly: use L2 activation max as input
    max_act2 = float(np.max(np.maximum(np.dot(W2_abs, np.full(W2.shape[1], max_act1)) + b2, 0.0)))
    Wout = arrays["FCN_OUT_W"]
    Bout = arrays["FCN_OUT_b"]
    Wout_abs = np.abs(Wout)
    sum_abs_mul_act2 = np.dot(Wout_abs, np.full(Wout.shape[1], max_act2))
    pre_min_OUT = -sum_abs_mul_act2 + Bout
    pre_max_OUT = sum_abs_mul_act2 + Bout
    pre_stats["OUT"] = {"per_neuron_min": pre_min_OUT.tolist(), "per_neuron_max": pre_max_OUT.tolist(),
                        "min": float(np.min(pre_min_OUT)), "max": float(np.max(pre_max_OUT))}

    # Now compute recommended and minimum formats for inputs, weights, biases, pre-activation ranges
    inputs = []
    for i, label in enumerate(INPUT_LABELS):
        minv = float(input_min[i])
        maxv = float(input_max[i])
        maxabs = float(input_maxabs[i])
        rec = recommend_recommended_format(minv, maxv, total_bits=RECOMMENDED_TOTAL_BITS, min_frac=RECOMMENDED_MIN_FRAC)
        mn = recommend_minimum_format(minv, maxv, candidates=MIN_TOTAL_BITS_CANDIDATES)
        inputs.append({
            "label": label,
            "index": int(INPUT_INDICES[i]),
            "min": minv,
            "max": maxv,
            "max_abs": maxabs,
            "recommended": rec,
            "minimum": mn
        })

    def make_arr_stats(name, arr_min, arr_max):
        rec = recommend_recommended_format(float(np.min(arr_min)), float(np.max(arr_max)), total_bits=RECOMMENDED_TOTAL_BITS, min_frac=RECOMMENDED_MIN_FRAC)
        mn = recommend_minimum_format(float(np.min(arr_min)), float(np.max(arr_max)), candidates=MIN_TOTAL_BITS_CANDIDATES)
        return {"min": float(np.min(arr_min)), "max": float(np.max(arr_max)), "recommended": rec, "minimum": mn}

    # weights & biases recommendations (per-layer)
    weights_reco = {}
    biases_reco = {}
    for k in ["FCN_L1_W","FCN_L2_W","FCN_OUT_W"]:
        arr = arrays[k]
        weights_reco[k] = {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "max_abs": float(np.max(np.abs(arr))),
            "recommended": recommend_recommended_format(float(np.min(arr)), float(np.max(arr)), RECOMMENDED_TOTAL_BITS, RECOMMENDED_MIN_FRAC),
            "minimum": recommend_minimum_format(float(np.min(arr)), float(np.max(arr)), MIN_TOTAL_BITS_CANDIDATES)
        }
    for k in ["FCN_L1_b","FCN_L2_b","FCN_OUT_b"]:
        arr = arrays[k]
        biases_reco[k] = {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "max_abs": float(np.max(np.abs(arr))),
            "recommended": recommend_recommended_format(float(np.min(arr)), float(np.max(arr)), RECOMMENDED_TOTAL_BITS, RECOMMENDED_MIN_FRAC),
            "minimum": recommend_minimum_format(float(np.min(arr)), float(np.max(arr)), MIN_TOTAL_BITS_CANDIDATES)
        }

    # accumulator requirements: use product_frac = max(input_frac) + chosen_weight_frac for recommended path
    rec_input_fracs = [inp["recommended"]["fractional_bits"] for inp in inputs]
    max_input_frac = int(max(rec_input_fracs))
    product_frac_rec = max_input_frac + RECOMMENDED_WEIGHT_FRAC

    # Compute required total bits for accumulators using signed range of pre-activation (min,max)
    def accum_req_from_pre(min_pre, max_pre, prod_frac):
        # integer bits needed to represent signed integer part:
        int_bits = signed_required_integer_bits(min_pre, max_pre)
        total_bits = 1 + int_bits + int(prod_frac)
        return {"min_pre": float(min_pre), "max_pre": float(max_pre), "product_frac": int(prod_frac), "int_bits_needed": int(int_bits), "total_bits_needed": int(total_bits)}

    accum_L1_req = accum_req_from_pre(pre_stats["L1"]["min"], pre_stats["L1"]["max"], product_frac_rec)
    accum_L2_req = accum_req_from_pre(pre_stats["L2"]["min"], pre_stats["L2"]["max"], product_frac_rec)
    accum_OUT_req = accum_req_from_pre(pre_stats["OUT"]["min"], pre_stats["OUT"]["max"], product_frac_rec)

    # Build result
    result = {
        "input_indices": INPUT_INDICES,
        "input_labels": INPUT_LABELS,
        "inputs": inputs,
        "weights": weights_reco,
        "biases": biases_reco,
        "pre_activation_stats": pre_stats,
        "accumulators_recommended": {"L1": accum_L1_req, "L2": accum_L2_req, "OUT": accum_OUT_req},
        "notes": {
            "weight_fractional_recommended": RECOMMENDED_WEIGHT_FRAC,
            "recommended_storage_bits": RECOMMENDED_TOTAL_BITS,
            "minimum_candidates_bits": MIN_TOTAL_BITS_CANDIDATES,
            "behavior": "minimum format is the smallest container (from candidates) that can store values without overflow; recommended format keeps >= RECOMMENDED_MIN_FRAC fractional bits."
        }
    }

    OUT_JSON.write_text(json.dumps(result, indent=2))
    print("Saved JSON to", OUT_JSON)

    # Print concise summary
    print("\nINPUT summary (min / max / max_abs) and recommended / minimum formats:")
    for inp in inputs:
        print(f" - {inp['label']} (idx {inp['index']}): min={inp['min']:.6g}, max={inp['max']:.6g}, max_abs={inp['max_abs']:.6g}")
        print(f"    recommended: {inp['recommended']['notation']}  minimum: {inp['minimum']['notation']}")

    print("\nAccumulator recommended totals (bits):")
    print(" L1:", accum_L1_req)
    print(" L2:", accum_L2_req)
    print(" OUT:", accum_OUT_req)

if __name__ == "__main__":
    main()
