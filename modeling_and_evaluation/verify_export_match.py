#!/usr/bin/env python3
# verify_export_match.py
# Put it next to soc_fcn_weights.h and (optionally) soc_fcn_model.pth and run.

import os, re, math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

MODEL_PATH = "soc_fcn_model.pth"      # adjust if needed
HEADER_PATH = "soc_fcn_weights.h"

# --- Model definition (must match training model exactly) ---
class SoCFCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout = 0.1):
        super(SoCFCN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        # first
        self.hidden_layers.append(nn.Linear(input_size, hidden_size))
        self.batch_norm_layers.append(nn.BatchNorm1d(hidden_size))
        for i in range(1, num_layers):
            layer_size = hidden_size // (2 ** i)
            self.hidden_layers.append(nn.Linear(hidden_size // (2 ** (i - 1)), layer_size))
            self.batch_norm_layers.append(nn.BatchNorm1d(layer_size))
        self.output_layer = nn.Linear(hidden_size // (2 ** (num_layers - 1)), 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        for hidden_layer, batch_norm_layer in zip(self.hidden_layers, self.batch_norm_layers):
            x = self.relu(batch_norm_layer(hidden_layer(x)))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

# ---- helpers ----
def load_state_dict_robust(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict):
        return ckpt.state_dict(), {}, True
    # pick candidate wrappers
    possible_wrappers = ["state_dict","model_state_dict","weights","params","net","model"]
    state_dict = None
    wrapper = None
    for k in possible_wrappers:
        if k in ckpt and isinstance(ckpt[k], dict):
            state_dict = ckpt[k]
            wrapper = k
            break
    if state_dict is None:
        state_dict = ckpt
    # strip module.
    def strip_module(sd):
        out = {}
        for k,v in sd.items():
            out[k[len("module."):] if k.startswith("module.") else k] = v
        return out
    state_dict = strip_module(state_dict)
    meta = {}
    if wrapper is not None:
        for k,v in ckpt.items():
            if k != wrapper:
                meta[k] = v
    return state_dict, meta, False

def fuse_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d):
    with torch.no_grad():
        W = linear.weight.clone()
        b = linear.bias.clone() if linear.bias is not None else torch.zeros(W.size(0), dtype=W.dtype)
        gamma = bn.weight if bn.weight is not None else torch.ones_like(bn.running_mean)
        beta  = bn.bias   if bn.bias is not None else torch.zeros_like(bn.running_mean)
        mean  = bn.running_mean
        var   = bn.running_var
        eps   = bn.eps
        inv_std = gamma / torch.sqrt(var + eps)
        W_fused = W * inv_std.unsqueeze(1)
        b_fused = (b - mean) * inv_std + beta
        fused = nn.Linear(W.size(1), W.size(0), bias=True)
        fused.weight.data.copy_(W_fused)
        fused.bias.data.copy_(b_fused)
        return fused

def parse_header_arrays(header_path):
    text = Path(header_path).read_text()
    # collect defines
    defs = {m.group(1):int(m.group(2)) for m in re.finditer(r"#define\s+([A-Za-z0-9_]+)\s+([0-9]+)", text)}
    def extract(name):
        m = re.search(rf"float\s+{name}\s*\[.*?\]\s*=\s*\{{(.*?)\}};", text, re.S)
        if not m:
            return None
        s = m.group(1).replace('\n',' ').replace('{',' ').replace('}',' ')
        nums = [float(x) for x in re.split(r",\s*", s) if x.strip()]
        return np.array(nums, dtype=np.float32)
    arrays = {}
    for name in ["FCN_L1_W","FCN_L1_b","FCN_L2_W","FCN_L2_b","FCN_OUT_W","FCN_OUT_b"]:
        a = extract(name)
        if a is None:
            arrays[name] = None
            continue
        # reshape where possible using defines
        if name=="FCN_L1_W" and "FCN_L1_OUT" in defs and "FCN_L1_IN" in defs:
            arrays[name] = a.reshape(defs["FCN_L1_OUT"], defs["FCN_L1_IN"])
        elif name=="FCN_L2_W" and "FCN_L2_OUT" in defs and "FCN_L2_IN" in defs:
            arrays[name] = a.reshape(defs["FCN_L2_OUT"], defs["FCN_L2_IN"])
        elif name=="FCN_OUT_W" and "FCN_OUT_OUT" in defs and "FCN_OUT_IN" in defs:
            arrays[name] = a.reshape(defs["FCN_OUT_OUT"], defs["FCN_OUT_IN"])
        else:
            arrays[name] = a
    arrays["defines"] = defs
    return arrays

def c_simulate(xs, fused_weights, fused_biases, W_out, B_out):
    a = xs.copy()
    for W,b in zip(fused_weights, fused_biases):
        a = np.dot(a, W.T) + b  # W is (out, in)
        a = np.maximum(a, 0.0)
    y = np.dot(a, W_out.T) + B_out
    return y

# --- main verification ---
def main():
    hdr = Path(HEADER_PATH)
    if not hdr.exists():
        print("Header not found:", HEADER_PATH); return
    arrays = parse_header_arrays(HEADER_PATH)
    defs = arrays.pop("defines", {})
    print("Header defines:", defs)

    print("\nHeader arrays present and shapes:")
    for k,v in arrays.items():
        if v is None:
            print(" ", k, "MISSING")
        else:
            print(" ", k, np.shape(v))

    # load checkpoint if present
    if Path(MODEL_PATH).exists():
        sd, meta, whole = load_state_dict_robust(MODEL_PATH)
        print("\nLoaded checkpoint:", MODEL_PATH, "whole_model_saved?", whole)
        print("meta keys:", list(meta.keys()))
        # infer sizes
        IN = int(meta.get("input_size", defs.get("FCN_INPUT_SIZE", 5)))
        HS = int(meta.get("hidden_size", defs.get("FCN_L1_OUT", 128)))
        NL = int(meta.get("num_layers", 2))
        print("Using inferred sizes:", "input",IN,"hidden",HS,"num_layers",NL)
        model = SoCFCN(IN, HS, NL, dropout=float(meta.get("dropout",0.0)))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        model.eval()
        # fuse in PyTorch and read weights
        fused_linears = []
        for lin,bn in zip(model.hidden_layers, model.batch_norm_layers):
            fused_linears.append(fuse_linear_bn(lin, bn))
        fused_weights = [f.weight.detach().cpu().numpy().astype(np.float32) for f in fused_linears]
        fused_biases  = [f.bias.detach().cpu().numpy().astype(np.float32) for f in fused_linears]
        out_lin = model.output_layer
        W_out = out_lin.weight.detach().cpu().numpy().astype(np.float32)
        B_out = out_lin.bias.detach().cpu().numpy().astype(np.float32)
        # compare with header arrays (if present)
        ok = True
        for i,(Wh_name,Wh_pt) in enumerate(zip(["FCN_L1_W","FCN_L2_W"], fused_weights)):
            Whdr = arrays.get(f"FCN_L{i+1}_W")
            bhdr = arrays.get(f"FCN_L{i+1}_b")
            if Whdr is None:
                print("Header missing", f"FCN_L{i+1}_W"); ok=False; continue
            wdiff = np.max(np.abs(Whdr - Wh_pt))
            print(f"Layer {i+1} W max abs diff header vs fused-pytorch: {wdiff:.6g}")
            if bhdr is not None:
                bdiff = np.max(np.abs(bhdr - fused_biases[i]))
                print(f"Layer {i+1} b max abs diff header vs fused-pytorch: {bdiff:.6g}")
            if wdiff > 1e-5:
                ok=False
        # output
        if arrays.get("FCN_OUT_W") is not None:
            out_diff = np.max(np.abs(arrays["FCN_OUT_W"] - W_out))
            bout_diff = np.max(np.abs(arrays["FCN_OUT_b"] - B_out))
            print("OUT W diff:", out_diff, "OUT b diff:", bout_diff)
            if out_diff > 1e-5:
                ok=False
        print("Header matches fused pytorch model?" , ok)
        # forward-check: orig vs fused vs c-sim on random inputs
        x = torch.randn(32, IN)
        with torch.no_grad():
            y_orig = model(x).cpu().numpy()
        fused_model = lambda x_t: c_simulate(x_t, fused_weights, fused_biases, W_out, B_out)
        y_fused = fused_model(x.cpu().numpy())
        print("orig vs fused max abs diff:", np.max(np.abs(y_orig - y_fused)))
        # simulate using header arrays directly
        hdr_fws = [arrays["FCN_L1_W"], arrays["FCN_L2_W"]]
        hdr_fbs = [arrays["FCN_L1_b"], arrays["FCN_L2_b"]]
        y_csim = c_simulate(x.cpu().numpy(), hdr_fws, hdr_fbs, arrays["FCN_OUT_W"], arrays["FCN_OUT_b"])
        print("fused pytorch vs c-sim (header) max abs diff:", np.max(np.abs(y_fused - y_csim)))
    else:
        # If no checkpoint, at least sanity-check header shapes & do c-sim on random inputs
        IN = int(defs.get("FCN_INPUT_SIZE", 5))
        W1 = arrays.get("FCN_L1_W"); b1=arrays.get("FCN_L1_b")
        W2 = arrays.get("FCN_L2_W"); b2=arrays.get("FCN_L2_b")
        Wout = arrays.get("FCN_OUT_W"); bout=arrays.get("FCN_OUT_b")
        print("\nNo PyTorch checkpoint found; doing header-only forward pass sanity check.")
        x = np.random.randn(64, IN).astype(np.float32)
        y_csim = c_simulate(x, [W1, W2], [b1, b2], Wout, bout)
        print("Header-only csim outputs mean/min/max:", np.mean(y_csim), np.min(y_csim), np.max(y_csim))
        # small shape checks
        print("W1 shape, expected out=in-hidden:", W1.shape if W1 is not None else None)
        print("W2 shape:", W2.shape if W2 is not None else None)
        print("Wout shape:", Wout.shape if Wout is not None else None)
        print("If shapes look wrong re-check input_size/hidden_size/num_layers when recreating model.")

if __name__ == "__main__":
    main()
