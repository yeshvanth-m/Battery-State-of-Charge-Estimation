# export_fcn_weights.py
import os
import torch
import torch.nn as nn
import numpy as np

MODEL_PATH  = "soc_fcn_model.pth"   # <- adjust if needed
HEADER_PATH = "soc_fcn_weights.h"   # output header

# ========= Your model definition (unchanged) =========
class SoCFCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout = 0.1):
        super(SoCFCN, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()

        # First layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_size))
        self.batch_norm_layers.append(nn.BatchNorm1d(hidden_size))

        # Dynamically add hidden layers
        for i in range(1, num_layers):
            layer_size = hidden_size // (2 ** i)
            self.hidden_layers.append(nn.Linear(hidden_size // (2 ** (i - 1)), layer_size))
            self.batch_norm_layers.append(nn.BatchNorm1d(layer_size))

        # Output layer
        self.output_layer = nn.Linear(hidden_size // (2 ** (num_layers - 1)), 1)

        # Activation and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for hidden_layer, batch_norm_layer in zip(self.hidden_layers, self.batch_norm_layers):
            x = self.relu(batch_norm_layer(hidden_layer(x)))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

# ========= Robust checkpoint loader =========
def load_checkpoint_state_dict(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)

    # Whole-model save (rare)
    if not isinstance(ckpt, dict):
        model_obj = ckpt
        return model_obj.state_dict(), {}, True  # state_dict, meta, whole_model_saved

    # Unwrap common containers
    possible_wrappers = ["state_dict", "model_state_dict", "weights", "params", "net", "model"]
    state_dict = None
    wrapper_used = None
    for k in possible_wrappers:
        if k in ckpt and isinstance(ckpt[k], dict):
            state_dict = ckpt[k]
            wrapper_used = k
            break
    if state_dict is None:
        # assume plain state_dict
        state_dict = ckpt

    # Strip 'module.' (DataParallel / DDP)
    def strip_module_prefix(sd):
        out = {}
        for k, v in sd.items():
            out[k[7:]] = v if k.startswith("module.") else v if not k.startswith("module.") else v
            if k.startswith("module."):
                out[k[len("module."):]] = v
            else:
                out[k] = v
        return out
    # The above duplicated logic could add both keys; fix to only once:
    def strip_module_prefix(sd):
        out = {}
        for k, v in sd.items():
            out[k[len("module."):] if k.startswith("module.") else k] = v
        return out

    state_dict = strip_module_prefix(state_dict)

    # Meta = all non-state dict fields (sizes, hparams, etc.)
    meta = {}
    if wrapper_used is not None:
        for k, v in ckpt.items():
            if k != wrapper_used:
                meta[k] = v

    return state_dict, meta, False

# ========= BN folding =========
@torch.no_grad()
def fuse_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d):
    """
    Fold a BatchNorm1d into the preceding Linear layer for inference:
      W_fused = (gamma / sqrt(var + eps))[:,None] * W
      b_fused = (gamma / sqrt(var + eps)) * (b - mean) + beta
    """
    W = linear.weight.clone()
    b = linear.bias.clone() if linear.bias is not None else torch.zeros(W.size(0), dtype=W.dtype)

    gamma = bn.weight
    beta  = bn.bias
    mean  = bn.running_mean
    var   = bn.running_var
    eps   = bn.eps

    inv_std = gamma / torch.sqrt(var + eps)      # [out]
    W_fused = W * inv_std.unsqueeze(1)           # scale rows
    b_fused = (b - mean) * inv_std + beta

    fused = nn.Linear(W.size(1), W.size(0), bias=True)
    fused.weight.data.copy_(W_fused)
    fused.bias.data.copy_(b_fused)
    return fused

# ========= C array formatting =========
def c_float_array(name, arr, values_per_line=8, indent="    "):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        lines = []
        flat = arr.ravel()
        for i in range(0, flat.size, values_per_line):
            chunk = ", ".join(f"{v:.9g}" for v in flat[i:i+values_per_line])
            lines.append(indent + chunk)
        return f"float {name}[{flat.size}] = {{\n" + ",\n".join(lines) + "\n};\n"
    elif arr.ndim == 2:
        rows, cols = arr.shape
        lines = [f"float {name}[{rows}][{cols}] = {{"]
        for r in range(rows):
            row_vals = ", ".join(f"{v:.9g}" for v in arr[r])
            lines.append(indent + "{" + row_vals + "}" + ("," if r != rows - 1 else ""))
        lines.append("};\n")
        return "\n".join(lines)
    else:
        raise ValueError("Only 1D or 2D arrays are supported for C export.")

def main():
    # ---- Load checkpoint robustly
    state_dict, meta, was_whole = load_checkpoint_state_dict(MODEL_PATH)

    # ---- Pull sizes from meta if present; otherwise set your known values
    INPUT_SIZE  = int(meta.get("input_size", 5))
    HIDDEN_SIZE = int(meta.get("hidden_size", 128))
    NUM_LAYERS  = int(meta.get("num_layers", 2))
    # We export params; dropout not needed for params, set to 0 for determinism
    DROPOUT     = float(meta.get("dropout", 0.0))

    # ---- Recreate model and load weights
    model = SoCFCN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, dropout=DROPOUT)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    model.eval()

    # ---- Fuse BN into the hidden linears
    fused_linears = []
    sizes = []  # (in, out) per fused hidden, then output
    for lin, bn in zip(model.hidden_layers, model.batch_norm_layers):
        fused = fuse_linear_bn(lin, bn)
        fused_linears.append(fused)
        sizes.append((fused.in_features, fused.out_features))

    out_lin = model.output_layer
    sizes.append((out_lin.in_features, out_lin.out_features))

    # ---- Gather params as float32 numpy arrays
    fused_weights = [f.weight.detach().cpu().numpy().astype(np.float32) for f in fused_linears]
    fused_biases  = [f.bias.detach().cpu().numpy().astype(np.float32) for f in fused_linears]
    W_out = out_lin.weight.detach().cpu().numpy().astype(np.float32)
    B_out = out_lin.bias.detach().cpu().numpy().astype(np.float32)

    # ---- Emit header
    guard = "SOC_FCN_WEIGHTS_H_"
    with open(HEADER_PATH, "w") as f:
        f.write("// Auto-generated by export_fcn_weights.py\n")
        f.write("// Layout: row-major; y[j] = sum_i W[j][i]*x[i] + b[j]\n\n")
        f.write("#ifndef " + guard + "\n#define " + guard + "\n\n")
        f.write("#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n")

        f.write(f"#define FCN_INPUT_SIZE   {INPUT_SIZE}\n")
        for k, (ins, outs) in enumerate(sizes[:-1]):
            f.write(f"#define FCN_L{k+1}_IN   {ins}\n")
            f.write(f"#define FCN_L{k+1}_OUT  {outs}\n")
        f.write(f"#define FCN_OUT_IN       {sizes[-1][0]}\n")
        f.write(f"#define FCN_OUT_OUT      {sizes[-1][1]}\n\n")

        for k, (W, b) in enumerate(zip(fused_weights, fused_biases), start=1):
            f.write(c_float_array(f"FCN_L{k}_W", W))
            f.write(c_float_array(f"FCN_L{k}_b", b))
        f.write(c_float_array("FCN_OUT_W", W_out))
        f.write(c_float_array("FCN_OUT_b", B_out))

        f.write("\n#ifdef __cplusplus\n}\n#endif\n")
        f.write("#endif // " + guard + "\n")

    print(f"Wrote header: {os.path.abspath(HEADER_PATH)}")
    print("Layer sizes (in -> out):", sizes)

if __name__ == "__main__":
    main()
