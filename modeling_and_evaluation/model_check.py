# model_check.py
# Place this next to model.py and run: python model_check.py
import torch
import torch.nn as nn
import numpy as np
from model import load_checkpoint_state_dict, SoCFCN, fuse_linear_bn  # <- import from your file

MODEL_PATH = "soc_fcn_model.pth"  # change if checkpoint is elsewhere
MAP_LOCATION = "cpu"

def build_original_model(state_dict, meta):
    INPUT_SIZE  = int(meta.get("input_size", 5))
    HIDDEN_SIZE = int(meta.get("hidden_size", 128))
    NUM_LAYERS  = int(meta.get("num_layers", 2))
    DROPOUT     = float(meta.get("dropout", 0.0))
    model = SoCFCN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, dropout=DROPOUT)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    model.eval()
    return model, (INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)

def build_folded_model(model):
    # fuse each hidden linear + bn into a single Linear
    fused_linears = []
    for lin, bn in zip(model.hidden_layers, model.batch_norm_layers):
        fused = fuse_linear_bn(lin, bn)
        fused_linears.append(fused)
    # define a small folded model using fused_linears and the original output layer
    class FoldedModel(nn.Module):
        def __init__(self, fused_linears, out_layer):
            super().__init__()
            self.hidden = nn.ModuleList(fused_linears)
            self.out = out_layer
            self.relu = nn.ReLU()
        def forward(self, x):
            for lin in self.hidden:
                x = self.relu(lin(x))
            return self.out(x)
    folded = FoldedModel(fused_linears, model.output_layer)
    folded.eval()
    return folded

def compare_models(original, folded, in_size, n_tests=8, batch_size=16, tol=1e-5):
    for t in range(n_tests):
        x = torch.randn(batch_size, in_size, dtype=torch.float32)
        with torch.no_grad():
            out_orig = original(x)
            out_fold = folded(x)
        diff = (out_orig - out_fold).abs()
        maxdiff = float(diff.max().item())
        mean = float(diff.mean().item())
        print(f"Test {t+1}: max abs diff = {maxdiff:.6e}, mean diff = {mean:.6e}")
        if maxdiff > tol:
            print("WARNING: difference is larger than tolerance (possible problem).")
    print("Comparison complete.")

def main():
    state_dict, meta, was_whole = load_checkpoint_state_dict(MODEL_PATH, map_location=MAP_LOCATION)
    print("Loaded checkpoint. meta keys:", list(meta.keys()))
    original_model, sizes = build_original_model(state_dict, meta)
    folded_model = build_folded_model(original_model)
    in_size = sizes[0]
    compare_models(original_model, folded_model, in_size)

if __name__ == "__main__":
    main()
