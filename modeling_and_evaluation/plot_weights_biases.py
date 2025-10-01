#!/usr/bin/env python3
"""
plot_weights_biases.py

Reads /mnt/data/soc_fcn_weights.h, extracts numeric constants (weights & biases)
and displays two plots (interactive window):
  1) index vs value (line + scatter)
  2) histogram of values

No file is saved.
"""
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

INFILE = Path("soc_fcn_weights.h")

if not INFILE.exists():
    raise FileNotFoundError(f"{INFILE} not found. Place your header at: {INFILE}")

text = INFILE.read_text()

# Regex to capture floats/ints (including scientific notation), allow trailing f/F
num_re = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?f?', re.IGNORECASE)
matches = num_re.findall(text)

# Convert to floats (strip trailing 'f' if present)
values = []
for m in matches:
    s = m.rstrip('f').rstrip('F')
    try:
        values.append(float(s))
    except ValueError:
        # ignore anything unparsable
        pass

values = np.array(values, dtype=np.float64)
if values.size == 0:
    raise ValueError("No numeric values found in the input file.")

indices = np.arange(values.size)

# Create figure and axes
fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [2, 1]})

# Top: index vs value (line + scatter)
ax = axes[0]
ax.plot(indices, values, linestyle='-', linewidth=0.6, alpha=0.7, label='value (line)')
# scatter by sign
pos_mask = values >= 0
neg_mask = values < 0
ax.scatter(indices[pos_mask], values[pos_mask], s=8, marker='o', label='>= 0', alpha=0.8)
ax.scatter(indices[neg_mask], values[neg_mask], s=8, marker='x', label='< 0', alpha=0.8)
ax.set_ylabel('value')
ax.set_xlabel('index (order in file)')
ax.set_title('Weights & Biases — index vs value')
ax.legend(loc='upper right')
ax.grid(True, linestyle=':', alpha=0.6)

# Optional inset if outliers exist
abs_vals = np.abs(values)
median_abs = np.median(abs_vals)
max_abs = np.max(abs_vals)
if median_abs > 0 and (max_abs / (median_abs + 1e-30)) > 50:
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, width="30%", height="30%", loc='lower left', borderpad=2)
        lowp = np.percentile(values, 1)
        hip = np.percentile(values, 99)
        mask = (values >= lowp) & (values <= hip)
        axins.plot(indices[mask], values[mask], linestyle='-', linewidth=0.6)
        axins.scatter(indices[mask], values[mask], s=6)
        axins.set_title('Zoom (1..99 percentile)', fontsize=8)
        axins.grid(True, linestyle=':', alpha=0.4)
    except Exception:
        # inset failed (matplotlib older version), continue without inset
        pass

# Bottom: histogram
ax2 = axes[1]
n_bins = 120
ax2.hist(values, bins=n_bins)
ax2.set_xlabel('value')
ax2.set_ylabel('count')
ax2.set_title('Histogram of values (weights & biases)')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()
