#!/usr/bin/env python3
"""
simple_mem_merger.py

Reads multiple .mem files (each line = one element)
and writes a single merged file, in the order specified.

No reshaping, grouping, padding, or formatting — it just
concatenates lines from the input files exactly in order.

Usage:
    python3 simple_mem_merger.py
"""

from pathlib import Path

# ======== EDIT THIS SECTION =========
# List your files here in the *exact* order they should appear in output
input_files = [
    "fcn_l1_w.mem",
    "fcn_l1_b.mem",
    "fcn_l2_w.mem",
    "fcn_l2_b.mem",
    "fcn_out_w.mem",
    "fcn_out_b.mem"
]

output_file_w = "weights.mem"
output_file_b = "biases.mem"
# ====================================

weights_transformed = "weights_transformed.mem"
bias_transformed = "biases_transformed.mem"

def read_lines_strip_comments(path: Path):
    """Read lines, ignore empty or comment-only lines."""
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(";") or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def main():
    out_lines_w = []
    out_lines_b = []
    for fname in input_files:
        path = Path(fname)
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {fname}")
        lines = read_lines_strip_comments(path)
        if fname.endswith("_w.mem"):
            print(f"Processing weights file: {fname}")
            out_lines_w.extend(lines)
        elif fname.endswith("_b.mem"):
            out_lines_b.extend(lines)
        print(f"Read {len(lines)} lines from {fname}")

    # Write output
    Path(output_file_w).write_text("\n".join(out_lines_w) + "\n")
    Path(output_file_b).write_text("\n".join(out_lines_b) + "\n")

    print(f"Wrote {len(out_lines_w)} total lines to {output_file_w}")
    print(f"Wrote {len(out_lines_b)} total lines to {output_file_b}")



if __name__ == "__main__":
    main()
