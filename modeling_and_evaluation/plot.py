import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
csv_path = "../dataset/LG_HG2_processed/25degC/549_HPPC_processed.csv"

# Read the CSV file
df = pd.read_csv(csv_path)

# Plot each column
for col in df.columns:
    plt.figure()
    plt.plot(df[col])
    plt.title(col)
    plt.xlabel("Row Index")
    plt.ylabel(col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()