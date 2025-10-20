import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("results.csv")
plt.figure(figsize=(10,6))
sns.lineplot(df, x="Size", y="Single(ms)", label="Single-threaded", marker="o")
sns.lineplot(df, x="Size", y="PerRow(ms)", label="Per-row Threads", marker="o")
sns.lineplot(df, x="Size", y="CUDA(ms)", label="CUDA GPU", marker="o")
plt.title("Matrix Multiplication Performance Comparison")
plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Time (ms)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results_comparison.png", dpi=300)
plt.show()

plt.figure(figsize=(10,6))
for col in ["PerRow(ms)", "CUDA(ms)"]:
    sns.lineplot(df, x="Size", y=df["Single(ms)"]/df[col], label=f"Speedup ({col})", marker="o")
plt.title("Speedup over Single-threaded")
plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Speedup (x faster)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("speedup.png", dpi=300)
plt.show()
