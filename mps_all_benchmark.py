import os
import zipfile
import gzip
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pyscipopt import Model

# Define directories
BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_ZIP = BASE_DIR / "benchmark.zip"
BENCHMARK_DIR = BASE_DIR / "benchmark"
MPS_DIR = BASE_DIR / "mps"
OUTPUT_DIR = BASE_DIR / "output"
SUMMARY_FILE = OUTPUT_DIR / "summary_statistics.csv"
summary_records = []

# Create necessary directories
BENCHMARK_DIR.mkdir(exist_ok=True)
MPS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Step 1: Unzip benchmark.zip
print("🔍 Step 1: Unzipping benchmark.zip...")
with zipfile.ZipFile(BENCHMARK_ZIP, 'r') as zip_ref:
    zip_ref.extractall(BENCHMARK_DIR)
print("✅ Unzipped benchmark.zip to ./benchmark")

# Step 2: Unpack all .gz files into mps directory
print("🔍 Step 2: Decompressing .gz files into ./mps folder...")
for gz_file in BENCHMARK_DIR.rglob("*.gz"):
    mps_target = MPS_DIR / gz_file.stem
    print(f"📂 Decompressing: {gz_file.name}")
    if not mps_target.exists():
        with gzip.open(gz_file, 'rb') as f_in:
            with open(mps_target, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
print("✅ Decompressed all .gz files")

# Step 3: Process MPS files
def process_mps(file_path, stem):
    print(f"🔄 Processing MPS file: {stem}")
    output_path = OUTPUT_DIR / stem
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the problem file using PySCIPOpt
    model = Model()
    model.readProblem(str(file_path))

    vars_ = model.getVars()
    conss = model.getConss()

    # Skip very large files
    if len(vars_) > 10000 or len(conss) > 10000:
        print(f"⚠️ Skipping {stem} due to size > 10,000")
        return

    var_names = [var.name for var in vars_]
    var_index = {name: i for i, name in enumerate(var_names)}
    A = np.zeros((len(conss), len(vars_)))
    b = []
    senses = []
    c = np.array([var.getObj() for var in vars_])

    # Fill matrix A and vector b
    for i, cons in enumerate(conss):
        terms = model.getValsLinear(cons)
        for name, val in terms.items():
            j = var_index[name]
            A[i, j] = val
        lhs = model.getLhs(cons)
        rhs = model.getRhs(cons)
        if lhs == rhs:
            senses.append("=")
            b.append(rhs)
        elif np.isfinite(rhs):
            senses.append("<=")
            b.append(rhs)
        else:
            senses.append(">=")
            b.append(lhs)

    # Save matrices and vectors
    pd.DataFrame(A, columns=var_names).to_pickle(output_path / f"{stem}_A.pkl")
    pd.DataFrame({'RHS': b, 'Sense': senses}).to_pickle(output_path / f"{stem}_b.pkl")
    pd.DataFrame({'Variable': var_names, 'Objective Coef': c}).to_pickle(output_path / f"{stem}_c.pkl")

    # Save basic statistics
    stats = {
        "file": stem,
        "variables": len(vars_),
        "constraints": len(conss),
        "nonzeros": np.count_nonzero(A),
        "density": np.count_nonzero(A) / A.size
    }
    pd.Series(stats).to_csv(output_path / f"{stem}_stats.csv")
    summary_records.append(stats)

    # Plot nonzero scatterplot
    nonzero_coords = np.argwhere(A != 0)
    plt.figure(figsize=(10, 6))
    plt.scatter(nonzero_coords[:, 1], nonzero_coords[:, 0], s=1, color='black')
    plt.gca().invert_yaxis()
    plt.title("Constraint Matrix A - Nonzero Pattern")
    plt.xlabel("Variable Index")
    plt.ylabel("Constraint Index")
    plt.tight_layout()
    plt.savefig(output_path / f"{stem}_scatter.png")
    plt.clf()

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(A, cmap='viridis', cbar=False)
    plt.title("Constraint Matrix A - Heatmap")
    plt.tight_layout()
    plt.savefig(output_path / f"{stem}_heatmap.png")
    plt.clf()

    print(f"✅ Finished processing {stem}\n")

# Step 4: Loop over all .mps files
print("🔍 Step 3: Extracting constraint matrices and statistics...")
for mps_file in MPS_DIR.glob("*.mps"):
    try:
        process_mps(mps_file, mps_file.stem)
    except Exception as e:
        print(f"❌ Error processing {mps_file.name}: {e}")

# Step 5: Save overall summary
if summary_records:
    pd.DataFrame(summary_records).to_csv(SUMMARY_FILE, index=False)
    print(f"📊 Summary written to {SUMMARY_FILE}")

print("🎉 All benchmark files processed successfully!")
