import os
import zipfile
import gzip
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pyscipopt import Model
import time
from scipy.sparse import csr_matrix

# Define directories
BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_ZIP = BASE_DIR / "benchmark.zip"
BENCHMARK_DIR = BASE_DIR / "benchmark"
MPS_DIR = BASE_DIR / "mps"
OUTPUT_DIR = BASE_DIR / "output"
TIME_FILE = OUTPUT_DIR / "time_statistics.csv"
time_records = []

# Create necessary directories
MPS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Step 1: Unzip benchmark.zip
print("🔍 Step 1: Unzipping benchmark.zip...")
try:
    BENCHMARK_DIR.mkdir(exist_ok=False)
    with zipfile.ZipFile(BENCHMARK_ZIP, 'r') as zip_ref:
        zip_ref.extractall(BENCHMARK_DIR)
    print("✅ Unzipped benchmark.zip to ./benchmark")
except FileExistsError:
    print("📁 ./benchmark already exists!")

# Step 2: Decompress all .gz files into ./mps folder
print("🔍 Step 2: Decompressing .gz files in ./benchmark into ./mps...")
for gz_file in BENCHMARK_DIR.rglob("*.gz"):
    mps_target = MPS_DIR / gz_file.stem
    print(f"📂 Decompressing: {gz_file.name}")
    if not mps_target.exists():
        with gzip.open(gz_file, 'rb') as f_in:
            with open(mps_target, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print(f"✅ {gz_file.name} already decompressed!")

print("✅ All .gz files decompressed.\n")

# Step 3: Process each MPS file
def process_mps(file_path, stem):
    print(f"🔄 Processing MPS file: {stem}")
    output_path = OUTPUT_DIR / stem
    output_path.mkdir(parents=True, exist_ok=True)

    # Time the MPS loading and matrix extraction
    start_time = time.time()
    model = Model()
    model.readProblem(str(file_path))

    variables = model.getVars()
    constraints = model.getConss()

    var_names = [var.name for var in variables]
    var_index = {name: idx for idx, name in enumerate(var_names)}
    n_vars = len(variables)
    n_cons = len(constraints)

    row_inds, col_inds, data = [], [], []
    for i, cons in enumerate(constraints):
        terms = model.getValsLinear(cons)
        for var_name, coef in terms.items():
            j = var_index[var_name]
            if coef != 0:
                row_inds.append(i)
                col_inds.append(j)
                data.append(coef)

    A_sparse = csr_matrix((data, (row_inds, col_inds)), shape=(n_cons, n_vars))
    sparsity = A_sparse.nnz / (n_vars * n_cons)
    load_time = time.time() - start_time

    # Time the scatter plot
    scatter_start = time.time()
    try:
        coo = A_sparse.tocoo()
        plt.figure(figsize=(10, 6))
        plt.scatter(coo.col, coo.row, s=0.1)
        plt.title(f"Scatter Plot of {stem}")
        plt.xlabel("Variables")
        plt.ylabel("Constraints")
        plt.savefig(output_path / f"{stem}_scatter_plot.png", bbox_inches='tight')
        plt.close()
        scatter_time = time.time() - scatter_start
    except Exception as e:
        print(f"⚠️ Failed to plot scatter for {stem}: {e}")
        scatter_time = None

    # Record stats
    stats = {
        "file": stem,
        "variables": n_vars,
        "constraints": n_cons,
        "sparsity": sparsity,
        "time": load_time,
        "scatter_plot_time": scatter_time
    }
    time_records.append(stats)

    print(f"✅ Finished {stem} | Load: {load_time:.3f}s | Plot: {scatter_time:.3f}s\n")

# Step 4: Apply to all .mps files
print("🔍 Step 3: Extracting constraint matrices and scatter plots...")
for mps_file in MPS_DIR.glob("*.mps"):
    try:
        process_mps(mps_file, mps_file.stem)
    except Exception as e:
        print(f"❌ Error processing {mps_file.name}: {e}")

# Step 5: Save summary
if time_records:
    df = pd.DataFrame(time_records)
    df.to_csv(TIME_FILE, index=False)
    print(f"📊 Time summary saved to {TIME_FILE}")

print("🎉 All benchmark files processed with scatter plots!")

