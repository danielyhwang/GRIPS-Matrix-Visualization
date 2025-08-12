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
from numpy.linalg import matrix_rank

# Define directories
BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_ZIP = BASE_DIR / "benchmark.zip"
BENCHMARK_DIR = BASE_DIR / "benchmark"
MPS_DIR = BASE_DIR / "mps"
OUTPUT_DIR = BASE_DIR / "output"
SUMMARY_FILE = OUTPUT_DIR / "matrix_statistics.csv"
summary_records = []

# Create necessary directories
BENCHMARK_DIR.mkdir(exist_ok=True)
MPS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Step 1: Unzip benchmark.zip
print("🔍 Step 1: Unzipping benchmark.zip...")
# Make benchmark directory
try:
    BENCHMARK_DIR.mkdir(exist_ok = False) #If exist_ok is False, creating an existing directory in benchmark
    # will raise a FileExistsError, to which we simply say that the benchmark directory exists. 
    with zipfile.ZipFile(BENCHMARK_ZIP, 'r') as zip_ref:
        zip_ref.extractall(BENCHMARK_DIR)
    print("✅ Unzipped benchmark.zip to ./benchmark")
except FileExistsError:
    print("./benchmark already exists!")

# Step 2: Unpack all .gz files into mps directory
print("🔍 Step 2: Decompressing .gz files in ./benchmark into ./mps folder...")
for gz_file in BENCHMARK_DIR.rglob("*.gz"):
    mps_target = MPS_DIR / gz_file.stem
    print(f"📂 Decompressing: {gz_file.name}")
    if not mps_target.exists():
        with gzip.open(gz_file, 'rb') as f_in:
            with open(mps_target, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print(f"{gz_file.name} already exists!")
print("✅ Decompressed all .gz files")

# Step 3: Process MPS files
def process_mps(file_path, stem):
    print(f"🔄 Processing MPS file: {stem}")
    output_path = OUTPUT_DIR / stem
    output_path.mkdir(parents=True, exist_ok=True)

    model = Model()
    model.readProblem(str(file_path))

    vars_ = model.getVars()
    conss = model.getConss()

    # Tried to test this for files greater than 15000 variables or 15000 constraints, ran into issues.
    if len(vars_) > 15000 or len(conss) > 15000:
        print(f"⚠️ Skipping {stem} due to size > 15,000")
        print()
        return

    # Commented out b, senses, c, these were taking too much space. Feel free to uncomment if you want 
    # local matrix data.
    var_names = [var.name for var in vars_]
    var_index = {name: i for i, name in enumerate(var_names)}
    A = np.zeros((len(conss), len(vars_)))
    #b = []
    #senses = []
    #c = np.array([var.getObj() for var in vars_])

    for i, cons in enumerate(conss):
        terms = model.getValsLinear(cons)
        for name, val in terms.items():
            j = var_index[name]
            A[i, j] = val
        #lhs = model.getLhs(cons)
        #rhs = model.getRhs(cons)
        #if lhs == rhs:
        #    senses.append("=")
        #    b.append(rhs)
        #elif np.isfinite(rhs):
        #    senses.append("<=")
        #    b.append(rhs)
        #else:
        #    senses.append(">=")
        #    b.append(lhs)

    # Commented out, as these were taking too much space. Feel free to uncomment if you want 
    # local matrix data.
    #df_A = pd.DataFrame(A, columns=var_names)
    #df_A.to_pickle(output_path / f"{stem}_A.pkl")
    #pd.DataFrame({'RHS': b, 'Sense': senses}).to_pickle(output_path / f"{stem}_b.pkl")
    #pd.DataFrame({'Variable': var_names, 'Objective Coef': c}).to_pickle(output_path / f"{stem}_c.pkl")

    # Advanced matrix statistics
    nonzeros = np.count_nonzero(A)
    shape = A.shape
    sparsity = 100 * (1 - nonzeros / A.size)
    row_variance = np.var(np.count_nonzero(A, axis=1))
    col_variance = np.var(np.count_nonzero(A, axis=0))
    A_nonzero = A[np.nonzero(A)]
    min_coef = A_nonzero.min() if A_nonzero.size else 0
    max_coef = A_nonzero.max() if A_nonzero.size else 0
    mean_coef = A_nonzero.mean() if A_nonzero.size else 0
    std_coef = A_nonzero.std() if A_nonzero.size else 0
    integer_like = 100 * np.mean(np.isclose(A_nonzero, np.round(A_nonzero))) if A_nonzero.size else 0
    rank = matrix_rank(A)
    row_norms = np.linalg.norm(A, axis=1)
    avg_row_norm = row_norms.mean() if row_norms.size else 0
    max_row_norm = row_norms.max() if row_norms.size else 0
    zero_rows = np.sum(np.count_nonzero(A, axis=1) == 0)
    zero_cols = np.sum(np.count_nonzero(A, axis=0) == 0)

    try:
        from networkx.algorithms.approximation.treewidth import treewidth_min_fill_in
        treewidth_calculated, _ = treewidth_min_fill_in(primal_graph)
    except:
        treewidth_calculated = "N/A"

    stats = {
        "file": stem,
        "variables": shape[1],
        "constraints": shape[0],
        "nonzeros": nonzeros,
        "density": nonzeros / A.size,
        "sparsity_%": sparsity,
        "row_nnz_variance": row_variance,
        "col_nnz_variance": col_variance,
        "min_coefficient": min_coef,
        "max_coefficient": max_coef,
        "mean_coefficient": mean_coef,
        "std_coefficient": std_coef,
        "integer_like_%": integer_like,
        "matrix_rank": rank,
        "avg_row_L2_norm": avg_row_norm,
        "max_row_L2_norm": max_row_norm,
        "zero_rows": zero_rows,
        "zero_columns": zero_cols,
        "treewidth" : treewidth_calculated
    }

    pd.Series(stats).to_csv(output_path / f"{stem}_stats.csv")
    summary_records.append(stats)

    # This code computes the binary scatterplot and a heatmap version of the scatterplot, but
    # we commented this out due to taking too much time. Feel free to uncomment if you want
    # these figures stored locally on your computer.

    #plt.figure(figsize=(10, 6))
    #nonzero_coords = np.argwhere(A != 0)
    #plt.scatter(nonzero_coords[:, 1], nonzero_coords[:, 0], s=1, color='black')
    #plt.gca().invert_yaxis()
    #plt.title("Constraint Matrix A - Nonzero Pattern")
    #plt.xlabel("Variable Index")
    #plt.ylabel("Constraint Index")
    #plt.tight_layout()
    #plt.savefig(output_path / f"{stem}_scatter.png")
    #plt.clf()
    #plt.close()

    #plt.figure(figsize=(10, 6))
    #sns.heatmap(A, cmap='viridis', cbar=False)
    #plt.title("Constraint Matrix A - Heatmap")
    #plt.tight_layout()
    #plt.savefig(output_path / f"{stem}_heatmap.png")
    #plt.clf()
    #plt.close()

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
