import time
import numpy as np
import pandas as pd
from pathlib import Path
from pyscipopt import Model
from scipy.sparse import csr_matrix

# === Setup directories ===
OUTPUT_DIR = Path("output_gui_benchmark")
OUTPUT_DIR.mkdir(exist_ok=True)
RESULT_CSV = OUTPUT_DIR / "gui_component_timings.csv"
MPS_DIR = Path("mps")

# === Data collection ===
time_records = []

def process_gui_timings(file_path):
    stem = file_path.stem
    print(f"🔍 Benchmarking GUI simulation for: {stem}")
    stats = {"file": stem}

    try:
        # Step 1: Upload/load GUI matrix
        t0 = time.time()
        model = Model()
        model.readProblem(str(file_path))
        variables = model.getVars()
        constraints = model.getConss()
        var_names = [var.name for var in variables]
        var_index = {name: idx for idx, name in enumerate(var_names)}
        n_vars, n_cons = len(variables), len(constraints)

        row_inds, col_inds, data = [], [], []
        for i, cons in enumerate(constraints):
            terms = model.getValsLinear(cons)
            for var_name, coef in terms.items():
                j = var_index[var_name]
                row_inds.append(i)
                col_inds.append(j)
                data.append(coef)

        A_sparse = csr_matrix((data, (row_inds, col_inds)), shape=(n_cons, n_vars))
        stats["upload_gui_time"] = time.time() - t0

        # Step 2: GUI statistics rendering
        t1 = time.time()
        row_nnz = np.diff(A_sparse.indptr)
        col_nnz = np.diff(A_sparse.T.indptr)
        _ = {
            "shape": A_sparse.shape,
            "nnz": A_sparse.nnz,
            "avg_row_nnz": np.mean(row_nnz),
            "avg_col_nnz": np.mean(col_nnz),
            "sparsity_percent": 100 * (1 - A_sparse.nnz / (A_sparse.shape[0] * A_sparse.shape[1])),
            "row_nnz_var": np.var(row_nnz)
        }
        stats["stats_time"] = time.time() - t1

        # Step 3: Binary scatter plot
        t2 = time.time()
        _ = list(zip(*A_sparse.nonzero()))
        stats["scatter_plot_time"] = time.time() - t2

        # Step 4: Magnitude scatter plot
        t3 = time.time()
        _ = [np.log10(abs(v)) for v in A_sparse.data if v != 0]
        stats["magnitude_plot_time"] = time.time() - t3

        # Step 5: Row-scaled heatmap
        t4 = time.time()
        A = A_sparse.toarray()
        row_scaled = np.zeros_like(A)
        for i, row in enumerate(A):
            nonzero = row[np.nonzero(row)]
            if len(nonzero) == 0:
                continue
            min_val, max_val = np.min(abs(nonzero)), np.max(abs(nonzero))
            if max_val > min_val:
                row_scaled[i] = (abs(row) - min_val) / (max_val - min_val)
        stats["row_scaled_heatmap_time"] = time.time() - t4

        time_records.append(stats)
        print(f"✅ Finished {stem}")
    except Exception as e:
        print(f"❌ Error in {stem}: {e}")

# === Run benchmark ===
for mps_file in MPS_DIR.glob("*.mps"):
    process_gui_timings(mps_file)

# === Save to CSV ===
df = pd.DataFrame(time_records)
df.to_csv(RESULT_CSV, index=False)
print(f"📊 Saved timing data to {RESULT_CSV}")
