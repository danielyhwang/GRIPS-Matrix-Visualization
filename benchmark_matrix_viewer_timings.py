import sys, os, time, math, random, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor
from PySide6.QtCharts import QChart, QScatterSeries
from pyscipopt import Model
from scipy.sparse import csr_matrix

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

timing_logs = []

def process_file(filepath):
    record = {"file": os.path.basename(filepath)}
    start_total = time.time()

    start = time.time()
    model = Model()
    model.readProblem(filepath)
    record["upload_gui_time"] = time.time() - start

    variables = model.getVars()
    constraints = model.getConss()
    var_names = [var.name for var in variables]
    var_index = {name: idx for idx, name in enumerate(var_names)}
    row_inds, col_inds, data = [], [], []
    for i, cons in enumerate(constraints):
        terms = model.getValsLinear(cons)
        for var_name, coef in terms.items():
            j = var_index[var_name]
            row_inds.append(i)
            col_inds.append(j)
            data.append(coef)
    A_sparse = csr_matrix((data, (row_inds, col_inds)), shape=(len(constraints), len(variables)))

    # Stats
    start = time.time()
    _ = {
        "Shape": str(A_sparse.shape),
        "Total entries": A_sparse.shape[0] * A_sparse.shape[1],
        "Non-zero entries": A_sparse.nnz,
        "Avg non-zeros per row": np.mean(np.diff(A_sparse.indptr)),
        "Avg non-zeros per column": np.mean(np.diff(A_sparse.T.indptr)),
        "Sparsity (%)": 100 * (1 - A_sparse.nnz / (A_sparse.shape[0] * A_sparse.shape[1])),
        "Row NNZ Variance": np.var(np.diff(A_sparse.indptr))
    }
    record["stats_time"] = time.time() - start

    # Binary plot
    start = time.time()
    rows, cols = A_sparse.nonzero()
    chart = QChart()
    series = QScatterSeries()
    for r, c in zip(rows, cols):
        series.append(c, r)
    chart.addSeries(series)
    record["scatter_plot_time"] = time.time() - start

    # Magnitude plot
    start = time.time()
    vals = A_sparse.data
    entries = list(zip(rows, cols, vals))
    if len(entries) > 50000:
        entries = random.sample(entries, 50000)
    abs_vals = [abs(v) for _, _, v in entries if v != 0]
    logs = [math.log10(abs(v)) for v in abs_vals]
    min_log, max_log = min(logs), max(logs)
    log_range = max_log - min_log if max_log > min_log else 1.0
    for r, c, v in entries:
        if v == 0: continue
        log_mag = math.log10(abs(v))
        norm = (log_mag - min_log) / log_range
        red = int(255 * norm)
        blue = 255 - red
        color = QColor(red, 0, blue)
        dummy = QScatterSeries()
        dummy.setColor(color)
        dummy.append(c, r)
    record["magnitude_plot_time"] = time.time() - start

    # Row-scaled heatmap
    start = time.time()
    A = A_sparse.toarray()
    row_scaled = np.zeros_like(A)
    for i, row in enumerate(A):
        nonzero = row[np.nonzero(row)]
        if len(nonzero) == 0: continue
        min_val, max_val = np.min(abs(nonzero)), np.max(abs(nonzero))
        if max_val > min_val:
            row_scaled[i] = (abs(row) - min_val) / (max_val - min_val)
    fig = Figure()
    ax = fig.subplots()
    ax.imshow(row_scaled, cmap='bwr', aspect='auto', origin='lower')
    record["row_scaled_heatmap_time"] = time.time() - start

    record["total_time"] = time.time() - start_total
    timing_logs.append(record)
    print(f"✅ {record['file']} done.")

def main():
    input_dir = "mps"  # Folder with .mps files
    files = [f for f in os.listdir(input_dir) if f.endswith(".mps")]
    if not files:
        print("❌ No MPS files found in benchmark/mps")
        return

    app = QApplication([])  # Required for QtCharts
    for fname in files:
        fpath = os.path.join(input_dir, fname)
        process_file(fpath)

    outpath = "timing_statistics.csv"
    pd.DataFrame(timing_logs).to_csv(outpath, index=False)
    print(f"📊 All timings saved to {outpath}")

if __name__ == "__main__":
    main()