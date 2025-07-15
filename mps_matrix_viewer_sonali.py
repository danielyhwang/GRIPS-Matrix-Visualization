import sys
import numpy as np
import random
import math
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox
)
from PySide6.QtCharts import QChart, QChartView, QScatterSeries
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainter, QColor
from pyscipopt import Model
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


class MatrixViewer(QWidget):
    def __init__(self, filename):
        super().__init__()
        self.A_sparse = None
        self.last_plot_data = []
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.stats_table)

        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.layout.addWidget(self.chart_view)

        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()

        self.export_image_button = QPushButton("Export Plot to JPEG")
        self.export_image_button.clicked.connect(self.export_chart_as_image)
        bottom_layout.addWidget(self.export_image_button)

        self.binary_test_button = QPushButton("Binary Scatterplot")
        self.magnitude_test_button = QPushButton("Magnitude Scatterplot")
        self.binary_test_button.clicked.connect(lambda: self.update_plot("Binary Scatterplot"))
        self.magnitude_test_button.clicked.connect(lambda: self.update_plot("Magnitude Scatterplot"))
        bottom_layout.addWidget(self.binary_test_button)
        bottom_layout.addWidget(self.magnitude_test_button)
        self.layout.addLayout(bottom_layout)

        model = Model()
        model.readProblem(filename)
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
                row_inds.append(i)
                col_inds.append(j)
                data.append(coef)

        self.A_sparse = csr_matrix((data, (row_inds, col_inds)), shape=(n_cons, n_vars))

        total_entries = n_cons * n_vars
        non_zero = len(data)
        sparsity = 100 * (1 - non_zero / total_entries)
        row_nnz = np.diff(self.A_sparse.indptr)
        col_nnz = np.diff(csr_matrix(self.A_sparse.T).indptr)
        row_indices, col_indices = self.A_sparse.nonzero()
        bandwidth = np.max(np.abs(row_indices - col_indices)) if self.A_sparse.nnz > 0 else 0

        try:
            k = min(self.A_sparse.shape) - 1
            _, s, _ = svds(self.A_sparse, k=k)
            tol = 1e-10
            rank_val = int(np.sum(s > tol))
        except Exception:
            rank_val = "N/A"

        l2_norms = np.linalg.norm(self.A_sparse.toarray(), axis=1)
        props = [
            ("📁 File:", filename),
            ("Shape", str(self.A_sparse.shape)),
            ("Total entries", total_entries),
            ("Non-zero entries", non_zero),
            ("Avg non-zeros per row", np.mean(row_nnz)),
            ("Avg non-zeros per column", np.mean(col_nnz)),
            ("Sparsity (%)", round(sparsity, 3)),
            ("Row NNZ Variance", np.var(row_nnz)),
            ("Column NNZ Variance", np.var(col_nnz)),
            ("Min coefficient", np.min(data)),
            ("Max coefficient", np.max(data)),
            ("Mean coefficient", np.mean(data)),
            ("Std coefficient", np.std(data)),
            ("Integer-like (%)", round(100 * np.mean(np.mod(data, 1) == 0), 3)),
            ("Matrix rank", rank_val),
            ("Matrix bandwidth", bandwidth),
            ("Diag Dominant Rows (%)", "N/A" if self.A_sparse.shape[0] != self.A_sparse.shape[1] else
             round(100 * np.mean([
                 abs(self.A_sparse[i, i]) >= np.sum(np.abs(self.A_sparse[i, :])) - abs(self.A_sparse[i, i])
                 for i in range(self.A_sparse.shape[0])
             ]), 3)),
            ("Avg row L2 norm", float(np.mean(l2_norms))),
            ("Max row L2 norm", float(np.max(l2_norms))),
            ("Zero rows", int(np.sum(row_nnz == 0))),
            ("Zero columns", int(np.sum(col_nnz == 0)))
        ]

        self.stats_table.setRowCount(len(props))
        for i, (k, v) in enumerate(props):
            self.stats_table.setItem(i, 0, QTableWidgetItem(str(k)))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(v)))

        self.update_plot("Binary Scatterplot")

    def update_plot(self, type_of_plot):
        if self.A_sparse is None:
            return
        if type_of_plot == "Binary Scatterplot":
            self.plot_binary_scatterplot()
        elif type_of_plot == "Magnitude Scatterplot":
            self.plot_magnitude_scatterplot()

    def plot_binary_scatterplot(self):
        rows, cols = self.A_sparse.nonzero()
        indices = list(zip(rows, cols))
        if len(indices) > 100_000:
            indices = random.sample(indices, 100_000)
        self.last_plot_data = [(r, c, self.A_sparse[r, c]) for r, c in indices]

        chart = QChart()
        chart.setTitle("Binary Scatterplot of Constraint Matrix A (black = non-zero)")
        chart.legend().hide()

        series = QScatterSeries()
        series.setMarkerSize(6)
        series.setColor(QColor("black"))
        for r, c in indices:
            series.append(QPointF(c, r))
        series.clicked.connect(self.on_point_clicked)
        chart.addSeries(series)

        chart.createDefaultAxes()
        chart.axisX().setTitleText("Variables (Columns)")
        chart.axisY().setTitleText("Constraints (Rows)")
        chart.axisY().setReverse(True)
        self.chart_view.setChart(chart)

    def plot_magnitude_scatterplot(self):
        rows, cols = self.A_sparse.nonzero()
        vals = self.A_sparse.data
        entries = list(zip(rows, cols, vals))
        if len(entries) > 50_000:
            entries = random.sample(entries, 50_000)

        self.last_plot_data = [(r, c, v) for r, c, v in entries]

        chart = QChart()
        chart.setTitle("Signed Heatmap (Blue = −, Red = +, Circle = +, Square = −, Log Magnitude)")
        chart.legend().hide()

        abs_vals = [abs(v) for _, _, v in entries if v != 0]
        log_vals = [math.log10(x) if x > 0 else 0 for x in abs_vals]
        min_log = min(log_vals)
        max_log = max(log_vals)
        log_range = max_log - min_log if max_log > min_log else 1

        for r, c, v in entries:
            if v == 0:
                continue
            log_mag = math.log10(abs(v)) if abs(v) > 0 else min_log
            norm = (log_mag - min_log) / log_range
            red = int(255 * norm)
            blue = 255 - red
            color = QColor(red, 0, blue)

            series = QScatterSeries()
            series.setMarkerSize(7)
            series.setColor(color)
            if v > 0:
                series.setMarkerShape(QScatterSeries.MarkerShapeCircle)
            else:
                series.setMarkerShape(QScatterSeries.MarkerShapeRectangle)
            series.append(QPointF(c, r))
            series.clicked.connect(self.on_point_clicked)
            chart.addSeries(series)

        chart.createDefaultAxes()
        chart.axisX().setTitleText("Variables (Columns)")
        chart.axisY().setTitleText("Constraints (Rows)")
        chart.axisY().setReverse(True)
        self.chart_view.setChart(chart)

    def on_point_clicked(self, point):
        row = int(point.y())
        col = int(point.x())
        A = self.A_sparse.toarray()
        r_vals = A[row]
        c_vals = A[:, col]

        def stats(arr):
            return {
                "Min": np.min(arr),
                "Max": np.max(arr),
                "Mean": np.mean(arr),
                "Std": np.std(arr),
                "L2 norm": np.linalg.norm(arr),
                "Non-zeros": int(np.count_nonzero(arr)),
                "Zeros": int(len(arr) - np.count_nonzero(arr))
            }

        row_stats = stats(r_vals)
        col_stats = stats(c_vals)

        message = (
            f"<b>Clicked Entry</b>: Row {row}, Column {col}<br><br>"
            f"<b>Row {row} Stats:</b><br>" +
            "".join(f"{k}: {v:.4g}<br>" for k, v in row_stats.items()) +
            f"<br><b>Column {col} Stats:</b><br>" +
            "".join(f"{k}: {v:.4g}<br>" for k, v in col_stats.items())
        )
        QMessageBox.information(self, "Matrix Entry Statistics", message)

    def export_chart_as_image(self):
        if not self.chart_view.chart():
            QMessageBox.information(self, "❌", "No chart to export.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Chart as JPEG", "plot.jpeg", "JPEG Image (*.jpeg *.jpg)")
        if filename:
            pixmap = self.chart_view.grab()
            if not pixmap.save(filename, "JPEG"):
                QMessageBox.information(self, "", "❌ Failed to save image.")
            else:
                QMessageBox.information(self, "", f"✅ Saved: {filename}")


class FileLoader(QWidget):
    def __init__(self):
        super().__init__()
        self.filename, _ = QFileDialog.getOpenFileName(self, "Open MPS File", "", "MPS Files (*.mps *.MPS);;All Files (*)")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    file_window = FileLoader()
    if file_window.filename:
        window = MatrixViewer(file_window.filename)
        window.show()
    sys.exit(app.exec())
