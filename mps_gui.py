
## NEXT STEPS:
# Make it so we can take in larger MPS files, at leat 10k variables
# Add comments to code and functions to explain things
# Make rows and columns clickable so user can get row/column numerical properties
# Make both scatterplots export to CSV/CSR
# Add more functionality to the GUI
# --- For example, be able to expand/contract the upper text part


import sys
import numpy as np
import csv
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit,
    QFileDialog, QComboBox, QHBoxLayout, QToolTip, QSpacerItem, QSizePolicy
)
from PySide6.QtCharts import QChart, QChartView, QScatterSeries
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainter, QColor, QCursor
from pyscipopt import Model
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import random


class MPSLoaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive MPS Viewer with QtCharts")
        self.resize(1000, 700)

        self.A_sparse = None
        self.last_plot_data = []

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        control_layout = QHBoxLayout()

        self.load_button = QPushButton("Load MPS File")
        self.load_button.clicked.connect(self.load_mps_file)
        control_layout.addWidget(self.load_button)

        self.plot_selector = QComboBox()
        self.plot_selector.addItems(["Binary Scatterplot", "Magnitude Scatterplot"])
        self.plot_selector.currentIndexChanged.connect(self.update_plot)
        control_layout.addWidget(self.plot_selector)

        self.export_button = QPushButton("Export Matrix to CSV")
        self.export_button.clicked.connect(self.export_matrix_to_csv)
        control_layout.addWidget(self.export_button)

        self.layout.addLayout(control_layout)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.layout.addWidget(self.text_area)

        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.layout.addWidget(self.chart_view)

        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()

        self.export_image_button = QPushButton("Export Plot to JPEG")
        self.export_image_button.clicked.connect(self.export_chart_as_image)
        bottom_layout.addWidget(self.export_image_button)

        self.layout.addLayout(bottom_layout)

        # Highlight series
        self.row_highlight_series = QScatterSeries()
        self.row_highlight_series.setMarkerSize(6)
        self.row_highlight_series.setColor(QColor(255, 255, 150, 180))  # Light yellow

        self.col_highlight_series = QScatterSeries()
        self.col_highlight_series.setMarkerSize(6)
        self.col_highlight_series.setColor(QColor(255, 255, 150, 180))  # Light yellow

    def load_mps_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open MPS File", "", "MPS Files (*.mps *.MPS);;All Files (*)")
        if not filename:
            return

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
            u, s, vt = svds(self.A_sparse, k=k)
            tol = 1e-10
            rank_val = np.sum(s > tol)
        except Exception:
            rank_val = "N/A"

        stats = (
            f"📁 File: {filename}\n"
            f"🔹 Matrix shape: {n_cons} rows x {n_vars} columns\n"
            f"🔹 Total entries: {total_entries}\n"
            f"🔹 Non-zero entries: {non_zero}\n"
            f"🔹 Sparsity: {sparsity:.2f}%\n"
            f"🔹 Avg non-zeros per row: {np.mean(row_nnz):.2f}\n"
            f"🔹 Avg non-zeros per column: {np.mean(col_nnz):.2f}\n"
            f"🔹 Matrix Bandwidth: {bandwidth}\n"
            f"🔹 Matrix Rank (est.): {rank_val}\n"
        )
        self.text_area.setPlainText(stats)

        self.update_plot()

    def update_plot(self):
        if self.A_sparse is None:
            return

        plot_type = self.plot_selector.currentText()
        if plot_type == "Binary Scatterplot":
            self.plot_binary_scatterplot()
        elif plot_type == "Magnitude Scatterplot":
            self.plot_magnitude_scatterplot()

    def export_matrix_to_csv(self):
        if not self.last_plot_data:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Matrix Data", "", "CSV Files (*.csv)")
        if not filename:
            return
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Row", "Column", "Value"])
            writer.writerows(self.last_plot_data)

    def on_point_hovered(self, point, state):
        if state:
            row = int(point.y())
            col = int(point.x())
            val = self.A_sparse[row, col]
            QToolTip.showText(QCursor.pos(), f"Row: {row}\nCol: {col}\nVal: {val:.4g}", self.chart_view)


    # COMMENT OUT STARTING HERE IF YOU WANT TO SEE INDIVIDUAL VALUES

    def clear_highlight_series(self, chart):
        chart.removeSeries(self.row_highlight_series)
        chart.removeSeries(self.col_highlight_series)
        self.row_highlight_series.clear()
        self.col_highlight_series.clear()

    def on_row_hovered(self, row_idx, state):
        if not state or self.A_sparse is None:
            return
        chart = self.chart_view.chart()
        self.clear_highlight_series(chart)

        row = self.A_sparse.getrow(row_idx).toarray().flatten()
        cols = np.nonzero(row)[0]
        vals = row[cols]
        if len(vals) == 0:
            return

        for col in cols:
            self.row_highlight_series.append(QPointF(col, row_idx))

        chart.addSeries(self.row_highlight_series)

        max_val = np.max(vals)
        min_val = np.min(vals)
        ratio = abs(max_val) / abs(min_val) if abs(min_val) > 0 else float('inf')

        QToolTip.showText(
            QCursor.pos(),
            f"Row {row_idx}\nMax: {max_val:.4g}\nMin: {min_val:.4g}\nMax/Min Ratio: {ratio:.2f}",
            self.chart_view
        )

    def on_col_hovered(self, col_idx, state):
        if not state or self.A_sparse is None:
            return
        chart = self.chart_view.chart()
        self.clear_highlight_series(chart)

        col = self.A_sparse.getcol(col_idx).toarray().flatten()
        rows = np.nonzero(col)[0]
        vals = col[rows]
        if len(vals) == 0:
            return

        for row in rows:
            self.col_highlight_series.append(QPointF(col_idx, row))

        chart.addSeries(self.col_highlight_series)

        max_val = np.max(vals)
        min_val = np.min(vals)
        ratio = abs(max_val) / abs(min_val) if abs(min_val) > 0 else float('inf')

        QToolTip.showText(
            QCursor.pos(),
            f"Column {col_idx}\nMax: {max_val:.4g}\nMin: {min_val:.4g}\nMax/Min Ratio: {ratio:.2f}",
            self.chart_view
        )

    def add_hoverable_axis_areas(self, chart, count, is_row):
        # Get axis ranges to cover entire axis length
        x_max = int(chart.axisX().max())
        y_max = int(chart.axisY().max())

        for idx in range(count):
            series = QScatterSeries()
            series.setMarkerSize(20)  # Bigger marker for easier hover detection
            series.setColor(QColor(0, 0, 0, 0))  

            if is_row:
                # Add points along the entire row
                for col in range(x_max + 1):
                    series.append(QPointF(col, idx))

                def make_row_cb(row_idx):
                    return lambda point, state: self.on_row_hovered(row_idx, state)
                series.hovered.connect(make_row_cb(idx))

            else:
                # Add points along the entire column
                for row in range(y_max + 1):
                    series.append(QPointF(idx, row))

                def make_col_cb(col_idx):
                    return lambda point, state: self.on_col_hovered(col_idx, state)
                series.hovered.connect(make_col_cb(idx))

            chart.addSeries(series)
            series.attachAxis(chart.axisX())
            series.attachAxis(chart.axisY())


    # COMMENTING STOPS HERE. UNCOMMENT ABOVE IF YOU WANT TO SEE COLUMN VALUES
    # To comment out large chunks of code, highlight the code and use Command / on Mac

    def plot_binary_scatterplot(self):
        rows, cols = self.A_sparse.nonzero()
        indices = list(zip(rows, cols))

        if len(indices) > 100_000:
            indices = random.sample(indices, 100_000)

        self.last_plot_data = [(r, c, self.A_sparse[r, c]) for r, c in indices]

        chart = QChart()
        chart.setTitle("Binary Scatterplot of Constraint Matrix A (white = 0, black = non-zero)")
        chart.legend().hide()

        series = QScatterSeries()
        series.setMarkerSize(6)
        series.setColor(QColor("black"))

        for r, c in indices:
            point = QPointF(c, r)
            series.append(point)

        series.hovered.connect(self.on_point_hovered)

        chart.addSeries(series)
        chart.createDefaultAxes()
        chart.axisX().setTitleText("Variables (Columns)")
        chart.axisY().setTitleText("Constraints (Rows)")
        chart.axisY().setReverse(True)

        self.chart_view.setChart(chart)

        self.clear_highlight_series(chart)
        self.add_hoverable_axis_areas(chart, self.A_sparse.shape[0], is_row=True)
        self.add_hoverable_axis_areas(chart, self.A_sparse.shape[1], is_row=False)

    def plot_magnitude_scatterplot(self):
        rows, cols = self.A_sparse.nonzero()
        vals = self.A_sparse.data

        entries = list(zip(rows, cols, vals))
        if len(entries) > 50_000:
            entries = random.sample(entries, 50_000)

        self.last_plot_data = [(r, c, v) for r, c, v in entries]

        chart = QChart()
        chart.setTitle("Magnitude Scatterplot of Constraint Matrix A (Blue = Negative, Red = Positive, Darker = Larger Magnitude)")
        chart.legend().hide()

        max_val = max(abs(v) for _, _, v in entries) if entries else 1.0

        for r, c, v in entries:
            normalized = min(abs(v) / max_val, 1.0)
            alpha = int(255 * (normalized ** 0.5))
            alpha = max(alpha, 50)

            color = QColor(255, 0, 0, alpha) if v > 0 else QColor(0, 0, 255, alpha)

            series = QScatterSeries()
            series.setMarkerSize(6)
            series.setColor(color)
            series.append(QPointF(c, r))
            series.hovered.connect(self.on_point_hovered)
            chart.addSeries(series)

        chart.createDefaultAxes()
        chart.axisX().setTitleText("Variables (Columns)")
        chart.axisY().setTitleText("Constraints (Rows)")
        chart.axisY().setReverse(True)

        self.chart_view.setChart(chart)

        self.clear_highlight_series(chart)
        self.add_hoverable_axis_areas(chart, self.A_sparse.shape[0], is_row=True)
        self.add_hoverable_axis_areas(chart, self.A_sparse.shape[1], is_row=False)

    def export_chart_as_image(self):
        if not self.chart_view.chart():
            QToolTip.showText(QCursor.pos(), "❌ No chart to export.", self.chart_view)
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Chart as JPEG",
            "plot.jpeg",
            "JPEG Image (*.jpeg *.jpg)"
        )
        if not filename:
            return

        pixmap = self.chart_view.grab()
        if not pixmap.save(filename, "JPEG"):
            QToolTip.showText(QCursor.pos(), "❌ Failed to save image.", self.chart_view)
        else:
            QToolTip.showText(QCursor.pos(), f"✅ Saved: {filename}", self.chart_view)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MPSLoaderApp()
    window.show()
    sys.exit(app.exec())

