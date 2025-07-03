import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit,
    QFileDialog, QComboBox, QHBoxLayout, QMessageBox
)
from PySide6.QtCharts import QChart, QChartView, QScatterSeries
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainter, QColor
from pyscipopt import Model
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


class MPSLoaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive MPS Viewer with QtCharts")
        self.resize(1000, 700)

        self.A_sparse = None

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

        self.layout.addLayout(control_layout)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.layout.addWidget(self.text_area)

        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.layout.addWidget(self.chart_view)

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
            f"🔷 Matrix shape: {n_cons} rows x {n_vars} columns\n"
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

    def show_popup(self, title, content):
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(content)
        msg.exec()

    def plot_binary_scatterplot(self):
        rows, cols = self.A_sparse.nonzero()

        chart = QChart()
        chart.setTitle("Binary Scatterplot of Constraint Matrix A (white = 0, black = non-zero)")
        chart.legend().hide()

        series = QScatterSeries()
        series.setMarkerSize(6)
        series.setColor(QColor("black"))
        for x, y in zip(cols, rows):
            point = QPointF(x, y)
            series.append(point)

        def on_point_clicked(point):
            row = int(point.y())
            col = int(point.x())
            val = self.A_sparse[row, col]
            self.show_popup("Matrix Entry Clicked", f"Row: {row}\nColumn: {col}\nValue: {val:.4g}")

        series.clicked.connect(on_point_clicked)

        chart.addSeries(series)
        chart.createDefaultAxes()
        chart.axisX().setTitleText("Variables (Columns)")
        chart.axisY().setTitleText("Constraints (Rows)")
        chart.axisY().setReverse(True)

        self.chart_view.setChart(chart)

    def plot_magnitude_scatterplot(self):
        rows, cols = self.A_sparse.nonzero()
        vals = self.A_sparse.data

        chart = QChart()
        chart.setTitle("Magnitude Scatterplot of Constraint Matrix A (Blue = Negative, Red = Positive, Darker = Larger Magnitude)")
        chart.legend().hide()

        max_val = abs(vals).max() if self.A_sparse.nnz > 0 else 1.0

        for x, y, v in zip(cols, rows, vals):
            normalized = min(abs(v) / max_val, 1.0)
            alpha = int(255 * (normalized ** 0.5))  # use square root for darker values
            alpha = max(alpha, 50)  # ensure even small values are visible

            if v > 0:
                color = QColor(255, 0, 0, alpha)
            else:
                color = QColor(0, 0, 255, alpha)

            series = QScatterSeries()
            series.setMarkerSize(6)
            series.setColor(color)
            series.append(QPointF(x, y))

            def make_handler(row=y, col=x, val=v):
                def on_click(_):
                    self.show_popup("Matrix Entry Clicked", f"Row: {row}\nColumn: {col}\nValue: {val:.4g}")
                return on_click

            series.clicked.connect(make_handler())
            chart.addSeries(series)

        chart.createDefaultAxes()
        chart.axisX().setTitleText("Variables (Columns)")
        chart.axisY().setTitleText("Constraints (Rows)")
        chart.axisY().setReverse(True)

        self.chart_view.setChart(chart)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MPSLoaderApp()
    window.show()
    sys.exit(app.exec())