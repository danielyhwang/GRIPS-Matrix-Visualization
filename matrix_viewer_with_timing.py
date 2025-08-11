
import sys, math, random, io, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QLabel
)
from PySide6.QtCharts import QChart, QChartView, QScatterSeries
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPainter, QColor, QPixmap, QImage
from pyscipopt import Model
from scipy.sparse import csr_matrix
from PIL import Image
from pathlib import Path

timing_logs = []  # Collect timings for CSV


class ClickableFigureCanvas(FigureCanvas):
    def __init__(self, parent, fig, data_matrix):
        super().__init__(fig)
        self.setParent(parent)
        self.data_matrix = data_matrix
        self.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event):
        if event.inaxes:
            row = int(event.ydata)
            if 0 <= row < self.data_matrix.shape[0]:
                stats = {
                    "Min": np.min(self.data_matrix[row]),
                    "Max": np.max(self.data_matrix[row]),
                    "Mean": np.mean(self.data_matrix[row]),
                    "Std": np.std(self.data_matrix[row]),
                    "L2 norm": np.linalg.norm(self.data_matrix[row]),
                    "Non-zeros": np.count_nonzero(self.data_matrix[row]),
                    "Zeros": len(self.data_matrix[row]) - np.count_nonzero(self.data_matrix[row])
                }
                msg = f"<b>Row {row} Statistics:</b><br>" + "".join(f"{k}: {v:.3g}<br>" for k, v in stats.items())
                QMessageBox.information(self, "Row Info", msg)


class MatrixViewer(QWidget):
    def __init__(self, filename, log_timings=True):
        super().__init__()
        self.setWindowTitle("Enhanced MPS Matrix Viewer")
        self.A_sparse = None
        self.last_plot_data = []
        self.benchmark_record = {"file": Path(filename).stem} if log_timings else None
        self.layout = QVBoxLayout(self)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.stats_table)

        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.layout.addWidget(self.chart_view)

        self.heatmap_label = QLabel()
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.heatmap_label)
        self.heatmap_label.setVisible(False)

        # Buttons
        bottom_layout = QHBoxLayout()
        self.export_button = QPushButton("Export Plot to JPEG")
        self.binary_button = QPushButton("Binary Scatterplot")
        self.magnitude_button = QPushButton("Magnitude Scatterplot")
        self.row_scaled_button = QPushButton("Row-Scaled Heatmap")
        bottom_layout.addWidget(self.binary_button)
        bottom_layout.addWidget(self.magnitude_button)
        bottom_layout.addWidget(self.row_scaled_button)
        bottom_layout.addWidget(self.export_button)
        self.layout.addLayout(bottom_layout)

        self.export_button.clicked.connect(self.export_chart_as_image)
        self.binary_button.clicked.connect(lambda: self.update_plot("Binary"))
        self.magnitude_button.clicked.connect(lambda: self.update_plot("Magnitude"))
        self.row_scaled_button.clicked.connect(lambda: self.update_plot("RowScaled"))

        self.legend_label = QLabel()
        self.legend_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.legend_label)
        self.legend_label.setVisible(False)

        self.load_matrix(filename)
        self.update_plot("Binary")
        self.destroyed.connect(self.save_benchmark_results)

    def generate_gradient_qpixmap(self, width=300, height=20):
        gradient = np.linspace(-1, 1, width)
        gradient = np.tile(gradient, (height, 1))
        cmap = plt.get_cmap('bwr')
        rgba_img = cmap((gradient + 1) / 2)
        rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)
        image = Image.fromarray(rgb_img)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        qimg = QImage.fromData(buffer.getvalue(), "PNG")
        return QPixmap.fromImage(qimg)

    def load_matrix(self, filename):
        start = time.time()
        model = Model()
        model.readProblem(filename)
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
        self.A_sparse = csr_matrix((data, (row_inds, col_inds)), shape=(len(constraints), len(variables)))
        if self.benchmark_record is not None:
            self.benchmark_record["upload_gui_time"] = time.time() - start

        # Stats timing
        stat_start = time.time()
        props = [
            ("File", filename),
            ("Shape", str(self.A_sparse.shape)),
            ("Total entries", self.A_sparse.shape[0] * self.A_sparse.shape[1]),
            ("Non-zero entries", self.A_sparse.nnz),
            ("Avg non-zeros per row", np.mean(np.diff(self.A_sparse.indptr))),
            ("Avg non-zeros per column", np.mean(np.diff(self.A_sparse.T.indptr))),
            ("Sparsity (%)", 100 * (1 - self.A_sparse.nnz / (self.A_sparse.shape[0] * self.A_sparse.shape[1]))),
            ("Row NNZ Variance", np.var(np.diff(self.A_sparse.indptr)))
        ]
        if self.benchmark_record is not None:
            self.benchmark_record["stats_time"] = time.time() - stat_start

        self.stats_table.setRowCount(len(props))
        for i, (k, v) in enumerate(props):
            self.stats_table.setItem(i, 0, QTableWidgetItem(str(k)))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(v)))

    def update_plot(self, plot_type):
        self.legend_label.setVisible(False)
        self.heatmap_label.setVisible(False)
        self.chart_view.setVisible(True)
        if plot_type == "Binary":
            self.plot_binary()
        elif plot_type == "Magnitude":
            self.plot_magnitude()
            self.legend_label.setPixmap(self.generate_gradient_qpixmap(350, 25))
            self.legend_label.setVisible(True)
        elif plot_type == "RowScaled":
            self.plot_row_scaled_heatmap()

    def plot_binary(self):
        start = time.time()
        rows, cols = self.A_sparse.nonzero()
        chart = QChart()
        chart.setTitle("Binary Scatterplot (Black = Non-zero)")
        chart.legend().hide()
        series = QScatterSeries()
        series.setMarkerSize(10)
        series.setColor(QColor("black"))
        for r, c in zip(rows, cols):
            series.append(QPointF(c, r))
        series.clicked.connect(self.on_point_clicked)
        chart.addSeries(series)
        chart.createDefaultAxes()
        chart.axisX().setTitleText("Variables (Columns)")
        chart.axisY().setTitleText("Constraints (Rows)")
        chart.axisY().setReverse(True)
        self.chart_view.setChart(chart)
        if self.benchmark_record is not None:
            self.benchmark_record["scatter_plot_time"] = time.time() - start

    def plot_magnitude(self):
        start = time.time()
        rows, cols = self.A_sparse.nonzero()
        vals = self.A_sparse.data
        entries = list(zip(rows, cols, vals))
        if len(entries) > 50000:
            entries = random.sample(entries, 50000)
        chart = QChart()
        chart.setTitle("Signed Magnitude Heatmap")
        chart.legend().hide()
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
            series = QScatterSeries()
            series.setColor(color)
            series.setMarkerSize(12)
            shape = QScatterSeries.MarkerShapeCircle if v > 0 else QScatterSeries.MarkerShapeRectangle
            series.setMarkerShape(shape)
            series.append(QPointF(c, r))
            series.clicked.connect(self.on_point_clicked)
            chart.addSeries(series)
        chart.createDefaultAxes()
        chart.axisX().setTitleText("Variables (Columns)")
        chart.axisY().setTitleText("Constraints (Rows)")
        chart.axisY().setReverse(True)
        self.chart_view.setChart(chart)
        if self.benchmark_record is not None:
            self.benchmark_record["magnitude_plot_time"] = time.time() - start

    def plot_row_scaled_heatmap(self):
        start = time.time()
        self.chart_view.setVisible(False)
        self.heatmap_label.setVisible(False)
        A = self.A_sparse.toarray()
        row_scaled = np.zeros_like(A)
        for i, row in enumerate(A):
            nonzero = row[np.nonzero(row)]
            if len(nonzero) == 0: continue
            min_val, max_val = np.min(abs(nonzero)), np.max(abs(nonzero))
            if max_val > min_val:
                row_scaled[i] = (abs(row) - min_val) / (max_val - min_val)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(row_scaled, cmap='bwr', aspect='auto', origin='lower')
        ax.set_title("Row-Scaled Heatmap")
        ax.set_xlabel("Variables (Columns)")
        ax.set_ylabel("Constraints (Rows)")
        fig.colorbar(im, orientation='horizontal')
        self.canvas = ClickableFigureCanvas(self, fig, A)
        self.layout.replaceWidget(self.heatmap_label, self.canvas)
        self.heatmap_label.setParent(None)
        self.heatmap_label = self.canvas
        self.heatmap_label.setVisible(True)
        self.canvas.draw()
        if self.benchmark_record is not None:
            self.benchmark_record["row_scaled_heatmap_time"] = time.time() - start

    def on_point_clicked(self, point):
        row, col = int(point.y()), int(point.x())
        A = self.A_sparse.toarray()
        r_vals, c_vals = A[row], A[:, col]
        def stats(arr):
            return {
                "Min": np.min(arr), "Max": np.max(arr), "Mean": np.mean(arr),
                "Std": np.std(arr), "L2 norm": np.linalg.norm(arr),
                "Non-zeros": np.count_nonzero(arr), "Zeros": len(arr) - np.count_nonzero(arr)
            }
        msg = (
            f"<b>Entry:</b> Row {row}, Col {col}<br><br>"
            f"<b>Row Stats:</b><br>" + "".join(f"{k}: {v:.3g}<br>" for k, v in stats(r_vals).items()) +
            f"<br><b>Column Stats:</b><br>" + "".join(f"{k}: {v:.3g}<br>" for k, v in stats(c_vals).items())
        )
        QMessageBox.information(self, "Matrix Entry Info", msg)

    def export_chart_as_image(self):
        if self.heatmap_label.isVisible():
            pixmap = self.heatmap_label.grab()
        elif self.chart_view.chart():
            pixmap = self.chart_view.grab()
        else:
            QMessageBox.information(self, "Export", "No chart to export.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Chart", "plot.jpeg", "JPEG (*.jpeg *.jpg)")
        if filename and pixmap:
            pixmap.save(filename, "JPEG")

    def save_benchmark_results(self):
        if hasattr(self, 'benchmark_record'):
            timing_logs.append(self.benchmark_record)
        if timing_logs:
            pd.DataFrame(timing_logs).to_csv("gui_interaction_timings.csv", index=False)


class FileLoader(QWidget):
    def __init__(self):
        super().__init__()
        self.filename, _ = QFileDialog.getOpenFileName(self, "Open MPS File", "", "MPS Files (*.mps)")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    loader = FileLoader()
    if loader.filename:
        viewer = MatrixViewer(loader.filename)
        viewer.show()
    sys.exit(app.exec())
