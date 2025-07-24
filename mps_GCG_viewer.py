import sys, math, random, io
import numpy as np
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
from pygcgopt import Model as GCGModel
from collections import defaultdict
import os


def parse_block_assignments(gcg_model):
    block_row_map = {}
    block_col_map = {}

    for cons in gcg_model.getConss():
        block = gcg_model.getBlockCons(cons)
        if block >= 0:
            block_row_map[cons.name] = block

    for var in gcg_model.getVars():
        block = gcg_model.getBlockVar(var)
        if block >= 0:
            block_col_map[var.name] = block

    return block_row_map, block_col_map


def export_dec_file(path, block_row_map, block_col_map):
    block_rows = defaultdict(list)
    block_cols = defaultdict(list)

    for row, block in block_row_map.items():
        block_rows[block].append(row)
    for col, block in block_col_map.items():
        block_cols[block].append(col)

    with open(path, 'w') as f:
        for block in sorted(block_rows.keys()):
            f.write(f"BLOCK {block}\n")
            f.write("ROWS\n")
            for row in block_rows[block]:
                f.write(f"{row}\n")
            f.write("COLUMNS\n")
            for col in block_cols.get(block, []):
                f.write(f"{col}\n")


def generate_color_palette(n):
    colors = plt.cm.get_cmap("tab10", n)
    return [QColor(*(int(c * 255) for c in colors(i)[:3])) for i in range(n)]


def save_matrix_image(matrix, filename, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.spy(matrix, markersize=0.5)
    ax.set_title(title)
    ax.set_xlabel("Variables")
    ax.set_ylabel("Constraints")
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


class MatrixBlockViewer(QWidget):
    def __init__(self, mps_file):
        super().__init__()
        self.setWindowTitle("GCG-Enhanced MPS Matrix Viewer")

        self.mps_file = mps_file
        self.layout = QVBoxLayout(self)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.stats_table)

        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.layout.addWidget(self.chart_view)

        self.legend_label = QLabel()
        self.legend_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.legend_label)
        self.legend_label.setVisible(False)

        self.export_button = QPushButton("Export Chart to JPEG")
        self.export_button.clicked.connect(self.export_chart_as_image)
        self.layout.addWidget(self.export_button)

        self.load_matrix()
        self.update_plot()

    def load_matrix(self):
        model = Model()
        model.readProblem(self.mps_file)
        self.variables = model.getVars()
        self.constraints = model.getConss()
        var_names = [var.name for var in self.variables]
        var_index = {name: idx for idx, name in enumerate(var_names)}

        row_inds, col_inds, data = [], [], []
        for i, cons in enumerate(self.constraints):
            terms = model.getValsLinear(cons)
            for var_name, coef in terms.items():
                j = var_index[var_name]
                row_inds.append(i)
                col_inds.append(j)
                data.append(coef)

        self.A_sparse = csr_matrix((data, (row_inds, col_inds)), shape=(len(self.constraints), len(self.variables)))
        self.row_names = [c.name for c in self.constraints]
        self.col_names = var_names

        base = os.path.splitext(self.mps_file)[0]
        save_matrix_image(self.A_sparse, base + "_original.png", "Nonzero entries of original problem")

        gcg_model = GCGModel(self.mps_file)
        gcg_model.setIntParam("detection/maxrounds", 10)
        gcg_model.detect()
        self.block_row_map, self.block_col_map = parse_block_assignments(gcg_model)

        dec_path = base + ".dec"
        export_dec_file(dec_path, self.block_row_map, self.block_col_map)
        print(f"Saved decomposition file to {dec_path}")

        self.total_blocks = len(set(self.block_row_map.values()).union(set(self.block_col_map.values())))
        self.block_colors = generate_color_palette(self.total_blocks)

        props = [
            ("MPS File", self.mps_file),
            ("Shape", str(self.A_sparse.shape)),
            ("Non-zero entries", self.A_sparse.nnz),
            ("Sparsity (%)", 100 * (1 - self.A_sparse.nnz / (self.A_sparse.shape[0] * self.A_sparse.shape[1]))),
            ("Detected GCG Blocks", self.total_blocks)
        ]
        self.stats_table.setRowCount(len(props))
        for i, (k, v) in enumerate(props):
            self.stats_table.setItem(i, 0, QTableWidgetItem(str(k)))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(v)))

    def update_plot(self):
        rows, cols = self.A_sparse.nonzero()
        chart = QChart()
        chart.setTitle("GCG Block Visualization")
        chart.legend().hide()

        series_map = {}
        for r, c in zip(rows, cols):
            rname = self.row_names[r]
            cname = self.col_names[c]
            row_block = self.block_row_map.get(rname, -1)
            col_block = self.block_col_map.get(cname, -1)
            block_id = row_block if row_block == col_block else -1

            if block_id not in series_map:
                s = QScatterSeries()
                s.setMarkerSize(8)
                s.setColor(self.block_colors[block_id] if block_id >= 0 else QColor("gray"))
                series_map[block_id] = s

            series_map[block_id].append(QPointF(c, r))

        for s in series_map.values():
            chart.addSeries(s)

        chart.createDefaultAxes()
        chart.axisX().setTitleText("Variables (Columns)")
        chart.axisY().setTitleText("Constraints (Rows)")
        chart.axisY().setReverse(True)
        self.chart_view.setChart(chart)

        html = "<b>Block Colors:</b><br>"
        for i, color in enumerate(self.block_colors):
            html += f"<span style='color:{color.name()}'>■</span> Block {i}<br>"
        html += "<span style='color:gray'>■</span> Mixed/Unassigned"
        self.legend_label.setText(html)
        self.legend_label.setVisible(True)

    def export_chart_as_image(self):
        if self.chart_view.chart():
            pixmap = self.chart_view.grab()
            filename, _ = QFileDialog.getSaveFileName(self, "Save Chart", "plot.jpeg", "JPEG (*.jpeg *.jpg)")
            if filename:
                pixmap.save(filename, "JPEG")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mps_file, _ = QFileDialog.getOpenFileName(None, "Open MPS File", "", "MPS Files (*.mps)")
    if mps_file:
        viewer = MatrixBlockViewer(mps_file)
        viewer.show()
        sys.exit(app.exec())
    else:
        sys.exit(0)



