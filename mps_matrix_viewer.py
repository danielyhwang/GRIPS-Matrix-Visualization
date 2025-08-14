# MPS Matrix Viewer with Stats + Clickable Scatterplot
import sys, math, random, io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QLabel
)
from PySide6.QtCharts import QChart, QChartView, QScatterSeries, QValueAxis
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPainter, QColor, QPixmap, QImage
from pyscipopt import Model
from scipy.sparse import csr_matrix
from PIL import Image


# -----------------------------
# Custom Matplotlib canvas that allows row-click interaction
# -----------------------------
class ClickableFigureCanvas(FigureCanvas):
    def __init__(self, parent, fig, data_matrix):
        super().__init__(fig)
        self.setParent(parent)
        self.data_matrix = data_matrix
        # Connect mouse click events to handler
        self.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event):
        """When user clicks on the heatmap, show stats for that row."""
        if event.inaxes:
            row = int(event.ydata)
            if 0 <= row < self.data_matrix.shape[0]:
                # Compute basic statistics for the clicked row
                stats = {
                    "Min": np.min(self.data_matrix[row]),
                    "Max": np.max(self.data_matrix[row]),
                    "Mean": np.mean(self.data_matrix[row]),
                    "Std": np.std(self.data_matrix[row]),
                    "L2 norm": np.linalg.norm(self.data_matrix[row]),
                    "Non-zeros": np.count_nonzero(self.data_matrix[row]),
                    "Zeros": len(self.data_matrix[row]) - np.count_nonzero(self.data_matrix[row])
                }
                # Display the stats in a popup
                msg = f"<b>Row {row} Statistics:</b><br>" + "".join(f"{k}: {v:.3g}<br>" for k, v in stats.items())
                QMessageBox.information(self, "Row Info", msg)


# -----------------------------
# Main viewer widget
# -----------------------------
class MatrixViewer(QWidget):
    def __init__(self, filename):
        super().__init__()
        self.setWindowTitle("Enhanced MPS Matrix Viewer")

        self.A_sparse = None  # Sparse matrix representation of constraints
        self.last_plot_data = []  # Placeholder for storing plot data if needed

        self.layout = QVBoxLayout(self) # Layout to hold QtWidgets.

        # Table for matrix statistics
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.stats_table)

        # Chart view for scatterplots
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.layout.addWidget(self.chart_view)

        # Label for displaying heatmaps
        self.heatmap_label = QLabel()
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.heatmap_label)
        self.heatmap_label.setVisible(False)

        # Bottom control buttons, contained in bottom_layout. 
        bottom_layout = QHBoxLayout()
        self.export_button = QPushButton("Export Plot to JPEG")
        self.binary_button = QPushButton("Binary Scatterplot")
        self.magnitude_button = QPushButton("Signed Magnitude Scatterplot")
        self.row_scaled_button = QPushButton("Row-Scaled Heatmap")
        bottom_layout.addWidget(self.binary_button)
        bottom_layout.addWidget(self.magnitude_button)
        bottom_layout.addWidget(self.row_scaled_button)
        bottom_layout.addWidget(self.export_button)
        self.layout.addLayout(bottom_layout)

        # Connect buttons to functions
        self.export_button.clicked.connect(self.export_chart_as_image)
        self.binary_button.clicked.connect(lambda: self.update_plot("Binary"))
        self.magnitude_button.clicked.connect(lambda: self.update_plot("Magnitude"))
        self.row_scaled_button.clicked.connect(lambda: self.update_plot("RowScaled"))

        # Label for colorbar legend
        self.legend_label = QLabel()
        self.legend_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.legend_label)
        self.legend_label.setVisible(False)

        # Load the matrix from MPS file and show initial plot
        self.load_matrix(filename)
        self.update_plot("Binary")

    # -----------------------------
    # Utility to convert Matplotlib figure to QPixmap
    # -----------------------------
    def fig_to_qpixmap(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        qimg = QImage.fromData(buf.getvalue())
        return QPixmap.fromImage(qimg)

    # -----------------------------
    # Load MPS file into a sparse matrix
    # -----------------------------
    def load_matrix(self, filename):
        model = Model()
        model.readProblem(filename)
        variables = model.getVars()
        constraints = model.getConss()
        var_names = [var.name for var in variables]
        var_index = {name: idx for idx, name in enumerate(var_names)}

        row_inds, col_inds, data = [], [], []
        # Extract coefficients for each constraint-variable pair
        for i, cons in enumerate(constraints):
            terms = model.getValsLinear(cons)
            for var_name, coef in terms.items():
                j = var_index[var_name]
                # We only append nonzero entries in our data
                if coef != 0:
                    row_inds.append(i)
                    col_inds.append(j)
                    data.append(coef)

        # Build SciPy CSR sparse matrix
        self.A_sparse = csr_matrix((data, (row_inds, col_inds)), shape=(len(constraints), len(variables)))

        # Compute matrix properties for display
        props = [
            ("\U0001F4C1 File:", filename),
            ("Shape", str(self.A_sparse.shape)),
            ("Total entries", self.A_sparse.shape[0] * self.A_sparse.shape[1]),
            ("Non-zero entries", self.A_sparse.nnz),
            ("Avg non-zeros per row", np.mean(np.diff(self.A_sparse.indptr))),
            ("Avg non-zeros per column", np.mean(np.diff(self.A_sparse.indptr)) * self.A_sparse.shape[0] / self.A_sparse.shape[1]), 
            #NOTE ABOVE: For some reason, np.mean(np.diff(self.A_sparse.T.indptr)) does not work, so we manually rescale by num_rows/num_cols.
            ("Sparsity (%)", 100 * (1 - self.A_sparse.nnz / (self.A_sparse.shape[0] * self.A_sparse.shape[1]))),
            ("Row NNZ Variance", np.var(np.diff(self.A_sparse.indptr))),
            ("Relative Rank", np.linalg.matrix_rank(self.A_sparse.toarray()) / min(self.A_sparse.shape))
        ]

        self.stats_table.setRowCount(len(props))
        for i, (k, v) in enumerate(props):
            self.stats_table.setItem(i, 0, QTableWidgetItem(str(k)))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(v)))

    # -----------------------------
    # Main dispatcher for plot types
    # -----------------------------
    def update_plot(self, plot_type):
        self.legend_label.setVisible(False)
        self.heatmap_label.setVisible(False)
        self.chart_view.setVisible(True)

        if plot_type == "Binary":
            self.plot_binary()
        elif plot_type == "Magnitude":
            self.plot_magnitude()
        elif plot_type == "RowScaled":
            self.plot_row_scaled_heatmap()

    # -----------------------------
    # Plot binary scatterplot (nonzeros in black)
    # -----------------------------
    def plot_binary(self):
        rows, cols = self.A_sparse.nonzero()
        chart = QChart()
        chart.setTitle("Binary Scatterplot (Black = Non-Zero)")
        chart.legend().hide()

        series = QScatterSeries()
        series.setMarkerSize(10)
        series.setColor(QColor("black"))
        for r, c in zip(rows, cols):
            series.append(QPointF(c, r))
        series.clicked.connect(self.on_point_clicked)
        chart.addSeries(series)

        # Calculate max_col and max_row in axes.

        max_col = self.A_sparse.shape[1]
        max_row = self.A_sparse.shape[0]

        # Compute axis bounds with exactly 1 unit padding
        x_min = 0
        x_max = max_col - 1
        y_min = 0
        y_max = max_row - 1

        # Creates X Axis
        axisX = QValueAxis()
        axisX.setTitleText("Variables (Columns)")
        axisX.setRange(x_min, x_max)
        axisX.setTickInterval(1)
        axisX.setLabelFormat("%d")
        axisX.setMinorTickCount(0)
        axisX.setTickType(QValueAxis.TicksFixed)

        # Creates Y Axis
        axisY = QValueAxis()
        axisY.setTitleText("Constraints (Rows)")
        axisY.setRange(y_min, y_max)
        axisY.setTickInterval(1)
        axisY.setLabelFormat("%d")
        axisY.setMinorTickCount(0)
        axisY.setTickType(QValueAxis.TicksFixed)
        axisY.setReverse(True)

        # Remove grid lines
        axisX.setGridLineVisible(False)
        axisY.setGridLineVisible(False)
        axisX.setMinorGridLineVisible(False)
        axisY.setMinorGridLineVisible(False)

        # Set titles and tick count to axes.
        axisX.setTitleText("Variables (Columns)")
        axisY.setTitleText("Constraints (Rows)")
        axisX.setTickCount(self.A_sparse.shape[1])
        axisY.setTickCount(self.A_sparse.shape[0])
        axisY.setReverse(True)
        self.chart_view.setChart(chart)

        # Add axes to chart.
        chart.addAxis(axisX, Qt.AlignBottom)
        chart.addAxis(axisY, Qt.AlignLeft)
        series.attachAxis(axisX)
        series.attachAxis(axisY)

    # -----------------------------
    # Plot signed magnitude scatterplot with no-white colorbar
    # -----------------------------
    def plot_magnitude(self):
        rows, cols = self.A_sparse.nonzero()
        vals = self.A_sparse.data
        entries = list(zip(rows, cols, vals))
        # If there are greater than 50,000 nonzero entries, then sample 50,000 of them to generate the magnitude scatterplot.
        if len(entries) > 50000:
            entries = random.sample(entries, 50000)

        chart = QChart()
        chart.setTitle("Signed Magnitude Scatterplot (Circle = +, Square = –; Color by log|A|)")
        chart.legend().hide()

        # Compute log10(|A|) for color scaling
        abs_vals = [abs(v) for _, _, v in entries if v != 0]
        logs = np.log10(abs_vals)
        min_log, max_log = logs.min(), logs.max()
        norm = Normalize(vmin=min_log, vmax=max_log)

        # Creates a Blue→Red ColorMap without white midpoint
        cmap = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])

        # Create 20 separate series for positive and negative entries, as each series can only be colored once.
        num_bins = 20
        bin_edges = np.linspace(min_log, max_log, num_bins + 1)
        pos_bins, neg_bins = [], []
        for i in range(num_bins):
            mid_val = (bin_edges[i] + bin_edges[i + 1]) / 2
            color = cmap(norm(mid_val))
            qcolor = QColor(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            s_pos = QScatterSeries()
            s_pos.setMarkerShape(QScatterSeries.MarkerShapeCircle)
            s_pos.setMarkerSize(8)
            s_pos.setColor(qcolor)
            s_pos.setOpacity(0.6)
            s_pos.clicked.connect(self.on_point_clicked)
            s_neg = QScatterSeries()
            s_neg.setMarkerShape(QScatterSeries.MarkerShapeRectangle)
            s_neg.setMarkerSize(8)
            s_neg.setColor(qcolor)
            s_neg.setOpacity(0.6)
            s_neg.clicked.connect(self.on_point_clicked)
            pos_bins.append(s_pos)
            neg_bins.append(s_neg)

        for r, c, v in entries:
            if v == 0:
                continue
            logv = np.log10(abs(v))
            bin_idx = np.searchsorted(bin_edges, logv, side='right') - 1
            bin_idx = max(0, min(num_bins - 1, bin_idx))
            if v > 0:
                pos_bins[bin_idx].append(QPointF(c, r))
            else:
                neg_bins[bin_idx].append(QPointF(c, r))

        for s in pos_bins + neg_bins:
            if s.count() > 0:
                chart.addSeries(s)

        # Calculate max_col and max_row in axes.
        max_col = self.A_sparse.shape[1]
        max_row = self.A_sparse.shape[0]

        chart.createDefaultAxes()
        chart.axisX().setTitleText("Variables (Columns)")
        chart.axisY().setTitleText("Constraints (Rows)")
        chart.axisX().setTickCount(self.A_sparse.shape[1])
        chart.axisY().setTickCount(self.A_sparse.shape[0])
        chart.axisY().setReverse(True)
        self.chart_view.setChart(chart)

        # Colorbar showing log10(|A|) range
        fig, ax = plt.subplots(figsize=(3.5, 0.3))
        cb = ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
        self.legend_label.setPixmap(self.fig_to_qpixmap(fig))
        plt.close(fig)
        self.legend_label.setVisible(True)

    # -----------------------------
    # Plot row-scaled heatmap with no-white colorbar
    # -----------------------------
    def plot_row_scaled_heatmap(self):
        self.chart_view.setVisible(False)
        self.heatmap_label.setVisible(False)

        A = self.A_sparse.toarray()
        row_scaled = np.full_like(A, np.nan, dtype=float)

        # Normalize each row over nonzeros only
        for i, row in enumerate(A):
            nz = row[row != 0]
            if len(nz) == 0:
                continue
            min_val, max_val = np.min(abs(nz)), np.max(abs(nz))
            if max_val > min_val:
                row_scaled[i] = (abs(row) - min_val) / (max_val - min_val)
            else:
                row_scaled[i] = 0

        masked = np.ma.masked_invalid(row_scaled)

        # Blue→Red without white midpoint; white only for masked (zero) entries
        cmap = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])
        cmap.set_bad(color='white')

        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(masked, cmap=cmap, aspect='auto', origin='lower', vmin=0, vmax=1)
        ax.set_title("Row-Scaled Heatmap (row-wise min-max over nonzeros)")
        ax.set_xlabel("Variables (Columns)")
        ax.set_ylabel("Constraints (Rows)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        fig.colorbar(im, ax=ax, orientation='horizontal', label="Normalized |A| per row")
        self.canvas = ClickableFigureCanvas(self, fig, A)
        self.layout.replaceWidget(self.heatmap_label, self.canvas)
        self.heatmap_label.setParent(None)
        self.heatmap_label = self.canvas
        self.heatmap_label.setVisible(True)
        self.canvas.draw()

    # -----------------------------
    # When a point in a scatterplot is clicked
    # -----------------------------
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

        # Display stats for the selected row and column
        msg = (
            f"<b>Entry:</b> Row {row}, Col {col}<br><br>"
            f"<b>Row {row} Stats:</b><br>" + "".join(f"{k}: {v:.3g}<br>" for k, v in stats(r_vals).items()) +
            f"<br><b>Column {col} Stats:</b><br>" + "".join(f"{k}: {v:.3g}<br>" for k, v in stats(c_vals).items())
        )
        QMessageBox.information(self, "Matrix Entry Info", msg)

    # -----------------------------
    # Export currently visible chart or heatmap to JPEG
    # -----------------------------
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


# -----------------------------
# File loader dialog for MPS file
# -----------------------------
class FileLoader(QWidget):
    def __init__(self):
        super().__init__()
        self.filename, _ = QFileDialog.getOpenFileName(self, "Open MPS File", "", "MPS Files (*.mps)")


# -----------------------------
# Main entry point
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    loader = FileLoader()
    if loader.filename:
        viewer = MatrixViewer(loader.filename)
        viewer.show()
    sys.exit(app.exec())

