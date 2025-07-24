# MPS Matrix Viewer with Stats + Clickable Scatterplot
import sys, math, random, io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QLabel
)
from PySide6.QtCharts import QChart, QChartView, QScatterSeries
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPainter, QColor, QPixmap, QImage
from pygcgopt import Model
from scipy.sparse import csr_matrix
#from scipy.sparse.linalg import svds
from PIL import Image

from pathlib import Path
# Define directories, create partial_decomps directory if it doesn't exist, we'll store our stuff there.
BASE_DIR = Path(__file__).resolve().parent
PARTIAL_DECOMP_DIR = BASE_DIR / "partial_decomps"
PARTIAL_DECOMP_DIR.mkdir(exist_ok=True)


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
    def __init__(self, filename, include_toggle_buttons = True):
        super().__init__()
        self.setWindowTitle("Enhanced MPS Matrix Viewer")

        self.A_sparse = None

        self.layout = QVBoxLayout(self) #combines two lines together

        # Load in stats table, and chart view, along with export button.
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

        # Add in bottom layout for buttons.
        bottom_layout = QHBoxLayout()

        # Testing functionality of MatrixViewer, only if include_toggle_buttons is on.
        if include_toggle_buttons:
            self.binary_button = QPushButton("Binary Scatterplot")
            self.magnitude_button = QPushButton("Magnitude Scatterplot")
            self.row_scaled_button = QPushButton("Row-Scaled Heatmap")
            # Ensure these match their corresponding parts in update_plot!
            self.binary_button.clicked.connect(lambda: self.update_plot("Binary Scatterplot")) 
            self.magnitude_button.clicked.connect(lambda: self.update_plot("Magnitude Scatterplot"))
            self.row_scaled_button.clicked.connect(lambda: self.update_plot("Row-Scaled Heatmap"))
            bottom_layout.addWidget(self.binary_button)
            bottom_layout.addWidget(self.magnitude_button)
            bottom_layout.addWidget(self.row_scaled_button)

        # Include export plot to JPEG functionality.
        bottom_layout.addStretch()
        self.export_image_button = QPushButton("Export Plot to JPEG")
        self.export_image_button.clicked.connect(self.export_chart_as_image)
        bottom_layout.addWidget(self.export_image_button)
        self.layout.addLayout(bottom_layout)

        self.legend_label = QLabel()
        self.legend_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.legend_label)
        self.legend_label.setVisible(False)

        self.load_matrix(filename)
        self.update_plot("Binary Scatterplot")

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
        model = Model()
        model.readProblem(filename)

        # Prevent upgrade as suggested by https://stackoverflow.com/questions/67828685/how-to-get-constraint-matrix-after-presolving-in-pyscipopt#67832364
        param_dict = {"constraints/linear/upgrade/logicor" : False,
                      "constraints/linear/upgrade/indicator" : False,
                      "constraints/linear/upgrade/knapsack" : False,
                      "constraints/linear/upgrade/setppc" : False,
                      "constraints/linear/upgrade/xor" : False,
                      "constraints/linear/upgrade/varbound" : False
                      }
        model.setParams(param_dict)

        # The following code gives us a list of partial decompositions and writes them locally.

        # These commands suppress output.
        #model.optimize() # Added to test capability of GCG. EDIT: Results in code breaking for small cases, also slow in general because it actually solves the LP, when we just want the decomps.
        model.printVersion()
        model.redirectOutput()
        model.setMinimize()

        # These commands do the actual decomposition. 
        #model.presolve() # Added to test capability of GCG.
        model.detect()
        decomps = model.listDecompositions() 
        print("GCG found {} finished decompositions.".format(len(decomps)))
        #print(decomps) # Currently does not work due to bug in PyGCGOpt.
        model.writeAllDecomps() # WILL NOT WORK WITH MODEL.PRESOLVE ENABLED.

        # These commands write all partial decompositions to disk.
        STEM_OF_FILENAME = Path(filename).stem
        #https://github.com/scipopt/PyGCGOpt/issues/27. writeAllDecomps fails.
        for i in range(len(decomps)):
            d = decomps[i]
            d.isSelected = True
            
            PARTIAL_DECOMP_NAME = PARTIAL_DECOMP_DIR / f"{STEM_OF_FILENAME}_partial_decomposition_{i}.dec"
            
            # Apparently you have to print PARTIAL_DECOMP_NAME before calling model.writeProblem. 
            # Words cannot explain my confusion, but leave both these commands alone.
            print(PARTIAL_DECOMP_NAME)
            model.writeProblem(str(PARTIAL_DECOMP_NAME))

        # All code from mps_matrix_viewer follows regularly from here.

        variables = model.getVars(transformed=True) #transformed=True is necessary due to presolve.
        constraints = model.getConss()
        var_names = [var.name for var in variables]
        var_index = {name: idx for idx, name in enumerate(var_names)}

        # Enumerate over each of the constraints. 
        row_inds, col_inds, data = [], [], []
        for i, cons in enumerate(constraints):
            #try:
            #    row = model.getRowLinear(cons)
            #    print(row)
            #except:
            #    print("")

            terms = model.getValsLinear(cons) 
            for var_name, coef in terms.items():
                j = var_index[var_name]
                if coef != 0:
                    row_inds.append(i)
                    col_inds.append(j)
                    data.append(coef)

        self.A_sparse = csr_matrix((data, (row_inds, col_inds)), shape=(len(constraints), len(variables)))

        #Preserving old properties in case we need them later
        # Rank - requires svds, from scipy.sparse.linalg import svds
        #try:
        #    k = min(self.A_sparse.shape) - 1
        #    _, s, _ = svds(self.A_sparse, k=k)
        #    tol = 1e-10
        #    rank_val = int(np.sum(s > tol))
        #except Exception:
        #    rank_val = "N/A"

        # Additional properties are commented out under props. They may use these below.
        #row_indices, col_indices = self.A_sparse.nonzero()
        #l2_norms = np.linalg.norm(self.A_sparse.toarray(), axis=1)

        props = [
            ("\U0001F4C1 File:", filename),
            ("Shape", str(self.A_sparse.shape)),
            ("Total entries", self.A_sparse.shape[0] * self.A_sparse.shape[1]),
            ("Non-zero entries", self.A_sparse.nnz),
            ("Avg non-zeros per row", np.mean(np.diff(self.A_sparse.indptr))),
            ("Avg non-zeros per column", np.mean(np.diff(self.A_sparse.indptr)) * self.A_sparse.shape[0] / self.A_sparse.shape[1]), #For some reason, np.mean(np.diff(self.A_sparse.T.indptr)) does not work, so we manually rescale by num_rows/num_cols.
            ("Sparsity (%)", 100 * (1 - self.A_sparse.nnz / (self.A_sparse.shape[0] * self.A_sparse.shape[1]))),
            ("Row NNZ Variance", np.var(np.diff(self.A_sparse.indptr)))
            # Following properties are commented out for historical preservation
            #("Column NNZ Variance", np.var(np.diff(csr_matrix(self.A_sparse.T).indptr))),
            #("Min coefficient", np.min(data)),
            #("Max coefficient", np.max(data)),
            #("Mean coefficient", np.mean(data)),
            #("Std coefficient", np.std(data)),
            #("Integer-like (%)", round(100 * np.mean(np.mod(data, 1) == 0), 3)),
            #("Matrix rank", rank_val),
            #("Matrix bandwidth", np.max(np.abs(row_indices - col_indices)) if self.A_sparse.nnz > 0 else 0),
            #("Diag Dominant Rows (%)", "N/A" if self.A_sparse.shape[0] != self.A_sparse.shape[1] else
            #    round(100 * np.mean([
            #        abs(self.A_sparse[i, i]) >= np.sum(np.abs(self.A_sparse[i, :])) - abs(self.A_sparse[i, i])
            #        for i in range(self.A_sparse.shape[0])
            #    ]), 3)),
            #("Avg row L2 norm", float(np.mean(l2_norms))),
            #("Max row L2 norm", float(np.max(l2_norms))),
            #("Zero rows", int(np.sum(row_nnz == 0))),
            #("Zero columns", int(np.sum(col_nnz == 0)))
        ]

        self.stats_table.setRowCount(len(props))
        for i, (k, v) in enumerate(props):
            self.stats_table.setItem(i, 0, QTableWidgetItem(str(k)))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(v)))

    def update_plot(self, type_of_plot):
        self.legend_label.setVisible(False)
        self.heatmap_label.setVisible(False)
        self.chart_view.setVisible(True)

        if self.A_sparse is None:
            return
        # If you are using merged viewer, make sure that these match the items in self.selector and update_selection.
        if type_of_plot == "Binary Scatterplot":
            self.plot_binary_scatterplot()
        elif type_of_plot == "Magnitude Scatterplot":
            self.plot_magnitude_scatterplot()
            self.legend_label.setPixmap(self.generate_gradient_qpixmap(350, 25))
            self.legend_label.setVisible(True)
        elif type_of_plot == "Row-Scaled Heatmap":
            self.plot_row_scaled_heatmap()
        else:
            msgBox = QMessageBox()
            msgBox.setWindowTitle("")
            msgBox.setText("ERROR: The type of scatterplot you have requested is not supported. Please try something else. "
            + "(Devs: This means that you tried calling upload_plot with an option that is currently not implemented.)")
            msgBox.exec()

    def plot_binary_scatterplot(self):
        rows, cols = self.A_sparse.nonzero()
        #If you want to do random sampling of a matrix instead of visualizing the whole thing.
        #indices = list(zip(rows, cols))
        #if len(indices) > 100_000:
        #    indices = random.sample(indices, 100_000)

        chart = QChart()
        chart.setTitle("Binary Scatterplot of Constraint Matrix A (black = non-zero)")
        chart.legend().hide()

        series = QScatterSeries()
        series.setMarkerSize(10)
        series.setColor(QColor("black"))
        for r, c in zip(rows, cols):
            series.append(QPointF(c, r))
        series.clicked.connect(self.on_point_clicked)
        chart.addSeries(series)

        chart.createDefaultAxes()
        axisX, axisY = chart.axes()
        axisX.setTitleText("Variables (Columns)") 
        axisY.setTitleText("Constraints (Rows)")
        axisY.setReverse(True)
        self.chart_view.setChart(chart)

    def plot_magnitude_scatterplot(self):
        rows, cols = self.A_sparse.nonzero()
        vals = self.A_sparse.data
        entries = list(zip(rows, cols, vals))
        if len(entries) > 50_000:
            entries = random.sample(entries, 50_000)

        chart = QChart()
        chart.setTitle("Signed Magnitude Heatmap (Circle = +, Square = −, Blue→Red by Log-Magnitude)")
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
        axisX, axisY = chart.axes()
        axisX.setTitleText("Variables (Columns)") 
        axisY.setTitleText("Constraints (Rows)")
        axisY.setReverse(True)
        self.chart_view.setChart(chart)

    def plot_row_scaled_heatmap(self):
        self.chart_view.setVisible(False)
        self.heatmap_label.setVisible(False)

        A = self.A_sparse.toarray()
        row_scaled = np.zeros_like(A)
        for i, row in enumerate(A):
            nonzero = row[np.nonzero(row)]
            if len(nonzero) == 0:
                continue
            min_val, max_val = np.min(abs(nonzero)), np.max(abs(nonzero))
            if max_val > min_val:
                row_scaled[i] = (abs(row) - min_val) / (max_val - min_val)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(row_scaled, cmap='bwr', aspect='auto', origin='lower')
        ax.set_title("Row-Scaled Heatmap (Normalized per Row)")
        ax.set_xlabel("Variables (Columns)")
        ax.set_ylabel("Constraints (Rows)")
        fig.colorbar(im, orientation='horizontal')

        self.canvas = ClickableFigureCanvas(self, fig, A)
        self.layout.replaceWidget(self.heatmap_label, self.canvas)
        self.heatmap_label.setParent(None)
        self.heatmap_label = self.canvas
        self.heatmap_label.setVisible(True)
        self.canvas.draw()

    def on_point_clicked(self, point):
        row, col = int(point.y()), int(point.x())
        A = self.A_sparse.toarray()
        r_vals, c_vals = A[row], A[:, col]

        def stats(arr):
            return {
                "Min": np.min(arr),
                #"Max": np.max(arr),
                #"Mean": np.mean(arr),
                "Std": np.std(arr),
                #"L2 norm": np.linalg.norm(arr),
                #"Zeros": int(len(arr) - np.count_nonzero(arr)),
                "Non-zeros": int(np.count_nonzero(arr))
            }

        msg = (
            f"<b>Entry:</b> Row {row}, Col {col}<br>" + 
            f"<b>Value:</b> {A[row][col]}<br><br>"
            f"<b>Row Stats:</b><br>" + "".join(f"{k}: {v:.3g}<br>" for k, v in stats(r_vals).items()) +
            f"<br><b>Column Stats:</b><br>" + "".join(f"{k}: {v:.3g}<br>" for k, v in stats(c_vals).items())
        )
        QMessageBox.information(self, "Matrix Entry Info", msg)

    def export_chart_as_image(self):
        print("STILL IN DEBUG MODE.")
        # Get pixmap.
        if self.heatmap_label.isVisible():
            pixmap = self.heatmap_label.grab()
        elif self.chart_view.chart():
            pixmap = self.chart_view.grab()
        else:
            msgBox = QMessageBox()
            msgBox.setWindowTitle("")
            msgBox.setText("❌ No chart to export.")
            msgBox.exec()
            # THIS SHOULD NEVER RUN
            return
        
        # Now save it.
        filename, _ = QFileDialog.getSaveFileName(self, "Save Chart as JPEG", "plot.jpeg", "JPEG Image (*.jpeg *.jpg)")
        if filename:
            if not pixmap.save(filename, "JPEG"):
                msgBox = QMessageBox()
                msgBox.setWindowTitle("")
                msgBox.setText("❌ Failed to save image.")
                msgBox.exec()
            else:
                msgBox = QMessageBox()
                msgBox.setWindowTitle("")
                msgBox.setText(f"✅ Saved: {filename}")
                msgBox.exec()

class FileLoader(QWidget):
    def __init__(self):
        super().__init__()
        self.filename, _ = QFileDialog.getOpenFileName(self, "Open MPS File", "", "MPS Files (*.mps *.MPS);;All Files (*)")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    file_window = FileLoader()
    if file_window.filename:
        window = MatrixViewer(file_window.filename, include_toggle_buttons=True)
        window.show()
    sys.exit(app.exec())