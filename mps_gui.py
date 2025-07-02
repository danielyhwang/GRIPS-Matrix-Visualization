import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QFileDialog
from pyscipopt import Model
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MPSLoaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPS Loader and Sparse Matrix Viewer")
        self.resize(800, 600)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.load_button = QPushButton("Load MPS File")
        self.load_button.clicked.connect(self.load_mps_file)
        self.layout.addWidget(self.load_button)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.layout.addWidget(self.text_area)

        # Setup matplotlib Figure and Canvas for scatter plot
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

    def load_mps_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open MPS File", "", "MPS Files (*.mps *.MPS);;All Files (*)")
        if not filename:
            return
         
        self.text_area.clear()
        self.ax.clear()

        # Load model and build sparse matrix
        model = Model()
        model.readProblem(filename)

        variables = model.getVars()
        constraints = model.getConss()

        var_names = [var.name for var in variables]
        var_index = {name: idx for idx, name in enumerate(var_names)}

        n_vars = len(variables)
        n_cons = len(constraints)

        row_inds = []
        col_inds = []
        data = []

        b = []
        senses = []
        c = np.zeros(n_vars)

        # Objective coefficients
        for j, var in enumerate(variables):
            c[j] = var.getObj()

        # Build sparse matrix
        for i, cons in enumerate(constraints):
            terms = model.getValsLinear(cons)
            for var_name, coef in terms.items():
                j = var_index[var_name]
                row_inds.append(i)
                col_inds.append(j)
                data.append(coef)

            lhs = model.getLhs(cons)
            rhs = model.getRhs(cons)

            if np.isclose(lhs, rhs):
                senses.append("=")
                b.append(rhs)
            elif np.isfinite(rhs):
                senses.append("<=")
                b.append(rhs)
            else:
                senses.append(">=")
                b.append(lhs)

        A_sparse = csr_matrix((data, (row_inds, col_inds)), shape=(n_cons, n_vars))

        # Display basic info
        info_text = (
            f"Loaded MPS file: {filename}\n"
            f"Number of variables: {n_vars}\n"
            f"Number of constraints: {n_cons}\n"
            f"Sparse matrix shape: {A_sparse.shape}\n"
            f"Number of nonzeros in A: {A_sparse.nnz}\n"
        )
        self.text_area.setPlainText(info_text)

        # Scatter plot: visualize nonzero pattern

        # Clear previous plot
        self.ax.clear()

        # Get row, col indices of nonzeros
        rows, cols = A_sparse.nonzero()

        # Scatter plot of nonzero entries (cols on x-axis, rows on y-axis)
        self.ax.scatter(cols, rows, s=5, marker='.', color='blue')

        self.ax.set_xlabel('Variables (Columns)')
        self.ax.set_ylabel('Constraints (Rows)')
        self.ax.set_title('Nonzero pattern of constraint matrix A')

        # Invert y-axis so row 0 is at top (like matrix style)
        self.ax.invert_yaxis()

        # Show grid
        self.ax.grid(True)

        # Tight layout to fit labels nicely
        self.fig.tight_layout()

        # Refresh canvas
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MPSLoaderApp()
    window.show()
    sys.exit(app.exec())

