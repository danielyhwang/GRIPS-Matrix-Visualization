import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QMessageBox, QHBoxLayout
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter

from PySide6.QtCharts import QChartView
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from pyscipopt import Model
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, leaves_list


class BlockDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectral Co-Clustering for Block Detection")
        self.layout = QVBoxLayout(self)

        self.A_sparse = None
        self.constraint_types = None
        self.cluster_model = None

        self.load_button = QPushButton("Load MPS File")
        self.load_button.clicked.connect(self.load_and_cluster)
        self.layout.addWidget(self.load_button)

        # Placeholders for charts
        self.canvas = None
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.chart_view.setVisible(False)  # Start hidden
        self.layout.addWidget(self.chart_view)

        # Buttons to toggle views
        button_layout = QHBoxLayout()
        self.binary_button = QPushButton("Show Binary Scatterplot")
        self.clustered_button = QPushButton("Show Clustered Matrix")
        self.binary_button.clicked.connect(self.show_binary_scatter)
        self.clustered_button.clicked.connect(self.show_clustered_matrix)
        button_layout.addWidget(self.binary_button)
        button_layout.addWidget(self.clustered_button)
        self.layout.addLayout(button_layout)

        self.agglomerative_button = QPushButton("Agglomerative Clustering")
        self.agglomerative_button.clicked.connect(self.run_agglomerative_clustering)
        button_layout.addWidget(self.agglomerative_button)

        self.setWindowTitle("Block Matrix Visualizer")


    def load_and_cluster(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open MPS File", "", "MPS Files (*.mps)")
        if not filename:
            return

        self.A_sparse, self.constraint_types = self.extract_binary_matrix(filename)
        if self.A_sparse.nnz == 0:
            QMessageBox.warning(self, "Empty Matrix", "The constraint-variable matrix is empty.")
            return

        binary_matrix = self.A_sparse.toarray()
        # Remove all-zero rows
        nonzero_row_indices = ~np.all(binary_matrix == 0, axis=1)
        binary_matrix = binary_matrix[nonzero_row_indices, :]

        # Remove all-zero columns
        nonzero_col_indices = ~np.all(binary_matrix == 0, axis=0)
        binary_matrix = binary_matrix[:, nonzero_col_indices]

        # Debug: Check final shape
        print("Filtered binary matrix shape:", binary_matrix.shape)

        # Now run spectral co-clustering
        self.cluster_model = SpectralCoclustering(n_clusters=5, random_state=0)
        self.cluster_model.fit(binary_matrix)
       
        # OR: agglomerative clustering
        
       
        QMessageBox.information(self, "File Loaded", f"File loaded and clustered.\nMatrix shape: {self.A_sparse.shape}")

    def extract_binary_matrix(self, filename):
        model = Model()
        model.readProblem(filename)

        variables = model.getVars()
        constraints = model.getConss()

        var_names = [var.name for var in variables]
        var_index = {name: idx for idx, name in enumerate(var_names)}

        row_inds, col_inds = [], []
        constraint_types = []

        for i, cons in enumerate(constraints):
            constype = cons.getConshdlrName()
            constraint_types.append(constype)

            lin_expr = model.getValsLinear(cons)
            for var, coef in lin_expr.items():
                j = var_index[var]
                row_inds.append(i)
                col_inds.append(j)

        shape = (len(constraints), len(variables))
        data = np.ones(len(row_inds))
        A_sparse = csr_matrix((data, (row_inds, col_inds)), shape=shape)

        return A_sparse, constraint_types

    def clear_canvas(self):
        if self.canvas:
            self.layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None
        self.chart_view.setVisible(False)

    def show_binary_scatter(self):
        if self.A_sparse is None:
            QMessageBox.warning(self, "Error", "Load a file first.")
            return

        self.clear_canvas()

        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        rows, cols = self.A_sparse.nonzero()
        ax.scatter(cols, rows, s=10, color='black')
        ax.set_title("Binary Scatterplot (Black = Non-zero)")
        ax.set_xlabel("Variables (Columns)")
        ax.set_ylabel("Constraints (Rows)")
        ax.invert_yaxis()

        self.canvas = FigureCanvas(fig)
        self.layout.addWidget(self.canvas)
        self.canvas.draw()
        self.canvas.setVisible(True)

    def show_clustered_matrix(self):
        if self.A_sparse is None or self.cluster_model is None:
            QMessageBox.warning(self, "Error", "Load and cluster a file first.")
            return

        self.clear_canvas()

        binary_matrix = self.A_sparse.toarray()
        row_order = np.argsort(self.cluster_model.row_labels_)
        col_order = np.argsort(self.cluster_model.column_labels_)
        A_reordered = binary_matrix[np.ix_(row_order, col_order)]

        reordered_types = np.array(self.constraint_types)[row_order]
        type_counts = {}
        for t in reordered_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(A_reordered, aspect='auto', cmap='binary', interpolation='none')
        ax.set_title("Reordered Matrix with Spectral Co-Clustering")
        ax.set_xlabel("Variables (Reordered)")
        ax.set_ylabel("Constraints (Reordered)")

        fig.colorbar(im, ax=ax, label="Nonzero pattern")
        fig.tight_layout()

        self.canvas = FigureCanvas(fig)
        self.layout.addWidget(self.canvas)
        self.canvas.draw()
        self.canvas.setVisible(True)

        QMessageBox.information(self, "Clustering Info",
                                f"5 block(s) detected.\n"
                                f"Matrix shape: {self.A_sparse.shape}\n"
                                f"Row cluster sizes: {np.bincount(self.cluster_model.row_labels_)}\n"
                                f"Col cluster sizes: {np.bincount(self.cluster_model.column_labels_)}\n\n"
                                f"Constraint types (first cluster):\n" +
                                "\n".join(f"{t}: {c}" for t, c in type_counts.items()))


    def run_agglomerative_clustering(self):
        if self.A_sparse is None:
            QMessageBox.warning(self, "Error", "Load a file first.")
            return

        binary_matrix = self.A_sparse.toarray()

        # Remove all-zero rows and columns
        nonzero_row_indices = ~np.all(binary_matrix == 0, axis=1)
        binary_matrix = binary_matrix[nonzero_row_indices, :]
        nonzero_col_indices = ~np.all(binary_matrix == 0, axis=0)
        binary_matrix = binary_matrix[:, nonzero_col_indices]

        if binary_matrix.shape[0] < 2 or binary_matrix.shape[1] < 2:
            QMessageBox.warning(self, "Too Few Data Points", "Matrix too small for clustering.")
            return

        try:
            # Hierarchical clustering for row/column order
            row_linkage = linkage(binary_matrix, method='ward')
            row_order = leaves_list(row_linkage)

            col_linkage = linkage(binary_matrix.T, method='ward')
            col_order = leaves_list(col_linkage)

            # Apply the ordering
            A_reordered = binary_matrix[np.ix_(row_order, col_order)]

            # Plot result
            self.clear_canvas()
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            im = ax.imshow(A_reordered, aspect='auto', cmap='binary', interpolation='none')
            ax.set_title("Agglomerative Clustering: Reordered Matrix")
            ax.set_xlabel("Variables (Reordered)")
            ax.set_ylabel("Constraints (Reordered)")

            fig.colorbar(im, ax=ax, label="Nonzero pattern")
            fig.tight_layout()

            self.canvas = FigureCanvas(fig)
            self.layout.addWidget(self.canvas)
            self.canvas.draw()
            self.canvas.setVisible(True)

            QMessageBox.information(self, "Agglomerative Clustering",
                                    f"Reordering complete using hierarchical clustering.\n"
                                    f"Reordered matrix shape: {A_reordered.shape}")

        except Exception as e:
            QMessageBox.critical(self, "Clustering Error", str(e))



if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = BlockDetector()
    viewer.show()
    sys.exit(app.exec())
