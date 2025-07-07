import sys
import networkx as nx
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QComboBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QLabel, QHeaderView
)
from PySide6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from pyscipopt import Model
from scipy.sparse import csr_matrix

class MIPGraphStatsApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MIP Graph Statistics Viewer")
        self.setGeometry(100, 100, 1200, 800)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Top buttons
        self.buttons_layout = QHBoxLayout()
        self.load_button = QPushButton("Load MPS File")
        self.load_button.clicked.connect(self.load_mps_file)

        self.graph_type_combo = QComboBox()
        self.graph_type_combo.addItems(["Primal graph", "Dual graph", "Incidence graph"])
        self.graph_type_combo.currentTextChanged.connect(self.analyze_graph)

        self.buttons_layout.addWidget(self.load_button)
        self.buttons_layout.addWidget(self.graph_type_combo)
        self.layout.addLayout(self.buttons_layout)

        # Info display
        self.info_box = QTextEdit()
        self.info_box.setFont(QFont("Courier", 10))
        self.info_box.setReadOnly(True)
        self.layout.addWidget(self.info_box)

        # === Stats Tables ===
        self.stats_table = QTableWidget()
        self.stats_table.setMinimumWidth(400)
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.matrix_stats_table = QTableWidget()
        self.matrix_stats_table.setMinimumWidth(400)
        self.matrix_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Labels
        graph_label = QLabel("Graph Properties")
        matrix_label = QLabel("Matrix Properties")

        graph_stats_layout = QVBoxLayout()
        graph_stats_layout.addWidget(graph_label)
        graph_stats_layout.addWidget(self.stats_table)

        matrix_stats_layout = QVBoxLayout()
        matrix_stats_layout.addWidget(matrix_label)
        matrix_stats_layout.addWidget(self.matrix_stats_table)

        tables_layout = QHBoxLayout()
        tables_layout.addLayout(graph_stats_layout)
        tables_layout.addLayout(matrix_stats_layout)
        self.layout.addLayout(tables_layout)

        # === Graph Canvas ===
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

    def load_mps_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open MPS File", "", "MPS Files (*.mps *.MPS)")
        if not filename:
            return

        model = Model()
        model.readProblem(filename)

        self.variables = model.getVars()
        self.constraints = model.getConss()
        self.model = model

        self.var_names = [var.name for var in self.variables]
        self.con_names = [con.name for con in self.constraints]

        self.info_box.setText(f"Loaded MPS file: {filename}\nVars: {len(self.var_names)}, Constraints: {len(self.con_names)}")

        self.analyze_graph()

    def analyze_graph(self):
        gtype = self.graph_type_combo.currentText()
        G = nx.Graph()

        if gtype == "Primal graph":
            G.add_nodes_from(self.var_names)
            for cons in self.constraints:
                terms = self.model.getValsLinear(cons)
                involved = [v for v in terms if terms[v] != 0]
                for i in range(len(involved)):
                    for j in range(i + 1, len(involved)):
                        G.add_edge(involved[i], involved[j])

        elif gtype == "Dual graph":
            G.add_nodes_from(self.con_names)
            var_map = {v: [] for v in self.var_names}
            for cons in self.constraints:
                terms = self.model.getValsLinear(cons)
                for v in terms:
                    var_map[v].append(cons.name)
            for v, clist in var_map.items():
                for i in range(len(clist)):
                    for j in range(i + 1, len(clist)):
                        G.add_edge(clist[i], clist[j])

        elif gtype == "Incidence graph":
            G.add_nodes_from(self.var_names + self.con_names)
            for cons in self.constraints:
                terms = self.model.getValsLinear(cons)
                for v in terms:
                    if terms[v] != 0:
                        G.add_edge(cons.name, v)

        self.display_graph_stats(G)
        self.display_matrix_stats()

    def display_matrix_stats(self):
        row_inds, col_inds, data = [], [], []

        for i, cons in enumerate(self.constraints):
            terms = self.model.getValsLinear(cons)
            for var, coef in terms.items():
                if coef != 0:
                    row_inds.append(i)
                    col_inds.append(self.var_names.index(var))
                    data.append(coef)

        A = csr_matrix((data, (row_inds, col_inds)), shape=(len(self.constraints), len(self.variables)))

        stats = [
            ("Shape", A.shape),
            ("Sparsity (%)", round(100 * (1.0 - A.nnz / (A.shape[0] * A.shape[1])), 4)),
            ("Row NNZ Variance", np.var(A.getnnz(axis=1))),
            ("Column NNZ Variance", np.var(A.getnnz(axis=0))),
            ("Min coefficient", np.min(data)),
            ("Max coefficient", np.max(data)),
            ("Mean coefficient", np.mean(data)),
            ("Std coefficient", np.std(data)),
            ("Integer-like (%)", round(100 * np.mean(np.all(np.mod(data, 1) == 0)), 4)),
            ("Matrix rank", np.linalg.matrix_rank(A.toarray())),
            ("Diag Dominant Rows (%)", "N/A" if A.shape[0] != A.shape[1] else round(
                100 * np.mean([
                    abs(A[i, i]) >= np.sum(np.abs(A[i, :])) - abs(A[i, i])
                    for i in range(A.shape[0])
                ]), 4)),
            ("Avg row L2 norm", np.mean(np.linalg.norm(A.toarray(), axis=1))),
            ("Max row L2 norm", np.max(np.linalg.norm(A.toarray(), axis=1))),
            ("Zero rows", np.sum(A.getnnz(axis=1) == 0)),
            ("Zero columns", np.sum(A.getnnz(axis=0) == 0))
        ]

        self.matrix_stats_table.setRowCount(len(stats))
        self.matrix_stats_table.setColumnCount(2)
        self.matrix_stats_table.setHorizontalHeaderLabels(["Property", "Value"])
        for i, (k, v) in enumerate(stats):
            self.matrix_stats_table.setItem(i, 0, QTableWidgetItem(str(k)))
            self.matrix_stats_table.setItem(i, 1, QTableWidgetItem(str(v)))
        self.matrix_stats_table.resizeColumnsToContents()

    def display_graph_stats(self, G):
        stats = []

        degrees = [deg for _, deg in G.degree()]
        if degrees:
            stats.append(("Min Degree", min(degrees)))
            stats.append(("Max Degree", max(degrees)))
            stats.append(("Average Degree", round(sum(degrees)/len(degrees), 3)))

        if nx.is_connected(G) if isinstance(G, nx.Graph) else False:
            stats.append(("Connected Components", 1))
        else:
            comps = list(nx.connected_components(G))
            stats.append(("Connected Components", len(comps)))
            stats.append(("Largest Component Size", max(len(c) for c in comps)))

        if isinstance(G, nx.Graph):
            clustering = nx.average_clustering(G)
            stats.append(("Clustering Coefficient", round(clustering, 4)))
            cliques = list(nx.find_cliques(G))
            stats.append(("Number of Cliques", len(cliques)))
            stats.append(("Max Clique Size", max(len(c) for c in cliques)))

        self.stats_table.setRowCount(len(stats))
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Property", "Value"])
        for i, (k, v) in enumerate(stats):
            self.stats_table.setItem(i, 0, QTableWidgetItem(str(k)))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(v)))
        self.stats_table.resizeColumnsToContents()

        # === Draw Graph ===
        self.ax.clear()
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=self.ax, with_labels=True, node_color='skyblue', edge_color='gray', font_size=7)
        self.canvas.draw()

if __name__ == '__main__':
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    window = MIPGraphStatsApp()
    window.show()
    sys.exit(app.exec())

