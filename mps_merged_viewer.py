import sys
import numpy as np
import csv
import random
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QFileDialog,
    QComboBox, QHBoxLayout, QToolTip, QSizePolicy, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QStackedWidget
)
from PySide6.QtCharts import QChart, QChartView, QScatterSeries
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainter, QColor, QCursor
from pyscipopt import Model
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Import viewers from other files
from mps_matrix_viewer import MatrixViewer
from mps_graph_viewer import GraphViewer


class MergedMPSViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive MPS Viewer with Matrix Stats + QtCharts + Graph Viewer")
        self.resize(1000, 800)

        self.A_sparse = None
        self.last_plot_data = []

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Add toolbar
        control_layout = QHBoxLayout()
        self.load_button = QPushButton("Load MPS File")
        self.load_button.clicked.connect(self.load_mps_file)
        control_layout.addWidget(self.load_button)

        self.selector = QComboBox()
        self.selector.addItems(["Primal graph", "Dual graph", "Incidence graph", "Binary Scatterplot", "Magnitude Scatterplot"])
        self.selector.currentIndexChanged.connect(self.update_selection)
        control_layout.addWidget(self.selector)

        self.export_button = QPushButton("Export Matrix to CSV")
        self.export_button.clicked.connect(self.export_matrix_to_csv)
        control_layout.addWidget(self.export_button)
        self.layout.addLayout(control_layout)

        # Add stacked widget
        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget)

    def load_mps_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open MPS File", "", "MPS Files (*.mps *.MPS);;All Files (*)")
        if not filename:
            return
        
        # Initialize stacked widget with matrix view.
        self.matrix_viewer = MatrixViewer(filename)
        self.graph_viewer = GraphViewer(filename)
        self.stacked_widget.addWidget(self.matrix_viewer)
        self.stacked_widget.addWidget(self.graph_viewer)
        self.stacked_widget.setCurrentWidget(self.matrix_viewer)
        self.selector.setCurrentText("Binary Scatterplot")

    def update_selection(self):
        if self.selector.currentText() == "Primal graph":
            if self.stacked_widget.currentWidget() != self.graph_viewer:
                self.stacked_widget.setCurrentWidget(self.graph_viewer)
            self.graph_viewer.load_graph_type("Primal graph")
        elif self.selector.currentText() == "Dual graph":
            if self.stacked_widget.currentWidget() != self.graph_viewer:
                self.stacked_widget.setCurrentWidget(self.graph_viewer)
            self.graph_viewer.load_graph_type("Dual graph")
        elif self.selector.currentText() == "Incidence graph":
            if self.stacked_widget.currentWidget() != self.graph_viewer:
                self.stacked_widget.setCurrentWidget(self.graph_viewer)
            self.graph_viewer.load_graph_type("Incidence graph")
        elif self.selector.currentText() == "Binary Scatterplot":
            if self.stacked_widget.currentWidget() != self.matrix_viewer:
                self.stacked_widget.setCurrentWidget(self.matrix_viewer)
            self.matrix_viewer.update_plot("Binary Scatterplot")
        elif self.selector.currentText() == "Magnitude Scatterplot":
            if self.stacked_widget.currentWidget() != self.matrix_viewer:
                self.stacked_widget.setCurrentWidget(self.matrix_viewer)
            self.matrix_viewer.update_plot("Magnitude Scatterplot")
        
        else:
            msgBox = QMessageBox()
            msgBox.setWindowTitle("")
            msgBox.setText("ERROR: The selection you have requested is not supported. Please try something else. "
            + "(Devs: This means that you tried calling upload_item with an option that is currently not implemented.)")
            msgBox.exec()
        
        return None

    def export_matrix_to_csv(self):
        if not self.last_plot_data:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Matrix Data", "", "CSV Files (*.csv)")
        if filename:
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Row", "Column", "Value"])
                writer.writerows(self.last_plot_data)
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MergedMPSViewer()
    window.show()
    sys.exit(app.exec())

