from __future__ import annotations

import math, sys
from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QTextEdit, QFileDialog
from pyscipopt import Model
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# NetworkX code copied from https://doc.qt.io/qtforpython-6/examples/example_external_networkx.html
# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause
from PySide6.QtCore import (QEasingCurve, QLineF,
                            QParallelAnimationGroup, QPointF,
                            QPropertyAnimation, QRectF, Qt)
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QApplication, QComboBox, QGraphicsItem,
                               QGraphicsObject, QGraphicsScene, QGraphicsView,
                               QStyleOptionGraphicsItem, QVBoxLayout, QWidget)
import networkx as nx


class Node(QGraphicsObject):

    """A QGraphicsItem representing node in a graph"""

    def __init__(self, name: str, parent=None):
        """Node constructor

        Args:
            name (str): Node label
        """
        super().__init__(parent)
        self._name = name
        self._edges = []
        self._color = "#5AD469"
        self._radius = 30
        self._rect = QRectF(0, 0, self._radius * 2, self._radius * 2)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

    def boundingRect(self) -> QRectF:
        """Override from QGraphicsItem

        Returns:
            QRect: Return node bounding rect
        """
        return self._rect

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget = None):
        """Override from QGraphicsItem

        Draw node

        Args:
            painter (QPainter)
            option (QStyleOptionGraphicsItem)
        """
        painter.setRenderHints(QPainter.RenderHint.Antialiasing)
        painter.setPen(
            QPen(
                QColor(self._color).darker(),
                2,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        painter.setBrush(QBrush(QColor(self._color)))
        painter.drawEllipse(self.boundingRect())
        painter.setPen(QPen(QColor("white")))
        painter.drawText(self.boundingRect(), 0, str(self._name)) 

    def add_edge(self, edge):
        """Add an edge to this node

        Args:
            edge (Edge)
        """
        self._edges.append(edge)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):
        """Override from QGraphicsItem

        Args:
            change (QGraphicsItem.GraphicsItemChange)
            value (Any)

        Returns:
            Any
        """
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for edge in self._edges:
                edge.adjust()

        return super().itemChange(change, value)


class Edge(QGraphicsItem):
    def __init__(self, source: Node, dest: Node, parent: QGraphicsItem = None):
        """Edge constructor

        Args:
            source (Node): source node
            dest (Node): destination node
        """
        super().__init__(parent)
        self._source = source
        self._dest = dest

        self._tickness = 2
        self._color = "#2BB53C"
        self._arrow_size = 20

        self._source.add_edge(self)
        self._dest.add_edge(self)

        self._line = QLineF()
        self.setZValue(-1)
        self.adjust()

    def boundingRect(self) -> QRectF:
        """Override from QGraphicsItem

        Returns:
            QRect: Return node bounding rect
        """
        return (
            QRectF(self._line.p1(), self._line.p2())
            .normalized()
            .adjusted(
                -self._tickness - self._arrow_size,
                -self._tickness - self._arrow_size,
                self._tickness + self._arrow_size,
                self._tickness + self._arrow_size,
            )
        )

    def adjust(self):
        """
        Update edge position from source and destination node.
        This method is called from Node::itemChange
        """
        self.prepareGeometryChange()
        self._line = QLineF(
            self._source.pos() + self._source.boundingRect().center(),
            self._dest.pos() + self._dest.boundingRect().center(),
        )

    def _draw_arrow(self, painter: QPainter, start: QPointF, end: QPointF):
        """Draw arrow from start point to end point.

        Args:
            painter (QPainter)
            start (QPointF): start position
            end (QPointF): end position
        """
        painter.setBrush(QBrush(self._color))

        line = QLineF(end, start)

        angle = math.atan2(-line.dy(), line.dx())
        arrow_p1 = line.p1() + QPointF(
            math.sin(angle + math.pi / 3) * self._arrow_size,
            math.cos(angle + math.pi / 3) * self._arrow_size,
        )
        arrow_p2 = line.p1() + QPointF(
            math.sin(angle + math.pi - math.pi / 3) * self._arrow_size,
            math.cos(angle + math.pi - math.pi / 3) * self._arrow_size,
        )

        arrow_head = QPolygonF()
        arrow_head.clear()
        arrow_head.append(line.p1())
        arrow_head.append(arrow_p1)
        arrow_head.append(arrow_p2)
        painter.drawLine(line)
        painter.drawPolygon(arrow_head)

    def _arrow_target(self) -> QPointF:
        """Calculate the position of the arrow taking into account the size of the destination node

        Returns:
            QPointF
        """
        target = self._line.p1()
        center = self._line.p2()
        radius = self._dest._radius
        vector = target - center
        length = math.sqrt(vector.x() ** 2 + vector.y() ** 2)
        if length == 0:
            return target
        normal = vector / length
        target = QPointF(center.x() + (normal.x() * radius), center.y() + (normal.y() * radius))

        return target

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None):
        """Override from QGraphicsItem

        Draw Edge. This method is called from Edge.adjust()

        Args:
            painter (QPainter)
            option (QStyleOptionGraphicsItem)
        """

        if self._source and self._dest:
            painter.setRenderHints(QPainter.RenderHint.Antialiasing)

            painter.setPen(
                QPen(
                    QColor(self._color),
                    self._tickness,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            painter.drawLine(self._line)
            self._draw_arrow(painter, self._line.p1(), self._arrow_target())
            self._arrow_target()


class GraphView(QGraphicsView):
    def __init__(self, graph: nx.DiGraph, parent=None):
        """GraphView constructor

        This widget can display a directed graph

        Args:
            graph (nx.DiGraph): a networkx directed graph
        """
        super().__init__()
        self._graph = graph
        self._scene = QGraphicsScene()
        self.setScene(self._scene)

        # Used to add space between nodes
        self._graph_scale = 200

        # Map node name to Node object {str=>Node}
        self._nodes_map = {}

        # List of networkx layout function
        self._nx_layout = {
            "circular": nx.circular_layout,
            "planar": nx.planar_layout,
            "random": nx.random_layout,
            "shell_layout": nx.shell_layout,
            "kamada_kawai_layout": nx.kamada_kawai_layout,
            "spring_layout": nx.spring_layout,
            "spiral_layout": nx.spiral_layout,
        }

        self._load_graph()
        self.set_nx_layout("circular")

    def get_nx_layouts(self) -> list:
        """Return all layout names

        Returns:
            list: layout name (str)
        """
        return self._nx_layout.keys()

    def set_nx_layout(self, name: str):
        """Set networkx layout and start animation

        Args:
            name (str): Layout name
        """
        if name in self._nx_layout:
            self._nx_layout_function = self._nx_layout[name]

            # Compute node position from layout function
            positions = self._nx_layout_function(self._graph)

            # Change position of all nodes using an animation
            self.animations = QParallelAnimationGroup()
            for node, pos in positions.items():
                x, y = pos
                x *= self._graph_scale
                y *= self._graph_scale
                item = self._nodes_map[node]

                animation = QPropertyAnimation(item, b"pos")
                animation.setDuration(1000)
                animation.setEndValue(QPointF(x, y))
                animation.setEasingCurve(QEasingCurve.Type.OutExpo)
                self.animations.addAnimation(animation)

            self.animations.start()

    def _load_graph(self):
        """Load graph into QGraphicsScene using Node class and Edge class"""

        self.scene().clear()
        self._nodes_map.clear()

        # Add nodes
        for node in self._graph:
            item = Node(node)
            self.scene().addItem(item)
            self._nodes_map[node] = item

        # Add edges
        for a, b in self._graph.edges:
            source = self._nodes_map[a]
            dest = self._nodes_map[b]
            self.scene().addItem(Edge(source, dest))

class MPSLoaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPS Loader and Sparse Matrix Viewer")
        self.resize(800, 600)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.load_button = QPushButton("Load MPS File")
        self.load_button.clicked.connect(self.load_mps_file)
        self.graph_choice_combo = QComboBox()
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.load_button)
        self.hbox.addWidget(self.graph_choice_combo)
        self.layout.addLayout(self.hbox)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.layout.addWidget(self.text_area)

        self.view = GraphView(nx.complete_graph(5))
        self.choice_combo = QComboBox()
        self.choice_combo.addItems(self.view.get_nx_layouts())
        self.choice_combo.currentTextChanged.connect(self.view.set_nx_layout)
        self.layout.addWidget(self.choice_combo)
        self.layout.addWidget(self.view)

    def load_mps_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open MPS File", "", "MPS Files (*.mps *.MPS);;All Files (*)")
        if not filename:
            return

        # Load model and build sparse matrix
        self.model = Model()
        self.model.readProblem(filename)

        self.variables = self.model.getVars()
        self.constraints = self.model.getConss()

        self.var_names = [var.name for var in self.variables]
        self.con_names = [con.name for con in self.constraints]
        self.var_index = {name: idx for idx, name in enumerate(self.var_names)}

        self.n_vars = len(self.variables)
        self.n_cons = len(self.constraints)

         # Load in details into text.
        self.text_area.clear()

        # Display basic info and store it in text area.
        info_text = (
            f"Loaded MPS file: {filename}\n"
            f"Number of variables: {self.n_vars}\n"
            f"Number of constraints: {self.n_cons}\n"
        )
        self.text_area.setPlainText(info_text)

        # Build primal graph. The set of vertices is the set of columns c_k, the set of edges is the set (c_k, c_l) such that there exists 
        # a row such that A[r][c_k] != 0 and A[r][c_l] != 0
        primal_graph = nx.Graph()
        # Add vertices
        primal_graph.add_nodes_from(self.var_names)

        for i, cons in enumerate(self.constraints):
            terms = self.model.getValsLinear(cons)
            nonzero_columns_this_row = []
            for var_name, coef in terms.items():
                # j = var_index[var_name] # we get A[i][j] = coef here
                if coef != 0:
                    nonzero_columns_this_row.append(var_name)
            # For all nonzero elements this row, connect them together
            if len(nonzero_columns_this_row) >= 2:
                for k in nonzero_columns_this_row:
                    for l in nonzero_columns_this_row:
                        if k != l:
                            primal_graph.add_edge(k, l)

        # Reset self.view and self.choice_combo before reupdating it based on graph of choice.
        #For some reason, calling self.view = GraphView(primal_graph) does not work. Had to do this instead.
        self.view._graph = primal_graph
        self.view._scene = QGraphicsScene()
        self.view.setScene(self.view._scene)
        self.view._load_graph()
        self.view.set_nx_layout("circular")

        self.graph_choice_combo.clear()
        self.graph_choice_combo.addItems(["Primal graph", "Dual graph", "Incidence graph"]) #Primal graph must come first
        self.graph_choice_combo.currentTextChanged.connect(self.load_graph_type)

    def load_graph_type(self):
        updated_graph = nx.Graph()
        
        if self.graph_choice_combo.currentText() == "Primal graph":
            # Build primal graph. The set of vertices is the set of columns c_k, the set of edges is the set (c_k, c_l) such that there exists 
            # a row such that A[r][c_k] != 0 and A[r][c_l] != 0
            # Add vertices
            updated_graph.add_nodes_from(self.var_names)
            # Iterate through and add edges.
            for i, cons in enumerate(self.constraints):
                terms = self.model.getValsLinear(cons)
                nonzero_columns_this_row = []
                for var_name, coef in terms.items():
                    # j = var_index[var_name] # we get A[i][j] = coef here
                    if coef != 0:
                        nonzero_columns_this_row.append(var_name)
                # For all nonzero elements this row, connect them together
                if len(nonzero_columns_this_row) >= 2:
                    for k in nonzero_columns_this_row:
                        for l in nonzero_columns_this_row:
                            if k != l:
                                updated_graph.add_edge(k, l)

        elif self.graph_choice_combo.currentText() == "Dual graph":
            print("Warning: Because PySciPoPT does not support searching over each column, we manually reconstruct a binary version of the transpose matrix, this may lead to large runtime")
            # Add vertices
            updated_graph.add_nodes_from(self.con_names)
            
            # What we do is store a dictionary with keys matching variables and values being the list of constraints with nonzero coefficients for each variable. 
            manual_matrix_dictionary = {}
            for var_name in self.var_names:
                manual_matrix_dictionary[var_name] = []

            # Because you cannot iterate over columns via PySciPoPT, you have to store data
            for i, cons in enumerate(self.constraints):
                terms = self.model.getValsLinear(cons)
                for var_name, coef in terms.items():
                    # j = var_index[var_name] # we get A[i][j] = coef here
                    if coef != 0:
                        manual_matrix_dictionary[var_name].append(cons.name)

            # For all nonzero constraints for each variable, connect them together
            for var_name in manual_matrix_dictionary.keys():
                nonzero_constraints_this_column = manual_matrix_dictionary[var_name]
                if len(nonzero_constraints_this_column) >= 2:
                    for k in nonzero_constraints_this_column:
                        for l in nonzero_constraints_this_column:
                            if k != l:
                                updated_graph.add_edge(k, l)

        elif self.graph_choice_combo.currentText() == "Incidence graph":
            # Build incidence graph = The set of vertices is the set of rows r_k and cols c_l, the set of edges connects r_k and c_l if A[r_k][c_l] != 0.
            updated_graph.add_nodes_from(self.var_names)
            updated_graph.add_nodes_from(self.con_names)
            for i, cons in enumerate(self.constraints):
                terms = self.model.getValsLinear(cons)
                for var_name, coef in terms.items():
                    #j = self.var_index[var_name] 
                    # we get A[i][j] = coef here
                    if coef != 0:
                        updated_graph.add_edge(cons.name, var_name)
        else:
            raise ValueError("Items in combo box do not match those in load_graph_type")

        # Reset self.view and self.choice_combo before reupdating it based on graph of choice.
        #For some reason, calling self.view = GraphView(primal_graph) does not work. Had to do this instead.
        self.view._graph = updated_graph
        self.view._scene = QGraphicsScene()
        self.view.setScene(self.view._scene)
        self.view._load_graph()
        self.view.set_nx_layout("circular")

        # AFTER LUNCH: Dual graph.
        # BONUS: Integrate Sonali's code. TREEWIDTH, TREEDEPTH, STRUCTURE GRAPH, CLIQUES, DEGREE DISTRIBUTIONS, CLUSTERING COEFFICIENTS, CONNECTED COMPONENTS


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MPSLoaderApp()
    window.show()
    sys.exit(app.exec())