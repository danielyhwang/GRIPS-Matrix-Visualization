from __future__ import annotations

import math, sys
from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QTextEdit, QFileDialog, QMessageBox
from pyscipopt import Model
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Zoom/Pan code copied from https://stackoverflow.com/questions/35508711/how-to-enable-pan-and-zoom-in-a-qgraphicsview
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
        self._color = "#6495ED"
        self._radius = 30 #ADJUST THIS VARIABLE TO CHANGE NODE SIZE.
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
        painter.setPen(QPen(QColor("black"))) #MODIFY TO CHANGE TEXT COLOR
        painter.drawText(self.boundingRect(), Qt.AlignmentFlag.AlignCenter, str(self._name)) 

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
        self._color = "#898989"
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
            #self._draw_arrow(painter, self._line.p1(), self._arrow_target()) #our edges are undirected
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

        # Enable dragging
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        """
        Following code sets up pan/zoom functionality.
        """

        #self._zoom = 0
        #self._pinned = False
        #self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        #self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        #self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        #self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        #self.setBackgroundBrush(QBrush(QColor(30, 30, 30))) # This code sets the color.
        #self.setFrameShape(QtWidgets.Qframe.Shape.NoFrame)

        # Pause here, continue stuff. Keep working on graph, 
        # move onto row and column clickable functionality.

        """
        Following code creates graph
        """

        # Used to add space between nodes
        self._graph_scale = 500 # MODIFY THIS TO CHANGE SPACING

        # Map node name to Node object {str=>Node}
        self._nodes_map = {}

        # List of networkx layout function
        # To make things consistent, please keep circular as the first layout.
        # Tried looking at graphviz, but doesn't like it is supported by Python.
        self._nx_layout = {
            "circular": nx.circular_layout,
            "planar": nx.planar_layout,
            "random": nx.random_layout,
            "shell": nx.shell_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "spring": nx.spring_layout,
            "spiral": nx.spiral_layout,
            "spectral": nx.spectral_layout,
            "bipartite": nx.bipartite_layout
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

            # Try computing node position from layout function, if not possible, try something else.
            try:
                # Compute node position from layout function
                positions = self._nx_layout_function(self._graph)
            except:
                msgBox = QMessageBox()
                msgBox.setWindowTitle("")
                msgBox.setText("Error: Your layout is not possible for this graph! Try choosing another layout.")
                msgBox.exec()
                return None

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

        self._scene.clear()
        self._nodes_map.clear()

        # Add nodes
        for node in self._graph:
            item = Node(node)
            self._scene.addItem(item)
            self._nodes_map[node] = item

        # Add edges
        for a, b in self._graph.edges:
            source = self._nodes_map[a]
            dest = self._nodes_map[b]
            self._scene.addItem(Edge(source, dest))

        # print(self.size())
        # print(self.viewport().size())
        # Remaining things to do
        # self.setScene(self._scene)
        # self.fitInView(self.sceneRect())
        # Rescale the QGraphicsView to fit whole scene in view.
        # Enable zoom functionality

# The following code contains a self-contained graph visualization widget, which allows it
# to be imported as a module in our mps_merged_viewer.py file. Note that it is initialized by feeding
# in a filename.
class GraphViewer(QWidget):
    """
    Initializes a GraphView and TextBox where we can store important information.
    Input: filename - name of a file passed in from QFileDialog.
    """
    def __init__(self, filename, include_toggle_buttons = True):
        self.filename = filename

        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        #Create a text box where you can store information.

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.layout.addWidget(self.text_area)

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

        # Build primal graph. The set of vertices is the set of columns c_k, the set of edges is the set (c_k, c_l) such that there exists 
        # a row such that A[r][c_k] != 0 and A[r][c_l] != 0
        primal_graph = nx.Graph()
        # Add vertices
        primal_graph.add_nodes_from(self.var_names)
        # Add edges
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

        # Create GraphView based on primal graph.
        self.view = GraphView(primal_graph)

        # Create GraphCanvas using matplotlib. Old code kept for testing purposes.
        # self.figure, self.ax = plt.subplots()
        # self.canvas = FigureCanvas(self.figure)
        # self.layout.addWidget(self.canvas)

         # Calculate basic statistics and store it in text.
        self.text_area.clear()
        degrees = [deg for _, deg in primal_graph.degree()]
        avg_degree = np.mean(degrees)
        max_degree = np.max(degrees)
        clustering = nx.clustering(primal_graph)
        clustering_avg = np.mean(list(clustering.values()))
        components = list(nx.connected_components(primal_graph))
        num_components = len(components)
        component_sizes = [len(c) for c in components]
        largest_component = max(component_sizes)
        
        try:
            from networkx.algorithms.approximation.treewidth import treewidth_min_fill_in
            treewidth_calculated, _ = treewidth_min_fill_in(primal_graph)
        except:
            treewidth_calculated = "N/A"
            print("⚠️ Treewidth estimation not available — requires `networkx >= 2.6`.")
            print("🧠 Explanation: Treewidth is NP-hard to compute exactly, so approximation is used.\n")

        # Set text based on statistics above.
        info_text = (
            f"Loaded MPS file: {self.filename}\n"
            f"Number of variables: {self.n_vars}\n"
            f"Number of constraints: {self.n_cons}\n"
            f"Average node degree: {avg_degree:.2f}\n"
            f"Maximum node degree: {max_degree}\n"
            f"Average clustering coefficient: {clustering_avg:.4f}\n"
            f"Number of connected components: {num_components}\n"
            f"Largest component size: {largest_component}\n"
            f"Treewidth (approxmimate): {treewidth_calculated}"
        )
        self.text_area.setPlainText(info_text)

        # Create a choice combo box where you can toggle between different layouts (based off layouts of self view).

        self.choice_combo = QComboBox()
        self.choice_combo.addItems(self.view.get_nx_layouts())
        self.choice_combo.currentTextChanged.connect(self.view.set_nx_layout)
        self.layout.addWidget(self.choice_combo)
        self.layout.addWidget(self.view)
        
        # Set current layout to circular.
        self.choice_combo.setCurrentText("circular")
        #self.view.set_nx_layout("circular") # Not necessary?

        # Create a button where you can reset view AND another button where you can export current graph to JPEG.
        bottom_layout = QHBoxLayout()
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self.reset_graph_view)
        bottom_layout.addWidget(self.reset_view_button)
        bottom_layout.addStretch()

        # Include toggle buttons, only if include_toggle_buttons is on.
        self.primal_graph_test_button = QPushButton("Primal Graph")
        self.dual_graph_test_button = QPushButton("Dual Graph")
        self.incidence_graph_test_button = QPushButton("Incidence Graph")
        # Make sure that these match the options in self.selector under the mps_merged_viewer.py file.
        self.primal_graph_test_button.clicked.connect(lambda: self.load_graph_type("Primal graph"))
        self.dual_graph_test_button.clicked.connect(lambda: self.load_graph_type("Dual graph"))
        self.incidence_graph_test_button.clicked.connect(lambda: self.load_graph_type("Incidence graph"))
        bottom_layout.addWidget(self.primal_graph_test_button)
        bottom_layout.addWidget(self.dual_graph_test_button)
        bottom_layout.addWidget(self.incidence_graph_test_button)
        bottom_layout.addStretch()

        self.export_image_button = QPushButton("Export Graph to JPEG")
        self.export_image_button.clicked.connect(self.export_graph_as_image)
        bottom_layout.addWidget(self.export_image_button)

        self.layout.addLayout(bottom_layout)

    def load_graph_type(self, type_of_graph):
        updated_graph = nx.Graph()

        # If you are using merged viewer, make sure that these match the items in self.selector and update_selection.
        if type_of_graph == "Primal graph":
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

        elif type_of_graph == "Dual graph":
            msgBox = QMessageBox()
            msgBox.setWindowTitle("")
            msgBox.setText("Warning: Because PySciPoPT does not support searching over each column, we manually reconstruct a binary version of the transpose matrix, this may lead to large runtime")
            msgBox.exec()
            
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

        elif type_of_graph == "Incidence graph":
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
            msgBox = QMessageBox()
            msgBox.setWindowTitle("")
            msgBox.setText("ERROR: The type of graph you have requested is not supported. Please try something else. "
            + "(Devs: This means that you tried calling load_graph_type with an option that is currently not implemented.)")
            msgBox.exec()
            return None
        
        ## === Old Draw Graph Code kept for preservation. ===
        #self.ax.clear()
        #pos = nx.spring_layout(G, seed=42)
        #nx.draw(G, pos, ax=self.ax, with_labels=True, node_color='skyblue', edge_color='gray', font_size=7)
        #self.canvas.draw()

        # Reset self.view and self.choice_combo before reupdating it based on graph of choice.
        #For some reason, calling self.view = GraphView(primal_graph) does not work. Had to do this instead.
        self.view._graph = updated_graph
        self.view._scene = QGraphicsScene()
        self.view.setScene(self.view._scene)
        self.view._load_graph()
        self.view.set_nx_layout("circular") # added just in case...
        self.choice_combo.setCurrentText("circular") 

         # Calculate basic statistics and store it in text.
        self.text_area.clear()
        degrees = [deg for _, deg in updated_graph.degree()]
        avg_degree = np.mean(degrees)
        max_degree = np.max(degrees)
        clustering = nx.clustering(updated_graph)
        clustering_avg = np.mean(list(clustering.values()))
        components = list(nx.connected_components(updated_graph))
        num_components = len(components)
        component_sizes = [len(c) for c in components]
        largest_component = max(component_sizes)
        
        try:
            from networkx.algorithms.approximation.treewidth import treewidth_min_fill_in
            treewidth_calculated, _ = treewidth_min_fill_in(updated_graph)
        except:
            treewidth_calculated = "⚠️ Treewidth estimation not available — requires `networkx >= 2.6`."

        info_text = (
            f"Loaded MPS file: {self.filename}\n"
            f"Number of variables: {self.n_vars}\n"
            f"Number of constraints: {self.n_cons}\n"
            f"Average node degree: {avg_degree:.2f}\n"
            f"Maximum node degree: {max_degree}\n"
            f"Average clustering coefficient: {clustering_avg:.4f}\n"
            f"Number of connected components: {num_components}\n"
            f"Largest component size: {largest_component}\n"
            f"Treewidth (approxmimate): {treewidth_calculated}"
        )
        self.text_area.setPlainText(info_text)

    # Exports current graph view as JPEG.
    def export_graph_as_image(self):
        if not self.view:
            msgBox = QMessageBox()
            msgBox.setWindowTitle("")
            msgBox.setText("❌ No graph to export.")
            msgBox.exec()
            # THIS SHOULD NEVER COME UP.
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Chart as JPEG", "plot.jpeg", "JPEG Image (*.jpeg *.jpg)")
        if filename:
            pixmap = self.view.grab()
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
    
    # Reset graph view.
    def reset_graph_view(self):
        self.view.set_nx_layout(self.choice_combo.currentText())

class FileLoader(QWidget):
    def __init__(self):
        super().__init__()
        self.filename, _ = QFileDialog.getOpenFileName(self, "Open MPS File", "", "MPS Files (*.mps *.MPS);;All Files (*)")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    file_window = FileLoader()
    if file_window.filename:
        window = GraphViewer(file_window.filename, include_toggle_buttons=True)
        window.show()
    sys.exit(app.exec())