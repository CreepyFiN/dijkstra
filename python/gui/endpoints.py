"""Editable route stops, shared by start, middle points, and destination."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QSizePolicy, QSpinBox, QWidget


class EndpointRow(QWidget):
    removeRequested = Signal()

    def __init__(self, caption: str, pick_text: str = "Pick on map", *, removable=False):
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setColumnStretch(0, 1)
        self.title = QLabel()
        layout.addWidget(self.title, 0, 0, 1, 2)
        self.node = QSpinBox()
        self.pick_button = QPushButton(pick_text)
        self.pick_button.setCheckable(True)
        layout.addWidget(self.node, 1, 0)
        layout.addWidget(self.pick_button, 1, 1)
        self.details = QLabel()
        self.details.setWordWrap(True)
        if removable:
            self.remove_button = QPushButton("Remove")
            self.remove_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.remove_button.clicked.connect(self.removeRequested)
            layout.addWidget(self.details, 2, 0)
            layout.addWidget(self.remove_button, 2, 1, alignment=Qt.AlignmentFlag.AlignVCenter)
        else:
            layout.addWidget(self.details, 2, 0, 1, 2)
        self.set_caption(caption)

    def set_caption(self, caption: str) -> None:
        self.title.setText(caption)
        self.node.setAccessibleName(caption + " index")
        self.pick_button.setAccessibleName("Pick " + caption + " on map")
        if hasattr(self, "remove_button"):
            self.remove_button.setAccessibleName("Remove " + caption)
