"""A discrete nonlinear slider with exact numeric entry and tuned presets."""
from bisect import bisect_left

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QVBoxLayout,
)

BUCKET_VALUES = tuple(sorted({value for start, stop, step in (
    (100, 1000, 100), (1000, 10_000, 1000), (10_000, 100_000, 10_000),
    (100_000, 1_000_000, 100_000), (1_000_000, 10_000_000, 1_000_000))
    for value in range(start, stop + 1, step)}))


def default_bucket(algorithm: str, early_stop: bool) -> int | None:
    if algorithm == "delta-cpu":
        return 300 if early_stop else 1000
    if algorithm == "delta-cuda":
        return 10_000 if early_stop else 1_000_000
    return None


class BucketControl(QGroupBox):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Bucket width", parent)
        self.algorithm, self.early_stop = "", True
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(BUCKET_VALUES) - 1)
        self.slider.setAccessibleName("Delta bucket width slider")
        self.slider.setToolTip("Steps grow from 100 m to 1,000,000 m across the range")
        self.spin = QSpinBox()
        self.spin.setRange(BUCKET_VALUES[0], BUCKET_VALUES[-1])
        self.spin.setSingleStep(100)
        self.spin.setSuffix(" m")
        self.spin.setGroupSeparatorShown(True)
        self.spin.setAccessibleName("Delta bucket width in metres")
        self.reset_button = QPushButton("Reset to default")
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.slider)
        row = QHBoxLayout()
        row.addWidget(self.spin, 1)
        row.addWidget(self.reset_button)
        layout.addLayout(row)
        layout.addWidget(self.note)
        self.opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.valueChanged.connect(self._spin_changed)
        self.reset_button.clicked.connect(self.reset_default)
        self.configure("heap", True)

    @property
    def value(self) -> int | None:
        return self.spin.value() if self.algorithm.startswith("delta-") else None

    def configure(self, algorithm: str, early_stop: bool) -> None:
        if (algorithm, early_stop) == (self.algorithm, self.early_stop):
            return
        self.algorithm, self.early_stop = algorithm, early_stop
        default = default_bucket(algorithm, early_stop)
        self.setEnabled(default is not None)
        self.opacity.setOpacity(1.0 if default is not None else 0.4)
        self.note.setText(f"Tuned default: {default:,} m" if default is not None
                          else "Available for delta-stepping algorithms")
        self.spin.setValue(default or BUCKET_VALUES[0])
        # Also align the slider when the numeric value happened to stay unchanged.
        self._sync_slider()

    def reset_default(self) -> None:
        default = default_bucket(self.algorithm, self.early_stop)
        if default is not None:
            self.spin.setValue(default)

    def _slider_changed(self, position: int) -> None:
        self.spin.setValue(BUCKET_VALUES[position])

    def _sync_slider(self) -> None:
        value = self.spin.value()
        index = min(bisect_left(BUCKET_VALUES, value), len(BUCKET_VALUES) - 1)
        if index and value - BUCKET_VALUES[index - 1] < BUCKET_VALUES[index] - value:
            index -= 1
        previous = self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(previous)

    def _spin_changed(self, value: int) -> None:
        self._sync_slider()
        self.spin.setSingleStep(next(step for bound, step in (
            (1000, 100), (10_000, 1000), (100_000, 10_000),
            (1_000_000, 100_000), (float("inf"), 1_000_000)) if value < bound))
        self.changed.emit()
