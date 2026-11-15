"""Launch from the repository root: python -m python.gui."""
import argparse
from pathlib import Path
import sys

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from .window import MainWindow


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Pohjois-Savo route planner")
    parser.add_argument("--graph", type=Path, help="CSR graph (defaults to the regional dataset)")
    parser.add_argument("--nodes", type=Path, help="Matching node CSV (defaults to <graph stem>_nodes.csv)")
    parser.add_argument("--executable", type=Path, help="Routing executable (prefers CUDA Release if present)")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Pohjois-Savo Dijkstra")
    app.setStyle("Fusion")
    palette = QPalette()
    for role, color in ((QPalette.ColorRole.Window, "#f5f7fa"), (QPalette.ColorRole.WindowText, "#203442"),
                        (QPalette.ColorRole.Base, "white"), (QPalette.ColorRole.Text, "#203442"),
                        (QPalette.ColorRole.Button, "#eef2f5"), (QPalette.ColorRole.ButtonText, "#203442"),
                        (QPalette.ColorRole.Highlight, "#176f9b"), (QPalette.ColorRole.HighlightedText, "white")):
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    window = MainWindow(args.graph, args.nodes, args.executable.resolve() if args.executable else None)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
