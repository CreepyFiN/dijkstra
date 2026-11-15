"""Asynchronous, cancellable JSON subprocesses; no shell or Python routing."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

ROOT = Path(__file__).resolve().parents[2]
ALGORITHMS = ("heap", "phased", "cpu-parallel", "cuda", "delta-cpu", "delta-cuda")


def find_executable() -> Path | None:
    configured = os.environ.get("PSV_EXECUTABLE")
    if configured:
        return Path(configured).resolve()
    name = "psv_dijkstra.exe" if os.name == "nt" else "psv_dijkstra"
    directories = ("build-cuda/Release", "build/Release", "build-cuda", "build",
                   "build-cuda/Debug", "build/Debug")
    return next((ROOT / directory / name for directory in directories
                 if (ROOT / directory / name).is_file()), None)


@dataclass(frozen=True)
class RouteRequest:
    graph: Path
    source: int
    target: int
    algorithm: str = "heap"
    early_stop: bool = True
    delta: int | None = None
    threads: int = 4

    def arguments(self) -> list[str]:
        if self.algorithm not in ALGORITHMS:
            raise ValueError("Unknown algorithm")
        args = ["--graph", str(self.graph.resolve()), "--source", str(self.source),
                "--target", str(self.target), "--algorithm", self.algorithm]
        if self.early_stop:
            args.append("--early-stop")
        if self.algorithm.startswith("delta-") and self.delta is not None:
            if self.delta <= 0:
                raise ValueError("Bucket width must be positive")
            args += ["--delta", str(self.delta)]
        return args


class JsonProcess(QObject):
    completed = Signal(object, float)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # Reuse process ownership and signal connections across retries/cancellation.
        self.process = QProcess(self)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self._active = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._timeout)
        self.kill_timer = QTimer(self)
        self.kill_timer.setSingleShot(True)
        self.kill_timer.timeout.connect(self.process.kill)
        self.cancel_requested = False
        self.timed_out = False
        self.started_at = 0.0

    @property
    def busy(self) -> bool:
        return self._active

    def start(self, program: Path, arguments: list[str], timeout_ms: int = 120_000,
              threads: int = 4) -> None:
        if self.busy:
            raise RuntimeError("A calculation is already running")
        self.cancel_requested = self.timed_out = False
        process = self.process
        self._active = True
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("OMP_NUM_THREADS", str(threads))
        environment.insert("OMP_DYNAMIC", "FALSE")
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(ROOT))
        self.started_at = time.perf_counter()
        self.timer.start(timeout_ms)
        process.start(str(program), arguments)

    def _release(self) -> None:
        self.timer.stop()
        self.kill_timer.stop()
        self._active = False

    def _error(self, error) -> None:
        if self.busy and error == QProcess.ProcessError.FailedToStart:
            message = self.process.errorString()
            self._release()
            self.failed.emit("Cannot start executable: " + message)

    def _finished(self, code: int, status) -> None:
        if not self.busy:
            return
        process = self.process
        elapsed = (time.perf_counter() - self.started_at) * 1000
        output = bytes(process.readAllStandardOutput())
        error = bytes(process.readAllStandardError()).decode("utf-8", errors="replace").strip()
        self._release()
        if self.timed_out:
            self.failed.emit("Calculation exceeded the configured timeout")
        elif self.cancel_requested:
            self.cancelled.emit()
        elif code != 0 or status != QProcess.ExitStatus.NormalExit:
            self.failed.emit(error or f"Executable exited with code {code}")
        else:
            try:
                report = json.loads(output)
                if not isinstance(report, dict):
                    raise ValueError("Expected a JSON object")
            except (ValueError, UnicodeDecodeError) as exception:
                self.failed.emit("Invalid executable output: " + str(exception))
            else:
                self.completed.emit(report, elapsed)

    def cancel(self) -> None:
        if not self.busy:
            return
        self.cancel_requested = True
        self.process.terminate()
        # Console processes on Windows may ignore terminate().
        if self.busy:
            self.kill_timer.start(1000)

    def _timeout(self) -> None:
        if self.busy:
            self.timed_out = True
            self.process.kill()
