"""Ordered, cancellable route legs using the unchanged C++ CLI."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

from PySide6.QtCore import QObject, QTimer, Signal

from .backend import JsonProcess, RouteRequest
from .data import MapData


@dataclass(frozen=True)
class JourneyRequest:
    graph: Path
    nodes: tuple[int, ...]
    algorithm: str = "heap"
    early_stop: bool = True
    delta: int | None = None
    threads: int = 4

    @property
    def leg_count(self) -> int:
        return len(self.nodes) - 1

    def leg(self, index: int) -> RouteRequest:
        return RouteRequest(self.graph, self.nodes[index], self.nodes[index + 1],
                            self.algorithm, self.early_stop, self.delta, self.threads)

    def stop_label(self, index: int) -> str:
        if index == 0:
            return "Start"
        if index == len(self.nodes) - 1:
            return "Destination"
        return f"Middle node #{index}"

    def leg_label(self, index: int) -> str:
        return (f"Leg {index + 1}/{self.leg_count}: {self.stop_label(index)} → "
                f"{self.stop_label(index + 1)} (nodes {self.nodes[index]} → {self.nodes[index + 1]})")


class RouteRunner(QObject):
    completed = Signal(object, float)
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.job = JsonProcess(self)
        self.job.completed.connect(self._leg_completed)
        self.job.failed.connect(self._leg_failed)
        self.job.cancelled.connect(self._job_cancelled)
        self.next_leg = QTimer(self)
        self.next_leg.setSingleShot(True)
        self.next_leg.timeout.connect(self._start_leg)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._timeout)
        self._active = False
        self._cancel_requested = False
        self._abort_reason: str | None = None
        self.reports: list[dict] = []

    @property
    def busy(self) -> bool:
        # Includes gaps between legs, so controls/cancellation remain coherent.
        return self._active

    def start(self, executable: Path, data: MapData, request: JourneyRequest,
              timeout_ms: int = 120_000) -> None:
        if self.busy:
            raise RuntimeError("A route is already running")
        if (len(request.nodes) < 2 or any(type(node) is not int or not 0 <= node < data.node_count
                                        for node in request.nodes)):
            raise ValueError("A route requires valid start, destination, and middle nodes")
        if request.graph.resolve() != data.graph_path:
            raise ValueError("Route graph differs from the loaded map")
        request.leg(0).arguments()  # Validate the algorithm/bucket before starting.
        self.executable, self.data, self.request = executable, data, request
        self.reports = []
        self._active = True
        self._cancel_requested = False
        self._abort_reason = None
        self._started_at = time.perf_counter()
        self.timer.start(timeout_ms)
        self.next_leg.start(0)

    def _start_leg(self) -> None:
        if not self.busy:
            return
        if self._cancel_requested:
            self._finish_cancel()
            return
        index = len(self.reports)
        self.progress.emit("Calculating " + self.request.leg_label(index))
        # A progress receiver may cancel before the subprocess is started.
        if not self.busy:
            return
        leg = self.request.leg(index)
        self.job.start(self.executable, leg.arguments(), max(1, self.timer.remainingTime()), leg.threads)

    def _leg_completed(self, report: dict, _elapsed: float) -> None:
        if not self.busy:
            return
        if self._cancel_requested or self._abort_reason:
            self._job_cancelled()
            return
        index = len(self.reports)
        leg = self.request.leg(index)
        try:
            self.data.validate_route(report, leg.source, leg.target)
            if (report.get("algorithm") != leg.algorithm or report.get("early_stop") != leg.early_stop
                    or type(report.get("search_complete")) is not bool
                    or type(report.get("settled_nodes")) is not int
                    or not 0 <= report["settled_nodes"] <= self.data.node_count):
                raise ValueError("Unexpected algorithm, mode, or finalized-node count")
            for field in ("algorithm_ms", "gpu_compute_ms", "gpu_upload_ms", "gpu_download_ms"):
                value = report.get(field)
                if value is None and field != "algorithm_ms":
                    continue
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid calculation timing")
            self.reports.append(report)
            if not report["reachable"] or len(self.reports) == self.request.leg_count:
                combined = self._combine()
                elapsed = (time.perf_counter() - self._started_at) * 1000
                self._release()
                self.completed.emit(combined, elapsed)
            else:
                self.next_leg.start(0)
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            self._finish_error(self.request.leg_label(index) + ": " + str(error))

    def _combine(self) -> dict:
        reachable = all(report["reachable"] for report in self.reports)
        nodes, edges = [], []
        if reachable:
            for report in self.reports:
                nodes.extend(report["path_nodes"] if not nodes else report["path_nodes"][1:])
                edges.extend(report["path_edges"])
        result = dict(self.reports[-1])
        result.update(source=self.request.nodes[0], target=self.request.nodes[-1], reachable=reachable,
                      length_m=math.fsum(r["length_m"] for r in self.reports) if reachable else None,
                      path_nodes=nodes, path_edges=edges, leg_count=self.request.leg_count,
                      attempted_legs=len(self.reports),
                      settled_nodes=sum(r["settled_nodes"] for r in self.reports),
                      search_complete=all(r["search_complete"] for r in self.reports))
        result["legs"] = [dict(label=f"Leg {index + 1}: {self.request.stop_label(index)} → "
                               f"{self.request.stop_label(index + 1)}",
                               length_m=report["length_m"], path_nodes=list(report["path_nodes"]))
                          for index, report in enumerate(self.reports)] if reachable else []
        for field in ("algorithm_ms", "gpu_compute_ms", "gpu_upload_ms", "gpu_download_ms"):
            values = [r.get(field) for r in self.reports]
            result[field] = math.fsum(values) if all(v is not None for v in values) else None
        if not reachable:
            result["failed_leg"] = self.request.leg_label(len(self.reports) - 1)
        # A journey may revisit vertices across legs. Each leg was validated
        # independently; applying the single-source acyclic-path check here
        # would incorrectly reject valid return trips and repeated stops.
        return result

    def _release(self) -> None:
        self.timer.stop()
        self.next_leg.stop()
        self._active = False
        self.data = None  # Do not retain a previous dataset after the map is replaced.

    def _finish_error(self, message: str) -> None:
        self._release()
        self.failed.emit(message)

    def _leg_failed(self, message: str) -> None:
        if self.busy:
            if self._abort_reason:
                message = self._abort_reason
            elif self._cancel_requested:
                self._finish_cancel()
                return
            else:
                message = self.request.leg_label(len(self.reports)) + ": " + message
            self._finish_error(message)

    def cancel(self) -> None:
        if self.busy:
            self._cancel_requested = True
            if self.job.busy:
                self.job.cancel()
            else:
                self._finish_cancel()

    def _finish_cancel(self) -> None:
        self._release()
        self.cancelled.emit()

    def _job_cancelled(self) -> None:
        if self.busy:
            if self._abort_reason:
                self._finish_error(self._abort_reason)
            else:
                self._finish_cancel()

    def _timeout(self) -> None:
        if self.busy:
            self._abort_reason = "Route calculation exceeded the configured timeout"
            if self.job.busy:
                self.job.cancel()
            else:
                self._finish_error(self._abort_reason)
