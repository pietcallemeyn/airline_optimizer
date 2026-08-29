
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PROGRESS = ROOT / "output" / "live_progress.json"

POLL_INTERVAL_SEC = 0.35


class LiveParetoMonitor:
    def __init__(self):
        plt.ion()

        self.fig, self.ax = plt.subplots(
            figsize=(9, 6.5)
        )

        try:
            self.fig.canvas.manager.set_window_title(
                "Live optimizer — Cost vs Complexity"
            )
        except Exception:
            pass

        self.last_timestamp = None
        self.keep_running = True

        self.fig.canvas.mpl_connect(
            "close_event",
            self.on_close,
        )

        self.ax.set_xlabel(
            "Total operational cost (EUR)"
        )
        self.ax.set_ylabel(
            "Operational complexity score"
        )
        self.ax.grid(alpha=0.25)

        self.status_text = self.ax.text(
            0.01,
            0.99,
            "Waiting for optimizer...",
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_close(self, _event):
        self.keep_running = False

    def read_snapshot(self):
        if not PROGRESS.exists():
            return None

        try:
            return json.loads(
                PROGRESS.read_text(
                    encoding="utf-8"
                )
            )
        except (
            json.JSONDecodeError,
            OSError,
        ):
            return None

    def draw_snapshot(self, data):
        self.ax.clear()

        background = data.get(
            "background",
            [],
        )

        front = data.get(
            "front",
            [],
        )

        cheapest = data.get(
            "cheapest"
        )

        if background:
            self.ax.scatter(
                [
                    p["cost"]
                    for p in background
                ],
                [
                    p["complexity"]
                    for p in background
                ],
                s=18,
                alpha=0.22,
                color="gray",
                label="Other discovered schedules",
                zorder=1,
            )

        if front:
            self.ax.plot(
                [
                    p["cost"]
                    for p in front
                ],
                [
                    p["complexity"]
                    for p in front
                ],
                marker="o",
                markersize=6,
                linewidth=1.7,
                label="Live 2D Pareto front",
                zorder=3,
            )

        if cheapest:
            self.ax.scatter(
                [cheapest["cost"]],
                [cheapest["complexity"]],
                s=165,
                facecolors="none",
                edgecolors="black",
                linewidths=1.7,
                zorder=5,
            )

            self.ax.annotate(
                (
                    f'CHEAPEST\n'
                    f'€{cheapest["cost"]:,.0f}\n'
                    f'complexity {cheapest["complexity"]}'
                ),
                (
                    cheapest["cost"],
                    cheapest["complexity"],
                ),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8.5,
            )

        run = data.get("run", 0)
        runs = data.get("runs", 0)
        generation = data.get("generation", 0)
        generations = data.get("generations", 0)
        evaluations = data.get("evaluations", 0)
        unique = data.get("unique_schedules", 0)
        status = data.get("status", "running")

        self.ax.set_title(
            (
                "Live Cost vs Operational Complexity\n"
                f'run {run}/{runs} · '
                f'generation {generation}/{generations} · '
                f'{unique:,} unique schedules · '
                f'{evaluations:,} evaluations'
            )
        )

        self.ax.set_xlabel(
            "Total operational cost (EUR)"
        )
        self.ax.set_ylabel(
            "Operational complexity score"
        )
        self.ax.grid(alpha=0.25)

        if background or front:
            self.ax.legend(
                loc="best"
            )

        if status == "complete":
            self.ax.text(
                0.01,
                0.01,
                "Optimization complete",
                transform=self.ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
            )

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self):
        while self.keep_running:
            snapshot = self.read_snapshot()

            if snapshot is not None:
                timestamp = snapshot.get(
                    "timestamp"
                )

                if timestamp != self.last_timestamp:
                    self.last_timestamp = timestamp
                    self.draw_snapshot(snapshot)

                if (
                    snapshot.get("status")
                    == "complete"
                ):
                    # Keep the completed plot open until the user closes it.
                    while self.keep_running:
                        plt.pause(
                            POLL_INTERVAL_SEC
                        )
                    break

            plt.pause(
                POLL_INTERVAL_SEC
            )


if __name__ == "__main__":
    LiveParetoMonitor().run()
