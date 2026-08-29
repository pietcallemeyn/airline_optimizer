
from __future__ import annotations

import json
from pathlib import Path

from visualize import (
    load_gallery,
    load_pareto,
    metric,
    named_solutions,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"


def main():
    pareto = load_pareto()
    gallery = load_gallery()
    named = named_solutions(
        pareto
    )

    print()
    print("DECISION COMPARISON")
    print("=" * 112)

    print(
        f'{"OPTION":22s} '
        f'{"COST EUR":>14s} '
        f'{"COMPLEX":>9s} '
        f'{"EMPTY":>7s} '
        f'{"PILOT-D":>8s} '
        f'{"PARK":>6s} '
        f'{"SCHEDULE":>12s}'
    )

    print("-" * 112)

    for name, solution in named.items():
        print(
            f'{name:22s} '
            f'{metric(solution,"total_operational_cost_eur"):14,.0f} '
            f'{metric(solution,"complexity_score"):9d} '
            f'{metric(solution,"empty_legs"):7d} '
            f'{metric(solution,"charged_pilot_days"):8d} '
            f'{metric(solution,"aircraft_parking_days"):6d} '
            f'{solution["schedule_signature"]:>12s}'
        )

    print("-" * 112)

    print(
        f'{"ACTUAL":22s} '
        f'{477700:14,.0f} '
        f'{"proxy":>9s}'
    )

    print()
    print(
        f'Unique alternative schedules saved for browsing: '
        f'{len(gallery)}'
    )

    summary_path = (
        OUTPUT
        / "run_summary.json"
    )

    if summary_path.exists():
        summary = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )

        print(
            f'Unique aircraft schedules discovered by search: '
            f'{summary["unique_aircraft_schedules"]}'
        )

        print(
            f'Pareto unique aircraft schedules: '
            f'{summary["pareto_unique_aircraft_schedules"]}'
        )


if __name__ == "__main__":
    main()
