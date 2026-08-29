
from __future__ import annotations

import csv
import gc
import json
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.widgets import Button


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
DATA = ROOT / "data"

TURNAROUND_MIN = 20
POSITION_SPEED_KT = 410.0
POSITION_FIXED_MIN = 10.0

FLIGHT_HOUR_COST_EUR = 6000.0
PILOT_DAY_COST_EUR = 600.0
AIRCRAFT_AWAY_PARKING_DAY_COST_EUR = 1000.0
BASES = ("EBAW", "EBLG")

CACHE_SIZE = 20
MIN_CLICK_INTERVAL_SEC = 0.12


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_dict(path, key):
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row[key]] = row
    return out


def distance_nm(a, b):
    R = 3440.065
    p1 = radians(float(a["lat"]))
    p2 = radians(float(b["lat"]))
    dp = radians(float(b["lat"]) - float(a["lat"]))
    dl = radians(float(b["lon"]) - float(a["lon"]))
    h = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * R * atan2(sqrt(h), sqrt(1 - h))


def positioning_duration(distance):
    return timedelta(
        minutes=POSITION_FIXED_MIN + 60.0 * distance / POSITION_SPEED_KT
    )


def load_pareto():
    solutions = []
    for path in sorted(OUTPUT.glob("pareto_*.json")):
        try:
            solutions.append(load_json(path))
        except Exception:
            pass

    if not solutions:
        raise FileNotFoundError(
            "No optimizer output found. Run `python optimizer.py` first."
        )

    return solutions


def load_gallery():
    path = OUTPUT / "diverse_solutions.json"
    if not path.exists():
        return []
    return load_json(path)


def metric(solution, name):
    for section in ("objectives", "metrics", "cost_breakdown"):
        if name in solution.get(section, {}):
            return solution[section][name]
    return None


def named_solutions(solutions):
    cheapest = min(
        solutions,
        key=lambda s: (
            metric(s, "total_operational_cost_eur"),
            metric(s, "complexity_score"),
        ),
    )

    min_complexity = min(
        solutions,
        key=lambda s: (
            metric(s, "complexity_score"),
            metric(s, "total_operational_cost_eur"),
        ),
    )

    min_empty = min(
        solutions,
        key=lambda s: (
            metric(s, "empty_legs"),
            metric(s, "total_operational_cost_eur"),
        ),
    )

    min_pilot = min(
        solutions,
        key=lambda s: (
            metric(s, "charged_pilot_days"),
            metric(s, "total_operational_cost_eur"),
        ),
    )

    min_parking = min(
        solutions,
        key=lambda s: (
            metric(s, "aircraft_parking_days"),
            metric(s, "total_operational_cost_eur"),
        ),
    )

    dimensions = [
        "total_operational_cost_eur",
        "complexity_score",
        "empty_legs",
        "charged_pilot_days",
        "aircraft_parking_days",
    ]

    mins = {d: min(metric(s, d) for s in solutions) for d in dimensions}
    maxs = {d: max(metric(s, d) for s in solutions) for d in dimensions}

    def balance_score(solution):
        values = []
        for d in dimensions:
            den = maxs[d] - mins[d]
            values.append(
                0.0 if den == 0 else (metric(solution, d) - mins[d]) / den
            )
        return sum(v * v for v in values) ** 0.5

    balanced = min(solutions, key=balance_score)

    return {
        "CHEAPEST": cheapest,
        "BALANCED": balanced,
        "MIN EMPTY LEGS": min_empty,
        "MIN PILOT DAYS": min_pilot,
        "MIN AIRCRAFT PARKING": min_parking,
        "MIN COMPLEXITY": min_complexity,
    }


def action_lookup(solution):
    lookup = {}

    for reg, actions in solution.get("aircraft_actions", {}).items():
        for action in actions:
            if action.get("mission") == "__END__":
                continue
            lookup[(reg, action["mission"])] = action

    return lookup


def reconstruct_routes(solution, missions, airports):
    # New optimizer outputs contain a self-contained mission snapshot and an
    # explicit, hard-validated movement log. Prefer those over reconstructing
    # positions from the current data/missions.csv.
    if solution.get("mission_snapshot"):
        missions = solution["mission_snapshot"]

    explicit = solution.get("aircraft_movements")

    if explicit:
        routes = {}

        for reg, movements in explicit.items():
            legs = []

            for movement in movements:
                start = datetime.fromisoformat(
                    movement["start"]
                )
                end = datetime.fromisoformat(
                    movement["end"]
                )

                if movement["type"] == "MISSION":
                    mid = movement["mission"]
                    mission = missions[mid]

                    legs.append({
                        "type": "MISSION",
                        "mission": mid,
                        "from": movement["from"],
                        "to": movement["to"],
                        "departure": start,
                        "arrival": end,
                        "pax": int(mission["pax"]),
                        "captain": movement.get("captain", "?"),
                        "fo": movement.get("fo", "?"),
                    })
                else:
                    legs.append({
                        "type": "EMPTY",
                        "from": movement["from"],
                        "to": movement["to"],
                        "departure": start,
                        "arrival": end,
                        "nm": float(movement.get("nm", 0.0)),
                        "captain": movement.get("captain", "?"),
                        "fo": movement.get("fo", "?"),
                    })

            # Keep parking notes from the existing action logic only.
            routes[reg] = {
                "legs": legs,
                "notes": [],
            }

        return routes

    lookup = action_lookup(solution)
    routes = {}

    for reg, mission_ids in solution["routes"].items():
        legs = []
        notes = []
        previous = None

        for i, mid in enumerate(mission_ids):
            mission = missions[mid]
            dep = datetime.fromisoformat(mission["departure"])
            arr = datetime.fromisoformat(mission["arrival"])
            action = lookup.get((reg, mid), {})
            mode = action.get("transition_mode", "")

            if i == 0 and mode.startswith("START_"):
                base = mode.replace("START_", "")

                if base != mission["origin"]:
                    d = distance_nm(airports[base], airports[mission["origin"]])
                    empty_arrival = dep - timedelta(minutes=TURNAROUND_MIN)
                    empty_departure = empty_arrival - positioning_duration(d)

                    legs.append({
                        "type": "EMPTY",
                        "from": base,
                        "to": mission["origin"],
                        "departure": empty_departure,
                        "arrival": empty_arrival,
                        "nm": d,
                    })

            if i > 0:
                previous_mission = missions[previous]
                previous_arrival = datetime.fromisoformat(
                    previous_mission["arrival"]
                )
                previous_destination = previous_mission["destination"]

                if mode == "STAY":
                    if previous_destination != mission["origin"]:
                        d = distance_nm(
                            airports[previous_destination],
                            airports[mission["origin"]],
                        )

                        empty_arrival = dep - timedelta(minutes=TURNAROUND_MIN)
                        empty_departure = (
                            empty_arrival - positioning_duration(d)
                        )

                        legs.append({
                            "type": "EMPTY",
                            "from": previous_destination,
                            "to": mission["origin"],
                            "departure": empty_departure,
                            "arrival": empty_arrival,
                            "nm": d,
                        })

                    parking_days = int(action.get("parking_days", 0))

                    if parking_days > 0:
                        notes.append({
                            "start": (
                                previous_arrival
                                + timedelta(minutes=TURNAROUND_MIN)
                            ),
                            "end": (
                                dep - timedelta(minutes=TURNAROUND_MIN)
                            ),
                            "airport": previous_destination,
                            "days": parking_days,
                            "cost": float(action.get("parking_cost_eur", 0)),
                        })

                elif mode.startswith("VIA_"):
                    base = mode.replace("VIA_", "")

                    if previous_destination != base:
                        d1 = distance_nm(
                            airports[previous_destination],
                            airports[base],
                        )

                        start1 = (
                            previous_arrival
                            + timedelta(minutes=TURNAROUND_MIN)
                        )

                        end1 = start1 + positioning_duration(d1)

                        legs.append({
                            "type": "EMPTY",
                            "from": previous_destination,
                            "to": base,
                            "departure": start1,
                            "arrival": end1,
                            "nm": d1,
                        })

                    if base != mission["origin"]:
                        d2 = distance_nm(
                            airports[base],
                            airports[mission["origin"]],
                        )

                        end2 = dep - timedelta(minutes=TURNAROUND_MIN)
                        start2 = end2 - positioning_duration(d2)

                        legs.append({
                            "type": "EMPTY",
                            "from": base,
                            "to": mission["origin"],
                            "departure": start2,
                            "arrival": end2,
                            "nm": d2,
                        })

            crew = solution.get("crew_assignments", {}).get(mid, {})

            legs.append({
                "type": "MISSION",
                "mission": mid,
                "from": mission["origin"],
                "to": mission["destination"],
                "departure": dep,
                "arrival": arr,
                "pax": int(mission["pax"]),
                "captain": crew.get("captain", "?"),
                "fo": crew.get("fo", "?"),
            })

            previous = mid

        routes[reg] = {
            "legs": legs,
            "notes": notes,
        }

    return routes


def actual_metrics(actual, airports):
    empty_nm = 0.0
    empty_legs = 0
    mission_legs = 0
    by_reg = {}

    for reg, legs in actual.items():
        parsed = []

        for leg in legs:
            d = distance_nm(
                airports[leg["from"]],
                airports[leg["to"]],
            )

            is_mission = leg.get("mission_hint") is not None

            if is_mission:
                mission_legs += 1
            else:
                empty_legs += 1
                empty_nm += d

            parsed.append({
                **leg,
                "nm": d,
                "is_mission": is_mission,
            })

        by_reg[reg] = parsed

    return {
        "empty_nm": empty_nm,
        "empty_legs": empty_legs,
        "mission_legs": mission_legs,
        "by_reg": by_reg,
    }


class RouteCache:
    def __init__(self, max_size=CACHE_SIZE):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, solution, missions, airports):
        key = solution.get(
            "schedule_signature",
            str(id(solution)),
        )

        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value

        value = reconstruct_routes(
            solution,
            missions,
            airports,
        )

        self.cache[key] = value

        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

        return value



def estimate_actual_cost(actual, airports):
    total_flight_hours = 0.0
    parking_days = 0
    pilot_days = 0
    empty_nm = 0.0
    empty_legs = 0

    for reg, legs in actual.items():
        charged_days = set()

        for i, leg in enumerate(legs):
            d = distance_nm(
                airports[leg["from"]],
                airports[leg["to"]],
            )

            total_flight_hours += (
                POSITION_FIXED_MIN / 60.0
                + d / POSITION_SPEED_KT
            )

            day = datetime.fromisoformat(
                leg["date"]
            ).date()
            charged_days.add(day)

            if leg.get("mission_hint") is None:
                empty_legs += 1
                empty_nm += d

            if i < len(legs) - 1:
                next_day = datetime.fromisoformat(
                    legs[i + 1]["date"]
                ).date()
                gap = (next_day - day).days

                if gap > 0 and leg["to"] not in BASES:
                    parking_days += gap

                    for offset in range(1, gap):
                        charged_days.add(
                            day + timedelta(days=offset)
                        )

        pilot_days += 2 * len(charged_days)

    flight_cost = (
        total_flight_hours
        * FLIGHT_HOUR_COST_EUR
    )
    pilot_cost = (
        pilot_days
        * PILOT_DAY_COST_EUR
    )
    parking_cost = (
        parking_days
        * AIRCRAFT_AWAY_PARKING_DAY_COST_EUR
    )

    return {
        "total_operational_cost_eur": (
            flight_cost
            + pilot_cost
            + parking_cost
        ),
        "flight_cost_eur": flight_cost,
        "pilot_cost_eur": pilot_cost,
        "parking_cost_eur": parking_cost,
        "flight_hours": total_flight_hours,
        "pilot_days": pilot_days,
        "parking_days": parking_days,
        "empty_nm": empty_nm,
        "empty_legs": empty_legs,
    }


def crew_action_lookup(solution):
    return {
        action.get("mission"): action
        for action in solution.get(
            "crew_actions",
            [],
        )
        if action.get("mission")
    }



def render_solution_info(
    info,
    solution,
    title,
    detail=None,
):
    info.cla()
    info.axis("off")

    if detail is not None:
        info.text(
            0.02,
            0.98,
            detail.get("kind", "MOVEMENT"),
            va="top",
            ha="left",
            fontsize=11,
            fontweight="bold",
        )

        y = 0.88

        fields = [
            ("ROUTE", detail.get("route", "—")),
            ("TIMING", detail.get("timing", "—")),
            ("AIRCRAFT", detail.get("aircraft", "—")),
            ("ATTACHED CREW", detail.get("crew", "—")),
            ("DETAIL", detail.get("extra", "—")),
        ]

        for label, value in fields:
            info.text(
                0.02,
                y,
                label,
                va="top",
                ha="left",
                fontsize=7.2,
                color="0.45",
                fontweight="bold",
            )

            y -= 0.045

            info.text(
                0.02,
                y,
                str(value),
                va="top",
                ha="left",
                fontsize=9.0,
                wrap=True,
            )

            y -= 0.13

            info.axhline(
                y + 0.035,
                xmin=0.02,
                xmax=0.98,
                linewidth=0.6,
                alpha=0.15,
            )

        info.text(
            0.02,
            0.04,
            "Click another card to inspect it.",
            va="bottom",
            ha="left",
            fontsize=7.0,
            color="0.45",
        )

        return

    objectives = solution.get(
        "objectives",
        {},
    )

    metrics = solution.get(
        "metrics",
        {},
    )

    validation = solution.get(
        "validation",
        {},
    )

    lines = [
        title,
        "",
        "TOTAL OPERATIONAL COST",
        f'€{objectives.get("total_operational_cost_eur",0):,.0f}',
        "",
        "COMPLEXITY",
        str(
            objectives.get(
                "complexity_score",
                0,
            )
        ),
        "",
        "ROTATION",
        f'{metrics.get("empty_legs",0)} empty legs',
        f'{metrics.get("empty_nm",0):,.0f} empty NM',
        f'{metrics.get("aircraft_parking_days",0)} parking-days',
        "",
        "CREW",
        f'{metrics.get("charged_pilot_days",0)} pilot-days',
        f'{metrics.get("nonhomebase_swap_events",0)} outstation swaps',
        f'{metrics.get("inbound_crew_deadhead_pilots",0)} inbound DH pilots',
        "",
        "HARD VALIDATION",
        (
            "✓ aircraft timeline"
            if validation.get(
                "aircraft_timeline_valid"
            )
            else "— aircraft timeline"
        ),
        (
            "✓ location continuity"
            if validation.get(
                "location_continuity_valid"
            )
            else "— location continuity"
        ),
        (
            "✓ crew on every leg"
            if validation.get(
                "crew_on_every_movement"
            )
            else "— crew on every leg"
        ),
        "",
        "HOW TO USE",
        "Click a card for details.",
        "Use DAYS ← / DAYS →",
        "to move through the plan.",
    ]

    info.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=8.2,
        linespacing=1.28,
    )


def draw_optimizer(
    ax,
    info,
    solution,
    routes,
    title,
    gallery_position=None,
    view_start=None,
    days_visible=5,
):
    """
    Operations-board visualization.

    Time is organized into DAY COLUMNS rather than using flight duration as
    pixel width. Movements inside each aircraft/day cell are laid out as cards
    in chronological sequence. Exact UTC timing remains available by clicking
    a card.

    This deliberately prioritizes planning readability over a dense Gantt view.
    """
    regs = [
        reg
        for reg in routes
        if routes[reg]["legs"]
    ]

    if not regs:
        ax.text(
            0.5,
            0.5,
            "No aircraft movements",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return None

    all_legs = [
        leg
        for reg in regs
        for leg in routes[reg]["legs"]
    ]

    first_date = min(
        leg["departure"].date()
        for leg in all_legs
    )

    last_date = max(
        leg["departure"].date()
        for leg in all_legs
    )

    if view_start is None:
        view_start = first_date

    max_start = max(
        first_date,
        last_date
        - timedelta(
            days=max(
                0,
                days_visible - 1,
            )
        ),
    )

    if view_start < first_date:
        view_start = first_date

    if view_start > max_start:
        view_start = max_start

    visible_days = [
        view_start
        + timedelta(days=i)
        for i in range(days_visible)
    ]

    day_index = {
        day: i
        for i, day in enumerate(
            visible_days
        )
    }

    reg_to_y = {
        reg: i
        for i, reg in enumerate(
            regs
        )
    }

    # Board background / day cells.
    for i, day in enumerate(visible_days):
        if i % 2:
            ax.axvspan(
                i,
                i + 1,
                alpha=0.035,
                zorder=0,
            )

        ax.axvline(
            i,
            linewidth=0.75,
            alpha=0.18,
            zorder=0,
        )

        ax.text(
            i + 0.5,
            -0.61,
            day.strftime(
                "%a %d %b"
            ),
            ha="center",
            va="center",
            fontsize=8.5,
            color="0.35",
            fontweight="bold",
        )

    ax.axvline(
        days_visible,
        linewidth=0.75,
        alpha=0.18,
        zorder=0,
    )

    # Lane separators.
    for y in range(
        len(regs) + 1
    ):
        ax.axhline(
            y - 0.5,
            linewidth=0.75,
            alpha=0.14,
            zorder=0,
        )

    movement_cards = []

    action_by_mission = {
        action.get("mission"): action
        for action in solution.get(
            "crew_actions",
            [],
        )
        if action.get("mission")
    }

    for reg in regs:
        y = reg_to_y[reg]

        # Group movements by departure day.
        per_day = {}

        for leg in sorted(
            routes[reg]["legs"],
            key=lambda x: x[
                "departure"
            ],
        ):
            day = leg[
                "departure"
            ].date()

            if day not in day_index:
                continue

            per_day.setdefault(
                day,
                [],
            ).append(
                leg
            )

        # Crew shown under aircraft name = first visible movement crew.
        visible_crew = "—"

        visible_leg_list = [
            leg
            for legs in per_day.values()
            for leg in legs
        ]

        if visible_leg_list:
            first_leg = min(
                visible_leg_list,
                key=lambda x: x[
                    "departure"
                ],
            )

            visible_crew = (
                f'{first_leg.get("captain","?")}/'
                f'{first_leg.get("fo","?")}'
            )

        # Put aircraft label just outside plot, mimicking a frozen first column.
        ax.text(
            -0.10,
            y - 0.06,
            reg,
            ha="right",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            clip_on=False,
        )

        ax.text(
            -0.10,
            y + 0.18,
            visible_crew,
            ha="right",
            va="center",
            fontsize=7.2,
            color="0.43",
            clip_on=False,
        )

        # Parking as a quiet day-cell background when away between movements.
        for note in routes[
            reg
        ].get(
            "notes",
            [],
        ):
            park_start = note[
                "start"
            ].date()

            park_end = note[
                "end"
            ].date()

            for day in visible_days:
                if (
                    park_start
                    <= day
                    <= park_end
                ):
                    x = day_index[
                        day
                    ]

                    parking_patch = FancyBboxPatch(
                        (
                            x + 0.06,
                            y - 0.31,
                        ),
                        0.88,
                        0.62,
                        boxstyle=(
                            "round,pad=0.01,"
                            "rounding_size=0.035"
                        ),
                        linewidth=0.7,
                        edgecolor="0.70",
                        facecolor="0.94",
                        alpha=0.60,
                        zorder=1,
                    )

                    ax.add_patch(
                        parking_patch
                    )

                    if day == park_start:
                        ax.text(
                            x + 0.50,
                            y + 0.28,
                            (
                                f'PARK {note["airport"]} · '
                                f'{note["days"]}d'
                            ),
                            ha="center",
                            va="bottom",
                            fontsize=5.5,
                            color="0.45",
                            zorder=2,
                        )

        for day, legs in per_day.items():
            x_day = day_index[
                day
            ]

            count = len(
                legs
            )

            gap = 0.035
            usable = (
                0.88
                - gap
                * max(
                    0,
                    count - 1,
                )
            )

            card_width = (
                usable / count
                if count
                else usable
            )

            for slot, leg in enumerate(
                legs
            ):
                x = (
                    x_day
                    + 0.06
                    + slot
                    * (
                        card_width
                        + gap
                    )
                )

                is_mission = (
                    leg["type"]
                    == "MISSION"
                )

                if is_mission:
                    height = 0.50
                    y0 = (
                        y
                        - height / 2
                    )
                    face = "C2"
                    edge = "0.35"
                    linestyle = "-"
                else:
                    height = 0.28
                    y0 = (
                        y
                        - height / 2
                    )
                    face = "0.96"
                    edge = "0.55"
                    linestyle = "--"

                patch = FancyBboxPatch(
                    (
                        x,
                        y0,
                    ),
                    card_width,
                    height,
                    boxstyle=(
                        "round,pad=0.01,"
                        "rounding_size=0.04"
                    ),
                    linewidth=1.0,
                    edgecolor=edge,
                    facecolor=face,
                    linestyle=linestyle,
                    alpha=0.92,
                    zorder=4,
                    picker=True,
                )

                ax.add_patch(
                    patch
                )

                crew = (
                    f'{leg.get("captain","?")}/'
                    f'{leg.get("fo","?")}'
                )

                if is_mission:
                    route_label = (
                        f'{leg["from"]} → '
                        f'{leg["to"]}'
                    )

                    if card_width >= 0.22:
                        text = (
                            f'{route_label}\n'
                            f'{leg["mission"]} · '
                            f'{leg["pax"]} pax\n'
                            f'{crew}'
                        )
                    else:
                        text = (
                            f'{leg["mission"]}\n'
                            f'{crew}'
                        )

                    kind = "CUSTOMER MISSION"

                    extra = (
                        f'{leg["pax"]} passengers'
                    )

                    action = action_by_mission.get(
                        leg["mission"],
                        {},
                    )

                    if (
                        action.get(
                            "action"
                        )
                        == "HANDOVER"
                    ):
                        nonhome = int(
                            action.get(
                                "nonhome_swap_event",
                                0,
                            )
                        )

                        ax.scatter(
                            [
                                x
                                + card_width
                                - 0.035
                            ],
                            [
                                y
                                - height / 2
                                - 0.08
                            ],
                            marker="D",
                            s=28,
                            facecolors="white",
                            edgecolors="0.20",
                            linewidths=1.0,
                            zorder=7,
                        )

                        swap_text = (
                            "OUTSTATION SWAP"
                            if nonhome
                            else "HOME SWAP"
                        )

                        extra += (
                            f' · {swap_text}'
                        )

                else:
                    route_label = (
                        f'{leg["from"]} → '
                        f'{leg["to"]}'
                    )

                    if card_width >= 0.22:
                        text = (
                            f'EMPTY\n'
                            f'{route_label}\n'
                            f'{crew}'
                        )
                    else:
                        text = (
                            f'EMPTY\n'
                            f'{crew}'
                        )

                    kind = (
                        "EMPTY POSITIONING"
                    )

                    extra = (
                        f'{leg["nm"]:.0f} NM positioning'
                    )

                ax.text(
                    x + card_width / 2,
                    y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=(
                        6.6
                        if card_width
                        >= 0.22
                        else 5.6
                    ),
                    fontweight=(
                        "bold"
                        if is_mission
                        else "normal"
                    ),
                    color="0.13",
                    clip_on=True,
                    zorder=5,
                )

                patch._ops_detail = {
                    "kind": kind,
                    "route": route_label,
                    "timing": (
                        f'{leg["departure"].strftime("%a %d %b · %H:%M")} '
                        f'→ {leg["arrival"].strftime("%H:%M")} UTC'
                    ),
                    "aircraft": reg,
                    "crew": crew,
                    "extra": extra,
                }

                movement_cards.append(
                    patch
                )

    ax.set_xlim(
        -0.02,
        days_visible,
    )

    ax.set_ylim(
        len(regs) - 0.5,
        -0.78,
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(
            False
        )

    objectives = solution.get(
        "objectives",
        {},
    )

    metrics = solution.get(
        "metrics",
        {},
    )

    alt = ""

    if gallery_position:
        alt = (
            f' · alternative '
            f'{gallery_position[0] + 1}/'
            f'{gallery_position[1]}'
        )

    ax.set_title(
        (
            f'{title}{alt}\n'
            f'€{objectives.get("total_operational_cost_eur",0):,.0f} · '
            f'{metrics.get("empty_legs",0)} empty legs · '
            f'{metrics.get("charged_pilot_days",0)} pilot-days · '
            f'{metrics.get("nonhomebase_swap_events",0)} outstation swaps'
        ),
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    render_solution_info(
        info,
        solution,
        title,
    )

    ax._movement_cards = (
        movement_cards
    )

    return {
        "first_date": first_date,
        "last_date": last_date,
        "view_start": view_start,
        "days_visible": days_visible,
    }

def draw_actual(ax, info, actual, airports):
    metrics = actual_metrics(actual, airports)
    costs = estimate_actual_cost(
        actual,
        airports,
    )
    regs = list(actual)

    for y, reg in enumerate(regs):
        day_counts = {}

        for leg in metrics["by_reg"][reg]:
            day = datetime.fromisoformat(
                leg["date"]
            )

            day_counts.setdefault(
                leg["date"],
                0,
            )
            slot = day_counts[leg["date"]]
            day_counts[leg["date"]] += 1

            start = day + timedelta(
                hours=1 + slot * 2
            )

            duration_hours = max(
                0.6,
                min(
                    3.2,
                    20 / 60 + leg["nm"] / 410,
                ),
            )

            end = start + timedelta(
                hours=duration_hours
            )

            left = mdates.date2num(start)
            width = (
                mdates.date2num(end)
                - left
            )

            ax.barh(
                y,
                width,
                left=left,
                height=0.58,
                alpha=(
                    0.9
                    if leg["is_mission"]
                    else 0.35
                ),
                hatch=(
                    None
                    if leg["is_mission"]
                    else "//"
                ),
                edgecolor="black",
                linewidth=0.65,
            )

    ax.set_yticks(range(len(regs)))
    ax.set_yticklabels(regs)
    ax.invert_yaxis()

    ax.xaxis_date()
    ax.xaxis.set_major_locator(
        mdates.DayLocator(interval=1)
    )
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%a %d/%m")
    )
    ax.grid(axis="x", alpha=0.25)

    ax.set_xlabel("UTC date")
    ax.set_ylabel("Aircraft")

    ax.set_title(
        "ACTUAL FLOWN — historical aircraft rotations\\n"
        f'Estimated total operational cost '
        f'€{costs["total_operational_cost_eur"]:,.0f}'
    )

    info.axis("off")

    info.text(
        0.02,
        0.98,
        "\\n".join([
            "ACTUAL — ESTIMATE",
            "",
            "TOTAL OPERATIONAL COST",
            f'€{costs["total_operational_cost_eur"]:,.0f}',
            "",
            "Aircraft flying",
            (
                f'{costs["flight_hours"]:.1f} h'
                f' → €{costs["flight_cost_eur"]:,.0f}'
            ),
            "",
            "Pilot-days",
            (
                f'{costs["pilot_days"]}'
                f' → €{costs["pilot_cost_eur"]:,.0f}'
            ),
            "",
            "Away parking-days",
            (
                f'{costs["parking_days"]}'
                f' → €{costs["parking_cost_eur"]:,.0f}'
            ),
            "",
            "Empty positioning",
            (
                f'{costs["empty_legs"]} legs / '
                f'{costs["empty_nm"]:,.0f} NM'
            ),
            "",
            "Crew assumption",
            "2 pilots attached / aircraft",
            "swaps only at EBAW/EBLG",
            "non-home swaps: 0",
            "crew deadheads: €0",
            "",
            "Estimate uses route-distance",
            "block times; exact historical",
            "crew roster is unavailable.",
        ]),
        va="top",
        fontsize=8.9,
    )


class ParetoWindow:
    def __init__(self, pareto_solutions):
        # Recompute a dedicated TWO-DIMENSIONAL Pareto front using only:
        #   1) total operational cost
        #   2) operational complexity
        #
        # The optimizer's stored Pareto set is multi-objective, so some stored
        # solutions can be dominated when projected onto only these two axes.
        self.all_solutions = pareto_solutions
        self.solutions = self.cost_complexity_front(
            pareto_solutions
        )

        front_points = {
            (
                metric(s, "total_operational_cost_eur"),
                metric(s, "complexity_score"),
            )
            for s in self.solutions
        }

        self.background_solutions = [
            s
            for s in self.all_solutions
            if (
                metric(s, "total_operational_cost_eur"),
                metric(s, "complexity_score"),
            ) not in front_points
        ]

        self.fig, self.ax = plt.subplots(
            figsize=(8.5, 6.5)
        )

        self.costs = [
            metric(
                s,
                "total_operational_cost_eur",
            )
            for s in self.solutions
        ]

        self.complexities = [
            metric(
                s,
                "complexity_score",
            )
            for s in self.solutions
        ]

        # Sort front from cheapest to most expensive so a line visually traces
        # the efficient trade-off boundary.
        ordered = sorted(
            zip(
                self.costs,
                self.complexities,
                self.solutions,
            ),
            key=lambda x: (
                x[0],
                x[1],
            ),
        )

        self.costs = [
            x[0]
            for x in ordered
        ]

        self.complexities = [
            x[1]
            for x in ordered
        ]

        self.solutions = [
            x[2]
            for x in ordered
        ]

        # Show the remaining multi-objective Pareto solutions in gray as
        # context. They are dominated specifically in the cost/complexity
        # projection and therefore are not part of the 2D efficient frontier.
        self.ax.scatter(
            [
                metric(s, "total_operational_cost_eur")
                for s in self.background_solutions
            ],
            [
                metric(s, "complexity_score")
                for s in self.background_solutions
            ],
            s=24,
            alpha=0.28,
            color="gray",
            label="Other optimizer solutions",
            zorder=1,
        )

        self.ax.plot(
            self.costs,
            self.complexities,
            marker="o",
            linewidth=1.6,
            markersize=6,
            alpha=0.9,
            label="2D Pareto front",
            zorder=3,
        )

        self.highlight = self.ax.scatter(
            [],
            [],
            s=190,
            facecolors="none",
            edgecolors="black",
            linewidths=1.8,
            zorder=5,
        )

        self.annotation = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
        )
        self.annotation.set_visible(False)

        self.ax.set_title(
            "Cost vs Operational Complexity — 2D Pareto Front"
        )

        self.ax.set_xlabel(
            "Total operational cost (EUR)"
        )

        self.ax.set_ylabel(
            "Operational complexity score"
        )

        self.ax.grid(
            alpha=0.25
        )

        self.ax.legend(
            loc="best"
        )

        self.fig.tight_layout()

        print()
        print(
            f"Cost/complexity 2D Pareto front: "
            f"{len(self.solutions)} points "
            f"(from {len(self.all_solutions)} multi-objective Pareto solutions)"
        )

    @staticmethod
    def cost_complexity_front(solutions):
        """
        Keep only solutions that are non-dominated on:
          cost <= other cost
          complexity <= other complexity

        At least one inequality must be strict for domination.
        If several solutions have exactly the same (cost, complexity), keep one.
        """
        unique = {}

        for s in solutions:
            point = (
                metric(
                    s,
                    "total_operational_cost_eur",
                ),
                metric(
                    s,
                    "complexity_score",
                ),
            )

            # Keep one representative for an identical 2D point.
            if point not in unique:
                unique[point] = s

        points = [
            (
                cost,
                complexity,
                solution,
            )
            for (
                cost,
                complexity
            ), solution
            in unique.items()
        ]

        front = []

        for cost, complexity, solution in points:
            dominated = False

            for other_cost, other_complexity, _ in points:
                if (
                    other_cost <= cost
                    and other_complexity <= complexity
                    and (
                        other_cost < cost
                        or other_complexity < complexity
                    )
                ):
                    dominated = True
                    break

            if not dominated:
                front.append(
                    solution
                )

        return front

    def update(self, solution, label):
        if solution is None:
            self.highlight.set_offsets(
                [[float("nan"), float("nan")]]
            )

            self.annotation.set_visible(
                False
            )

            self.fig.canvas.draw_idle()
            return

        cost = metric(
            solution,
            "total_operational_cost_eur",
        )

        complexity = metric(
            solution,
            "complexity_score",
        )

        # Only highlight directly if this exact point lies on the 2D efficient
        # frontier. Otherwise hide the marker: the selected schedule is
        # dominated in this particular 2D projection.
        on_front = any(
            abs(
                cost
                - metric(
                    s,
                    "total_operational_cost_eur",
                )
            ) < 1e-6
            and complexity
            == metric(
                s,
                "complexity_score",
            )
            for s in self.solutions
        )

        if not on_front:
            self.highlight.set_offsets(
                [[float("nan"), float("nan")]]
            )

            self.annotation.set_visible(
                False
            )

            self.fig.canvas.draw_idle()
            return

        self.highlight.set_offsets(
            [[cost, complexity]]
        )

        self.annotation.xy = (
            cost,
            complexity,
        )

        self.annotation.set_text(
            f'{label}\n'
            f'€{cost:,.0f}\n'
            f'complexity {complexity}\n'
            f'{solution.get("schedule_signature","")}'
        )

        self.annotation.set_visible(
            True
        )

        self.fig.canvas.draw_idle()




class GUI:
    def __init__(
        self,
        named,
        gallery,
        actual,
        missions,
        airports,
        pareto_window,
    ):
        self.named = named
        self.gallery = gallery
        self.actual = actual
        self.missions = missions
        self.airports = airports
        self.pareto_window = pareto_window

        self.current_name = "CHEAPEST"
        self.gallery_index = 0
        self.browsing_gallery = False
        self.route_cache = RouteCache()

        self.rendering = False
        self.last_click_time = 0.0

        self.days_visible = 5
        self.view_start = None
        self.last_board_state = None
        self.current_solution = None

        self.fig = plt.figure(
            figsize=(17.5, 9.5)
        )

        self.ax = self.fig.add_axes(
            [
                0.105,
                0.24,
                0.69,
                0.65,
            ]
        )

        self.info = self.fig.add_axes(
            [
                0.815,
                0.24,
                0.17,
                0.65,
            ]
        )

        self.info.axis(
            "off"
        )

        top_buttons = [
            ("CHEAPEST", 0.02, 0.145, 0.12),
            ("BALANCED", 0.145, 0.145, 0.12),
            ("MIN EMPTY LEGS", 0.27, 0.145, 0.14),
            ("MIN PILOT DAYS", 0.415, 0.145, 0.14),
            ("MIN AIRCRAFT PARKING", 0.56, 0.145, 0.17),
            ("MIN COMPLEXITY", 0.735, 0.145, 0.14),
            ("ACTUAL", 0.88, 0.145, 0.09),
        ]

        lower_buttons = [
            ("PREV ALT", 0.25, 0.065, 0.12),
            ("NEXT ALT", 0.38, 0.065, 0.12),
            ("DAYS ←", 0.56, 0.065, 0.10),
            ("DAYS →", 0.67, 0.065, 0.10),
        ]

        self.buttons = []

        for (
            label,
            x,
            y,
            w,
        ) in (
            top_buttons
            + lower_buttons
        ):
            axis = self.fig.add_axes(
                [
                    x,
                    y,
                    w,
                    0.055,
                ]
            )

            button = Button(
                axis,
                label,
            )

            if label == "PREV ALT":
                button.on_clicked(
                    self.previous_alternative
                )

            elif label == "NEXT ALT":
                button.on_clicked(
                    self.next_alternative
                )

            elif label == "DAYS ←":
                button.on_clicked(
                    self.previous_days
                )

            elif label == "DAYS →":
                button.on_clicked(
                    self.next_days
                )

            else:
                button.on_clicked(
                    self.named_callback(
                        label
                    )
                )

            self.buttons.append(
                button
            )

        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.on_key,
        )

        self.fig.canvas.mpl_connect(
            "button_press_event",
            self.on_board_click,
        )

        self.draw(
            force=True
        )

    def accept_click(self):
        now = time.monotonic()

        if self.rendering:
            return False

        if (
            now
            - self.last_click_time
            < MIN_CLICK_INTERVAL_SEC
        ):
            return False

        self.last_click_time = now
        return True

    def reset_days(self):
        self.view_start = None

    def named_callback(
        self,
        label,
    ):
        def callback(
            _event
        ):
            if not self.accept_click():
                return

            self.current_name = label
            self.browsing_gallery = False
            self.reset_days()
            self.draw()

        return callback

    def on_key(
        self,
        event,
    ):
        if event.key in (
            "right",
            "n",
        ):
            self.next_alternative(
                None
            )

        elif event.key in (
            "left",
            "p",
        ):
            self.previous_alternative(
                None
            )

        elif event.key == "]":
            self.next_days(
                None
            )

        elif event.key == "[":
            self.previous_days(
                None
            )

    def next_alternative(
        self,
        _event,
    ):
        if (
            not self.gallery
            or not self.accept_click()
        ):
            return

        self.browsing_gallery = True

        self.gallery_index = (
            self.gallery_index
            + 1
        ) % len(
            self.gallery
        )

        self.reset_days()
        self.draw()

    def previous_alternative(
        self,
        _event,
    ):
        if (
            not self.gallery
            or not self.accept_click()
        ):
            return

        self.browsing_gallery = True

        self.gallery_index = (
            self.gallery_index
            - 1
        ) % len(
            self.gallery
        )

        self.reset_days()
        self.draw()

    def previous_days(
        self,
        _event,
    ):
        if (
            self.current_name
            == "ACTUAL"
        ):
            return

        if (
            self.last_board_state
            is None
        ):
            return

        current = (
            self.view_start
            or self.last_board_state[
                "view_start"
            ]
        )

        self.view_start = (
            current
            - timedelta(
                days=self.days_visible
            )
        )

        self.draw()

    def next_days(
        self,
        _event,
    ):
        if (
            self.current_name
            == "ACTUAL"
        ):
            return

        if (
            self.last_board_state
            is None
        ):
            return

        current = (
            self.view_start
            or self.last_board_state[
                "view_start"
            ]
        )

        self.view_start = (
            current
            + timedelta(
                days=self.days_visible
            )
        )

        self.draw()

    def on_board_click(
        self,
        event,
    ):
        if (
            event.inaxes
            is not self.ax
        ):
            return

        cards = getattr(
            self.ax,
            "_movement_cards",
            [],
        )

        for patch in reversed(
            cards
        ):
            contains, _ = patch.contains(
                event
            )

            if not contains:
                continue

            detail = getattr(
                patch,
                "_ops_detail",
                None,
            )

            if (
                detail is None
                or self.current_solution
                is None
            ):
                return

            render_solution_info(
                self.info,
                self.current_solution,
                self.current_name,
                detail=detail,
            )

            self.fig.canvas.draw_idle()
            return

    def draw(
        self,
        force=False,
    ):
        if (
            self.rendering
            and not force
        ):
            return

        self.rendering = True

        try:
            self.ax.cla()
            self.info.cla()
            self.current_solution = None

            if (
                self.current_name
                == "ACTUAL"
            ):
                self.browsing_gallery = False

                draw_actual(
                    self.ax,
                    self.info,
                    self.actual,
                    self.airports,
                )

                self.pareto_window.update(
                    None,
                    "ACTUAL",
                )

            elif self.browsing_gallery:
                solution = self.gallery[
                    self.gallery_index
                ]

                self.current_solution = (
                    solution
                )

                routes = self.route_cache.get(
                    solution,
                    self.missions,
                    self.airports,
                )

                self.last_board_state = draw_optimizer(
                    self.ax,
                    self.info,
                    solution,
                    routes,
                    "ALTERNATIVE",
                    (
                        self.gallery_index,
                        len(
                            self.gallery
                        ),
                    ),
                    view_start=self.view_start,
                    days_visible=self.days_visible,
                )

                self.view_start = (
                    self.last_board_state[
                        "view_start"
                    ]
                )

                self.pareto_window.update(
                    solution,
                    (
                        "ALTERNATIVE "
                        f'{self.gallery_index + 1}'
                    ),
                )

            else:
                solution = self.named[
                    self.current_name
                ]

                self.current_solution = (
                    solution
                )

                routes = self.route_cache.get(
                    solution,
                    self.missions,
                    self.airports,
                )

                self.last_board_state = draw_optimizer(
                    self.ax,
                    self.info,
                    solution,
                    routes,
                    self.current_name,
                    view_start=self.view_start,
                    days_visible=self.days_visible,
                )

                self.view_start = (
                    self.last_board_state[
                        "view_start"
                    ]
                )

                self.pareto_window.update(
                    solution,
                    self.current_name,
                )

                for (
                    i,
                    candidate,
                ) in enumerate(
                    self.gallery
                ):
                    if (
                        candidate.get(
                            "schedule_signature"
                        )
                        == solution.get(
                            "schedule_signature"
                        )
                    ):
                        self.gallery_index = i
                        break

            self.fig.canvas.draw()

            try:
                self.fig.canvas.flush_events()
            except Exception:
                pass

        finally:
            self.rendering = False

        if (
            self.gallery_index
            % 25
            == 0
        ):
            gc.collect()

def main():
    pareto = load_pareto()
    gallery = load_gallery()
    named = named_solutions(pareto)

    # Prefer the optimizer-generated mission table because raw missions.csv
    # may intentionally contain blank arrival fields.
    estimated_missions_path = DATA / "missions_with_estimates.csv"
    raw_missions_path = DATA / "missions.csv"

    missions = load_csv_dict(
        (
            estimated_missions_path
            if estimated_missions_path.exists()
            else raw_missions_path
        ),
        "id",
    )

    airports = load_csv_dict(
        DATA / "airports.csv",
        "icao",
    )

    actual = load_json(
        DATA / "actual_flown.json"
    )

    summary_path = OUTPUT / "run_summary.json"

    if summary_path.exists():
        summary = load_json(summary_path)

        print()
        print("Search diversity")
        print("=" * 60)
        print(
            "Unique aircraft schedules discovered:",
            summary.get("unique_aircraft_schedules"),
        )
        print(
            "Pareto unique aircraft schedules:",
            summary.get("pareto_unique_aircraft_schedules"),
        )
        print(
            "Alternatives loaded in GUI:",
            len(gallery),
        )
        print("=" * 60)
        print()
        print("Tip: use LEFT/RIGHT arrow keys to browse alternatives quickly.")
        print()

    pareto_window = ParetoWindow(
        pareto
    )

    GUI(
        named,
        gallery,
        actual,
        missions,
        airports,
        pareto_window,
    )

    plt.show()


if __name__ == "__main__":
    main()
