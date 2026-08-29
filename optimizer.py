
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Make progress messages visible promptly even when stdout is redirected
# to a file or another process.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

DATA = ROOT / "data"
OUTPUT = ROOT / "output"

BASES = ("EBAW", "EBLG")

TURNAROUND_MIN = 20
REPORT_MIN = 30
RELEASE_MIN = 15

POSITION_SPEED_KT = 410.0
POSITION_FIXED_MIN = 10.0

# Cost allocation
FLIGHT_HOUR_COST_EUR = 6000.0
PILOT_DAY_COST_EUR = 600.0
NONHOMEBASE_SWAP_COST_EUR = 600.0
CREW_DEADHEAD_COST_EUR = 600.0
AIRCRAFT_AWAY_PARKING_DAY_COST_EUR = 1000.0

# Complexity score
COMPLEXITY_EMPTY_LEG = 1
COMPLEXITY_INBOUND_DEADHEAD = 2
COMPLEXITY_NONHOME_SWAP = 4
COMPLEXITY_AIRCRAFT_PARKING_DAY = 1
COMPLEXITY_CREW_RETURN = 1


@dataclass(frozen=True)
class Airport:
    icao: str
    lat: float
    lon: float


@dataclass(frozen=True)
class Aircraft:
    registration: str
    capacity: int
    available_from: datetime
    status: str


@dataclass(frozen=True)
class Mission:
    id: str
    origin: str
    destination: str
    departure: datetime
    arrival: datetime
    pax: int


@dataclass(frozen=True)
class Pilot:
    id: str
    role: str
    home_base: str
    max_duty_min: int


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def load_data():
    airports = {}
    aircraft = {}
    missions = {}
    pilots = {}

    with (DATA / "airports.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            airports[row["icao"]] = Airport(
                row["icao"],
                float(row["lat"]),
                float(row["lon"]),
            )

    with (DATA / "aircraft.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aircraft[row["registration"]] = Aircraft(
                row["registration"],
                int(row["capacity"]),
                parse_dt(row["available_from"]),
                row["status"],
            )

    estimated_mission_rows = []

    with (DATA / "missions.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            departure = parse_dt(row["departure"])

            arrival_text = (row.get("arrival") or "").strip()

            if arrival_text:
                arrival = parse_dt(arrival_text)
                arrival_source = "provided"
            else:
                origin = row["origin"]
                destination = row["destination"]

                if origin not in airports:
                    raise ValueError(
                        f'Mission {row["id"]}: airport {origin} is missing '
                        f'from data/airports.csv'
                    )

                if destination not in airports:
                    raise ValueError(
                        f'Mission {row["id"]}: airport {destination} is missing '
                        f'from data/airports.csv'
                    )

                mission_distance_nm = distance_nm(
                    airports[origin],
                    airports[destination],
                )

                estimated_block_hours = positioning_hours(
                    mission_distance_nm
                )

                arrival = departure + timedelta(
                    hours=estimated_block_hours
                )

                arrival_source = "estimated"

            missions[row["id"]] = Mission(
                row["id"],
                row["origin"],
                row["destination"],
                departure,
                arrival,
                int(row["pax"]),
            )

            estimated_mission_rows.append({
                "id": row["id"],
                "origin": row["origin"],
                "destination": row["destination"],
                "departure": departure.isoformat(),
                "arrival": arrival.isoformat(),
                "pax": int(row["pax"]),
                "arrival_source": arrival_source,
            })

    estimates_path = DATA / "missions_with_estimates.csv"

    with estimates_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "origin",
                "destination",
                "departure",
                "arrival",
                "pax",
                "arrival_source",
            ],
        )

        writer.writeheader()
        writer.writerows(estimated_mission_rows)

    with (DATA / "pilots.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pilots[row["id"]] = Pilot(
                row["id"],
                row["role"],
                row["home_base"],
                int(row["max_duty_min"]),
            )

    return airports, aircraft, missions, pilots


def distance_nm(a: Airport, b: Airport) -> float:
    earth_nm = 3440.065
    p1 = radians(a.lat)
    p2 = radians(b.lat)
    dp = radians(b.lat - a.lat)
    dl = radians(b.lon - a.lon)

    h = (
        sin(dp / 2) ** 2
        + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    )

    return (
        2
        * earth_nm
        * atan2(sqrt(h), sqrt(1 - h))
    )


def positioning_hours(distance: float) -> float:
    return (
        POSITION_FIXED_MIN
        + 60.0 * distance / POSITION_SPEED_KT
    ) / 60.0


def positioning_time(distance: float) -> timedelta:
    return timedelta(hours=positioning_hours(distance))


def mission_hours(mission: Mission) -> float:
    return (
        mission.arrival - mission.departure
    ).total_seconds() / 3600.0


def nights_between(start: datetime, end: datetime) -> int:
    return max(
        0,
        (end.date() - start.date()).days,
    )


def daterange(start_day, end_day):
    day = start_day
    while day <= end_day:
        yield day
        day += timedelta(days=1)


def random_individual(
    rng: random.Random,
    mission_ids: list[str],
    available_regs: list[str],
    all_pilots: list[str],
):
    aircraft_preferences = {}

    for mid in mission_ids:
        prefs = available_regs[:]
        rng.shuffle(prefs)
        aircraft_preferences[mid] = prefs

    return {
        "aircraft_preferences": aircraft_preferences,
        # 0=STAY, 1=VIA_EBAW, 2=VIA_EBLG
        "transition_mode": {
            mid: rng.randrange(3)
            for mid in mission_ids
        },
        # Select among the top crew alternatives generated for a mission.
        "crew_choice": {
            mid: rng.randrange(6)
            for mid in mission_ids
        },
        # end-of-horizon aircraft policy per aircraft:
        # 0=cheapest, 1=stay, 2=EBAW, 3=EBLG
        "final_aircraft_policy": {
            reg: rng.randrange(4)
            for reg in available_regs
        },
        # 0=cheapest, 1=stay, 2=return
        "final_crew_policy": rng.randrange(3),
        "objectives": None,
        "solution": None,
        "rank": 0,
        "crowding": 0.0,
    }




def transition_plan(
    state,
    reg: str,
    mission: Mission,
    aircraft: dict[str, Aircraft],
    airports: dict[str, Airport],
    mode: str,
):
    """
    Build an explicit, physically timed transition into `mission`.

    Supported transition concepts:
      START_EBAW / START_EBLG
          Initial aircraft attachment and direct positioning to first mission.

      DIRECT
          Stay at the previous destination during the idle gap, then position
          directly to the next mission origin if required.

      IDLE_RETURN_EBAW / IDLE_RETURN_EBLG
          During a sufficiently long idle gap, return the aircraft + attached
          crew to a company homebase as soon as practical, wait there, then
          position directly to the next mission origin. A crew handover may
          occur at that homebase before the outbound positioning leg.

    This is deliberately different from the old automatic VIA_HOMEBASE logic:
    the homebase visit is a real idle-period decision with explicit movements,
    full empty-flight cost, and a real opportunity for a homebase crew swap.
    """
    ac = aircraft[reg]

    if ac.status != "available":
        return None

    if mission.pax > ac.capacity:
        return None

    if ac.available_from > mission.departure:
        return None

    movements = []
    empty_nm = 0.0
    empty_hours = 0.0
    empty_legs = 0
    parking_days = 0
    parking_cost_eur = 0.0

    # ---------------------------------------------------------------
    # Initial aircraft use.
    # ---------------------------------------------------------------
    if not state["aircraft_started"][reg]:
        if not mode.startswith("START_"):
            return None

        base = mode.replace("START_", "")
        current_ready = ac.available_from

        if base != mission.origin:
            d = distance_nm(
                airports[base],
                airports[mission.origin],
            )

            empty_end = (
                mission.departure
                - timedelta(minutes=TURNAROUND_MIN)
            )
            empty_start = (
                empty_end
                - positioning_time(d)
            )

            if empty_start < current_ready:
                return None

            movements.append({
                "type": "EMPTY",
                "from": base,
                "to": mission.origin,
                "start": empty_start,
                "end": empty_end,
                "nm": d,
                "target_mission": mission.id,
                "purpose": "INITIAL_POSITION",
                "captain": None,
                "fo": None,
            })

            empty_nm += d
            empty_hours += positioning_hours(d)
            empty_legs += 1

        movements.append({
            "type": "MISSION",
            "mission": mission.id,
            "from": mission.origin,
            "to": mission.destination,
            "start": mission.departure,
            "end": mission.arrival,
            "pax": mission.pax,
            "target_mission": mission.id,
            "purpose": "CUSTOMER",
            "captain": None,
            "fo": None,
        })

        first_movement = movements[0]

        return {
            "mode": mode,
            "movements": movements,
            "empty_nm": empty_nm,
            "empty_hours": empty_hours,
            "empty_legs": empty_legs,
            "parking_days": 0,
            "parking_cost_eur": 0.0,
            "gap_location": base,
            "handover_airport": base,
            "handover_deadline": first_movement["start"],
            "pre_handover_empty_count": 0,
            "idle_return_base": None,
        }

    previous_location = state[
        "aircraft_location"
    ][reg]
    previous_time = state[
        "aircraft_available"
    ][reg]

    # ---------------------------------------------------------------
    # DIRECT: park where the aircraft finished, then position once.
    # ---------------------------------------------------------------
    if mode in ("DIRECT", "STAY"):
        if previous_location == mission.origin:
            if (
                previous_time
                + timedelta(minutes=TURNAROUND_MIN)
                > mission.departure
            ):
                return None

        else:
            d = distance_nm(
                airports[previous_location],
                airports[mission.origin],
            )

            empty_end = (
                mission.departure
                - timedelta(minutes=TURNAROUND_MIN)
            )
            empty_start = (
                empty_end
                - positioning_time(d)
            )

            if (
                empty_start
                < previous_time
                + timedelta(minutes=TURNAROUND_MIN)
            ):
                return None

            movements.append({
                "type": "EMPTY",
                "from": previous_location,
                "to": mission.origin,
                "start": empty_start,
                "end": empty_end,
                "nm": d,
                "target_mission": mission.id,
                "purpose": "DIRECT_POSITION",
                "captain": None,
                "fo": None,
            })

            empty_nm += d
            empty_hours += positioning_hours(d)
            empty_legs += 1

        parking_days = (
            nights_between(
                previous_time,
                mission.departure,
            )
            if previous_location not in BASES
            else 0
        )

        parking_cost_eur = (
            parking_days
            * AIRCRAFT_AWAY_PARKING_DAY_COST_EUR
        )

        movements.append({
            "type": "MISSION",
            "mission": mission.id,
            "from": mission.origin,
            "to": mission.destination,
            "start": mission.departure,
            "end": mission.arrival,
            "pax": mission.pax,
            "target_mission": mission.id,
            "purpose": "CUSTOMER",
            "captain": None,
            "fo": None,
        })

        return {
            "mode": "DIRECT",
            "movements": movements,
            "empty_nm": empty_nm,
            "empty_hours": empty_hours,
            "empty_legs": empty_legs,
            "parking_days": parking_days,
            "parking_cost_eur": parking_cost_eur,
            # Crew remains with aircraft at its previous destination during gap.
            "gap_location": previous_location,
            # Direct model only permits a swap at next mission origin.
            "handover_airport": mission.origin,
            "handover_deadline": None,
            "pre_handover_empty_count": len(
                [m for m in movements if m["type"] == "EMPTY"]
            ),
            "idle_return_base": None,
        }

    # ---------------------------------------------------------------
    # IDLE RETURN TO HOMEBASE.
    # ---------------------------------------------------------------
    if mode.startswith("IDLE_RETURN_"):
        base = mode.replace("IDLE_RETURN_", "")

        if base not in BASES:
            return None

        cursor_time = previous_time
        pre_handover_empty_count = 0

        # Return to base as soon as practical after the previous customer leg.
        if previous_location != base:
            d_home = distance_nm(
                airports[previous_location],
                airports[base],
            )

            home_start = (
                previous_time
                + timedelta(minutes=TURNAROUND_MIN)
            )
            home_end = (
                home_start
                + positioning_time(d_home)
            )

            movements.append({
                "type": "EMPTY",
                "from": previous_location,
                "to": base,
                "start": home_start,
                "end": home_end,
                "nm": d_home,
                "target_mission": mission.id,
                "purpose": "IDLE_RETURN_HOME",
                "captain": None,
                "fo": None,
            })

            empty_nm += d_home
            empty_hours += positioning_hours(d_home)
            empty_legs += 1
            cursor_time = home_end
            pre_handover_empty_count = 1

        # Then leave base as late as possible for the next mission.
        if base != mission.origin:
            d_out = distance_nm(
                airports[base],
                airports[mission.origin],
            )

            outbound_end = (
                mission.departure
                - timedelta(minutes=TURNAROUND_MIN)
            )
            outbound_start = (
                outbound_end
                - positioning_time(d_out)
            )

            # There must be a real feasible base layover, including turnaround.
            if (
                outbound_start
                < cursor_time
                + timedelta(minutes=TURNAROUND_MIN)
            ):
                return None

            movements.append({
                "type": "EMPTY",
                "from": base,
                "to": mission.origin,
                "start": outbound_start,
                "end": outbound_end,
                "nm": d_out,
                "target_mission": mission.id,
                "purpose": "POSITION_FROM_HOME",
                "captain": None,
                "fo": None,
            })

            empty_nm += d_out
            empty_hours += positioning_hours(d_out)
            empty_legs += 1

            handover_deadline = outbound_start

        else:
            if (
                cursor_time
                + timedelta(minutes=TURNAROUND_MIN)
                > mission.departure
            ):
                return None

            handover_deadline = (
                mission.departure
                - timedelta(minutes=REPORT_MIN)
            )

        movements.append({
            "type": "MISSION",
            "mission": mission.id,
            "from": mission.origin,
            "to": mission.destination,
            "start": mission.departure,
            "end": mission.arrival,
            "pax": mission.pax,
            "target_mission": mission.id,
            "purpose": "CUSTOMER",
            "captain": None,
            "fo": None,
        })

        # No away-aircraft parking charge while the aircraft waits at a
        # company homebase. Empty-flight cost is fully counted above.
        return {
            "mode": mode,
            "movements": movements,
            "empty_nm": empty_nm,
            "empty_hours": empty_hours,
            "empty_legs": empty_legs,
            "parking_days": 0,
            "parking_cost_eur": 0.0,
            "gap_location": base,
            "handover_airport": base,
            "handover_deadline": handover_deadline,
            "pre_handover_empty_count": pre_handover_empty_count,
            "idle_return_base": base,
        }

    return None


def aircraft_transition_candidates(
    state,
    reg: str,
    mission: Mission,
    aircraft: dict[str, Aircraft],
    airports: dict[str, Airport],
    preferred_mode: int,
):
    """
    Genetic transition choice:
      0 -> DIRECT
      1 -> IDLE_RETURN_EBAW
      2 -> IDLE_RETURN_EBLG

    If the genetically requested idle return is not physically feasible,
    DIRECT remains the fallback. This lets the genetic search genuinely
    compare park-away versus return-home economics.
    """
    ac = aircraft[reg]

    if ac.status != "available":
        return []

    if mission.pax > ac.capacity:
        return []

    if not state["aircraft_started"][reg]:
        base_order = (
            ["EBAW", "EBLG"]
            if preferred_mode != 2
            else ["EBLG", "EBAW"]
        )

        candidates = []

        for base in base_order:
            plan = transition_plan(
                state,
                reg,
                mission,
                aircraft,
                airports,
                f"START_{base}",
            )

            if plan is not None:
                candidates.append(plan)

        candidates.sort(
            key=lambda plan: (
                plan["empty_hours"] * FLIGHT_HOUR_COST_EUR,
                plan["empty_legs"],
            )
        )

        return candidates

    direct = transition_plan(
        state,
        reg,
        mission,
        aircraft,
        airports,
        "DIRECT",
    )

    requested_mode = {
        0: "DIRECT",
        1: "IDLE_RETURN_EBAW",
        2: "IDLE_RETURN_EBLG",
    }.get(
        int(preferred_mode) % 3,
        "DIRECT",
    )

    if requested_mode == "DIRECT":
        return [direct] if direct is not None else []

    requested = transition_plan(
        state,
        reg,
        mission,
        aircraft,
        airports,
        requested_mode,
    )

    candidates = []

    if requested is not None:
        candidates.append(requested)

    if direct is not None:
        candidates.append(direct)

    return candidates




def validate_aircraft_movements(
    state,
    aircraft,
):
    """
    Hard physical aircraft validator.

    Two consecutive empty legs are allowed only for an explicit idle-return
    transition:
        A -> HOMEBASE  [purpose=IDLE_RETURN_HOME]
        HOMEBASE -> B  [purpose=POSITION_FROM_HOME]

    Accidental old-style VIA_HOMEBASE routing remains infeasible.
    """
    seen_missions = set()

    for reg, movements in state[
        "movements"
    ].items():
        previous = None

        for idx, movement in enumerate(movements):
            if movement["end"] <= movement["start"]:
                return (
                    False,
                    f"{reg}: non-positive movement duration",
                )

            if previous is not None:
                if movement["from"] != previous["to"]:
                    return (
                        False,
                        (
                            f'{reg}: location discontinuity '
                            f'{previous["to"]} -> {movement["from"]}'
                        ),
                    )

                if (
                    movement["start"]
                    < previous["end"]
                    + timedelta(minutes=TURNAROUND_MIN)
                ):
                    return (
                        False,
                        (
                            f'{reg}: overlap/turnaround violation '
                            f'between {previous.get("mission", previous["type"])} '
                            f'and {movement.get("mission", movement["type"])}'
                        ),
                    )

                if (
                    previous["type"] == "EMPTY"
                    and movement["type"] == "EMPTY"
                ):
                    valid_idle_pair = (
                        previous.get("purpose") == "IDLE_RETURN_HOME"
                        and movement.get("purpose") == "POSITION_FROM_HOME"
                        and previous.get("target_mission")
                        == movement.get("target_mission")
                        and previous["to"] in BASES
                        and movement["from"] == previous["to"]
                    )

                    if not valid_idle_pair:
                        return (
                            False,
                            (
                                f'{reg}: consecutive empty legs are only '
                                "allowed for explicit idle-return-home routing"
                            ),
                        )

            if movement["type"] == "MISSION":
                mid = movement["mission"]

                if mid in seen_missions:
                    return (
                        False,
                        f"mission {mid} assigned more than once",
                    )

                seen_missions.add(mid)

            previous = movement

    expected = set(
        state["mission_aircraft"]
    )

    if seen_missions != expected:
        return (
            False,
            (
                "mission movement set does not match assigned missions: "
                f'missing={sorted(expected-seen_missions)} '
                f'extra={sorted(seen_missions-expected)}'
            ),
        )

    return True, None



def decode_aircraft(
    individual,
    mission_order: list[Mission],
    aircraft: dict[str, Aircraft],
    airports: dict[str, Airport],
):
    available_regs = [
        reg
        for reg, ac
        in aircraft.items()
        if ac.status == "available"
    ]

    state = {
        "aircraft_started": {
            reg: False
            for reg in available_regs
        },
        "aircraft_location": {
            reg: None
            for reg in available_regs
        },
        "aircraft_start_base": {
            reg: None
            for reg in available_regs
        },
        "aircraft_available": {
            reg: aircraft[reg].available_from
            for reg in available_regs
        },
        "routes": {
            reg: []
            for reg in available_regs
        },
        "aircraft_actions": {
            reg: []
            for reg in available_regs
        },
        "movements": {
            reg: []
            for reg in available_regs
        },
        "transition_details": {},
        "empty_nm": 0.0,
        "empty_hours": 0.0,
        "empty_legs": 0,
        "parking_days": 0,
        "parking_cost_eur": 0.0,
    }

    mission_aircraft = {}

    for mission in mission_order:
        assigned = False

        preferences = individual[
            "aircraft_preferences"
        ][mission.id]

        for reg in preferences:
            if reg not in state["routes"]:
                continue

            candidates = aircraft_transition_candidates(
                state,
                reg,
                mission,
                aircraft,
                airports,
                individual[
                    "transition_mode"
                ][mission.id],
            )

            if not candidates:
                continue

            action = candidates[0]

            if not state["aircraft_started"][reg]:
                if action["mode"].startswith("START_"):
                    state[
                        "aircraft_start_base"
                    ][reg] = action[
                        "mode"
                    ].replace(
                        "START_",
                        "",
                    )

            state[
                "aircraft_started"
            ][reg] = True

            state[
                "aircraft_location"
            ][reg] = mission.destination

            state[
                "aircraft_available"
            ][reg] = mission.arrival

            state[
                "routes"
            ][reg].append(
                mission.id
            )

            state[
                "movements"
            ][reg].extend(
                action["movements"]
            )

            state[
                "transition_details"
            ][
                (
                    reg,
                    mission.id,
                )
            ] = {
                "mode": action["mode"],
                "gap_location": action["gap_location"],
                "handover_airport": action["handover_airport"],
                "handover_deadline": action["handover_deadline"],
                "pre_handover_empty_count": action[
                    "pre_handover_empty_count"
                ],
                "idle_return_base": action["idle_return_base"],
            }

            state[
                "aircraft_actions"
            ][reg].append({
                "mission": mission.id,
                "transition_mode": action["mode"],
                "parking_days": action["parking_days"],
                "parking_cost_eur": round(
                    action[
                        "parking_cost_eur"
                    ],
                    2,
                ),
                "idle_return_base": action["idle_return_base"],
                "crew_handover_airport": action["handover_airport"],
            })

            state["empty_nm"] += action[
                "empty_nm"
            ]

            state["empty_hours"] += action[
                "empty_hours"
            ]

            state["empty_legs"] += action[
                "empty_legs"
            ]

            state["parking_days"] += action[
                "parking_days"
            ]

            state[
                "parking_cost_eur"
            ] += action[
                "parking_cost_eur"
            ]

            mission_aircraft[
                mission.id
            ] = reg

            # Keep validator view up to date during decoding.
            state[
                "mission_aircraft"
            ] = mission_aircraft

            valid, _ = validate_aircraft_movements(
                state,
                aircraft,
            )

            if not valid:
                # This should normally be impossible because transition_plan
                # already enforces timing, but never allow repair-by-penalty.
                return None

            assigned = True
            break

        if not assigned:
            return None

    state[
        "mission_aircraft"
    ] = mission_aircraft

    valid, reason = validate_aircraft_movements(
        state,
        aircraft,
    )

    if not valid:
        state["invalid_reason"] = reason
        return None

    return state



def pilot_travel_options(
    crew_state,
    pid: str,
    mission: Mission,
    pilot: Pilot,
    airports: dict[str, Airport],
    continuing: bool,
):
    report = (
        mission.departure
        - timedelta(minutes=REPORT_MIN)
    )

    release = (
        mission.arrival
        + timedelta(minutes=RELEASE_MIN)
    )

    current_location = crew_state[
        "pilot_location"
    ].get(
        pid,
        pilot.home_base,
    )

    available_at = crew_state[
        "pilot_available"
    ].get(
        pid,
        datetime.combine(
            mission.departure.date(),
            datetime.min.time(),
        ),
    )

    duty_start = crew_state[
        "duty_start"
    ].get(
        (pid, mission.departure.date()),
        report,
    )

    if (
        release - duty_start
    ).total_seconds() / 60.0 > pilot.max_duty_min:
        return []

    # Continuing crew is allowed to follow the aircraft
    # positioning leg with no commercial deadhead charge.
    if continuing:
        return [{
            "mode": "CONTINUE",
            "new_location_before_mission": mission.origin,
            "return_cost_eur": 0.0,
            "inbound_deadhead_cost_eur": 0.0,
            "return_pilots": 0,
            "inbound_deadhead_pilots": 0,
            "position_nm": 0.0,
            "away_days": 0,
        }]

    options = []

    # Stay where pilot currently is, then deadhead to mission origin if required.
    inbound_nm = distance_nm(
        airports[current_location],
        airports[mission.origin],
    )

    if (
        available_at
        + positioning_time(inbound_nm)
        <= report
    ):
        away_days = 0
        previous_day = crew_state[
            "pilot_last_day"
        ].get(pid)

        if (
            previous_day is not None
            and current_location
            != pilot.home_base
        ):
            away_days = max(
                0,
                (
                    mission.departure.date()
                    - previous_day
                ).days - 1,
            )

        options.append({
            "mode": "STAY",
            "new_location_before_mission": current_location,
            "return_cost_eur": 0.0,
            "inbound_deadhead_cost_eur": (
                CREW_DEADHEAD_COST_EUR
                if current_location
                != mission.origin
                else 0.0
            ),
            "return_pilots": 0,
            "inbound_deadhead_pilots": int(
                current_location
                != mission.origin
            ),
            "position_nm": inbound_nm,
            "away_days": away_days,
        })

    # Return to a base after prior activity, then deadhead to mission origin.
    for base in dict.fromkeys(
        [pilot.home_base, *BASES]
    ):
        to_base_nm = distance_nm(
            airports[current_location],
            airports[base],
        )

        from_base_nm = distance_nm(
            airports[base],
            airports[mission.origin],
        )

        base_arrival = (
            available_at
            + positioning_time(to_base_nm)
        )

        if (
            base_arrival
            + positioning_time(
                from_base_nm
            )
            <= report
        ):
            # If pilot returns to the *other* company base,
            # they are still away from personal home base.
            away_days = 0
            previous_day = crew_state[
                "pilot_last_day"
            ].get(pid)

            if (
                previous_day is not None
                and base != pilot.home_base
            ):
                away_days = max(
                    0,
                    (
                        mission.departure.date()
                        - previous_day
                    ).days - 1,
                )

            options.append({
                "mode": f"RETURN_{base}",
                "new_location_before_mission": base,
                "return_cost_eur": (
                    0.0
                    if current_location == base
                    else CREW_DEADHEAD_COST_EUR
                ),
                "inbound_deadhead_cost_eur": (
                    0.0
                    if base == mission.origin
                    else CREW_DEADHEAD_COST_EUR
                ),
                "return_pilots": int(
                    current_location != base
                ),
                "inbound_deadhead_pilots": int(
                    base != mission.origin
                ),
                "position_nm": from_base_nm,
                "away_days": away_days,
            })

    return options



def charge_attached_gap_days(
    charged_days,
    pid,
    pilot,
    aircraft_location,
    previous_day,
    new_day,
):
    if previous_day is None:
        return

    if aircraft_location == pilot.home_base:
        return

    day = previous_day + timedelta(days=1)
    while day < new_day:
        charged_days.add((pid, day))
        day += timedelta(days=1)


def mission_duty_feasible(
    crew_state,
    pid,
    mission,
    pilot,
):
    report = mission.departure - timedelta(minutes=REPORT_MIN)
    release = mission.arrival + timedelta(minutes=RELEASE_MIN)

    key = (pid, mission.departure.date())
    duty_start = crew_state["duty_start"].get(key, report)

    return (
        (release - duty_start).total_seconds() / 60.0
        <= pilot.max_duty_min
    )



def available_replacement_pilots(
    crew_state,
    role,
    aircraft_location,
    mission,
    pilots,
    airports,
    excluded,
    must_arrive_by=None,
):
    """
    Candidate pilot must be physically able to reach the handover airport.

    For an initial aircraft attachment, must_arrive_by is the departure time
    of the FIRST aircraft movement (including an empty positioning leg).
    """
    report = (
        must_arrive_by
        if must_arrive_by is not None
        else mission.departure - timedelta(minutes=REPORT_MIN)
    )

    options = []

    for pilot in pilots.values():
        if pilot.role != role:
            continue

        if pilot.id in excluded:
            continue

        if crew_state[
            "pilot_attached_to"
        ].get(pilot.id) is not None:
            continue

        if not mission_duty_feasible(
            crew_state,
            pilot.id,
            mission,
            pilot,
        ):
            continue

        current_location = crew_state[
            "pilot_location"
        ].get(
            pilot.id,
            pilot.home_base,
        )

        available_at = crew_state[
            "pilot_available"
        ].get(
            pilot.id,
            mission.departure
            - timedelta(days=30),
        )

        travel_nm = distance_nm(
            airports[current_location],
            airports[aircraft_location],
        )

        travel = positioning_time(
            travel_nm
        )

        if (
            available_at
            + travel
            > report
        ):
            continue

        inbound = int(
            current_location
            != aircraft_location
        )

        options.append({
            "pilot_id": pilot.id,
            "current_location": current_location,
            "inbound_deadhead_pilots": inbound,
            "inbound_deadhead_cost_eur": (
                inbound
                * CREW_DEADHEAD_COST_EUR
            ),
            "position_nm": travel_nm,
        })

    options.sort(
        key=lambda x: (
            x[
                "inbound_deadhead_cost_eur"
            ],
            x["position_nm"],
            x["pilot_id"],
        )
    )

    return options[:10]



def crew_candidates(
    crew_state,
    mission,
    assigned_aircraft,
    aircraft_location_before_mission,
    pilots,
    airports,
    initial_attach_airport=None,
    initial_attach_deadline=None,
    gap_location=None,
    handover_airport=None,
    handover_deadline=None,
):
    current_pair = crew_state["aircraft_crew"].get(assigned_aircraft)
    day = mission.departure.date()
    candidates = []

    effective_gap_location = (
        gap_location
        if gap_location is not None
        else aircraft_location_before_mission
    )

    effective_handover_airport = (
        handover_airport
        if handover_airport is not None
        else aircraft_location_before_mission
    )

    # Continue attached crew.
    if current_pair is not None:
        captain, fo = current_pair

        if (
            mission_duty_feasible(
                crew_state, captain, mission, pilots[captain]
            )
            and mission_duty_feasible(
                crew_state, fo, mission, pilots[fo]
            )
        ):
            charged = set(crew_state["charged_pilot_days"])
            previous_day = crew_state[
                "aircraft_crew_last_activity_day"
            ].get(assigned_aircraft)

            for pid in current_pair:
                charge_attached_gap_days(
                    charged,
                    pid,
                    pilots[pid],
                    effective_gap_location,
                    previous_day,
                    day,
                )
                charged.add((pid, day))

            candidates.append({
                "captain": captain,
                "fo": fo,
                "handover": False,
                "changed_pilots": 0,
                "nonhome_swap_event": 0,
                "charged_days": charged,
                "return_pilots": 0,
                "return_cost_eur": 0.0,
                "inbound_deadhead_pilots": 0,
                "inbound_cost_eur": 0.0,
                "crew_position_nm": 0.0,
                "captain_mode": "ATTACHED",
                "fo_mode": "ATTACHED",
            })

    replacement_airport = (
        initial_attach_airport
        if current_pair is None
        and initial_attach_airport is not None
        else effective_handover_airport
    )

    replacement_deadline = (
        initial_attach_deadline
        if current_pair is None
        else handover_deadline
    )

    captain_options = available_replacement_pilots(
        crew_state,
        "CAPT",
        replacement_airport,
        mission,
        pilots,
        airports,
        excluded=set(current_pair or ()),
        must_arrive_by=replacement_deadline,
    )

    fo_options = available_replacement_pilots(
        crew_state,
        "FO",
        replacement_airport,
        mission,
        pilots,
        airports,
        excluded=set(current_pair or ()),
        must_arrive_by=replacement_deadline,
    )

    if current_pair is None:
        for c in captain_options:
            for f in fo_options:
                charged = set(crew_state["charged_pilot_days"])
                charged.add((c["pilot_id"], day))
                charged.add((f["pilot_id"], day))

                candidates.append({
                    "captain": c["pilot_id"],
                    "fo": f["pilot_id"],
                    "handover": True,
                    "changed_pilots": 0,
                    "nonhome_swap_event": 0,
                    "charged_days": charged,
                    "return_pilots": 0,
                    "return_cost_eur": 0.0,
                    "inbound_deadhead_pilots": (
                        c["inbound_deadhead_pilots"]
                        + f["inbound_deadhead_pilots"]
                    ),
                    "inbound_cost_eur": (
                        c["inbound_deadhead_cost_eur"]
                        + f["inbound_deadhead_cost_eur"]
                    ),
                    "crew_position_nm": (
                        c["position_nm"] + f["position_nm"]
                    ),
                    "captain_mode": "INITIAL_ATTACH",
                    "fo_mode": "INITIAL_ATTACH",
                })
    else:
        for c in captain_options:
            for f in fo_options:
                charged = set(crew_state["charged_pilot_days"])
                previous_day = crew_state[
                    "aircraft_crew_last_activity_day"
                ].get(assigned_aircraft)

                # Old crew stayed with aircraft throughout the gap.
                for pid in current_pair:
                    charge_attached_gap_days(
                        charged,
                        pid,
                        pilots[pid],
                        aircraft_location_before_mission,
                        previous_day,
                        day,
                    )

                charged.add((c["pilot_id"], day))
                charged.add((f["pilot_id"], day))

                outgoing_returns = sum(
                    int(
                        effective_handover_airport
                        != pilots[pid].home_base
                    )
                    for pid in current_pair
                )

                nonhome_event = int(
                    effective_handover_airport not in BASES
                )

                candidates.append({
                    "captain": c["pilot_id"],
                    "fo": f["pilot_id"],
                    "handover": True,
                    "changed_pilots": 2,
                    "nonhome_swap_event": nonhome_event,
                    "charged_days": charged,
                    "return_pilots": outgoing_returns,
                    "return_cost_eur": (
                        outgoing_returns * CREW_DEADHEAD_COST_EUR
                    ),
                    "inbound_deadhead_pilots": (
                        c["inbound_deadhead_pilots"]
                        + f["inbound_deadhead_pilots"]
                    ),
                    "inbound_cost_eur": (
                        c["inbound_deadhead_cost_eur"]
                        + f["inbound_deadhead_cost_eur"]
                    ),
                    "crew_position_nm": (
                        c["position_nm"] + f["position_nm"]
                    ),
                    "captain_mode": "HANDOVER_IN",
                    "fo_mode": "HANDOVER_IN",
                })

    candidates.sort(
        key=lambda x: (
            (
                len(x["charged_days"])
                - len(crew_state["charged_pilot_days"])
            ) * PILOT_DAY_COST_EUR
            + x["return_cost_eur"]
            + x["inbound_cost_eur"]
            + x["nonhome_swap_event"]
            * NONHOMEBASE_SWAP_COST_EUR,
            x["handover"],
            x["inbound_deadhead_pilots"],
        )
    )

    return candidates[:18]



def validate_crew_movements(
    aircraft_state,
    crew_state,
):
    """
    HARD crew/aircraft continuity validation.

    Every aircraft movement, including every empty leg, must have a CAPT + FO.
    A pilot may not operate overlapping movements on different aircraft.
    """
    pilot_flights = {}

    for reg, movements in aircraft_state[
        "movements"
    ].items():
        for movement in movements:
            captain = movement.get(
                "captain"
            )
            fo = movement.get(
                "fo"
            )

            if not captain or not fo:
                return (
                    False,
                    (
                        f'{reg}: {movement["type"]} '
                        f'{movement["from"]}->{movement["to"]} '
                        "has no complete attached crew"
                    ),
                )

            if captain == fo:
                return (
                    False,
                    f"{reg}: captain and FO are the same pilot",
                )

            for pid in (
                captain,
                fo,
            ):
                pilot_flights.setdefault(
                    pid,
                    [],
                ).append(
                    (
                        movement["start"],
                        movement["end"],
                        reg,
                        movement,
                    )
                )

    for pid, flights in pilot_flights.items():
        flights.sort(
            key=lambda x: x[0]
        )

        for previous, current in zip(
            flights,
            flights[1:],
        ):
            if (
                current[0]
                < previous[1]
            ):
                return (
                    False,
                    (
                        f'{pid}: overlapping aircraft movements '
                        f'on {previous[2]} and {current[2]}'
                    ),
                )

    return True, None


def assign_crew(
    individual,
    aircraft_state,
    mission_order,
    pilots,
    airports,
):
    crew_state = {
        "pilot_location": {},
        "pilot_available": {},
        "pilot_last_day": {},
        "pilot_attached_to": {},
        "duty_start": {},
        "charged_pilot_days": set(),
        "aircraft_crew": {},
        "aircraft_crew_last_activity_day": {},
        "crew_assignments": {},
        "crew_actions": [],
        "changed_pilots": 0,
        "nonhome_swap_events": 0,
        "return_cost_eur": 0.0,
        "inbound_cost_eur": 0.0,
        "return_pilots": 0,
        "inbound_deadhead_pilots": 0,
        "crew_position_nm": 0.0,
    }

    # Convenient movement lookup by target customer mission.
    transition_movements = {}

    for reg, movements in aircraft_state[
        "movements"
    ].items():
        for movement in movements:
            transition_movements.setdefault(
                (
                    reg,
                    movement[
                        "target_mission"
                    ],
                ),
                [],
            ).append(
                movement
            )

    for mission in mission_order:
        reg = aircraft_state[
            "mission_aircraft"
        ][mission.id]

        old_pair = crew_state[
            "aircraft_crew"
        ].get(reg)

        related = transition_movements[
            (
                reg,
                mission.id,
            )
        ]

        transition_detail = aircraft_state.get(
            "transition_details",
            {},
        ).get(
            (
                reg,
                mission.id,
            ),
            {},
        )

        empty_before_mission = [
            movement
            for movement in related
            if movement["type"] == "EMPTY"
        ]

        mission_movement = next(
            movement
            for movement in related
            if movement["type"] == "MISSION"
        )

        # ----------------------------------------------------------
        # INITIAL ATTACH:
        # crew must join the aircraft BEFORE its very first movement.
        # ----------------------------------------------------------
        if old_pair is None:
            attach_airport = aircraft_state[
                "aircraft_start_base"
            ][reg]

            first_movement = related[0]

            attach_deadline = first_movement[
                "start"
            ]

            candidates = crew_candidates(
                crew_state,
                mission,
                reg,
                mission.origin,
                pilots,
                airports,
                initial_attach_airport=attach_airport,
                initial_attach_deadline=attach_deadline,
            )

        else:
            candidates = crew_candidates(
                crew_state,
                mission,
                reg,
                mission.origin,
                pilots,
                airports,
                gap_location=transition_detail.get(
                    "gap_location",
                    mission.origin,
                ),
                handover_airport=transition_detail.get(
                    "handover_airport",
                    mission.origin,
                ),
                handover_deadline=transition_detail.get(
                    "handover_deadline",
                ),
            )

        if not candidates:
            return None

        chosen = candidates[
            individual[
                "crew_choice"
            ][mission.id]
            % len(candidates)
        ]

        new_pair = (
            chosen["captain"],
            chosen["fo"],
        )

        # Crew on transition empty legs.
        #
        # Initial attach: new crew operates every positioning leg.
        #
        # Direct transition: old crew operates the positioning leg and an
        # optional swap happens at next mission origin.
        #
        # Idle-return transition:
        #   old crew operates A -> homebase;
        #   optional handover at homebase;
        #   new/current crew operates homebase -> next origin.
        pre_handover_empty_count = int(
            transition_detail.get(
                "pre_handover_empty_count",
                0,
            )
        )

        for idx, movement in enumerate(
            empty_before_mission
        ):
            if old_pair is None:
                pair_for_leg = new_pair
            elif idx < pre_handover_empty_count:
                pair_for_leg = old_pair
            else:
                pair_for_leg = new_pair

            movement["captain"] = pair_for_leg[0]
            movement["fo"] = pair_for_leg[1]

        actual_handover_airport = (
            transition_detail.get(
                "handover_airport"
            )
            or mission.origin
        )

        actual_handover_time = (
            transition_detail.get(
                "handover_deadline"
            )
            or mission.departure
        )

        # Explicit handover at the real transition handover point.
        if (
            old_pair is not None
            and new_pair != old_pair
        ):
            for pid in old_pair:
                crew_state[
                    "pilot_attached_to"
                ][pid] = None

                if (
                    actual_handover_airport
                    == pilots[pid].home_base
                ):
                    crew_state[
                        "pilot_location"
                    ][pid] = actual_handover_airport

                    crew_state[
                        "pilot_available"
                    ][pid] = actual_handover_time

                else:
                    crew_state[
                        "pilot_location"
                    ][pid] = pilots[
                        pid
                    ].home_base

                    crew_state[
                        "pilot_available"
                    ][pid] = (
                        actual_handover_time
                        + positioning_time(
                            distance_nm(
                                airports[
                                    actual_handover_airport
                                ],
                                airports[
                                    pilots[
                                        pid
                                    ].home_base
                                ],
                            )
                        )
                    )

        # New/current crew flies the customer mission.
        mission_movement[
            "captain"
        ] = new_pair[0]
        mission_movement[
            "fo"
        ] = new_pair[1]

        for pid in new_pair:
            crew_state[
                "pilot_attached_to"
            ][pid] = reg

            crew_state[
                "pilot_location"
            ][pid] = mission.destination

            crew_state[
                "pilot_available"
            ][pid] = (
                mission.arrival
                + timedelta(
                    minutes=RELEASE_MIN
                )
            )

            crew_state[
                "pilot_last_day"
            ][pid] = (
                mission.departure.date()
            )

            crew_state[
                "duty_start"
            ].setdefault(
                (
                    pid,
                    mission.departure.date(),
                ),
                mission.departure
                - timedelta(
                    minutes=REPORT_MIN
                ),
            )

        crew_state[
            "aircraft_crew"
        ][reg] = new_pair

        crew_state[
            "aircraft_crew_last_activity_day"
        ][reg] = (
            mission.departure.date()
        )

        crew_state[
            "charged_pilot_days"
        ] = set(
            chosen["charged_days"]
        )

        crew_state[
            "changed_pilots"
        ] += chosen[
            "changed_pilots"
        ]

        crew_state[
            "nonhome_swap_events"
        ] += chosen[
            "nonhome_swap_event"
        ]

        crew_state[
            "return_cost_eur"
        ] += chosen[
            "return_cost_eur"
        ]

        crew_state[
            "inbound_cost_eur"
        ] += chosen[
            "inbound_cost_eur"
        ]

        crew_state[
            "return_pilots"
        ] += chosen[
            "return_pilots"
        ]

        crew_state[
            "inbound_deadhead_pilots"
        ] += chosen[
            "inbound_deadhead_pilots"
        ]

        crew_state[
            "crew_position_nm"
        ] += chosen[
            "crew_position_nm"
        ]

        crew_state[
            "crew_assignments"
        ][mission.id] = {
            "captain": new_pair[0],
            "fo": new_pair[1],
        }

        handover_airport = (
            aircraft_state[
                "aircraft_start_base"
            ][reg]
            if old_pair is None
            else actual_handover_airport
        )

        crew_state[
            "crew_actions"
        ].append({
            "mission": mission.id,
            "aircraft": reg,
            "captain": new_pair[0],
            "fo": new_pair[1],
            "action": (
                "HANDOVER"
                if old_pair is not None
                and new_pair != old_pair
                else (
                    "INITIAL_ATTACH"
                    if old_pair is None
                    else "CONTINUE"
                )
            ),
            "handover_airport": handover_airport,
            "transition_mode": transition_detail.get(
                "mode",
                "INITIAL",
            ),
            "changed_pilots": chosen[
                "changed_pilots"
            ],
            "nonhome_swap_event": chosen[
                "nonhome_swap_event"
            ],
            "return_cost_eur": round(
                chosen[
                    "return_cost_eur"
                ],
                2,
            ),
            "inbound_deadhead_cost_eur": round(
                chosen[
                    "inbound_cost_eur"
                ],
                2,
            ),
            "return_pilots": chosen[
                "return_pilots"
            ],
            "inbound_deadhead_pilots": chosen[
                "inbound_deadhead_pilots"
            ],
        })

        valid, reason = validate_crew_movements(
            aircraft_state,
            crew_state,
        )

        # During partial assignment, future movements still have no crew.
        # Validate only movements through the current mission.
        if not valid:
            # Ignore only future-unassigned-movement complaints.
            if (
                reason is None
                or "has no complete attached crew"
                not in reason
            ):
                return None

    # Final hard check: now EVERY flight movement must have crew.
    valid, reason = validate_crew_movements(
        aircraft_state,
        crew_state,
    )

    if not valid:
        crew_state[
            "invalid_reason"
        ] = reason
        return None

    return crew_state




def apply_final_aircraft_policy(
    individual,
    aircraft_state,
    aircraft: dict[str, Aircraft],
    airports: dict[str, Airport],
    horizon_end: datetime,
):
    state = copy.deepcopy(
        aircraft_state
    )

    state["final_aircraft_locations"] = {
        reg: state["aircraft_location"][reg]
        for reg in state["routes"]
    }

    for reg in state["routes"]:
        if not state["routes"][reg]:
            continue

        current = state[
            "aircraft_location"
        ][reg]

        last_time = state[
            "aircraft_available"
        ][reg]

        options = []

        parking_days = (
            nights_between(
                last_time,
                horizon_end,
            )
            if current not in BASES
            else 0
        )

        options.append({
            "mode": "STAY_END",
            "empty_nm": 0.0,
            "empty_hours": 0.0,
            "empty_legs": 0,
            "parking_days": parking_days,
            "parking_cost_eur": (
                parking_days
                * AIRCRAFT_AWAY_PARKING_DAY_COST_EUR
            ),
            "movement": None,
        })

        for base in BASES:
            if current == base:
                options.append({
                    "mode": f"RETURN_END_{base}",
                    "empty_nm": 0.0,
                    "empty_hours": 0.0,
                    "empty_legs": 0,
                    "parking_days": 0,
                    "parking_cost_eur": 0.0,
                    "movement": None,
                })
                continue

            d = distance_nm(
                airports[current],
                airports[base],
            )

            start = (
                last_time
                + timedelta(minutes=TURNAROUND_MIN)
            )

            end = (
                start
                + positioning_time(d)
            )

            # A return selected as part of the planning horizon must physically
            # fit inside that horizon.
            if end > horizon_end:
                continue

            last_movement = state[
                "movements"
            ][reg][-1]

            captain = last_movement.get(
                "captain"
            )
            fo = last_movement.get(
                "fo"
            )

            if not captain or not fo:
                continue

            options.append({
                "mode": f"RETURN_END_{base}",
                "empty_nm": d,
                "empty_hours": positioning_hours(d),
                "empty_legs": 1,
                "parking_days": 0,
                "parking_cost_eur": 0.0,
                "movement": {
                    "type": "EMPTY",
                    "from": current,
                    "to": base,
                    "start": start,
                    "end": end,
                    "nm": d,
                    "target_mission": "__END__",
                    "captain": captain,
                    "fo": fo,
                },
            })

        policy = individual[
            "final_aircraft_policy"
        ][reg]

        if policy == 1:
            chosen = options[0]

        elif policy in (
            2,
            3,
        ):
            desired_base = (
                "EBAW"
                if policy == 2
                else "EBLG"
            )

            matching = [
                option
                for option in options
                if option["mode"]
                == f"RETURN_END_{desired_base}"
            ]

            # If the requested return cannot fit in the horizon, staying is the
            # only physically feasible repair.
            chosen = (
                matching[0]
                if matching
                else options[0]
            )

        else:
            chosen = min(
                options,
                key=lambda x: (
                    x["empty_hours"]
                    * FLIGHT_HOUR_COST_EUR
                    + x[
                        "parking_cost_eur"
                    ],
                    x["empty_legs"],
                )
            )

        state["empty_nm"] += chosen[
            "empty_nm"
        ]

        state["empty_hours"] += chosen[
            "empty_hours"
        ]

        state["empty_legs"] += chosen[
            "empty_legs"
        ]

        state["parking_days"] += chosen[
            "parking_days"
        ]

        state[
            "parking_cost_eur"
        ] += chosen[
            "parking_cost_eur"
        ]

        if chosen[
            "movement"
        ] is not None:
            state[
                "movements"
            ][reg].append(
                chosen["movement"]
            )

            state[
                "aircraft_available"
            ][reg] = chosen[
                "movement"
            ]["end"]

            state[
                "aircraft_location"
            ][reg] = chosen[
                "movement"
            ]["to"]

        if chosen[
            "mode"
        ] == "RETURN_END_EBAW":
            state[
                "final_aircraft_locations"
            ][reg] = "EBAW"

        elif chosen[
            "mode"
        ] == "RETURN_END_EBLG":
            state[
                "final_aircraft_locations"
            ][reg] = "EBLG"

        else:
            state[
                "final_aircraft_locations"
            ][reg] = current

        state[
            "aircraft_actions"
        ][reg].append({
            "mission": "__END__",
            "transition_mode": chosen[
                "mode"
            ],
            "parking_days": chosen[
                "parking_days"
            ],
            "parking_cost_eur": round(
                chosen[
                    "parking_cost_eur"
                ],
                2,
            ),
        })

    valid, reason = validate_aircraft_movements(
        state,
        aircraft,
    )

    if not valid:
        state[
            "invalid_reason"
        ] = reason
        return None

    return state



def apply_final_crew_policy(
    individual,
    crew_state,
    pilots,
    horizon_end_day,
    final_aircraft_locations=None,
):
    state = copy.deepcopy(crew_state)
    final_actions = []
    final_aircraft_locations = final_aircraft_locations or {}

    for reg, pair in state["aircraft_crew"].items():
        aircraft_location = final_aircraft_locations.get(reg)

        if aircraft_location is None:
            continue

        last_day = state["aircraft_crew_last_activity_day"].get(reg)

        if last_day is None:
            continue

        if aircraft_location in BASES:
            for pid in pair:
                state["pilot_attached_to"][pid] = None
                state["pilot_location"][pid] = pilots[pid].home_base

                final_actions.append({
                    "pilot_id": pid,
                    "aircraft": reg,
                    "decision": "RELEASE_AT_BASE",
                    "cost_eur": 0.0,
                    "extra_away_days": 0,
                })
        else:
            for pid in pair:
                day = last_day + timedelta(days=1)

                while day <= horizon_end_day:
                    if aircraft_location != pilots[pid].home_base:
                        state["charged_pilot_days"].add((pid, day))
                    day += timedelta(days=1)

                final_actions.append({
                    "pilot_id": pid,
                    "aircraft": reg,
                    "decision": "STAY_WITH_AIRCRAFT",
                    "cost_eur": 0.0,
                    "extra_away_days": max(
                        0,
                        (horizon_end_day - last_day).days,
                    ),
                })

    state["final_crew_decisions"] = final_actions
    return state




def mission_snapshot(
    mission_order,
):
    return {
        mission.id: {
            "id": mission.id,
            "origin": mission.origin,
            "destination": mission.destination,
            "departure": mission.departure.isoformat(),
            "arrival": mission.arrival.isoformat(),
            "pax": mission.pax,
        }
        for mission in mission_order
    }


def mission_dataset_fingerprint(
    mission_order,
):
    raw = json.dumps(
        mission_snapshot(
            mission_order
        ),
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    return hashlib.sha1(
        raw
    ).hexdigest()[:16]


def serialized_movements(
    aircraft_state,
):
    result = {}

    for reg, movements in aircraft_state[
        "movements"
    ].items():
        result[reg] = []

        for movement in movements:
            row = dict(
                movement
            )
            row["start"] = movement[
                "start"
            ].isoformat()
            row["end"] = movement[
                "end"
            ].isoformat()

            if "nm" in row:
                row["nm"] = round(
                    float(row["nm"]),
                    2,
                )

            result[reg].append(
                row
            )

    return result


def schedule_signature(solution) -> str:
    payload = {
        "routes": solution["routes"],
        "aircraft_actions": {
            reg: [
                (
                    a.get("mission"),
                    a.get("transition_mode"),
                )
                for a in actions
            ]
            for reg, actions
            in solution[
                "aircraft_actions"
            ].items()
        },
    }

    raw = json.dumps(
        payload,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha1(
        raw
    ).hexdigest()[:12]


def evaluate(
    individual,
    mission_order,
    aircraft,
    airports,
    pilots,
    commercial_hours,
    horizon_end,
):
    aircraft_state = decode_aircraft(
        individual,
        mission_order,
        aircraft,
        airports,
    )

    if aircraft_state is None:
        individual["objectives"] = (
            1e12,
            1e9,
            1e9,
            1e9,
            1e9,
        )
        individual["solution"] = None
        individual["invalid_reason"] = "AIRCRAFT_FEASIBILITY"
        return

    aircraft_valid, aircraft_reason = validate_aircraft_movements(
        aircraft_state,
        aircraft,
    )

    if not aircraft_valid:
        individual["objectives"] = (
            1e12,
            1e9,
            1e9,
            1e9,
            1e9,
        )
        individual["solution"] = None
        individual["invalid_reason"] = aircraft_reason
        return

    crew_state = assign_crew(
        individual,
        aircraft_state,
        mission_order,
        pilots,
        airports,
    )

    if crew_state is None:
        individual["objectives"] = (
            1e12,
            1e9,
            1e9,
            1e9,
            1e9,
        )
        individual["solution"] = None
        individual["invalid_reason"] = "CREW_FEASIBILITY"
        return

    crew_valid, crew_reason = validate_crew_movements(
        aircraft_state,
        crew_state,
    )

    if not crew_valid:
        individual["objectives"] = (
            1e12,
            1e9,
            1e9,
            1e9,
            1e9,
        )
        individual["solution"] = None
        individual["invalid_reason"] = crew_reason
        return

    aircraft_state = apply_final_aircraft_policy(
        individual,
        aircraft_state,
        aircraft,
        airports,
        horizon_end,
    )

    if aircraft_state is None:
        individual["objectives"] = (
            1e12,
            1e9,
            1e9,
            1e9,
            1e9,
        )
        individual["solution"] = None
        individual["invalid_reason"] = "FINAL_AIRCRAFT_FEASIBILITY"
        return

    final_crew_valid, final_crew_reason = validate_crew_movements(
        aircraft_state,
        crew_state,
    )

    if not final_crew_valid:
        individual["objectives"] = (
            1e12,
            1e9,
            1e9,
            1e9,
            1e9,
        )
        individual["solution"] = None
        individual["invalid_reason"] = final_crew_reason
        return

    crew_state = apply_final_crew_policy(
        individual,
        crew_state,
        pilots,
        horizon_end.date(),
        aircraft_state.get("final_aircraft_locations", {}),
    )

    commercial_cost = (
        commercial_hours
        * FLIGHT_HOUR_COST_EUR
    )

    empty_cost = (
        aircraft_state[
            "empty_hours"
        ]
        * FLIGHT_HOUR_COST_EUR
    )

    pilot_cost = (
        len(
            crew_state[
                "charged_pilot_days"
            ]
        )
        * PILOT_DAY_COST_EUR
    )

    nonhome_swap_cost = (
        crew_state[
            "nonhome_swap_events"
        ]
        * NONHOMEBASE_SWAP_COST_EUR
    )

    crew_return_cost = crew_state[
        "return_cost_eur"
    ]

    inbound_deadhead_cost = crew_state[
        "inbound_cost_eur"
    ]

    parking_cost = aircraft_state[
        "parking_cost_eur"
    ]

    total_operational_cost = (
        commercial_cost
        + empty_cost
        + pilot_cost
        + nonhome_swap_cost
        + crew_return_cost
        + inbound_deadhead_cost
        + parking_cost
    )

    complexity = (
        aircraft_state[
            "empty_legs"
        ]
        * COMPLEXITY_EMPTY_LEG
        + crew_state[
            "inbound_deadhead_pilots"
        ]
        * COMPLEXITY_INBOUND_DEADHEAD
        + crew_state[
            "nonhome_swap_events"
        ]
        * COMPLEXITY_NONHOME_SWAP
        + aircraft_state[
            "parking_days"
        ]
        * COMPLEXITY_AIRCRAFT_PARKING_DAY
        + crew_state[
            "return_pilots"
        ]
        * COMPLEXITY_CREW_RETURN
    )

    idle_return_events = sum(
        1
        for actions in aircraft_state[
            "aircraft_actions"
        ].values()
        for action in actions
        if str(
            action.get(
                "transition_mode",
                "",
            )
        ).startswith(
            "IDLE_RETURN_"
        )
    )

    solution = {
        "objectives": {
            "total_operational_cost_eur": round(
                total_operational_cost,
                2,
            ),
            "complexity_score": int(
                complexity
            ),
            "empty_legs": int(
                aircraft_state[
                    "empty_legs"
                ]
            ),
            "charged_pilot_days": len(
                crew_state[
                    "charged_pilot_days"
                ]
            ),
            "aircraft_parking_days": int(
                aircraft_state[
                    "parking_days"
                ]
            ),
        },
        "cost_breakdown": {
            "commercial_flight_cost_eur": round(
                commercial_cost,
                2,
            ),
            "empty_flight_cost_eur": round(
                empty_cost,
                2,
            ),
            "pilot_cost_eur": round(
                pilot_cost,
                2,
            ),
            "crew_deadhead_return_cost_eur": round(
                crew_return_cost,
                2,
            ),
            "inbound_crew_deadhead_cost_eur": round(
                inbound_deadhead_cost,
                2,
            ),
            "nonhome_swap_cost_eur": round(
                nonhome_swap_cost,
                2,
            ),
            "aircraft_parking_cost_eur": round(
                parking_cost,
                2,
            ),
            "commercial_flight_hours": round(
                commercial_hours,
                2,
            ),
            "empty_flight_hours": round(
                aircraft_state[
                    "empty_hours"
                ],
                2,
            ),
        },
        "metrics": {
            "empty_nm": round(
                aircraft_state[
                    "empty_nm"
                ],
                2,
            ),
            "empty_legs": int(
                aircraft_state[
                    "empty_legs"
                ]
            ),
            "charged_pilot_days": len(
                crew_state[
                    "charged_pilot_days"
                ]
            ),
            "aircraft_parking_days": int(
                aircraft_state[
                    "parking_days"
                ]
            ),
            "changed_pilot_count": int(
                crew_state[
                    "changed_pilots"
                ]
            ),
            "nonhomebase_swap_events": int(
                crew_state[
                    "nonhome_swap_events"
                ]
            ),
            "crew_return_pilots": int(
                crew_state[
                    "return_pilots"
                ]
            ),
            "inbound_crew_deadhead_pilots": int(
                crew_state[
                    "inbound_deadhead_pilots"
                ]
            ),
            "idle_return_events": int(
                idle_return_events
            ),
            "crew_positioning_nm": round(
                crew_state[
                    "crew_position_nm"
                ],
                1,
            ),
            "complexity_breakdown": {
                "empty_positioning": int(
                    aircraft_state[
                        "empty_legs"
                    ]
                    * COMPLEXITY_EMPTY_LEG
                ),
                "inbound_deadheads": int(
                    crew_state[
                        "inbound_deadhead_pilots"
                    ]
                    * COMPLEXITY_INBOUND_DEADHEAD
                ),
                "nonhome_swaps": int(
                    crew_state[
                        "nonhome_swap_events"
                    ]
                    * COMPLEXITY_NONHOME_SWAP
                ),
                "away_parking": int(
                    aircraft_state[
                        "parking_days"
                    ]
                    * COMPLEXITY_AIRCRAFT_PARKING_DAY
                ),
                "crew_returns": int(
                    crew_state[
                        "return_pilots"
                    ]
                    * COMPLEXITY_CREW_RETURN
                ),
            },
        },
        "dataset_fingerprint": mission_dataset_fingerprint(
            mission_order
        ),
        "mission_snapshot": mission_snapshot(
            mission_order
        ),
        "routes": aircraft_state[
            "routes"
        ],
        "aircraft_movements": serialized_movements(
            aircraft_state
        ),
        "aircraft_actions": aircraft_state[
            "aircraft_actions"
        ],
        "crew_assignments": crew_state[
            "crew_assignments"
        ],
        "crew_actions": crew_state[
            "crew_actions"
        ],
        "final_crew_decisions": crew_state.get(
            "final_crew_decisions",
            [],
        ),
        "validation": {
            "aircraft_timeline_valid": True,
            "location_continuity_valid": True,
            "crew_on_every_movement": True,
            "pilot_movement_overlap_valid": True,
            "dataset_self_contained": True,
        },
    }

    solution[
        "schedule_signature"
    ] = schedule_signature(
        solution
    )

    individual["objectives"] = (
        round(
            total_operational_cost,
            2,
        ),
        int(complexity),
        int(
            aircraft_state[
                "empty_legs"
            ]
        ),
        len(
            crew_state[
                "charged_pilot_days"
            ]
        ),
        int(
            aircraft_state[
                "parking_days"
            ]
        ),
    )

    individual["solution"] = solution


def dominates(a, b) -> bool:
    return (
        all(
            x <= y
            for x, y
            in zip(
                a["objectives"],
                b["objectives"],
            )
        )
        and any(
            x < y
            for x, y
            in zip(
                a["objectives"],
                b["objectives"],
            )
        )
    )


def nondominated_fronts(population):
    domination_sets = {
        id(p): []
        for p in population
    }

    domination_counts = {
        id(p): 0
        for p in population
    }

    first = []

    for i, p in enumerate(population):
        for q in population[i + 1:]:
            if dominates(p, q):
                domination_sets[
                    id(p)
                ].append(q)

                domination_counts[
                    id(q)
                ] += 1

            elif dominates(q, p):
                domination_sets[
                    id(q)
                ].append(p)

                domination_counts[
                    id(p)
                ] += 1

    for p in population:
        if domination_counts[
            id(p)
        ] == 0:
            p["rank"] = 0
            first.append(p)

    fronts = [first]
    level = 0

    while (
        level < len(fronts)
        and fronts[level]
    ):
        next_front = []

        for p in fronts[level]:
            for q in domination_sets[
                id(p)
            ]:
                domination_counts[
                    id(q)
                ] -= 1

                if domination_counts[
                    id(q)
                ] == 0:
                    q["rank"] = (
                        level + 1
                    )
                    next_front.append(q)

        if next_front:
            fronts.append(
                next_front
            )

        level += 1

    return fronts


def assign_crowding(front):
    if not front:
        return

    for p in front:
        p["crowding"] = 0.0

    if len(front) <= 2:
        for p in front:
            p["crowding"] = float("inf")
        return

    dimensions = len(
        front[0][
            "objectives"
        ]
    )

    for dimension in range(
        dimensions
    ):
        ordered = sorted(
            front,
            key=lambda p: p[
                "objectives"
            ][dimension],
        )

        ordered[0][
            "crowding"
        ] = float("inf")

        ordered[-1][
            "crowding"
        ] = float("inf")

        low = ordered[0][
            "objectives"
        ][dimension]

        high = ordered[-1][
            "objectives"
        ][dimension]

        if high == low:
            continue

        for i in range(
            1,
            len(ordered) - 1,
        ):
            if ordered[i][
                "crowding"
            ] == float("inf"):
                continue

            previous_value = (
                ordered[i - 1][
                    "objectives"
                ][dimension]
            )

            next_value = (
                ordered[i + 1][
                    "objectives"
                ][dimension]
            )

            ordered[i][
                "crowding"
            ] += (
                next_value
                - previous_value
            ) / (
                high - low
            )


def rank_population(population):
    fronts = nondominated_fronts(
        population
    )

    for front in fronts:
        assign_crowding(
            front
        )

    return fronts


def tournament(
    population,
    rng,
):
    a, b = rng.sample(
        population,
        2,
    )

    if a["rank"] != b["rank"]:
        return (
            a
            if a["rank"] < b["rank"]
            else b
        )

    return (
        a
        if a["crowding"]
        > b["crowding"]
        else b
    )


def clone_genes(individual):
    return {
        "aircraft_preferences": {
            mid: prefs[:]
            for mid, prefs
            in individual[
                "aircraft_preferences"
            ].items()
        },
        "transition_mode": dict(
            individual[
                "transition_mode"
            ]
        ),
        "crew_choice": dict(
            individual[
                "crew_choice"
            ]
        ),
        "final_aircraft_policy": dict(
            individual[
                "final_aircraft_policy"
            ]
        ),
        "final_crew_policy": individual[
            "final_crew_policy"
        ],
        "objectives": None,
        "solution": None,
        "rank": 0,
        "crowding": 0.0,
    }


def crossover(
    parent_a,
    parent_b,
    rng,
):
    child = clone_genes(
        parent_a
    )

    for mid in child[
        "aircraft_preferences"
    ]:
        if rng.random() < 0.5:
            child[
                "aircraft_preferences"
            ][mid] = parent_b[
                "aircraft_preferences"
            ][mid][:]

        if rng.random() < 0.5:
            child[
                "transition_mode"
            ][mid] = parent_b[
                "transition_mode"
            ][mid]

        if rng.random() < 0.5:
            child[
                "crew_choice"
            ][mid] = parent_b[
                "crew_choice"
            ][mid]

    for reg in child[
        "final_aircraft_policy"
    ]:
        if rng.random() < 0.5:
            child[
                "final_aircraft_policy"
            ][reg] = parent_b[
                "final_aircraft_policy"
            ][reg]

    if rng.random() < 0.5:
        child[
            "final_crew_policy"
        ] = parent_b[
            "final_crew_policy"
        ]

    return child


def mutate(
    individual,
    rng,
    mission_ids,
    available_regs,
):
    child = clone_genes(
        individual
    )

    # Multiple mutation types may occur.
    if rng.random() < 0.65:
        mid = rng.choice(
            mission_ids
        )

        prefs = child[
            "aircraft_preferences"
        ][mid]

        i, j = rng.sample(
            range(len(prefs)),
            2,
        )

        prefs[i], prefs[j] = (
            prefs[j],
            prefs[i],
        )

    if rng.random() < 0.45:
        mid = rng.choice(
            mission_ids
        )

        child[
            "transition_mode"
        ][mid] = rng.randrange(3)

    if rng.random() < 0.50:
        mid = rng.choice(
            mission_ids
        )

        child[
            "crew_choice"
        ][mid] = rng.randrange(6)

    if rng.random() < 0.25:
        reg = rng.choice(
            available_regs
        )

        child[
            "final_aircraft_policy"
        ][reg] = rng.randrange(4)

    if rng.random() < 0.15:
        child[
            "final_crew_policy"
        ] = rng.randrange(3)

    return child


def survivor_selection(
    combined,
    size,
):
    survivors = []

    for front in rank_population(
        combined
    ):
        if (
            len(survivors)
            + len(front)
            <= size
        ):
            survivors.extend(
                front
            )
        else:
            ordered = sorted(
                front,
                key=lambda p: p[
                    "crowding"
                ],
                reverse=True,
            )

            survivors.extend(
                ordered[
                    : size
                    - len(survivors)
                ]
            )

            break

    return survivors


def update_archive(
    archive_by_signature,
    population,
):
    for ind in population:
        solution = ind.get(
            "solution"
        )

        if solution is None:
            continue

        signature = solution[
            "schedule_signature"
        ]

        existing = archive_by_signature.get(
            signature
        )

        if (
            existing is None
            or ind["objectives"][0]
            < existing["objectives"][0]
        ):
            archive_by_signature[
                signature
            ] = {
                "objectives": ind[
                    "objectives"
                ],
                "solution": solution,
            }


def pareto_from_archive(
    archive_by_signature,
):
    items = list(
        archive_by_signature.values()
    )

    front = []

    for item in items:
        dominated = False

        for other in items:
            if other is item:
                continue

            if (
                all(
                    x <= y
                    for x, y
                    in zip(
                        other[
                            "objectives"
                        ],
                        item[
                            "objectives"
                        ],
                    )
                )
                and any(
                    x < y
                    for x, y
                    in zip(
                        other[
                            "objectives"
                        ],
                        item[
                            "objectives"
                        ],
                    )
                )
            ):
                dominated = True
                break

        if not dominated:
            front.append(item)

    front.sort(
        key=lambda x: (
            x["objectives"][0],
            x["objectives"][1],
            x["objectives"][2],
            x["objectives"][3],
            x["objectives"][4],
        )
    )

    return front



def cost_complexity_2d_front(points):
    """
    points: iterable of dicts with keys cost, complexity, signature.

    Returns the true 2D non-dominated frontier for:
      minimize cost
      minimize complexity
    """
    # One representative per identical 2D coordinate.
    unique = {}

    for point in points:
        key = (
            round(float(point["cost"]), 2),
            int(point["complexity"]),
        )
        if key not in unique:
            unique[key] = point

    candidates = list(unique.values())
    front = []

    for p in candidates:
        dominated = False

        for q in candidates:
            if q is p:
                continue

            if (
                q["cost"] <= p["cost"]
                and q["complexity"] <= p["complexity"]
                and (
                    q["cost"] < p["cost"]
                    or q["complexity"] < p["complexity"]
                )
            ):
                dominated = True
                break

        if not dominated:
            front.append(p)

    front.sort(
        key=lambda p: (
            p["cost"],
            p["complexity"],
        )
    )

    return front


def write_live_progress(
    archive_by_signature,
    run_index,
    runs,
    generation,
    generations,
    total_evaluations,
    status="running",
):
    """
    Write one compact atomic JSON snapshot for the separate live monitor.

    The optimizer never waits for the GUI. If the GUI is closed, optimization
    continues normally.
    """
    OUTPUT.mkdir(exist_ok=True)

    points = []

    for item in archive_by_signature.values():
        solution = item.get("solution")
        objectives = item.get("objectives")

        if solution is None or objectives is None:
            continue

        points.append({
            "cost": float(objectives[0]),
            "complexity": int(objectives[1]),
            "empty_legs": int(objectives[2]),
            "pilot_days": int(objectives[3]),
            "parking_days": int(objectives[4]),
            "signature": solution.get(
                "schedule_signature",
                "",
            ),
        })

    # Keep the live file light even after tens of thousands of discoveries.
    # Always retain all 2D-front points, plus a representative background cloud.
    front = cost_complexity_2d_front(points)
    front_keys = {
        (
            round(p["cost"], 2),
            p["complexity"],
            p["signature"],
        )
        for p in front
    }

    background = [
        p
        for p in points
        if (
            round(p["cost"], 2),
            p["complexity"],
            p["signature"],
        ) not in front_keys
    ]

    # Deterministic downsampling for visual context.
    max_background = 1200
    if len(background) > max_background:
        step = len(background) / max_background
        background = [
            background[int(i * step)]
            for i in range(max_background)
        ]

    cheapest = (
        min(
            points,
            key=lambda p: (
                p["cost"],
                p["complexity"],
            ),
        )
        if points
        else None
    )

    payload = {
        "status": status,
        "run": run_index,
        "runs": runs,
        "generation": generation,
        "generations": generations,
        "evaluations": total_evaluations,
        "unique_schedules": len(archive_by_signature),
        "archive_points": len(points),
        "background": background,
        "front": front,
        "cheapest": cheapest,
        "timestamp": time.time(),
    }

    target = OUTPUT / "live_progress.json"
    temporary = OUTPUT / "live_progress.tmp"

    temporary.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    temporary.replace(target)


def launch_live_monitor():
    monitor = ROOT / "live_pareto.py"

    if not monitor.exists():
        print(
            "Live monitor script not found; continuing without GUI."
        )
        return None

    try:
        return subprocess.Popen(
            [
                sys.executable,
                str(monitor),
            ],
            cwd=str(ROOT),
        )
    except Exception as exc:
        print(
            f"Could not start live Pareto monitor: {exc}"
        )
        return None


def save_results(
    archive_by_signature,
    pareto,
    run_stats,
):
    OUTPUT.mkdir(
        exist_ok=True
    )

    for path in OUTPUT.glob("*"):
        if path.name == "live_progress.json":
            continue
        path.unlink()

    # Save Pareto solutions.
    rows = []

    for index, item in enumerate(
        pareto
    ):
        solution = copy.deepcopy(
            item["solution"]
        )

        solution[
            "pareto_index"
        ] = index

        (
            OUTPUT
            / f"pareto_{index:03d}.json"
        ).write_text(
            json.dumps(
                solution,
                indent=2,
            ),
            encoding="utf-8",
        )

        rows.append({
            "solution_index": index,
            "schedule_signature": solution[
                "schedule_signature"
            ],
            "total_operational_cost_eur": solution[
                "objectives"
            ][
                "total_operational_cost_eur"
            ],
            "complexity_score": solution[
                "objectives"
            ][
                "complexity_score"
            ],
            "empty_legs": solution[
                "objectives"
            ][
                "empty_legs"
            ],
            "charged_pilot_days": solution[
                "objectives"
            ][
                "charged_pilot_days"
            ],
            "aircraft_parking_days": solution[
                "objectives"
            ][
                "aircraft_parking_days"
            ],
            "empty_nm": solution[
                "metrics"
            ][
                "empty_nm"
            ],
        })

    with (
        OUTPUT / "pareto.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            )
            if rows
            else [
                "solution_index",
                "schedule_signature",
                "total_operational_cost_eur",
                "complexity_score",
                "empty_legs",
                "charged_pilot_days",
                "aircraft_parking_days",
                "empty_nm",
            ],
        )

        writer.writeheader()

        if rows:
            writer.writerows(
                rows
            )

    # Save a gallery of unique schedules sorted by cost.
    gallery = sorted(
        archive_by_signature.values(),
        key=lambda x: (
            x["objectives"][0],
            x["objectives"][1],
        ),
    )

    # Cap JSON size while keeping a substantial diversity browser.
    gallery = gallery[:300]

    gallery_payload = []

    for index, item in enumerate(
        gallery
    ):
        solution = copy.deepcopy(
            item["solution"]
        )

        solution[
            "gallery_index"
        ] = index

        gallery_payload.append(
            solution
        )

    (
        OUTPUT
        / "diverse_solutions.json"
    ).write_text(
        json.dumps(
            gallery_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        **run_stats,
        "unique_aircraft_schedules": len(
            archive_by_signature
        ),
        "pareto_solutions": len(
            pareto
        ),
        "pareto_unique_aircraft_schedules": len({
            item[
                "solution"
            ][
                "schedule_signature"
            ]
            for item in pareto
        }),
    }

    (
        OUTPUT
        / "run_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    if pareto:
        cheapest = min(
            pareto,
            key=lambda x: x[
                "objectives"
            ][0],
        )

        (
            OUTPUT
            / "cheapest.json"
        ).write_text(
            json.dumps(
                cheapest[
                    "solution"
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

    print()
    print("=" * 84)
    print("GENETIC SEARCH COMPLETE")
    print("=" * 84)
    print(
        f'Unique aircraft schedules discovered: '
        f'{summary["unique_aircraft_schedules"]}'
    )
    print(
        f'Pareto solutions: '
        f'{summary["pareto_solutions"]}'
    )
    print(
        f'Pareto unique aircraft schedules: '
        f'{summary["pareto_unique_aircraft_schedules"]}'
    )

    if pareto:
        best = pareto[0][
            "solution"
        ]

        print(
            f'Cheapest total operational cost: '
            f'€{best["objectives"]["total_operational_cost_eur"]:,.0f}'
        )

    print("=" * 84)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Multi-run genetic airline scheduler "
            "with NSGA-II-style Pareto selection."
        )
    )

    parser.add_argument(
        "--population",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Open a separate live cost-vs-complexity Pareto window "
            "while optimization runs."
        ),
    )

    parser.add_argument(
        "--live-every",
        type=int,
        default=1,
        help=(
            "Write/update the live Pareto snapshot every N generations "
            "(default: 1)."
        ),
    )

    args = parser.parse_args()

    airports, aircraft, missions, pilots = load_data()

    mission_order = sorted(
        missions.values(),
        key=lambda m: m.departure,
    )

    mission_ids = [
        m.id
        for m in mission_order
    ]

    available_regs = [
        reg
        for reg, ac
        in aircraft.items()
        if ac.status == "available"
    ]

    commercial_hours = sum(
        mission_hours(m)
        for m in mission_order
    )

    horizon_end = (
        max(
            m.arrival
            for m in mission_order
        )
        + timedelta(days=1)
    )

    archive = {}

    total_evaluations = 0

    live_process = None

    # Start from a clean live snapshot.
    live_path = OUTPUT / "live_progress.json"
    if live_path.exists():
        try:
            live_path.unlink()
        except Exception:
            pass

    if args.live:
        live_process = launch_live_monitor()

        # Give the separate GUI process time to initialize before first snapshot.
        time.sleep(0.25)

    for run_index in range(
        args.runs
    ):
        rng = random.Random(
            args.seed
            + run_index * 10007
        )

        population = [
            random_individual(
                rng,
                mission_ids,
                available_regs,
                list(pilots),
            )
            for _ in range(
                args.population
            )
        ]

        for individual in population:
            evaluate(
                individual,
                mission_order,
                aircraft,
                airports,
                pilots,
                commercial_hours,
                horizon_end,
            )

        total_evaluations += len(
            population
        )

        update_archive(
            archive,
            population,
        )

        if args.live:
            write_live_progress(
                archive,
                run_index + 1,
                args.runs,
                0,
                args.generations,
                total_evaluations,
                status="running",
            )

        for generation in range(
            args.generations
        ):
            rank_population(
                population
            )

            children = []

            while len(children) < args.population:
                a = tournament(
                    population,
                    rng,
                )

                b = tournament(
                    population,
                    rng,
                )

                child = crossover(
                    a,
                    b,
                    rng,
                )

                child = mutate(
                    child,
                    rng,
                    mission_ids,
                    available_regs,
                )

                evaluate(
                    child,
                    mission_order,
                    aircraft,
                    airports,
                    pilots,
                    commercial_hours,
                    horizon_end,
                )

                children.append(
                    child
                )

            total_evaluations += len(
                children
            )

            population = survivor_selection(
                population
                + children,
                args.population,
            )

            update_archive(
                archive,
                population,
            )

            if (
                args.live
                and (
                    generation % max(1, args.live_every) == 0
                    or generation == args.generations - 1
                )
            ):
                write_live_progress(
                    archive,
                    run_index + 1,
                    args.runs,
                    generation + 1,
                    args.generations,
                    total_evaluations,
                    status="running",
                )

            if (
                generation % 10 == 0
                or generation
                == args.generations - 1
            ):
                valid = [
                    p
                    for p in population
                    if p["solution"]
                    is not None
                ]

                if valid:
                    cheapest = min(
                        p[
                            "objectives"
                        ][0]
                        for p in valid
                    )

                    signatures = len({
                        p[
                            "solution"
                        ][
                            "schedule_signature"
                        ]
                        for p in valid
                    })

                    print(
                        f'run={run_index + 1}/{args.runs} '
                        f'gen={generation:3d} '
                        f'population_schedules={signatures:3d} '
                        f'archive={len(archive):4d} '
                        f'best=€{cheapest:,.0f}'
                    )

    pareto = pareto_from_archive(
        archive
    )

    save_results(
        archive,
        pareto,
        {
            "runs": args.runs,
            "population": args.population,
            "generations": args.generations,
            "total_evaluations": total_evaluations,
            "seed": args.seed,
        },
    )

    if args.live:
        write_live_progress(
            archive,
            args.runs,
            args.runs,
            args.generations,
            args.generations,
            total_evaluations,
            status="complete",
        )


if __name__ == "__main__":
    main()
