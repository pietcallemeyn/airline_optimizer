
from __future__ import annotations

import csv
import html
import json
import os
import re
import signal
import subprocess
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
try:
    from streamlit_calendar import calendar as streamlit_calendar
except ImportError:
    streamlit_calendar = None
import plotly.express as px
import plotly.graph_objects as go
import visualize as viz


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"

# The planning calendar is a UTC-labelled wall-clock. FullCalendar renders in the
# browser's local timezone and streamlit-calendar serializes clicked/dragged JS
# Date objects as UTC instants. In Belgium this means a visible 15:00 click in
# summer arrives in Python as 13:00Z. Convert callback instants back to the
# Europe/Brussels wall-clock, then store that visible clock time as naive UTC.
CALENDAR_WALL_TZ = ZoneInfo("Europe/Brussels")

MISSIONS = DATA / "missions.csv"
AIRPORTS = DATA / "airports.csv"
AIRCRAFT = DATA / "aircraft.csv"
PILOTS = DATA / "pilots.csv"
PILOT_AVAILABILITY = DATA / "pilot_availability.csv"
PILOT_RULES = DATA / "pilot_rules.csv"
PILOT_MONTHLY_STATE = DATA / "pilot_monthly_state.csv"

PID_FILE = OUTPUT / "gui_optimizer.pid"
LOG_FILE = OUTPUT / "gui_optimizer.log"


st.set_page_config(
    page_title="Airline Scheduling Optimizer",
    page_icon="✈️",
    layout="wide",
)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(
        path,
        keep_default_na=False,
    )


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Streamlit can introduce numpy scalar types; CSV handles them fine,
    # but normalize blanks first.
    clean = df.copy()
    clean = clean.fillna("")
    clean.to_csv(
        path,
        index=False,
    )


def airports_df() -> pd.DataFrame:
    return load_csv(AIRPORTS)


def airport_codes() -> list[str]:
    df = airports_df()
    if df.empty or "icao" not in df.columns:
        return []
    return sorted(
        str(x)
        for x in df["icao"].tolist()
        if str(x)
    )


def next_mission_id(df: pd.DataFrame) -> str:
    maximum = 0

    if not df.empty and "id" in df.columns:
        for value in df["id"].astype(str):
            match = re.fullmatch(r"M(\d+)", value.strip())
            if match:
                maximum = max(
                    maximum,
                    int(match.group(1)),
                )

    return f"M{maximum + 1:03d}"


def normalize_missions(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "id",
        "origin",
        "destination",
        "departure",
        "arrival",
        "pax",
    ]

    for column in required:
        if column not in df.columns:
            df[column] = ""

    df = df[required].copy()

    df["id"] = df["id"].astype(str).str.strip()
    df["origin"] = df["origin"].astype(str).str.strip().str.upper()
    df["destination"] = (
        df["destination"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    df["departure"] = df["departure"].astype(str).str.strip()
    df["arrival"] = df["arrival"].astype(str).str.strip()

    def pax_value(value):
        try:
            return int(float(value))
        except Exception:
            return 0

    df["pax"] = df["pax"].map(pax_value)

    return df


def validate_missions(df: pd.DataFrame) -> list[str]:
    errors = []
    known_airports = set(airport_codes())
    seen_ids = set()

    for index, row in df.iterrows():
        line = index + 2

        mid = str(row["id"]).strip()
        origin = str(row["origin"]).strip().upper()
        destination = str(row["destination"]).strip().upper()
        departure = str(row["departure"]).strip()

        if not mid:
            errors.append(
                f"Row {line}: mission ID is empty."
            )
        elif mid in seen_ids:
            errors.append(
                f"Row {line}: duplicate mission ID {mid}."
            )
        else:
            seen_ids.add(mid)

        if origin not in known_airports:
            errors.append(
                f"Row {line}: unknown origin {origin}."
            )

        if destination not in known_airports:
            errors.append(
                f"Row {line}: unknown destination {destination}."
            )

        try:
            datetime.fromisoformat(departure)
        except Exception:
            errors.append(
                f"Row {line}: departure is not a valid ISO date/time."
            )

        try:
            if int(row["pax"]) < 0:
                raise ValueError
        except Exception:
            errors.append(
                f"Row {line}: pax must be a non-negative integer."
            )

    return errors


WEEKDAY_RE = re.compile(
    r"^(?:MA|DI|WO|DO|VR|ZA|ZO)\s+",
    re.IGNORECASE,
)

DATE_RE = re.compile(
    r"^(?:[A-Z]{2}\s+)?(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?$",
    re.IGNORECASE,
)

FLIGHT_RE = re.compile(
    r"^([A-Z0-9]{4})\s*-\s*([A-Z0-9]{4})\s+(\d{3,4})\s+(\d+)$",
    re.IGNORECASE,
)


def parse_bulk_missions(
    text: str,
    existing: pd.DataFrame,
    default_year: int,
) -> tuple[list[dict], list[str]]:
    """
    Accepts blocks such as:

        ZA 1/08
        EGGW-EDDC 1402 13

        MA 3/08
        EDDC-LDPL 1050 13
    """
    rows = []
    warnings = []

    current_date = None
    counter_df = existing.copy()

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    for line in lines:
        date_match = DATE_RE.match(
            line.upper()
        )

        if date_match:
            day = int(date_match.group(1))
            month = int(date_match.group(2))

            year_text = date_match.group(3)

            if year_text:
                year = int(year_text)
                if year < 100:
                    year += 2000
            else:
                year = default_year

            try:
                current_date = date(
                    year,
                    month,
                    day,
                )
            except ValueError:
                warnings.append(
                    f"Invalid date line: {line}"
                )

            continue

        flight_match = FLIGHT_RE.match(
            line.upper()
        )

        if flight_match:
            if current_date is None:
                warnings.append(
                    f"No date before flight line: {line}"
                )
                continue

            origin = flight_match.group(1)
            destination = flight_match.group(2)
            hhmm = flight_match.group(3).zfill(4)
            pax = int(flight_match.group(4))

            hour = int(hhmm[:2])
            minute = int(hhmm[2:])

            try:
                departure = datetime.combine(
                    current_date,
                    time(hour, minute),
                )
            except ValueError:
                warnings.append(
                    f"Invalid time in line: {line}"
                )
                continue

            mid = next_mission_id(
                counter_df
            )

            row = {
                "id": mid,
                "origin": origin,
                "destination": destination,
                "departure": departure.isoformat(),
                "arrival": "",
                "pax": pax,
            }

            rows.append(row)

            counter_df = pd.concat(
                [
                    counter_df,
                    pd.DataFrame([row]),
                ],
                ignore_index=True,
            )

            continue

        warnings.append(
            f"Could not parse: {line}"
        )

    return rows, warnings


def read_pid() -> int | None:
    if not PID_FILE.exists():
        return None

    try:
        return int(
            PID_FILE.read_text(
                encoding="utf-8"
            ).strip()
        )
    except Exception:
        return None


def process_running(pid: int | None) -> bool:
    if pid is None:
        return False

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def optimizer_status() -> tuple[bool, int | None]:
    pid = read_pid()
    running = process_running(pid)

    if not running and PID_FILE.exists():
        try:
            PID_FILE.unlink()
        except Exception:
            pass

    return running, pid


def start_optimizer(
    runs: int,
    population: int,
    generations: int,
    seed: int,
    live: bool,
    live_every: int,
):
    OUTPUT.mkdir(exist_ok=True)

    running, _ = optimizer_status()

    if running:
        raise RuntimeError(
            "An optimizer process is already running."
        )

    # -u = unbuffered stdout/stderr. This is important because optimizer.py
    # writes into gui_optimizer.log when launched from the GUI. Without -u,
    # Python may buffer many progress lines before they become visible.
    command = [
        sys.executable,
        "-u",
        str(ROOT / "optimizer.py"),
        "--runs",
        str(runs),
        "--population",
        str(population),
        "--generations",
        str(generations),
        "--seed",
        str(seed),
    ]

    if live:
        command.extend(
            [
                "--live",
                "--live-every",
                str(live_every),
            ]
        )

    log = LOG_FILE.open(
        "w",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    PID_FILE.write_text(
        str(process.pid),
        encoding="utf-8",
    )


def stop_optimizer():
    pid = read_pid()

    if pid and process_running(pid):
        try:
            os.killpg(
                os.getpgid(pid),
                signal.SIGTERM,
            )
        except Exception:
            try:
                os.kill(
                    pid,
                    signal.SIGTERM,
                )
            except Exception:
                pass

    try:
        PID_FILE.unlink()
    except Exception:
        pass


def launch_visualizer():
    subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "visualize.py"),
        ],
        cwd=str(ROOT),
        start_new_session=True,
    )


def log_tail(lines: int = 80) -> str:
    if not LOG_FILE.exists():
        return "No GUI optimizer log yet."

    content = LOG_FILE.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    return "\n".join(
        content[-lines:]
    )


def cost_complexity_front(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    work = df.copy()

    work = work.sort_values(
        [
            "total_operational_cost_eur",
            "complexity_score",
        ]
    )

    best_complexity = float("inf")
    keep = []

    for idx, row in work.iterrows():
        complexity = float(
            row["complexity_score"]
        )

        if complexity < best_complexity:
            keep.append(idx)
            best_complexity = complexity

    return work.loc[keep].copy()


def render_results():
    summary_path = OUTPUT / "run_summary.json"
    pareto_path = OUTPUT / "pareto.csv"

    if not summary_path.exists():
        st.info(
            "No optimizer results yet."
        )
        return

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    cols = st.columns(4)

    cols[0].metric(
        "Unique schedules",
        f'{summary.get("unique_aircraft_schedules", 0):,}',
    )

    cols[1].metric(
        "Pareto schedules",
        f'{summary.get("pareto_solutions", 0):,}',
    )

    cols[2].metric(
        "Evaluations",
        f'{summary.get("total_evaluations", 0):,}',
    )

    cols[3].metric(
        "Runs",
        f'{summary.get("runs", 0)}',
    )

    if not pareto_path.exists():
        return

    pareto = pd.read_csv(
        pareto_path
    )

    if pareto.empty:
        return

    cheapest = pareto.sort_values(
        "total_operational_cost_eur"
    ).iloc[0]

    st.subheader(
        "Cheapest discovered solution"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total operational cost",
        f'€{cheapest["total_operational_cost_eur"]:,.0f}',
    )

    c2.metric(
        "Complexity",
        int(
            cheapest["complexity_score"]
        ),
    )

    c3.metric(
        "Empty legs",
        int(
            cheapest["empty_legs"]
        ),
    )

    c4.metric(
        "Pilot-days",
        int(
            cheapest["charged_pilot_days"]
        ),
    )

    front = cost_complexity_front(
        pareto
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 4.8)
    )

    ax.scatter(
        pareto[
            "total_operational_cost_eur"
        ],
        pareto[
            "complexity_score"
        ],
        alpha=0.20,
        color="gray",
        label="Other Pareto schedules",
    )

    ax.plot(
        front[
            "total_operational_cost_eur"
        ],
        front[
            "complexity_score"
        ],
        marker="o",
        linewidth=1.6,
        label="2D efficient frontier",
    )

    ax.set_xlabel(
        "Total operational cost (EUR)"
    )

    ax.set_ylabel(
        "Operational complexity"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend()

    st.pyplot(
        fig,
        clear_figure=True,
    )



def _parse_iso_datetime(value: str) -> datetime | None:
    value = str(value or "").strip()
    if not value:
        return None
    try:
        # FullCalendar can return a trailing Z. Mission data is stored as
        # timezone-naive UTC, so normalize callback values to naive UTC-like
        # datetimes before writing missions.csv.
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is not None:
            parsed = parsed.replace(tzinfo=None)
        return parsed
    except Exception:
        return None


def _parse_calendar_callback_datetime(value: str) -> datetime | None:
    """Convert a FullCalendar callback instant to the visible calendar wall-clock.

    Mission timestamps are deliberately stored as timezone-naive UTC planning
    times. The calendar UI is also labelled UTC, so the hour the dispatcher sees
    must be the hour we store. streamlit-calendar serializes browser-local JS
    Dates as UTC (Z) instants; converting that instant to Europe/Brussels first
    reconstructs the visible wall-clock, including DST automatically.
    """
    value = str(value or "").strip()
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            # Some component versions may already return a floating wall-clock.
            return parsed
        return parsed.astimezone(CALENDAR_WALL_TZ).replace(tzinfo=None)
    except Exception:
        return None


def _airport_coordinate_lookup() -> dict[str, tuple[float, float]]:
    df = airports_df()
    lookup = {}
    if df.empty:
        return lookup
    for _, row in df.iterrows():
        try:
            lookup[str(row["icao"]).strip().upper()] = (
                float(row["lat"]),
                float(row["lon"]),
            )
        except Exception:
            continue
    return lookup


def _distance_nm_coords(a: tuple[float, float], b: tuple[float, float]) -> float:
    from math import atan2, cos, radians, sin, sqrt

    r_nm = 3440.065
    lat1, lon1 = a
    lat2, lon2 = b
    p1 = radians(lat1)
    p2 = radians(lat2)
    dp = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    h = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * r_nm * atan2(sqrt(h), sqrt(max(0.0, 1 - h)))


def estimated_mission_arrival(row: pd.Series | dict) -> datetime | None:
    departure = _parse_iso_datetime(row.get("departure", ""))
    if departure is None:
        return None

    explicit = _parse_iso_datetime(row.get("arrival", ""))
    if explicit is not None:
        return explicit

    coords = _airport_coordinate_lookup()
    origin = str(row.get("origin", "")).strip().upper()
    destination = str(row.get("destination", "")).strip().upper()
    if origin not in coords or destination not in coords:
        return departure + pd.Timedelta(minutes=90)

    nm = _distance_nm_coords(coords[origin], coords[destination])
    minutes = 10.0 + 60.0 * nm / 410.0
    return departure + pd.Timedelta(minutes=minutes)


def mission_calendar_events(missions: pd.DataFrame) -> list[dict]:
    events = []
    for _, row in missions.iterrows():
        departure = _parse_iso_datetime(row.get("departure", ""))
        if departure is None:
            continue
        arrival = estimated_mission_arrival(row)
        if arrival is None or arrival <= departure:
            arrival = departure + pd.Timedelta(minutes=60)

        events.append({
            "id": str(row["id"]),
            "title": (
                f'{row["id"]} · {row["origin"]} → {row["destination"]} '
                f'· {int(row["pax"])} pax'
            ),
            "start": departure.isoformat(),
            "end": arrival.isoformat(),
            "allDay": False,
            "editable": True,
            "extendedProps": {
                "origin": str(row["origin"]),
                "destination": str(row["destination"]),
                "pax": int(row["pax"]),
                "arrival_source": (
                    "manual" if str(row.get("arrival", "")).strip() else "estimated"
                ),
            },
        })
    return events


def optimizer_results_are_stale() -> bool:
    if not MISSIONS.exists() or not OUTPUT.exists():
        return False
    output_files = list(OUTPUT.glob("pareto_*.json"))
    for name in ("cheapest.json", "balanced.json", "diverse_solutions.json"):
        p = OUTPUT / name
        if p.exists():
            output_files.append(p)
    if not output_files:
        return False
    try:
        newest_result = max(p.stat().st_mtime for p in output_files)
        return MISSIONS.stat().st_mtime > newest_result
    except OSError:
        return False


def _calendar_callback_signature(state: dict) -> str:
    try:
        return json.dumps(state, sort_keys=True, default=str)
    except Exception:
        return repr(state)


def _remember_mission_calendar_position(state: dict) -> None:
    """Remember the visible calendar date/view before Streamlit remounts it.

    FullCalendar itself lives in the browser, so changing the component key clears
    a temporary selection but would normally also send the user back to the
    original initialDate. We keep the last interacted date and view in
    session_state and feed them back as initialDate/initialView after the remount.
    """
    callback = str(state.get("callback", ""))
    payload = state.get(callback, {}) or {}

    # Prefer the actual item/date the dispatcher interacted with. This is more
    # robust than using currentStart, which is the start of the whole week/month.
    raw_date = ""
    if callback == "dateClick":
        raw_date = payload.get("date", "")
    elif callback == "select":
        raw_date = payload.get("start", "")
    elif callback == "eventClick":
        raw_date = (payload.get("event", {}) or {}).get("start", "")
    elif callback == "eventChange":
        raw_date = (payload.get("event", {}) or {}).get("start", "")

    visible_dt = _parse_calendar_callback_datetime(raw_date)
    if visible_dt is not None:
        st.session_state["mission_calendar_focus_date"] = visible_dt.date().isoformat()

    # All callback payloads used by streamlit-calendar can include a FullCalendar
    # view object. Find it recursively so this also survives small wrapper changes.
    def find_view(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get("view"), dict):
                return obj["view"]
            for value in obj.values():
                found = find_view(value)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = find_view(value)
                if found is not None:
                    return found
        return None

    view = find_view(payload)
    if view:
        view_type = str(view.get("type", "")).strip()
        if view_type in {"timeGridDay", "timeGridWeek", "dayGridMonth"}:
            st.session_state["mission_calendar_view"] = view_type

        # Fallback when a callback has no directly usable interacted date.
        if "mission_calendar_focus_date" not in st.session_state:
            current_start = _parse_calendar_callback_datetime(view.get("currentStart", ""))
            if current_start is not None:
                st.session_state["mission_calendar_focus_date"] = current_start.date().isoformat()


def _reset_mission_calendar_component() -> None:
    """Remount FullCalendar to clear selection while preserving date/view."""
    st.session_state["mission_calendar_revision"] = (
        int(st.session_state.get("mission_calendar_revision", 0)) + 1
    )
    st.session_state.pop("last_mission_calendar_callback", None)


def _apply_calendar_event_change(missions: pd.DataFrame, payload: dict) -> tuple[pd.DataFrame, str | None]:
    event = payload.get("event", {}) or {}
    old_event = payload.get("oldEvent", {}) or {}
    mid = str(event.get("id", "")).strip()
    new_start = _parse_calendar_callback_datetime(event.get("start", ""))
    old_start = _parse_calendar_callback_datetime(old_event.get("start", ""))

    if not mid or new_start is None:
        return missions, None

    matches = missions.index[missions["id"].astype(str) == mid].tolist()
    if not matches:
        return missions, None

    idx = matches[0]
    previous_departure = _parse_iso_datetime(missions.at[idx, "departure"])
    missions.at[idx, "departure"] = new_start.isoformat()

    # Preserve the same manual block duration when a mission is dragged.
    manual_arrival = _parse_iso_datetime(missions.at[idx, "arrival"])
    reference_start = old_start or previous_departure
    if manual_arrival is not None and reference_start is not None:
        missions.at[idx, "arrival"] = (
            manual_arrival + (new_start - reference_start)
        ).isoformat()

    save_csv(normalize_missions(missions), MISSIONS)
    return missions, mid


def _set_calendar_editor_for_existing(missions: pd.DataFrame, mid: str) -> None:
    matches = missions[missions["id"].astype(str) == str(mid)]
    if matches.empty:
        return
    row = matches.iloc[0]
    st.session_state["mission_calendar_editor"] = {
        "mode": "edit",
        "id": str(row["id"]),
    }


def _set_calendar_editor_for_new(when: datetime) -> None:
    st.session_state["mission_calendar_editor"] = {
        "mode": "new",
        "start": when.isoformat(),
    }


def render_calendar_mission_form(missions: pd.DataFrame, airports: list[str]) -> None:
    editor = st.session_state.get("mission_calendar_editor")

    st.markdown("#### Mission details")
    if not editor:
        st.caption(
            "Click an empty calendar slot to add a mission, or click an existing "
            "mission to edit it. Drag a mission to change its departure time."
        )
        return

    mode = editor.get("mode")
    existing = None
    if mode == "edit":
        matches = missions[missions["id"].astype(str) == str(editor.get("id", ""))]
        if matches.empty:
            st.session_state.pop("mission_calendar_editor", None)
            st.warning("That mission no longer exists.")
            return
        existing = matches.iloc[0]
        mid = str(existing["id"])
        dep = _parse_iso_datetime(existing["departure"]) or datetime.now().replace(second=0, microsecond=0)
        origin_default = str(existing["origin"])
        destination_default = str(existing["destination"])
        pax_default = int(existing["pax"])
        manual_arrival_dt = _parse_iso_datetime(existing.get("arrival", ""))
    else:
        mid = next_mission_id(missions)
        dep = _parse_iso_datetime(editor.get("start", "")) or datetime.now().replace(second=0, microsecond=0)
        origin_default = airports[0] if airports else ""
        destination_default = airports[1] if len(airports) > 1 else origin_default
        pax_default = 10
        manual_arrival_dt = None

    def option_index(value: str, options: list[str], fallback: int = 0) -> int:
        try:
            return options.index(value)
        except ValueError:
            return min(fallback, max(0, len(options) - 1))

    with st.form("calendar_mission_editor_form"):
        st.caption(f"Mission ID: **{mid}**")
        origin = st.selectbox(
            "Origin",
            airports,
            index=option_index(origin_default, airports, 0) if airports else None,
        )
        destination = st.selectbox(
            "Destination",
            airports,
            index=option_index(destination_default, airports, 1) if airports else None,
        )
        c1, c2 = st.columns(2)
        mission_date = c1.date_input("Date", value=dep.date())
        mission_time = c2.time_input(
            "Departure UTC",
            value=dep.time().replace(second=0, microsecond=0),
            step=900,
        )
        pax = st.number_input(
            "Passengers",
            min_value=0,
            max_value=100,
            value=pax_default,
            step=1,
        )

        manual_arrival = st.checkbox(
            "Use manual arrival time",
            value=manual_arrival_dt is not None,
            help="Leave this off to let the optimizer estimate block time from route distance.",
        )
        if manual_arrival:
            arrival_seed = manual_arrival_dt or (dep + pd.Timedelta(minutes=90))
            a1, a2 = st.columns(2)
            arrival_date = a1.date_input(
                "Arrival date",
                value=arrival_seed.date(),
                key="calendar_arrival_date",
            )
            arrival_time = a2.time_input(
                "Arrival UTC",
                value=arrival_seed.time().replace(second=0, microsecond=0),
                step=900,
                key="calendar_arrival_time",
            )
        else:
            arrival_date = None
            arrival_time = None

        b1, b2 = st.columns(2)
        save_clicked = b1.form_submit_button("Save mission", type="primary")
        delete_clicked = b2.form_submit_button(
            "Delete mission" if mode == "edit" else "Cancel",
        )

    if save_clicked:
        if not airports:
            st.error("No airports are available in airports.csv.")
            return
        if origin == destination:
            st.error("Origin and destination must differ.")
            return

        departure = datetime.combine(mission_date, mission_time)
        arrival_value = ""
        if manual_arrival:
            manual_dt = datetime.combine(arrival_date, arrival_time)
            if manual_dt <= departure:
                st.error("Arrival must be after departure.")
                return
            arrival_value = manual_dt.isoformat()

        row = {
            "id": mid,
            "origin": origin,
            "destination": destination,
            "departure": departure.isoformat(),
            "arrival": arrival_value,
            "pax": int(pax),
        }

        if mode == "edit":
            idx = missions.index[missions["id"].astype(str) == mid][0]
            for column, value in row.items():
                missions.at[idx, column] = value
        else:
            missions = pd.concat([missions, pd.DataFrame([row])], ignore_index=True)

        normalized = normalize_missions(missions)
        errors = validate_missions(normalized)
        if errors:
            st.error("\n".join(errors[:10]))
            return

        save_csv(normalized, MISSIONS)
        st.session_state.pop("mission_calendar_editor", None)
        st.session_state["missions_changed_notice"] = True
        _reset_mission_calendar_component()
        st.rerun()

    if delete_clicked:
        if mode == "edit":
            missions = missions[missions["id"].astype(str) != mid].reset_index(drop=True)
            save_csv(normalize_missions(missions), MISSIONS)
            st.session_state["missions_changed_notice"] = True
        st.session_state.pop("mission_calendar_editor", None)
        _reset_mission_calendar_component()
        st.rerun()

def mission_editor():
    st.subheader("Mission planning")

    st.caption(
        "Plan demand on a calendar before aircraft are assigned. Click an empty "
        "time slot to create a mission, click a mission to edit it, or drag it "
        "to another time. Aircraft assignment remains entirely up to the optimizer."
    )

    missions = normalize_missions(load_csv(MISSIONS))
    airports = airport_codes()

    if optimizer_results_are_stale():
        st.warning(
            "Planning changed — the current optimizer results are older than missions.csv. "
            "Run the optimizer again before comparing schedules."
        )
    elif st.session_state.pop("missions_changed_notice", False):
        st.info("Mission planning saved.")

    calendar_tab, paste_tab, table_tab = st.tabs(
        ["Calendar", "Bulk import", "Table"]
    )

    with calendar_tab:
        if streamlit_calendar is None:
            st.error(
                "The interactive calendar component is not installed yet. Run "
                "`python -m pip install streamlit-calendar` once in this virtual environment, "
                "then restart Streamlit. The Table and Bulk import tabs still work."
            )
        else:
            events = mission_calendar_events(missions)
            valid_departures = [
                _parse_iso_datetime(x)
                for x in missions["departure"].tolist()
            ] if not missions.empty else []
            valid_departures = [x for x in valid_departures if x is not None]
            default_initial_date = (
                min(valid_departures).date().isoformat()
                if valid_departures
                else date.today().isoformat()
            )
            initial_date = st.session_state.get(
                "mission_calendar_focus_date",
                default_initial_date,
            )
            initial_view = st.session_state.get(
                "mission_calendar_view",
                "timeGridWeek",
            )

            options = {
                "initialView": initial_view,
                # Render floating mission times as browser wall-clock. The UI is
                # labelled UTC; callback conversion below prevents browser TZ
                # serialization from changing the stored mission hour.
                "timeZone": "local",
                "initialDate": initial_date,
                "firstDay": 1,
                "editable": True,
                "eventStartEditable": True,
                "eventDurationEditable": False,
                "selectable": True,
                "selectMirror": True,
                "nowIndicator": True,
                "allDaySlot": False,
                "slotMinTime": "00:00:00",
                "slotMaxTime": "24:00:00",
                "slotDuration": "00:30:00",
                "snapDuration": "00:15:00",
                "scrollTime": "06:00:00",
                "height": 760,
                "eventTimeFormat": {
                    "hour": "2-digit",
                    "minute": "2-digit",
                    "hour12": False,
                },
                "headerToolbar": {
                    "left": "today prev,next",
                    "center": "title",
                    "right": "timeGridDay,timeGridWeek,dayGridMonth",
                },
                "buttonText": {
                    "today": "Today",
                    "day": "Day",
                    "week": "Week",
                    "month": "Month",
                },
            }

            custom_css = """
            .fc { font-size: 0.88rem; }
            .fc .fc-toolbar-title { font-size: 1.2rem; font-weight: 650; }
            .fc .fc-timegrid-slot { height: 2.15em; }
            .fc .fc-event { border-radius: 5px; padding: 1px 3px; cursor: pointer; }
            .fc .fc-event-title { font-weight: 650; }
            .fc .fc-event-time { font-weight: 600; }
            .fc .fc-col-header-cell-cushion { padding: 6px 4px; }
            """

            left, right = st.columns([3.5, 1.25], gap="large")
            with left:
                calendar_revision = int(
                    st.session_state.get("mission_calendar_revision", 0)
                )
                state = streamlit_calendar(
                    events=events,
                    options=options,
                    custom_css=custom_css,
                    callbacks=["dateClick", "eventClick", "eventChange", "select"],
                    key=f"mission_planning_calendar_{calendar_revision}",
                ) or {}

                callback = state.get("callback")
                if callback:
                    # Save the browser-side date/view before any st.rerun() or
                    # component remount. This keeps the dispatcher on the same
                    # week/day/month after Save, Delete, Cancel or drag/drop.
                    _remember_mission_calendar_position(state)
                    signature = _calendar_callback_signature(state)
                    if signature != st.session_state.get("last_mission_calendar_callback"):
                        st.session_state["last_mission_calendar_callback"] = signature

                        if callback == "eventClick":
                            event = (state.get("eventClick") or {}).get("event", {})
                            mid = str(event.get("id", "")).strip()
                            if mid:
                                _set_calendar_editor_for_existing(missions, mid)
                                st.rerun()

                        elif callback == "dateClick":
                            clicked = _parse_calendar_callback_datetime(
                                (state.get("dateClick") or {}).get("date", "")
                            )
                            if clicked is not None:
                                _set_calendar_editor_for_new(clicked)
                                st.rerun()

                        elif callback == "select":
                            clicked = _parse_calendar_callback_datetime(
                                (state.get("select") or {}).get("start", "")
                            )
                            if clicked is not None:
                                _set_calendar_editor_for_new(clicked)
                                st.rerun()

                        elif callback == "eventChange":
                            missions, moved_mid = _apply_calendar_event_change(
                                missions,
                                state.get("eventChange") or {},
                            )
                            if moved_mid:
                                st.session_state["missions_changed_notice"] = True
                                st.session_state["mission_calendar_editor"] = {
                                    "mode": "edit",
                                    "id": moved_mid,
                                }
                                _reset_mission_calendar_component()
                                st.rerun()

                st.caption(
                    f"{len(events)} missions · drag = move departure · "
                    "15-minute snapping · times shown in UTC"
                )

            with right:
                render_calendar_mission_form(missions, airports)

    with paste_tab:
        st.markdown("#### Bulk import")
        st.caption(
            "Keep the fast paste workflow for larger planning updates. New missions "
            "appear in the calendar immediately after import."
        )
        default_year = st.number_input(
            "Default year",
            min_value=2020,
            max_value=2100,
            value=2026,
            step=1,
        )
        bulk = st.text_area(
            "Paste missions",
            height=280,
            placeholder=(
                "ZA 1/08\n"
                "EGGW-EDDC 1402 13\n\n"
                "MA 3/08\n"
                "EDDC-LDPL 1050 13"
            ),
        )
        if st.button("Parse & append", type="primary", key="bulk_append_missions"):
            rows, warnings = parse_bulk_missions(bulk, missions, int(default_year))
            known = set(airports)
            unknown = sorted({
                airport
                for row in rows
                for airport in (row["origin"], row["destination"])
                if airport not in known
            })
            if warnings:
                st.warning("\n".join(warnings))
            if unknown:
                st.error("These airports are not yet in airports.csv: " + ", ".join(unknown))
            elif rows:
                updated = pd.concat([missions, pd.DataFrame(rows)], ignore_index=True)
                save_csv(normalize_missions(updated), MISSIONS)
                st.session_state["missions_changed_notice"] = True
                st.success(f"Added {len(rows)} missions.")
                st.rerun()
            else:
                st.info("No missions found.")

    with table_tab:
        st.markdown("#### Table editor")
        st.caption(
            "Use this for exact data corrections or deleting/adding several rows. "
            "Arrival may stay blank; the optimizer estimates it automatically."
        )
        edited = st.data_editor(
            missions,
            width="stretch",
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "id": st.column_config.TextColumn("Mission ID"),
                "origin": st.column_config.SelectboxColumn("Origin", options=airports),
                "destination": st.column_config.SelectboxColumn("Destination", options=airports),
                "departure": st.column_config.TextColumn(
                    "Departure UTC", help="Example: 2026-08-07T06:50:00"
                ),
                "arrival": st.column_config.TextColumn(
                    "Arrival", help="Leave blank to estimate automatically."
                ),
                "pax": st.column_config.NumberColumn(
                    "Pax", min_value=0, max_value=100, step=1
                ),
            },
            key="mission_table",
        )
        c1, c2 = st.columns([1, 4])
        if c1.button("Save missions", type="primary", key="save_mission_table"):
            normalized = normalize_missions(edited)
            errors = validate_missions(normalized)
            if errors:
                st.error("\n".join(errors[:15]))
            else:
                save_csv(normalized, MISSIONS)
                st.session_state["missions_changed_notice"] = True
                st.success("missions.csv saved.")
                st.rerun()
        c2.caption(f"{len(edited)} missions currently in the table.")


def _ensure_pilot_planning_files() -> None:
    pilots = load_csv(PILOTS)
    pilot_ids = pilots["id"].astype(str).tolist() if not pilots.empty and "id" in pilots.columns else []

    if not PILOT_AVAILABILITY.exists():
        save_csv(pd.DataFrame(columns=["pilot_id", "date", "status", "note"]), PILOT_AVAILABILITY)

    if not PILOT_MONTHLY_STATE.exists():
        save_csv(pd.DataFrame(columns=["pilot_id", "month", "days_already_used", "note"]), PILOT_MONTHLY_STATE)

    if not PILOT_RULES.exists():
        rows = []
        for _, pilot in pilots.iterrows():
            rows.append({
                "pilot_id": str(pilot.get("id", "")),
                "active": "true",
                "daily_rate_eur": 600,
                "min_paid_days_per_month": 0,
                "max_planned_days_per_month": "",
                "preferred_home_base": str(pilot.get("home_base", "")),
                "note": "",
            })
        save_csv(pd.DataFrame(rows), PILOT_RULES)
    else:
        rules = load_csv(PILOT_RULES)
        existing = set(rules.get("pilot_id", pd.Series(dtype=str)).astype(str)) if not rules.empty else set()
        missing = []
        for _, pilot in pilots.iterrows():
            pid = str(pilot.get("id", ""))
            if pid and pid not in existing:
                missing.append({
                    "pilot_id": pid,
                    "active": "true",
                    "daily_rate_eur": 600,
                    "min_paid_days_per_month": 0,
                    "max_planned_days_per_month": "",
                    "preferred_home_base": str(pilot.get("home_base", "")),
                    "note": "",
                })
        if missing:
            save_csv(pd.concat([rules, pd.DataFrame(missing)], ignore_index=True), PILOT_RULES)


def _pilot_display_lookup() -> dict[str, str]:
    pilots = load_csv(PILOTS)
    result = {}
    for _, row in pilots.iterrows():
        pid = str(row.get("id", "")).strip()
        name = str(row.get("name", "")).strip()
        role = str(row.get("role", "")).strip()
        base = str(row.get("home_base", "")).strip()
        if pid:
            result[pid] = f"{pid} · {name} · {role} · {base}"
    return result


PILOT_AVAILABILITY_COLORS = {
    "AVAILABLE": {"bg": "#DCFCE7", "fg": "#166534"},
    "UNAVAILABLE": {"bg": "#FEE2E2", "fg": "#991B1B"},
    "LEAVE": {"bg": "#FFEDD5", "fg": "#9A3412"},
    "TRAINING": {"bg": "#DBEAFE", "fg": "#1E40AF"},
}


def _pilot_calendar_date_range() -> tuple[date, date]:
    """Return a generous year-based range so default AVAILABLE days are visible."""
    mission_df = normalize_missions(load_csv(MISSIONS))
    parsed = [_parse_iso_datetime(x) for x in mission_df.get("departure", pd.Series(dtype=str)).tolist()]
    parsed = [x for x in parsed if x is not None]
    if parsed:
        first_year = min(x.year for x in parsed)
        last_year = max(x.year for x in parsed)
    else:
        first_year = last_year = date.today().year
    return date(first_year, 1, 1), date(last_year, 12, 31)


def _pilot_availability_events(df: pd.DataFrame, pilot_id: str) -> list[dict]:
    """Render every planning day with a state color.

    AVAILABLE remains the data-model default (no CSV row required), but it is
    rendered explicitly as a green background so the planner can see the state.
    Exceptions override the default and also get a small labelled foreground pill.
    """
    work = df[df["pilot_id"].astype(str) == str(pilot_id)].copy() if not df.empty else pd.DataFrame()
    exceptions: dict[str, tuple[str, str]] = {}
    for _, row in work.iterrows():
        day = str(row.get("date", "")).strip()
        if not day:
            continue
        status = str(row.get("status", "AVAILABLE")).strip().upper() or "AVAILABLE"
        if status not in PILOT_AVAILABILITY_COLORS:
            status = "AVAILABLE"
        note = str(row.get("note", "")).strip()
        exceptions[day] = (status, note)

    start_day, end_day = _pilot_calendar_date_range()
    events: list[dict] = []
    for stamp in pd.date_range(start_day, end_day, freq="D"):
        day = stamp.date().isoformat()
        status, note = exceptions.get(day, ("AVAILABLE", ""))
        palette = PILOT_AVAILABILITY_COLORS[status]
        events.append({
            "id": f"{pilot_id}:{day}:background",
            "start": day,
            "allDay": True,
            "display": "background",
            "backgroundColor": palette["bg"],
            "extendedProps": {"pilot_id": pilot_id, "status": status, "note": note},
        })
        if status != "AVAILABLE":
            events.append({
                "id": f"{pilot_id}:{day}:label",
                "title": status if not note else f"{status} · {note}",
                "start": day,
                "allDay": True,
                "editable": False,
                "backgroundColor": palette["fg"],
                "borderColor": palette["fg"],
                "textColor": "#FFFFFF",
                "extendedProps": {"pilot_id": pilot_id, "status": status, "note": note},
            })
    return events


def _remember_pilot_calendar_position(state: dict, clicked_day: date | None = None) -> None:
    """Preserve visible month when the calendar is remounted after Save."""
    if clicked_day is not None:
        st.session_state["pilot_availability_calendar_focus_date"] = clicked_day.isoformat()

    callback = str(state.get("callback", ""))
    payload = state.get(callback, {}) or {}

    def find_view(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get("view"), dict):
                return obj["view"]
            for value in obj.values():
                found = find_view(value)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = find_view(value)
                if found is not None:
                    return found
        return None

    view = find_view(payload)
    if view:
        current_start = str(view.get("currentStart", "")).strip()
        if current_start:
            try:
                st.session_state["pilot_availability_calendar_focus_date"] = current_start[:10]
            except Exception:
                pass


def _reset_pilot_availability_calendar() -> None:
    """Force FullCalendar to refresh its events while keeping the current month."""
    st.session_state["pilot_availability_calendar_revision"] = (
        int(st.session_state.get("pilot_availability_calendar_revision", 0)) + 1
    )
    st.session_state.pop("last_pilot_availability_callback", None)


def _calendar_callback_date(payload: dict) -> date | None:
    """Return the calendar day the planner actually clicked.

    streamlit-calendar may serialize an all-day/dateClick value as a UTC JS
    instant. Around Europe/Brussels this can therefore be the previous UTC
    date (for example visible 21 Aug -> 20 Aug 22:00Z). Convert timestamp
    callbacks back to the visible Brussels wall-clock before taking .date().
    Plain YYYY-MM-DD values are already calendar dates and need no conversion.
    """
    if not payload:
        return None
    raw = str(payload.get("dateStr") or payload.get("date") or "").strip()
    if not raw:
        return None
    try:
        # FullCalendar all-day event starts are often already plain dates.
        if len(raw) >= 10 and len(raw) == 10:
            return date.fromisoformat(raw)

        visible_dt = _parse_calendar_callback_datetime(raw)
        if visible_dt is not None:
            return visible_dt.date()

        # Defensive fallback for component versions that return date-like text.
        return date.fromisoformat(raw[:10])
    except Exception:
        return None


def _save_availability_day(pilot_id: str, day: date, status: str, note: str) -> None:
    df = load_csv(PILOT_AVAILABILITY)
    for col in ["pilot_id", "date", "status", "note"]:
        if col not in df.columns:
            df[col] = ""
    day_text = day.isoformat()
    mask = (df["pilot_id"].astype(str) == str(pilot_id)) & (df["date"].astype(str) == day_text)
    df = df.loc[~mask].copy()
    status = str(status).strip().upper()
    if status != "AVAILABLE":
        df = pd.concat([df, pd.DataFrame([{
            "pilot_id": pilot_id,
            "date": day_text,
            "status": status,
            "note": note.strip(),
        }])], ignore_index=True)
    if not df.empty:
        df = df.sort_values(["pilot_id", "date"]).reset_index(drop=True)
    save_csv(df, PILOT_AVAILABILITY)


def _get_rule_row(rules: pd.DataFrame, pilot_id: str, pilots: pd.DataFrame) -> dict:
    if not rules.empty and "pilot_id" in rules.columns:
        rows = rules[rules["pilot_id"].astype(str) == str(pilot_id)]
        if not rows.empty:
            return rows.iloc[0].to_dict()
    home = ""
    if not pilots.empty:
        rows = pilots[pilots["id"].astype(str) == str(pilot_id)]
        if not rows.empty:
            home = str(rows.iloc[0].get("home_base", ""))
    return {
        "pilot_id": pilot_id,
        "active": "true",
        "daily_rate_eur": 600,
        "min_paid_days_per_month": 0,
        "max_planned_days_per_month": "",
        "preferred_home_base": home,
        "note": "",
    }


def _truthy(value) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def pilot_planning_page():
    _ensure_pilot_planning_files()
    st.subheader("Pilot planning")
    st.caption(
        "Availability and individual contract rules are stored as data, not hard-coded in optimizer.py. "
        "Unavailable / Leave / Training are hard planning blocks. Contract minima change the marginal pilot-day cost."
    )

    pilots = load_csv(PILOTS)
    if pilots.empty:
        st.warning("No pilots found in data/pilots.csv.")
        return

    display = _pilot_display_lookup()
    pilot_ids = [pid for pid in pilots["id"].astype(str).tolist() if pid]
    selected = st.selectbox(
        "Pilot",
        pilot_ids,
        format_func=lambda pid: display.get(pid, pid),
        key="pilot_planning_selected",
    )

    availability_tab, rules_tab, carry_tab, table_tab = st.tabs([
        "Availability calendar",
        "Contract rules",
        "Monthly carry-in",
        "All data",
    ])

    with availability_tab:
        availability = load_csv(PILOT_AVAILABILITY)
        events = _pilot_availability_events(availability, selected)

        st.caption(
            "Click any day to edit it. Every day has a visible state: green = Available, "
            "red = Unavailable, orange = Leave, blue = Training. "
            "Actual assigned work days still come from the optimizer result."
        )
        st.markdown(
            "<div style='display:flex;gap:14px;flex-wrap:wrap;margin:2px 0 10px 0'>"
            "<span>🟩 Available</span><span>🟥 Unavailable</span>"
            "<span>🟧 Leave</span><span>🟦 Training</span></div>",
            unsafe_allow_html=True,
        )

        cal_left, editor_right = st.columns([3.2, 1.2], gap="large")
        with cal_left:
            if streamlit_calendar is None:
                st.error("Install `streamlit-calendar` to use the availability calendar.")
                state = {}
            else:
                mission_df = normalize_missions(load_csv(MISSIONS))
                mission_dates = [_parse_iso_datetime(x) for x in mission_df.get("departure", pd.Series(dtype=str)).tolist()]
                mission_dates = [x for x in mission_dates if x is not None]
                default_initial_date = min(mission_dates).date().isoformat() if mission_dates else date.today().isoformat()
                initial_date = st.session_state.get(
                    "pilot_availability_calendar_focus_date", default_initial_date
                )
                options = {
                    "initialView": "dayGridMonth",
                    "initialDate": initial_date,
                    "firstDay": 1,
                    "selectable": True,
                    "editable": False,
                    "height": 690,
                    "headerToolbar": {
                        "left": "today prev,next",
                        "center": "title",
                        "right": "dayGridMonth",
                    },
                }
                calendar_revision = int(
                    st.session_state.get("pilot_availability_calendar_revision", 0)
                )
                state = streamlit_calendar(
                    events=events,
                    options=options,
                    custom_css="""
                    .fc { font-size: 0.9rem; }
                    .fc .fc-daygrid-day { cursor: pointer; }
                    .fc .fc-bg-event { opacity: 1.0; }
                    .fc .fc-event { border-radius: 5px; padding: 2px 4px; }
                    """,
                    callbacks=["dateClick", "eventClick"],
                    key=f"pilot_availability_calendar_{selected}_{calendar_revision}",
                ) or {}

                callback = state.get("callback")
                signature = _calendar_callback_signature(state) if callback else None
                if callback and signature != st.session_state.get("last_pilot_availability_callback"):
                    st.session_state["last_pilot_availability_callback"] = signature
                    clicked_day = None
                    if callback == "dateClick":
                        clicked_day = _calendar_callback_date(state.get("dateClick") or {})
                    elif callback == "eventClick":
                        event = (state.get("eventClick") or {}).get("event", {}) or {}
                        clicked_day = _calendar_callback_date({"dateStr": event.get("start", "")})
                    if clicked_day:
                        _remember_pilot_calendar_position(state, clicked_day)
                        st.session_state["pilot_availability_edit_date"] = clicked_day.isoformat()
                        # The editor date widget uses the clicked date as part of its key,
                        # so it cannot stay stuck on the previously selected day.
                        st.rerun()

        with editor_right:
            st.markdown("#### Day status")
            edit_day_text = st.session_state.get("pilot_availability_edit_date")
            if edit_day_text:
                edit_day = date.fromisoformat(edit_day_text)
            else:
                edit_day = date.today()
            edit_day = st.date_input(
                "Date",
                value=edit_day,
                key=f"pilot_availability_date_{selected}_{edit_day.isoformat()}",
            )

            existing = availability[
                (availability.get("pilot_id", pd.Series(dtype=str)).astype(str) == selected)
                & (availability.get("date", pd.Series(dtype=str)).astype(str) == edit_day.isoformat())
            ] if not availability.empty else pd.DataFrame()
            existing_status = str(existing.iloc[0].get("status", "AVAILABLE")).upper() if not existing.empty else "AVAILABLE"
            existing_note = str(existing.iloc[0].get("note", "")) if not existing.empty else ""
            statuses = ["AVAILABLE", "UNAVAILABLE", "LEAVE", "TRAINING"]
            status = st.selectbox(
                "Status",
                statuses,
                index=statuses.index(existing_status) if existing_status in statuses else 0,
                key=f"pilot_availability_status_{selected}_{edit_day.isoformat()}",
            )
            note = st.text_input(
                "Note",
                value=existing_note,
                key=f"pilot_availability_note_{selected}_{edit_day.isoformat()}",
            )
            if st.button("Save day", type="primary", key="save_pilot_availability_day"):
                _save_availability_day(selected, edit_day, status, note)
                st.session_state["pilot_availability_edit_date"] = edit_day.isoformat()
                st.session_state["pilot_availability_calendar_focus_date"] = edit_day.isoformat()
                st.session_state["pilot_availability_save_message"] = (
                    f"{selected} · {edit_day.isoformat()} saved as {status}."
                )
                _reset_pilot_availability_calendar()
                st.rerun()

            save_message = st.session_state.pop("pilot_availability_save_message", None)
            if save_message:
                st.success(save_message)

            if status in {"UNAVAILABLE", "LEAVE", "TRAINING"}:
                st.warning("This day is a HARD block for this pilot.")
            else:
                st.info("Available is the default and creates no explicit block row.")

    with rules_tab:
        rules = load_csv(PILOT_RULES)
        rule = _get_rule_row(rules, selected, pilots)
        st.markdown("#### Individual optimizer rules")
        st.caption(
            "Example: minimum paid days = 15 means the first guaranteed days have zero marginal cost "
            "until carry-in + planned days exceed 15. This makes the optimizer naturally prefer using already-paid capacity."
        )
        c1, c2 = st.columns(2)
        active = c1.checkbox("Active for planning", value=_truthy(rule.get("active", "true")), key=f"rule_active_{selected}")
        daily_rate = c2.number_input(
            "Daily rate (€)", min_value=0.0, value=float(rule.get("daily_rate_eur") or 600), step=50.0, key=f"rule_rate_{selected}"
        )
        c3, c4 = st.columns(2)
        min_paid = c3.number_input(
            "Minimum paid days / month", min_value=0, max_value=31,
            value=int(float(rule.get("min_paid_days_per_month") or 0)), step=1, key=f"rule_min_{selected}"
        )
        max_raw = str(rule.get("max_planned_days_per_month", "") or "").strip()
        max_days_text = c4.text_input(
            "Maximum planned days / month (blank = none)", value=max_raw, key=f"rule_max_{selected}"
        )
        bases = ["", "EBAW", "EBLG"]
        preferred = str(rule.get("preferred_home_base", "") or "").upper()
        preferred_base = st.selectbox(
            "Preferred homebase", bases,
            index=bases.index(preferred) if preferred in bases else 0,
            key=f"rule_base_{selected}",
            help="Stored as a preference field for future scoring; personal home_base in pilots.csv remains the physical reference for current crew cost logic.",
        )
        rule_note = st.text_area("Rule note", value=str(rule.get("note", "")), key=f"rule_note_{selected}")

        if st.button("Save contract rules", type="primary", key="save_pilot_rules"):
            max_days = ""
            if max_days_text.strip():
                try:
                    parsed = int(max_days_text)
                    if parsed < 0 or parsed > 31:
                        raise ValueError
                    max_days = parsed
                except Exception:
                    st.error("Maximum planned days must be blank or an integer from 0 to 31.")
                    st.stop()
            for col in ["pilot_id", "active", "daily_rate_eur", "min_paid_days_per_month", "max_planned_days_per_month", "preferred_home_base", "note"]:
                if col not in rules.columns:
                    rules[col] = ""
            mask = rules["pilot_id"].astype(str) == selected
            new_row = {
                "pilot_id": selected,
                "active": "true" if active else "false",
                "daily_rate_eur": float(daily_rate),
                "min_paid_days_per_month": int(min_paid),
                "max_planned_days_per_month": max_days,
                "preferred_home_base": preferred_base,
                "note": rule_note.strip(),
            }
            if mask.any():
                idx = rules.index[mask][0]
                for col, value in new_row.items():
                    rules.at[idx, col] = value
            else:
                rules = pd.concat([rules, pd.DataFrame([new_row])], ignore_index=True)
            save_csv(rules, PILOT_RULES)
            st.success("Contract rules saved. They will be loaded on the next optimizer run.")

    with carry_tab:
        state_df = load_csv(PILOT_MONTHLY_STATE)
        st.markdown("#### Monthly carry-in")
        st.caption(
            "Use this for days already used before the optimization horizon. Example: if the pilot has already worked 11 days in July "
            "and has a 15-day minimum, enter 11 for 2026-07."
        )
        default_month = "2026-07"
        mission_df = normalize_missions(load_csv(MISSIONS))
        if not mission_df.empty:
            first_dep = _parse_iso_datetime(mission_df.iloc[0].get("departure", ""))
            if first_dep:
                default_month = first_dep.strftime("%Y-%m")
        month = st.text_input("Month (YYYY-MM)", value=default_month, key=f"carry_month_{selected}")
        existing = state_df[
            (state_df.get("pilot_id", pd.Series(dtype=str)).astype(str) == selected)
            & (state_df.get("month", pd.Series(dtype=str)).astype(str) == month)
        ] if not state_df.empty else pd.DataFrame()
        current_days = int(float(existing.iloc[0].get("days_already_used", 0) or 0)) if not existing.empty else 0
        current_note = str(existing.iloc[0].get("note", "")) if not existing.empty else ""
        days_already = st.number_input("Days already used", min_value=0, max_value=31, value=current_days, step=1, key=f"carry_days_{selected}_{month}")
        carry_note = st.text_input("Note", value=current_note, key=f"carry_note_{selected}_{month}")
        if st.button("Save monthly carry-in", type="primary", key="save_monthly_carry"):
            try:
                datetime.strptime(month + "-01", "%Y-%m-%d")
            except Exception:
                st.error("Month must use YYYY-MM, for example 2026-07.")
            else:
                for col in ["pilot_id", "month", "days_already_used", "note"]:
                    if col not in state_df.columns:
                        state_df[col] = ""
                mask = (state_df["pilot_id"].astype(str) == selected) & (state_df["month"].astype(str) == month)
                state_df = state_df.loc[~mask].copy()
                state_df = pd.concat([state_df, pd.DataFrame([{
                    "pilot_id": selected,
                    "month": month,
                    "days_already_used": int(days_already),
                    "note": carry_note.strip(),
                }])], ignore_index=True)
                save_csv(state_df.sort_values(["pilot_id", "month"]), PILOT_MONTHLY_STATE)
                st.success("Monthly carry-in saved.")

    with table_tab:
        st.markdown("#### Availability rows")
        av = load_csv(PILOT_AVAILABILITY)
        av_edit = st.data_editor(av, width="stretch", hide_index=True, num_rows="dynamic", key="pilot_availability_table")
        if st.button("Save availability table", key="save_pilot_availability_table"):
            save_csv(av_edit, PILOT_AVAILABILITY)
            st.success("pilot_availability.csv saved.")

        st.markdown("#### Contract rules")
        rule_df = load_csv(PILOT_RULES)
        rule_edit = st.data_editor(rule_df, width="stretch", hide_index=True, num_rows="dynamic", key="pilot_rules_table")
        if st.button("Save rules table", key="save_pilot_rules_table"):
            save_csv(rule_edit, PILOT_RULES)
            st.success("pilot_rules.csv saved.")

        st.markdown("#### Monthly carry-in")
        carry_df = load_csv(PILOT_MONTHLY_STATE)
        carry_edit = st.data_editor(carry_df, width="stretch", hide_index=True, num_rows="dynamic", key="pilot_carry_table")
        if st.button("Save carry-in table", key="save_pilot_carry_table"):
            save_csv(carry_edit, PILOT_MONTHLY_STATE)
            st.success("pilot_monthly_state.csv saved.")


def data_editor_page():
    st.subheader(
        "Fleet, pilots & airports"
    )

    tabs = st.tabs(
        [
            "Aircraft",
            "Pilots",
            "Airports",
        ]
    )

    for tab, path, title in [
        (
            tabs[0],
            AIRCRAFT,
            "aircraft.csv",
        ),
        (
            tabs[1],
            PILOTS,
            "pilots.csv",
        ),
        (
            tabs[2],
            AIRPORTS,
            "airports.csv",
        ),
    ]:
        with tab:
            df = load_csv(path)

            edited = st.data_editor(
                df,
                width="stretch",
                hide_index=True,
                num_rows="dynamic",
                key=f"editor_{title}",
            )

            if st.button(
                f"Save {title}",
                key=f"save_{title}",
            ):
                save_csv(
                    edited,
                    path,
                )

                st.success(
                    f"{title} saved."
                )



@st.fragment(run_every=1.0)
def live_optimizer_terminal():
    """
    Live view of the exact stdout/stderr produced by optimizer.py.

    This fragment reruns independently every second, so the rest of the
    Streamlit page remains responsive.
    """
    running, pid = optimizer_status()

    header_cols = st.columns([2, 3])

    if running:
        header_cols[0].success(
            f"Optimizer running · PID {pid}"
        )
        header_cols[1].caption(
            "Live terminal output · auto-refresh every 1 second"
        )
    else:
        header_cols[0].info(
            "Optimizer not currently running"
        )
        header_cols[1].caption(
            "Showing the latest available optimizer output"
        )

    terminal_text = log_tail(
        lines=140
    )

    st.code(
        terminal_text,
        language="text",
    )

    if running:
        # Small visual heartbeat so it is obvious that the panel is live.
        st.caption(
            f"Last UI refresh: {datetime.now().strftime('%H:%M:%S')}"
        )


def optimizer_page():
    st.subheader(
        "Run optimizer"
    )

    missions = normalize_missions(
        load_csv(MISSIONS)
    )

    errors = validate_missions(
        missions
    )

    if errors:
        st.error(
            "Mission input has validation errors. "
            "Fix these before optimizing."
        )

        with st.expander(
            "Show errors"
        ):
            for error in errors:
                st.write(
                    "•",
                    error,
                )

    col1, col2, col3, col4 = st.columns(4)

    runs = col1.number_input(
        "Independent runs",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
    )

    population = col2.number_input(
        "Population",
        min_value=10,
        max_value=2000,
        value=200,
        step=10,
    )

    generations = col3.number_input(
        "Generations",
        min_value=1,
        max_value=5000,
        value=180,
        step=10,
    )

    seed = col4.number_input(
        "Random seed",
        min_value=0,
        max_value=10_000_000,
        value=42,
        step=1,
    )

    live = st.checkbox(
        "Open separate live Pareto window",
        value=False,
    )

    live_every = st.number_input(
        "Live update every N generations",
        min_value=1,
        max_value=100,
        value=2,
        step=1,
        disabled=not live,
    )

    running, pid = optimizer_status()

    status_cols = st.columns(
        [1, 1, 3]
    )

    if running:
        status_cols[0].success(
            f"Running · PID {pid}"
        )
    else:
        status_cols[0].info(
            "Optimizer idle"
        )

    if status_cols[1].button(
        "Start optimization",
        type="primary",
        disabled=running or bool(errors),
    ):
        try:
            start_optimizer(
                int(runs),
                int(population),
                int(generations),
                int(seed),
                bool(live),
                int(live_every),
            )

            st.success(
                "Optimizer started."
            )
            st.rerun()

        except Exception as exc:
            st.error(
                str(exc)
            )

    if status_cols[2].button(
        "Stop optimizer",
        disabled=not running,
    ):
        stop_optimizer()
        st.warning(
            "Stop signal sent."
        )
        st.rerun()

    st.caption(
        "The GUI launches the same optimizer.py as the Terminal. "
        "You can still run optimizer.py manually whenever the GUI is not running it."
    )

    st.subheader(
        "Live terminal"
    )

    st.caption(
        "This is the same stdout/stderr you would see when running "
        "optimizer.py directly in Terminal. The panel updates automatically."
    )

    live_optimizer_terminal()



def timeline_data():
    pareto = viz.load_pareto()
    gallery = viz.load_gallery()
    named = viz.named_solutions(pareto)

    estimated_missions_path = DATA / "missions_with_estimates.csv"
    raw_missions_path = DATA / "missions.csv"

    missions = viz.load_csv_dict(
        (
            estimated_missions_path
            if estimated_missions_path.exists()
            else raw_missions_path
        ),
        "id",
    )

    airports = viz.load_csv_dict(
        DATA / "airports.csv",
        "icao",
    )

    actual_path = DATA / "actual_flown.json"
    actual = (
        viz.load_json(actual_path)
        if actual_path.exists()
        else {}
    )

    return pareto, gallery, named, missions, airports, actual


def draw_streamlit_timeline(
    solution,
    title,
    missions,
    airports,
    gallery_position=None,
):
    fig = plt.figure(
        figsize=(16, 8.5)
    )

    ax = fig.add_axes(
        [
            0.07,
            0.12,
            0.76,
            0.80,
        ]
    )

    info = fig.add_axes(
        [
            0.845,
            0.12,
            0.145,
            0.80,
        ]
    )

    routes = viz.reconstruct_routes(
        solution,
        missions,
        airports,
    )

    viz.draw_optimizer(
        ax,
        info,
        solution,
        routes,
        title,
        gallery_position,
    )

    return fig


def draw_streamlit_actual(
    actual,
    airports,
):
    fig = plt.figure(
        figsize=(16, 8.5)
    )

    ax = fig.add_axes(
        [
            0.07,
            0.12,
            0.76,
            0.80,
        ]
    )

    info = fig.add_axes(
        [
            0.845,
            0.12,
            0.145,
            0.80,
        ]
    )

    viz.draw_actual(
        ax,
        info,
        actual,
        airports,
    )

    return fig



def solution_timeline_dataframe(
    solution,
    missions,
    airports,
):
    # Use the optimizer's explicit validated movement log whenever available.
    # This guarantees the GUI shows exactly the schedule that passed the hard
    # feasibility checks, including crew on empty legs.
    if solution.get("mission_snapshot"):
        missions = solution["mission_snapshot"]

    explicit = solution.get(
        "aircraft_movements"
    )

    if explicit:
        crew_actions = {
            action.get("mission"): action
            for action in solution.get(
                "crew_actions",
                [],
            )
            if action.get("mission")
        }

        rows = []

        for reg, movements in explicit.items():
            for movement in movements:
                start = pd.to_datetime(
                    movement["start"]
                )
                end = pd.to_datetime(
                    movement["end"]
                )

                captain = movement.get(
                    "captain",
                    "",
                )
                fo = movement.get(
                    "fo",
                    "",
                )

                if movement["type"] == "MISSION":
                    mid = movement["mission"]
                    mission = missions[mid]
                    action = crew_actions.get(
                        mid,
                        {},
                    )

                    rows.append({
                        "aircraft": reg,
                        "start": start,
                        "end": end,
                        "type": "Mission",
                        "label": mid,
                        "route": (
                            f'{movement["from"]} → '
                            f'{movement["to"]}'
                        ),
                        "detail": (
                            f'{mission["pax"]} pax'
                        ),
                        "crew": (
                            f'{captain}/{fo}'
                        ),
                        "pax": int(
                            mission["pax"]
                        ),
                        "nm": None,
                        "swap": (
                            action.get("action")
                            == "HANDOVER"
                        ),
                        "swap_type": (
                            "Outstation"
                            if int(
                                action.get(
                                    "nonhome_swap_event",
                                    0,
                                )
                            )
                            else (
                                "Homebase"
                                if action.get("action")
                                == "HANDOVER"
                                else ""
                            )
                        ),
                        "deadhead_in": int(
                            action.get(
                                "inbound_deadhead_pilots",
                                0,
                            )
                        ),
                        "deadhead_out": int(
                            action.get(
                                "return_pilots",
                                0,
                            )
                        ),
                    })

                else:
                    rows.append({
                        "aircraft": reg,
                        "start": start,
                        "end": end,
                        "type": "Empty positioning",
                        "label": "EMPTY",
                        "route": (
                            f'{movement["from"]} → '
                            f'{movement["to"]}'
                        ),
                        "detail": (
                            f'{float(movement.get("nm",0)):.0f} NM'
                        ),
                        "crew": (
                            f'{captain}/{fo}'
                        ),
                        "pax": None,
                        "nm": round(
                            float(
                                movement.get(
                                    "nm",
                                    0,
                                )
                            ),
                            0,
                        ),
                        "swap": False,
                        "swap_type": "",
                        "deadhead_in": 0,
                        "deadhead_out": 0,
                    })

        return pd.DataFrame(
            rows
        )

    # Legacy fallback for older solution files.
    routes = viz.reconstruct_routes(
        solution,
        missions,
        airports,
    )

    crew_actions = {
        action.get("mission"): action
        for action in solution.get(
            "crew_actions",
            [],
        )
        if action.get("mission")
    }

    rows = []

    for reg, route in routes.items():
        for leg in route["legs"]:
            if leg["type"] == "MISSION":
                action = crew_actions.get(
                    leg["mission"],
                    {},
                )

                rows.append({
                    "aircraft": reg,
                    "start": leg["departure"],
                    "end": leg["arrival"],
                    "type": "Mission",
                    "label": leg["mission"],
                    "route": (
                        f'{leg["from"]} → {leg["to"]}'
                    ),
                    "detail": (
                        f'{leg["pax"]} pax'
                    ),
                    "crew": (
                        f'{leg["captain"]}/{leg["fo"]}'
                    ),
                    "pax": leg["pax"],
                    "nm": None,
                    "swap": (
                        action.get("action") == "HANDOVER"
                    ),
                    "swap_type": "",
                    "deadhead_in": int(
                        action.get(
                            "inbound_deadhead_pilots",
                            0,
                        )
                    ),
                    "deadhead_out": int(
                        action.get(
                            "return_pilots",
                            0,
                        )
                    ),
                })
            else:
                rows.append({
                    "aircraft": reg,
                    "start": leg["departure"],
                    "end": leg["arrival"],
                    "type": "Empty positioning",
                    "label": "EMPTY",
                    "route": (
                        f'{leg["from"]} → {leg["to"]}'
                    ),
                    "detail": (
                        f'{leg["nm"]:.0f} NM'
                    ),
                    "crew": (
                        f'{leg.get("captain","?")}/'
                        f'{leg.get("fo","?")}'
                    ),
                    "pax": None,
                    "nm": round(
                        leg["nm"],
                        0,
                    ),
                    "swap": False,
                    "swap_type": "",
                    "deadhead_in": 0,
                    "deadhead_out": 0,
                })

    return pd.DataFrame(
        rows
    )



def actual_timeline_dataframe(
    actual,
    airports,
):
    rows = []

    for reg, legs in actual.items():
        day_counts = {}

        for leg in legs:
            day = pd.Timestamp(
                leg["date"]
            )

            day_counts.setdefault(
                leg["date"],
                0,
            )

            slot = day_counts[
                leg["date"]
            ]

            day_counts[
                leg["date"]
            ] += 1

            distance = viz.distance_nm(
                airports[leg["from"]],
                airports[leg["to"]],
            )

            # Historical screenshots do not contain machine-readable exact
            # block timestamps in actual_flown.json. Preserve sequence and use
            # the same approximate layout convention as the legacy visualizer.
            start = (
                day
                + pd.Timedelta(
                    hours=1 + slot * 2
                )
            )

            duration_hours = max(
                0.6,
                min(
                    3.2,
                    20 / 60 + distance / 410,
                ),
            )

            end = (
                start
                + pd.Timedelta(
                    hours=duration_hours
                )
            )

            is_mission = (
                leg.get("mission_hint")
                is not None
            )

            rows.append({
                "aircraft": reg,
                "start": start,
                "end": end,
                "type": (
                    "Mission"
                    if is_mission
                    else "Empty positioning"
                ),
                "label": (
                    leg.get("mission_hint")
                    or "EMPTY"
                ),
                "route": (
                    f'{leg["from"]} → {leg["to"]}'
                ),
                "detail": (
                    "Historical flown leg"
                ),
                "crew": "",
                "pax": None,
                "nm": round(distance, 0),
                "swap": False,
                "swap_type": "",
                "deadhead_in": 0,
                "deadhead_out": 0,
            })

    df = pd.DataFrame(
        rows
    )

    if not df.empty:
        df["start"] = pd.to_datetime(
            df["start"]
        )
        df["end"] = pd.to_datetime(
            df["end"]
        )

    return df



def operations_board_payload(
    solution,
    missions,
):
    """
    Convert a validated solution into a browser-friendly operations-board model.

    Prefer explicit aircraft_movements so the web board is always identical to
    the schedule that passed the hard feasibility checks.
    """
    if solution.get("mission_snapshot"):
        missions = solution["mission_snapshot"]

    actions = {
        action.get("mission"): action
        for action in solution.get(
            "crew_actions",
            [],
        )
        if action.get("mission")
    }

    aircraft_rows = []
    all_dates = set()

    explicit = solution.get(
        "aircraft_movements",
        {},
    )

    for reg, movements in explicit.items():
        cards = []

        for movement in movements:
            start = pd.to_datetime(
                movement["start"]
            )
            end = pd.to_datetime(
                movement["end"]
            )

            day = start.strftime(
                "%Y-%m-%d"
            )

            all_dates.add(
                day
            )

            captain = movement.get(
                "captain",
                "?",
            )
            fo = movement.get(
                "fo",
                "?",
            )
            crew = (
                f"{captain}/{fo}"
            )

            if movement[
                "type"
            ] == "MISSION":
                mid = movement[
                    "mission"
                ]
                mission = missions[
                    mid
                ]
                action = actions.get(
                    mid,
                    {},
                )

                swap = (
                    action.get(
                        "action"
                    )
                    == "HANDOVER"
                    and not str(
                        action.get(
                            "transition_mode",
                            "",
                        )
                    ).startswith(
                        "IDLE_RETURN_"
                    )
                )

                nonhome = int(
                    action.get(
                        "nonhome_swap_event",
                        0,
                    )
                )

                cards.append({
                    "day": day,
                    "kind": "mission",
                    "label": mid,
                    "route": (
                        f'{movement["from"]} → '
                        f'{movement["to"]}'
                    ),
                    "start": start.strftime(
                        "%H:%M"
                    ),
                    "end": end.strftime(
                        "%H:%M"
                    ),
                    "crew": crew,
                    "extra": (
                        f'{int(mission["pax"])} pax'
                    ),
                    "swap": swap,
                    "swap_type": (
                        "Outstation swap"
                        if swap and nonhome
                        else (
                            "Homebase swap"
                            if swap
                            else ""
                        )
                    ),
                    "sort": start.isoformat(),
                })

            else:
                purpose = movement.get(
                    "purpose",
                    "",
                )

                target_mission = movement.get(
                    "target_mission",
                )

                target_action = actions.get(
                    target_mission,
                    {},
                )

                is_idle_return = (
                    purpose
                    == "IDLE_RETURN_HOME"
                )

                is_from_home = (
                    purpose
                    == "POSITION_FROM_HOME"
                )

                home_swap = (
                    is_from_home
                    and target_action.get(
                        "action"
                    )
                    == "HANDOVER"
                )

                cards.append({
                    "day": day,
                    "kind": "returnhome" if is_idle_return else "empty",
                    "label": (
                        "RETURN HOME"
                        if is_idle_return
                        else (
                            "FROM HOMEBASE"
                            if is_from_home
                            else "EMPTY"
                        )
                    ),
                    "route": (
                        f'{movement["from"]} → '
                        f'{movement["to"]}'
                    ),
                    "start": start.strftime(
                        "%H:%M"
                    ),
                    "end": end.strftime(
                        "%H:%M"
                    ),
                    "crew": crew,
                    "extra": (
                        f'{float(movement.get("nm",0)):.0f} NM'
                    ),
                    "swap": bool(
                        home_swap
                    ),
                    "swap_type": (
                        "Homebase swap"
                        if home_swap
                        else ""
                    ),
                    "sort": start.isoformat(),
                })

        cards.sort(
            key=lambda x: x[
                "sort"
            ]
        )

        first_crew = (
            cards[0]["crew"]
            if cards
            else "—"
        )

        aircraft_rows.append({
            "registration": reg,
            "crew": first_crew,
            "cards": cards,
        })

    # Parking is represented by aircraft_actions in the solution.
    #
    # IMPORTANT: parking crew must be the crew physically attached to the
    # aircraft AFTER its previous movement, not the first crew ever seen on
    # the aircraft row. The old GUI incorrectly used row["crew"] here, which
    # made parking cards stale after a crew handover.
    for row in aircraft_rows:
        reg = row[
            "registration"
        ]

        explicit_movements = solution.get(
            "aircraft_movements",
            {},
        ).get(
            reg,
            [],
        )

        for action in solution.get(
            "aircraft_actions",
            {},
        ).get(
            reg,
            [],
        ):
            days = int(
                action.get(
                    "parking_days",
                    0,
                )
                or 0
            )

            if days <= 0:
                continue

            target = action.get(
                "mission"
            )

            # End-of-horizon parking: crew/location come from the final
            # aircraft movement.
            if target == "__END__":
                if not explicit_movements:
                    continue

                previous_movement = explicit_movements[-1]
                target_day = pd.Timestamp(
                    previous_movement["end"]
                ).normalize() + pd.Timedelta(
                    days=days
                )

            else:
                # Find the FIRST movement belonging to the transition into the
                # target mission. If positioning is required this is the empty
                # leg; otherwise it is the mission itself.
                target_index = next(
                    (
                        i
                        for i, movement
                        in enumerate(
                            explicit_movements
                        )
                        if movement.get(
                            "target_mission"
                        ) == target
                    ),
                    None,
                )

                if (
                    target_index is None
                    or target_index == 0
                ):
                    continue

                # Parking occurs after this preceding movement, before the
                # transition into the target mission begins.
                previous_movement = explicit_movements[
                    target_index - 1
                ]

                target_mission = solution.get(
                    "mission_snapshot",
                    {},
                ).get(
                    target,
                )

                if target_mission:
                    target_day = pd.Timestamp(
                        target_mission[
                            "departure"
                        ]
                    ).normalize()
                else:
                    target_day = pd.Timestamp(
                        explicit_movements[
                            target_index
                        ]["start"]
                    ).normalize()

            parking_crew = (
                f'{previous_movement.get("captain","?")}/'
                f'{previous_movement.get("fo","?")}'
            )

            parking_airport = previous_movement.get(
                "to",
                "Away from homebase",
            )

            for offset in range(
                days,
                0,
                -1,
            ):
                park_day = (
                    target_day
                    - pd.Timedelta(
                        days=offset
                    )
                ).strftime(
                    "%Y-%m-%d"
                )

                all_dates.add(
                    park_day
                )

                # Parking may coexist visually with a flight on the first/last
                # calendar day; don't suppress the parking card merely because
                # another movement exists that day.
                row[
                    "cards"
                ].append({
                    "day": park_day,
                    "kind": "parking",
                    "label": "PARK",
                    "route": (
                        f'PARK {parking_airport}'
                    ),
                    "start": "",
                    "end": "",
                    "crew": parking_crew,
                    "extra": (
                        "Aircraft + attached crew away"
                    ),
                    "swap": False,
                    "swap_type": "",
                    "sort": (
                        park_day
                        + "T23:59:00"
                    ),
                })

        row[
            "cards"
        ].sort(
            key=lambda x: x[
                "sort"
            ]
        )

    return {
        "aircraft": aircraft_rows,
        "dates": sorted(
            all_dates
        ),
    }


def render_operations_board_component(
    solution,
    missions,
    title,
):
    payload = operations_board_payload(
        solution,
        missions,
    )

    if not payload["dates"]:
        st.info(
            "No aircraft movements available."
        )
        return

    import json as _json

    data_json = _json.dumps(
        payload,
        ensure_ascii=False,
    ).replace(
        "</",
        "<\\/",
    )

    objectives = solution.get(
        "objectives",
        {},
    )
    metrics = solution.get(
        "metrics",
        {},
    )

    header = {
        "title": title,
        "cost": (
            f'€{objectives.get("total_operational_cost_eur",0):,.0f}'
        ),
        "complexity": int(
            objectives.get(
                "complexity_score",
                0,
            )
        ),
        "empty": int(
            metrics.get(
                "empty_legs",
                0,
            )
        ),
        "pilot_days": int(
            metrics.get(
                "charged_pilot_days",
                0,
            )
        ),
        "swaps": int(
            metrics.get(
                "nonhomebase_swap_events",
                0,
            )
        ),
    }

    header_json = _json.dumps(
        header,
        ensure_ascii=False,
    )

    component_html = f"""
<div id="ops-board-root">
<style>
* {{ box-sizing:border-box; }}

:root {{
  --row-label-width: 138px;
  --day-width: 190px;
  --row-height: 118px;
  --header-height: 48px;
}}

body {{
  margin:0;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  color:#161616;
  background:transparent;
  overflow:hidden;
}}

.board-shell {{
  width:100%;
}}

.kpis {{
  display:flex;
  flex-wrap:wrap;
  gap:10px;
  align-items:center;
  margin:0 0 12px 0;
}}

.kpi {{
  border:1px solid #ddd;
  border-radius:16px;
  padding:8px 13px;
  background:#fff;
  min-width:105px;
}}

.kpi b {{
  display:block;
  font-size:17px;
}}

.kpi span {{
  font-size:11px;
  color:#666;
}}

.toolbar {{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  margin-bottom:9px;
  flex-wrap:wrap;
}}

.toolbar-left {{
  display:flex;
  align-items:center;
  gap:10px;
  min-width:0;
}}

.toolbar .title {{
  font-size:18px;
  font-weight:750;
}}

.toolbar .hint {{
  font-size:11px;
  color:#777;
}}

.toolbar-controls {{
  display:flex;
  gap:6px;
  align-items:center;
  flex-wrap:wrap;
}}

.toolbar button {{
  appearance:none;
  border:1px solid #d5d5d5;
  background:#fff;
  padding:7px 10px;
  border-radius:9px;
  cursor:pointer;
  font-size:12px;
}}

.toolbar button:hover {{
  background:#f7f7f7;
}}

.toolbar button:disabled {{
  opacity:.35;
  cursor:not-allowed;
}}

.zoom-readout {{
  min-width:52px;
  text-align:center;
  font-size:12px;
  color:#555;
}}

.layout {{
  display:grid;
  grid-template-columns:minmax(0,1fr) 275px;
  gap:12px;
  align-items:start;
}}

.canvas-wrap {{
  border:1px solid #ddd;
  border-radius:14px;
  background:#fff;
  overflow:hidden;
}}

.canvas-scroll {{
  width:100%;
  height:640px;
  overflow:auto;
  position:relative;
  overscroll-behavior:contain;
  scrollbar-gutter:stable both-edges;
  background:#fff;
}}

.board {{
  position:relative;
  min-width:max-content;
  width:max-content;
}}

.header-row,
.aircraft-row {{
  display:grid;
  grid-template-columns:
    var(--row-label-width)
    repeat(var(--day-count), var(--day-width));
}}

.header-row {{
  position:sticky;
  top:0;
  z-index:30;
  background:#fafafa;
  border-bottom:1px solid #ddd;
  min-height:var(--header-height);
}}

.header-cell {{
  min-height:var(--header-height);
  padding:13px 10px;
  border-left:1px solid #e3e3e3;
  color:#5f5f5f;
  font-size:13px;
  white-space:nowrap;
}}

.header-cell:first-child {{
  position:sticky;
  left:0;
  z-index:40;
  border-left:0;
  background:#fafafa;
  border-right:1px solid #ddd;
}}

.aircraft-row {{
  min-height:var(--row-height);
  border-bottom:1px solid #e3e3e3;
}}

.aircraft-row:last-child {{
  border-bottom:0;
}}

.aircraft-label {{
  position:sticky;
  left:0;
  z-index:20;
  min-height:var(--row-height);
  padding:18px 12px;
  background:#fbfbfb;
  border-right:1px solid #ddd;
}}

.aircraft-label b {{
  display:block;
  font-size:16px;
  margin-bottom:5px;
}}

.aircraft-label span {{
  display:block;
  font-size:12px;
  color:#666;
}}

.day-cell {{
  position:relative;
  min-width:0;
  min-height:var(--row-height);
  padding:8px 7px;
  border-left:1px solid #ededed;
  display:flex;
  gap:6px;
  flex-wrap:wrap;
  align-content:center;
}}

.day-cell:nth-child(even) {{
  background:#fcfcfc;
}}

.card {{
  border:1px solid #aaa;
  border-radius:9px;
  padding:7px 8px;
  cursor:pointer;
  min-width:94px;
  max-width:100%;
  flex:1 1 94px;
  text-align:left;
  color:#1b1b1b;
  transition:
    transform .08s ease,
    box-shadow .08s ease,
    border-color .08s ease;
}}

.card:hover {{
  transform:translateY(-1px);
  box-shadow:0 2px 7px rgba(0,0,0,.08);
}}

.card:focus {{
  outline:2px solid #2878ff;
  outline-offset:1px;
}}

.card.selected {{
  border-color:#2878ff;
  box-shadow:0 0 0 2px rgba(40,120,255,.16);
}}

.card.mission {{
  background:#e7f3e8;
  border-color:#a8bea9;
}}

.card.empty {{
  background:#fafafa;
  border-style:dashed;
  color:#555;
}}

.card.returnhome {{
  background:#eef4fb;
  border-color:#9eb5cf;
  color:#334a62;
}}

.card.parking {{
  background:#f0e9f8;
  border-color:#cab9db;
  color:#5d5366;
}}

.card .route {{
  font-size:13px;
  font-weight:720;
  line-height:1.25;
}}

.card .meta {{
  margin-top:4px;
  font-size:11px;
  color:#666;
  line-height:1.25;
}}

.swap-dot {{
  display:inline-block;
  width:7px;
  height:7px;
  border-radius:50%;
  background:#2878ff;
  margin-right:5px;
  vertical-align:1px;
}}

.detail {{
  position:sticky;
  top:0;
  border:1px solid #ddd;
  border-radius:14px;
  background:#fff;
  padding:15px;
  max-height:640px;
  overflow:auto;
}}

.detail h3 {{
  margin:0 0 14px;
  font-size:18px;
}}

.detail-block {{
  padding:10px 0;
  border-bottom:1px solid #e5e5e5;
}}

.detail-block:last-child {{
  border-bottom:0;
}}

.label {{
  font-size:10px;
  letter-spacing:.08em;
  text-transform:uppercase;
  color:#777;
  margin-bottom:5px;
}}

.value {{
  font-size:14px;
  font-weight:600;
  overflow-wrap:anywhere;
}}

.legend {{
  display:flex;
  flex-wrap:wrap;
  gap:14px;
  margin-top:9px;
  color:#666;
  font-size:11px;
}}

.legend i {{
  display:inline-block;
  width:12px;
  height:12px;
  border-radius:3px;
  border:1px solid #aaa;
  vertical-align:-2px;
  margin-right:4px;
}}

.legend .m {{
  background:#e7f3e8;
}}

.legend .e {{
  background:#fafafa;
  border-style:dashed;
}}

.legend .p {{
  background:#f0e9f8;
}}

.empty-state {{
  font-size:12px;
  color:#aaa;
  padding:12px 4px;
}}

@media(max-width:950px) {{
  .layout {{
    grid-template-columns:1fr;
  }}

  .detail {{
    position:static;
    max-height:none;
    order:-1;
  }}

  .canvas-scroll {{
    height:560px;
  }}
}}
</style>

<div class="board-shell">

  <div class="kpis" id="kpis"></div>

  <div class="toolbar">
    <div class="toolbar-left">
      <div class="title" id="boardTitle"></div>
      <div class="hint">
        Scroll/trackpad to move · Shift+wheel for horizontal · Cmd/Ctrl+wheel to zoom
      </div>
    </div>

    <div class="toolbar-controls">
      <button type="button" id="jumpStart">Start</button>
      <button type="button" id="jumpToday">Fit start</button>
      <button type="button" id="zoomOut">−</button>
      <div class="zoom-readout" id="zoomReadout">100%</div>
      <button type="button" id="zoomIn">+</button>
      <button type="button" id="zoomReset">100%</button>
      <button type="button" id="zoomFit">Fit</button>
    </div>
  </div>

  <div class="layout">
    <div>
      <div class="canvas-wrap">
        <div class="canvas-scroll" id="canvasScroll">
          <div class="board" id="board"></div>
        </div>
      </div>

      <div class="legend">
        <span><i class="m"></i>Customer mission</span>
        <span><i class="e"></i>Empty positioning</span>
        <span>↩ Return homebase</span>
        <span><i class="p"></i>Away parking</span>
        <span>● Crew handover</span>
      </div>
    </div>

    <aside class="detail" aria-live="polite">
      <h3 id="detailTitle">Select a movement</h3>

      <div class="detail-block">
        <div class="label">Route</div>
        <div class="value" id="detailRoute">Click any card.</div>
      </div>

      <div class="detail-block">
        <div class="label">Timing</div>
        <div class="value" id="detailTiming">—</div>
      </div>

      <div class="detail-block">
        <div class="label">Aircraft</div>
        <div class="value" id="detailAircraft">—</div>
      </div>

      <div class="detail-block">
        <div class="label">Attached crew</div>
        <div class="value" id="detailCrew">—</div>
      </div>

      <div class="detail-block">
        <div class="label">Operational detail</div>
        <div class="value" id="detailExtra">—</div>
      </div>
    </aside>
  </div>

</div>

<script>
(() => {{
  const data = {data_json};
  const header = {header_json};

  const board = document.getElementById("board");
  const scroll = document.getElementById("canvasScroll");
  const title = document.getElementById("boardTitle");
  const kpis = document.getElementById("kpis");

  const zoomOut = document.getElementById("zoomOut");
  const zoomIn = document.getElementById("zoomIn");
  const zoomReset = document.getElementById("zoomReset");
  const zoomFit = document.getElementById("zoomFit");
  const zoomReadout = document.getElementById("zoomReadout");
  const jumpStart = document.getElementById("jumpStart");
  const jumpToday = document.getElementById("jumpToday");

  const dTitle = document.getElementById("detailTitle");
  const dRoute = document.getElementById("detailRoute");
  const dTiming = document.getElementById("detailTiming");
  const dAircraft = document.getElementById("detailAircraft");
  const dCrew = document.getElementById("detailCrew");
  const dExtra = document.getElementById("detailExtra");

  let zoom = 1.0;
  const minZoom = 0.55;
  const maxZoom = 2.2;
  const baseDayWidth = 190;

  title.textContent = header.title;

  [
    [header.cost, "Total cost"],
    [header.complexity, "Complexity"],
    [header.empty, "Empty legs"],
    [header.pilot_days, "Pilot-days"],
    [header.swaps, "Outstation swaps"]
  ].forEach(([v,l]) => {{
    const el = document.createElement("div");
    el.className = "kpi";
    el.innerHTML = `<b>${{v}}</b><span>${{l}}</span>`;
    kpis.appendChild(el);
  }});

  function prettyDay(iso) {{
    const d = new Date(iso + "T00:00:00Z");
    return d.toLocaleDateString(
      "en-GB",
      {{
        weekday:"short",
        day:"2-digit",
        month:"short",
        timeZone:"UTC"
      }}
    );
  }}

  function setZoom(nextZoom, anchorClientX=null) {{
    const oldZoom = zoom;
    zoom = Math.max(
      minZoom,
      Math.min(
        maxZoom,
        nextZoom
      )
    );

    const oldWidth = baseDayWidth * oldZoom;
    const newWidth = baseDayWidth * zoom;

    const scrollRect = scroll.getBoundingClientRect();
    const anchorX =
      anchorClientX == null
        ? scrollRect.width / 2
        : anchorClientX - scrollRect.left;

    const contentX = scroll.scrollLeft + anchorX;
    const ratio = newWidth / oldWidth;

    document.documentElement.style.setProperty(
      "--day-width",
      `${{newWidth}}px`
    );

    scroll.scrollLeft =
      contentX * ratio - anchorX;

    zoomReadout.textContent =
      `${{Math.round(zoom * 100)}}%`;
  }}

  function fitZoom() {{
    const available = Math.max(
      300,
      scroll.clientWidth - 138
    );

    const fit =
      available /
      Math.max(
        1,
        data.dates.length
      ) /
      baseDayWidth;

    setZoom(
      Math.max(
        minZoom,
        Math.min(
          1.15,
          fit
        )
      )
    );

    scroll.scrollLeft = 0;
  }}

  function showDetail(card, reg, el) {{
    document
      .querySelectorAll(".card.selected")
      .forEach(x => x.classList.remove("selected"));

    if (el) {{
      el.classList.add("selected");
    }}

    dTitle.textContent =
      card.kind === "mission"
        ? "Customer mission"
        : card.kind === "empty"
        ? "Empty positioning"
        : card.kind === "returnhome"
        ? "Return to homebase"
        : "Away parking";

    dRoute.textContent =
      card.route || "—";

    dTiming.textContent =
      card.start && card.end
        ? `${{card.start}} → ${{card.end}} UTC`
        : "Calendar day";

    dAircraft.textContent =
      reg;

    dCrew.textContent =
      card.crew || "—";

    let extra =
      card.extra || "—";

    if (card.swap) {{
      extra +=
        ` · ${{card.swap_type}}`;
    }}

    dExtra.textContent =
      extra;
  }}

  function makeCard(card, reg) {{
    const el =
      document.createElement("button");

    el.type = "button";
    el.className =
      `card ${{card.kind}}`;

    const swap =
      card.swap
        ? `<span class="swap-dot"></span>`
        : "";

    el.innerHTML = `
      <div class="route">
        ${{swap}}${{card.route}}
      </div>

      <div class="meta">
        ${{card.label}}
        ${{card.extra ? " · " + card.extra : ""}}
      </div>

      <div class="meta">
        ${{card.crew || ""}}
      </div>
    `;

    el.addEventListener(
      "click",
      () => {{
        showDetail(
          card,
          reg,
          el
        );
      }}
    );

    return el;
  }}

  function render() {{
    board.innerHTML = "";

    board.style.setProperty(
      "--day-count",
      data.dates.length
    );

    const headerRow =
      document.createElement("div");

    headerRow.className =
      "header-row";

    const blank =
      document.createElement("div");

    blank.className =
      "header-cell";

    blank.textContent =
      "Aircraft";

    headerRow.appendChild(
      blank
    );

    data.dates.forEach(
      day => {{
        const cell =
          document.createElement("div");

        cell.className =
          "header-cell";

        cell.textContent =
          prettyDay(day);

        headerRow.appendChild(
          cell
        );
      }}
    );

    board.appendChild(
      headerRow
    );

    data.aircraft.forEach(
      row => {{
        const line =
          document.createElement("div");

        line.className =
          "aircraft-row";

        const label =
          document.createElement("div");

        label.className =
          "aircraft-label";

        label.innerHTML = `
          <b>${{row.registration}}</b>
          <span>${{row.crew || "—"}}</span>
        `;

        line.appendChild(
          label
        );

        data.dates.forEach(
          day => {{
            const cell =
              document.createElement("div");

            cell.className =
              "day-cell";

            const cards =
              row.cards.filter(
                card =>
                  card.day === day
              );

            if (cards.length) {{
              cards.forEach(
                card => {{
                  cell.appendChild(
                    makeCard(
                      card,
                      row.registration
                    )
                  );
                }}
              );
            }} else {{
              const empty =
                document.createElement("div");

              empty.className =
                "empty-state";

              empty.textContent =
                "—";

              cell.appendChild(
                empty
              );
            }}

            line.appendChild(
              cell
            );
          }}
        );

        board.appendChild(
          line
        );
      }}
    );
  }}

  // Trackpad / mouse UX:
  // - regular wheel: natural vertical scrolling
  // - Shift+wheel: horizontal scrolling
  // - Cmd/Ctrl+wheel: zoom around pointer
  scroll.addEventListener(
    "wheel",
    (event) => {{
      if (
        event.ctrlKey
        || event.metaKey
      ) {{
        event.preventDefault();

        const direction =
          event.deltaY < 0
            ? 1.10
            : 0.90;

        setZoom(
          zoom * direction,
          event.clientX
        );

        return;
      }}

      if (event.shiftKey) {{
        event.preventDefault();

        scroll.scrollLeft +=
          event.deltaY
          + event.deltaX;

        return;
      }}

      // On trackpads, horizontal deltaX should naturally move the canvas.
      if (
        Math.abs(event.deltaX)
        > Math.abs(event.deltaY)
      ) {{
        scroll.scrollLeft +=
          event.deltaX;
      }}
    }},
    {{
      passive:false
    }}
  );

  // Drag-to-pan with middle mouse / Space+left click.
  let dragging = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let startScrollLeft = 0;
  let startScrollTop = 0;
  let spacePressed = false;

  window.addEventListener(
    "keydown",
    e => {{
      if (e.code === "Space") {{
        spacePressed = true;
      }}
    }}
  );

  window.addEventListener(
    "keyup",
    e => {{
      if (e.code === "Space") {{
        spacePressed = false;
      }}
    }}
  );

  scroll.addEventListener(
    "mousedown",
    e => {{
      if (
        e.button === 1
        || (
          e.button === 0
          && spacePressed
        )
      ) {{
        dragging = true;
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        startScrollLeft =
          scroll.scrollLeft;
        startScrollTop =
          scroll.scrollTop;

        scroll.style.cursor =
          "grabbing";

        e.preventDefault();
      }}
    }}
  );

  window.addEventListener(
    "mousemove",
    e => {{
      if (!dragging) return;

      scroll.scrollLeft =
        startScrollLeft
        - (
          e.clientX
          - dragStartX
        );

      scroll.scrollTop =
        startScrollTop
        - (
          e.clientY
          - dragStartY
        );
    }}
  );

  window.addEventListener(
    "mouseup",
    () => {{
      dragging = false;
      scroll.style.cursor = "";
    }}
  );

  zoomOut.addEventListener(
    "click",
    () => setZoom(
      zoom / 1.15
    )
  );

  zoomIn.addEventListener(
    "click",
    () => setZoom(
      zoom * 1.15
    )
  );

  zoomReset.addEventListener(
    "click",
    () => {{
      setZoom(1.0);
    }}
  );

  zoomFit.addEventListener(
    "click",
    fitZoom
  );

  jumpStart.addEventListener(
    "click",
    () => {{
      scroll.scrollTo(
        {{
          left:0,
          behavior:"smooth"
        }}
      );
    }}
  );

  jumpToday.addEventListener(
    "click",
    () => {{
      fitZoom();
      scroll.scrollTo(
        {{
          left:0,
          behavior:"smooth"
        }}
      );
    }}
  );

  render();
  setZoom(1.0);
}})();
</script>
"""

    # Keep the component itself at a stable viewport height. The planning
    # canvas scrolls internally in both directions.
    components.html(
        component_html,
        height=820,
        scrolling=False,
    )



def render_native_timeline_chart(
    df,
    title,
):
    if df.empty:
        st.info(
            "No timeline data available."
        )
        return

    type_order = [
        "Mission",
        "Empty positioning",
        "Away parking",
    ]

    # Stable categorical order: aircraft rows remain predictable.
    aircraft_order = list(
        dict.fromkeys(
            df["aircraft"].tolist()
        )
    )

    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="aircraft",
        color="type",
        text="label",
        category_orders={
            "aircraft": aircraft_order,
            "type": type_order,
        },
        hover_data={
            "aircraft": True,
            "route": True,
            "detail": True,
            "crew": True,
            "pax": True,
            "nm": True,
            "swap_type": True,
            "deadhead_in": True,
            "deadhead_out": True,
            "start": "|%a %d/%m %H:%M UTC",
            "end": "|%a %d/%m %H:%M UTC",
            "type": False,
        },
    )

    fig.update_yaxes(
        autorange="reversed",
        title=None,
        fixedrange=True,
    )

    fig.update_xaxes(
        title=None,
        showgrid=True,
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=[
                dict(
                    count=1,
                    label="1d",
                    step="day",
                    stepmode="backward",
                ),
                dict(
                    count=3,
                    label="3d",
                    step="day",
                    stepmode="backward",
                ),
                dict(
                    count=7,
                    label="7d",
                    step="day",
                    stepmode="backward",
                ),
                dict(
                    step="all",
                    label="All",
                ),
            ]
        ),
    )

    fig.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        cliponaxis=False,
    )

    # Explicit swap markers on top of the mission bars.
    swaps = df[
        df["swap"] == True
    ]

    if not swaps.empty:
        fig.add_trace(
            go.Scatter(
                x=swaps["start"],
                y=swaps["aircraft"],
                mode="markers",
                marker=dict(
                    symbol="diamond-open",
                    size=13,
                    line=dict(
                        width=2,
                    ),
                ),
                name="Crew handover",
                text=[
                    (
                        f'{row.swap_type} crew handover'
                        f'<br>{row.route}'
                        f'<br>DH in: {row.deadhead_in}'
                        f' · DH out: {row.deadhead_out}'
                    )
                    for row
                    in swaps.itertuples()
                ],
                hovertemplate=(
                    "%{text}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        height=max(
            500,
            125
            * len(aircraft_order)
            + 170,
        ),
        margin=dict(
            l=20,
            r=20,
            t=65,
            b=20,
        ),
        legend_title_text="",
        hoverlabel=dict(
            align="left",
        ),
        dragmode="pan",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "lasso2d",
                "select2d",
            ],
        },
    )


def render_solution_metrics(
    solution,
):
    c = solution.get(
        "cost_breakdown",
        {},
    )
    m = solution.get(
        "metrics",
        {},
    )
    o = solution.get(
        "objectives",
        {},
    )

    cols = st.columns(6)

    cols[0].metric(
        "Operational cost",
        f'€{o.get("total_operational_cost_eur", 0):,.0f}',
    )

    cols[1].metric(
        "Complexity",
        int(
            o.get(
                "complexity_score",
                0,
            )
        ),
    )

    cols[2].metric(
        "Empty legs",
        int(
            m.get(
                "empty_legs",
                0,
            )
        ),
    )

    cols[3].metric(
        "Pilot-days",
        int(
            m.get(
                "charged_pilot_days",
                0,
            )
        ),
    )

    cols[4].metric(
        "Parking-days",
        int(
            m.get(
                "aircraft_parking_days",
                0,
            )
        ),
    )

    cols[5].metric(
        "Non-home swaps",
        int(
            m.get(
                "nonhomebase_swap_events",
                0,
            )
        ),
    )

def render_timeline_browser():
    st.subheader(
        "Mission planning board"
    )

    st.caption(
        "Default view: CHEAPEST. The planning canvas now contains the full "
        "horizon: scroll horizontally/vertically, use Cmd/Ctrl + wheel to zoom, "
        "and click any card for crew, timing and operational details."
    )

    try:
        (
            pareto,
            gallery,
            named,
            missions,
            airports,
            actual,
        ) = timeline_data()

    except FileNotFoundError:
        st.info(
            "No optimizer output available yet. Run the optimizer first."
        )
        return

    if "timeline_mode" not in st.session_state:
        st.session_state.timeline_mode = "CHEAPEST"

    if "timeline_gallery_index" not in st.session_state:
        st.session_state.timeline_gallery_index = 0

    if "timeline_browsing_gallery" not in st.session_state:
        st.session_state.timeline_browsing_gallery = False

    named_options = [
        "CHEAPEST",
        "BALANCED",
        "MIN EMPTY LEGS",
        "MIN PILOT DAYS",
        "MIN AIRCRAFT PARKING",
        "MIN COMPLEXITY",
    ]

    selected_named = st.segmented_control(
        "Solution",
        options=(
            named_options
            + ["ACTUAL"]
        ),
        default=(
            st.session_state.timeline_mode
            if st.session_state.timeline_mode
            in named_options + ["ACTUAL"]
            else "CHEAPEST"
        ),
        selection_mode="single",
        key="timeline_strategy_selector",
    )

    if (
        selected_named
        and selected_named
        != st.session_state.timeline_mode
    ):
        st.session_state.timeline_mode = selected_named
        st.session_state.timeline_browsing_gallery = False

        if selected_named in named:
            target = named[
                selected_named
            ]

            for i, candidate in enumerate(gallery):
                if (
                    candidate.get(
                        "schedule_signature"
                    )
                    == target.get(
                        "schedule_signature"
                    )
                ):
                    st.session_state.timeline_gallery_index = i
                    break

    navigation = st.columns(
        [
            1,
            1,
            1.2,
            3,
        ]
    )

    if navigation[0].button(
        "◀ Previous solution",
        use_container_width=True,
        disabled=not bool(gallery),
    ):
        st.session_state.timeline_browsing_gallery = True
        st.session_state.timeline_mode = "ALTERNATIVE"
        st.session_state.timeline_gallery_index = (
            st.session_state.timeline_gallery_index - 1
        ) % len(gallery)

    if navigation[1].button(
        "Next solution ▶",
        use_container_width=True,
        disabled=not bool(gallery),
    ):
        st.session_state.timeline_browsing_gallery = True
        st.session_state.timeline_mode = "ALTERNATIVE"
        st.session_state.timeline_gallery_index = (
            st.session_state.timeline_gallery_index + 1
        ) % len(gallery)

    if gallery:
        jump_to = navigation[2].number_input(
            "Alternative",
            min_value=1,
            max_value=len(gallery),
            value=min(
                len(gallery),
                st.session_state.timeline_gallery_index + 1,
            ),
            step=1,
            label_visibility="collapsed",
        )

        if (
            st.session_state.timeline_browsing_gallery
            and int(jump_to) - 1
            != st.session_state.timeline_gallery_index
        ):
            st.session_state.timeline_gallery_index = (
                int(jump_to)
                - 1
            )

        navigation[3].caption(
            f'{len(gallery)} saved unique schedules · '
            f'alternative #{st.session_state.timeline_gallery_index + 1}'
        )

    if (
        st.session_state.timeline_mode == "ACTUAL"
        and actual
        and not st.session_state.timeline_browsing_gallery
    ):
        st.info(
            "ACTUAL currently uses the historical flown-leg view because the "
            "actual JSON does not contain the same exact timestamp/parking "
            "structure as optimizer solutions."
        )

        df = actual_timeline_dataframe(
            actual,
            airports,
        )

        render_native_timeline_chart(
            df,
            "ACTUAL — historical flown rotations",
        )
        return

    if (
        st.session_state.timeline_browsing_gallery
        and gallery
    ):
        index = (
            st.session_state.timeline_gallery_index
            % len(gallery)
        )

        solution = gallery[
            index
        ]

        render_operations_board_component(
            solution,
            missions,
            f'ALTERNATIVE {index + 1} / {len(gallery)}',
        )
        return

    mode = st.session_state.timeline_mode

    if mode not in named:
        mode = "CHEAPEST"

    solution = named[
        mode
    ]

    for i, candidate in enumerate(gallery):
        if (
            candidate.get(
                "schedule_signature"
            )
            == solution.get(
                "schedule_signature"
            )
        ):
            st.session_state.timeline_gallery_index = i
            break

    render_operations_board_component(
        solution,
        missions,
        mode,
    )



def results_page():
    st.subheader(
        "Results"
    )

    render_results()

    st.divider()

    render_timeline_browser()

    st.divider()

    with st.expander(
        "Open standalone timeline window"
    ):
        st.caption(
            "The separate Matplotlib visualizer remains available "
            "for the same results."
        )

        if st.button(
            "Open visualize.py",
            key="open_standalone_visualizer",
        ):
            launch_visualizer()


st.title(
    "✈️ Airline Scheduling Optimizer"
)

st.caption(
    "Graphical wrapper around the existing optimizer. "
    "All underlying CSV files and Python scripts remain directly usable from Terminal."
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Missions",
        "Pilot planning",
        "Fleet & crew",
        "Optimizer",
        "Results",
    ],
)

if page == "Missions":
    mission_editor()

elif page == "Pilot planning":
    pilot_planning_page()

elif page == "Fleet & crew":
    data_editor_page()

elif page == "Optimizer":
    optimizer_page()

elif page == "Results":
    results_page()
