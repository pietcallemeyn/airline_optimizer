
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
DEFAULT_MISSIONS_CSV = DATA / "missions_default.csv"
MISSION_SOURCE_INFO = DATA / "missions_source.json"
AIRPORTS = DATA / "airports.csv"
AIRCRAFT = DATA / "aircraft.csv"
PILOTS = DATA / "pilots.csv"
PILOT_AVAILABILITY = DATA / "pilot_availability.csv"

PID_FILE = OUTPUT / "gui_optimizer.pid"
LOG_FILE = OUTPUT / "gui_optimizer.log"


st.set_page_config(
    page_title="Airline Scheduling Optimizer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
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

    optional = ["fixed_aircraft"] if "fixed_aircraft" in df.columns else []
    df = df[required + optional].copy()
    if "fixed_aircraft" in df.columns:
        df["fixed_aircraft"] = (
            df["fixed_aircraft"].astype(str).replace({"nan": "", "None": ""})
            .str.strip().str.upper().str.replace("-", "", regex=False)
        )

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



def save_missions(df: pd.DataFrame, source_name: str | None = None) -> None:
    """Save the active mission set in the application's standard CSV format."""
    normalized = normalize_missions(df)
    save_csv(normalized, MISSIONS)

    # Force widgets that keep their own Streamlit state (especially the
    # mission table editor) to remount with the newly saved dataset. Without
    # this, activating another CSV updates data/missions.csv correctly but the
    # Table tab can continue displaying its previous widget state.
    st.session_state["mission_table_revision"] = int(
        st.session_state.get("mission_table_revision", 0)
    ) + 1

    if source_name is not None:
        MISSION_SOURCE_INFO.write_text(
            json.dumps({"source": source_name}, indent=2),
            encoding="utf-8",
        )


def mission_source_name() -> str:
    try:
        payload = json.loads(MISSION_SOURCE_INFO.read_text(encoding="utf-8"))
        return str(payload.get("source") or "Working mission set")
    except Exception:
        return "Standard mission set"


def load_flight_export_excel(raw: bytes) -> pd.DataFrame:
    """Convert the standard flight-export XLSX to the app mission schema."""
    from io import BytesIO

    # The flight export uses two header rows.
    raw_df = pd.read_excel(BytesIO(raw), header=[0, 1], engine="openpyxl")

    if raw_df.shape[1] < 43:
        raise ValueError(
            "This Excel file does not look like the expected flight export."
        )

    out = []
    for _, row in raw_df.iterrows():
        source_mission = row.iloc[1]
        aircraft = str(
            row.iloc[3] if pd.notna(row.iloc[3]) else ""
        ).strip().upper().replace("-", "")
        dep_date = row.iloc[8]
        dep_time = row.iloc[9]
        origin = str(
            row.iloc[11] if pd.notna(row.iloc[11]) else ""
        ).strip().upper()
        destination = str(
            row.iloc[13] if pd.notna(row.iloc[13]) else ""
        ).strip().upper()
        arr_time = row.iloc[16]
        pax_raw = row.iloc[42]

        if (
            not aircraft
            or not origin
            or not destination
            or pd.isna(dep_date)
            or pd.isna(dep_time)
        ):
            continue

        dep_day = pd.to_datetime(dep_date, dayfirst=True).date()
        dep_clock = pd.to_datetime(str(dep_time)).time()
        departure = pd.Timestamp.combine(dep_day, dep_clock)

        arrival = ""
        if pd.notna(arr_time) and str(arr_time).strip():
            arr_clock = pd.to_datetime(str(arr_time)).time()
            arr_dt = pd.Timestamp.combine(dep_day, arr_clock)
            if arr_dt <= departure:
                arr_dt += pd.Timedelta(days=1)
            arrival = arr_dt.isoformat()

        try:
            pax = (
                int(float(pax_raw))
                if pd.notna(pax_raw) and str(pax_raw).strip()
                else 0
            )
        except (TypeError, ValueError):
            pax = 0

        # Keep the historical mission-id convention used throughout the app.
        out.append(
            {
                "id": f"M{len(out) + 1:03d}",
                "origin": origin,
                "destination": destination,
                "departure": departure.isoformat(),
                "arrival": arrival,
                "pax": pax,
                "fixed_aircraft": aircraft,
                "source_mission_number": (
                    ""
                    if pd.isna(source_mission)
                    else str(source_mission)
                ),
            }
        )

    if not out:
        raise ValueError("No usable flight legs were found in the Excel export.")

    return normalize_missions(pd.DataFrame(out))


def load_missions_csv_bytes(raw: bytes) -> pd.DataFrame:
    from io import BytesIO
    df = pd.read_csv(BytesIO(raw), keep_default_na=False)
    required = {"id", "origin", "destination", "departure", "arrival", "pax"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))
    if df.empty:
        raise ValueError("Mission CSV contains no missions.")
    return normalize_missions(df)


def unknown_airport_codes(df: pd.DataFrame) -> list[str]:
    known = set(airport_codes())
    used = set(df["origin"].astype(str).str.upper()) | set(df["destination"].astype(str).str.upper())
    return sorted(code for code in used if code and code not in known)


def resolve_airports_from_ourairports(codes: list[str]) -> tuple[list[str], list[str]]:
    """Add missing ICAO airports from the public OurAirports dataset. Returns (added, unresolved)."""
    if not codes:
        return [], []
    url = "https://davidmegginson.github.io/ourairports-data/airports.csv"
    ref = pd.read_csv(url, low_memory=False)
    ref["gps_code"] = ref["gps_code"].fillna("").astype(str).str.upper().str.strip()
    wanted = ref[ref["gps_code"].isin(codes)].copy()
    wanted = wanted.dropna(subset=["latitude_deg", "longitude_deg"]).drop_duplicates("gps_code")
    current = airports_df()
    rows = pd.DataFrame({
        "icao": wanted["gps_code"],
        "name": wanted["name"].fillna(wanted["gps_code"]),
        "lat": wanted["latitude_deg"],
        "lon": wanted["longitude_deg"],
    })
    if not rows.empty:
        combined = pd.concat([current, rows], ignore_index=True)
        combined["icao"] = combined["icao"].astype(str).str.upper().str.strip()
        combined = combined.drop_duplicates("icao", keep="first").sort_values("icao")
        save_csv(combined, AIRPORTS)
    added = sorted(set(rows["icao"].astype(str))) if not rows.empty else []
    return added, sorted(set(codes) - set(added))


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

    # The optimizer is launched as a child of the Streamlit process.  A child
    # that has already exited can remain as a zombie until it is reaped;
    # os.kill(pid, 0) still succeeds for such a process and used to make the
    # GUI show "Running" after optimizer.py had printed DONE.
    try:
        waited_pid, _ = os.waitpid(pid, os.WNOHANG)
        if waited_pid == pid:
            return False
    except ChildProcessError:
        # Not our child (for example after a Streamlit restart); fall back to
        # the normal existence check below.
        pass
    except (AttributeError, OSError):
        # Keep the fallback portable on platforms without waitpid/WNOHANG.
        pass

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
    verbose: bool,
    debug: bool,
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

    if debug:
        command.append("--debug")
    elif verbose:
        command.append("--verbose")

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



def _pilot_display_map() -> dict[str, str]:
    """Map stable internal pilot IDs (C01/F01) to GUI trigrams."""
    path = PILOTS
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return {}
    if "id" not in df.columns:
        return {}
    trigram_col = "trigram" if "trigram" in df.columns else None
    result = {}
    for _, row in df.iterrows():
        pid = str(row.get("id", "")).strip()
        tri = str(row.get(trigram_col, "")).strip().upper() if trigram_col else ""
        if pid:
            result[pid] = tri or pid
    return result


def pilot_label(pilot_id) -> str:
    """GUI-only pilot label; optimizer/storage continue using the internal ID."""
    if pilot_id is None:
        return ""
    pid = str(pilot_id)
    return _pilot_display_map().get(pid, pid)


def _replace_pilot_ids_for_display(value):
    """Recursively replace known pilot IDs in GUI payloads only."""
    labels = _pilot_display_map()
    if isinstance(value, dict):
        return {
            k: _replace_pilot_ids_for_display(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_replace_pilot_ids_for_display(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_replace_pilot_ids_for_display(v) for v in value)
    if isinstance(value, str) and value in labels:
        return labels[value]
    return value


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

    save_missions(normalize_missions(missions))
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

        save_missions(normalized)
        st.session_state.pop("mission_calendar_editor", None)
        st.session_state["missions_changed_notice"] = True
        _reset_mission_calendar_component()
        st.rerun()

    if delete_clicked:
        if mode == "edit":
            missions = missions[missions["id"].astype(str) != mid].reset_index(drop=True)
            save_missions(normalize_missions(missions))
            st.session_state["missions_changed_notice"] = True
        st.session_state.pop("mission_calendar_editor", None)
        _reset_mission_calendar_component()
        st.rerun()


def _latest_mission_crew_lookup() -> dict[str, dict]:
    """Crew per mission from the newest available optimizer result.

    The mission page itself is demand data, so crew only exists after an optimizer
    run.  Keep this deliberately best-effort: stale/missing result files simply
    produce an empty lookup and the hover card says that crew is not assigned yet.
    """
    candidates = [OUTPUT / "balanced.json", OUTPUT / "cheapest.json"]
    candidates.extend(sorted(OUTPUT.glob("pareto_*.json"), key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True))
    for path in candidates:
        if not path.exists():
            continue
        try:
            solution = json.loads(path.read_text(encoding="utf-8"))
            lookup = {}
            for reg, movements in (solution.get("aircraft_movements", {}) or {}).items():
                for movement in movements or []:
                    if str(movement.get("type", "")).upper() != "MISSION":
                        continue
                    mid = str(movement.get("mission", "")).strip()
                    if not mid:
                        continue
                    lookup[mid] = {
                        "captain": str(movement.get("captain", "") or "").strip(),
                        "fo": str(movement.get("fo", "") or "").strip(),
                        "aircraft": str(reg),
                    }
            if lookup:
                return lookup
        except Exception:
            continue
    return {}


def render_mission_demand_timeline(missions: pd.DataFrame, solution: dict | None = None) -> None:
    """Aircraft-style UTC planning board for mission demand.

    Missions with a pre-assigned aircraft are shown on that aircraft row, exactly
    like the optimizer aircraft board. Missions without an assignment are shown
    in an UNASSIGNED row. This is deliberately not an artificial Lane 1/Lane 2
    packing view: the left-hand axis represents operational resources/status.
    """
    if solution is not None:
        crew_lookup = solution.get("crew_assignments", {}) or {}
    else:
        crew_lookup = _latest_mission_crew_lookup()
    items = []
    for _, row in missions.iterrows():
        dep = _parse_iso_datetime(row.get("departure", ""))
        if dep is None:
            continue
        arr = estimated_mission_arrival(row)
        if arr is None or arr <= dep:
            arr = dep + pd.Timedelta(minutes=60)
        fixed = str(row.get("fixed_aircraft", "")).strip()
        crew = crew_lookup.get(str(row.get("id", "")), {})
        items.append({
            "id": str(row.get("id", "")),
            "origin": str(row.get("origin", "")),
            "destination": str(row.get("destination", "")),
            "pax": int(row.get("pax", 0) or 0),
            "fixed_aircraft": fixed,
            "row": fixed if fixed else "UNASSIGNED",
            "start": dep.isoformat(),
            "end": arr.isoformat(),
            "arrival_source": "manual" if str(row.get("arrival", "")).strip() else "estimated",
            "captain": pilot_label(crew.get("captain", "")),
            "fo": pilot_label(crew.get("fo", "")),
            "crew_aircraft": crew.get("aircraft", ""),
        })
    if not items:
        st.info("No missions available for the timeline.")
        return

    items.sort(key=lambda x: (x["start"], x["end"], x["id"]))
    first = min(datetime.fromisoformat(x["start"]) for x in items).date()
    last = max(datetime.fromisoformat(x["end"]) for x in items).date()
    dates = [d.date().isoformat() for d in pd.date_range(first, last, freq="D")]

    # Preserve known aircraft order where possible, then show UNASSIGNED last.
    known = []
    try:
        ac = load_csv(AIRCRAFT)
        for col in ("registration", "aircraft", "tail", "id"):
            if col in ac.columns:
                known = [str(x).strip() for x in ac[col].tolist() if str(x).strip()]
                break
    except Exception:
        known = []
    assigned = list(dict.fromkeys(x["row"] for x in items if x["row"] != "UNASSIGNED"))
    row_order = [x for x in known if x in assigned] + [x for x in assigned if x not in known]
    if any(x["row"] == "UNASSIGNED" for x in items):
        row_order.append("UNASSIGNED")

    payload = {"missions": items, "dates": dates, "rows": row_order}
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    component_html = f"""
<div id="mission-demand-root">
<style>
*{{box-sizing:border-box}} :root{{--label-w:175px;--day-w:185px;--row-h:100px;--head-h:64px}}
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#202124;background:transparent;overflow:hidden}}
.toolbar{{display:flex;justify-content:space-between;gap:10px;align-items:center;margin:0;border-top:5px solid #7d7d7d;padding:6px 8px;background:#fff;flex-wrap:wrap}}
.toolbar-left{{display:flex;gap:10px;align-items:baseline}} .title{{font-size:18px;font-weight:750}} .hint{{color:#777;font-size:11px}}
.controls{{display:flex;gap:4px;align-items:center}} button.ctrl{{border:1px solid #c7c7c7;background:#f7f7f7;padding:5px 9px;border-radius:2px;cursor:pointer;color:#444}}
.layout{{display:block}}
.viewport{{height:650px;overflow:auto;border:1px solid #d2d2d2;border-radius:0;background:#fff;scrollbar-gutter:stable}}
.board{{position:relative;width:calc(var(--label-w) + var(--days) * var(--day-w));min-width:max-content}}
.header{{position:sticky;top:0;z-index:30;height:var(--head-h);margin-left:var(--label-w);background:#f3f3f3;border-bottom:1px solid #d2d2d2}}
.day-head{{position:absolute;top:0;height:var(--head-h);display:flex;align-items:center;justify-content:center;border-left:1px solid #dedede;color:#333;font-size:18px;white-space:nowrap}}
.corner{{position:sticky;left:0;top:0;z-index:50;width:var(--label-w);height:var(--head-h);margin-top:calc(-1 * var(--head-h));padding:20px 11px;background:#fff;border-right:1px solid #ccc;border-bottom:1px solid #d2d2d2;font-size:14px;font-weight:400}}
.row{{position:relative;height:var(--row-h);margin-left:var(--label-w);border-bottom:1px solid #cfcfcf;background:#fff}}
.label-cell{{position:sticky;left:0;z-index:20;width:var(--label-w);height:var(--row-h);margin-left:calc(-1 * var(--label-w));padding:25px 11px 8px;background:#fff;border-right:1px solid #ccc}}
.label-cell b{{display:block;font-size:20px;font-weight:400;color:#2d6d9d}} .label-cell span{{display:block;margin-top:8px;font-size:13px;color:#666}}
.gridline{{position:absolute;top:0;bottom:0;border-left:1px solid #d4d4d4;pointer-events:none}} .noon{{position:absolute;top:0;bottom:0;border-left:1px solid #eeeeee;pointer-events:none}}
.block{{position:absolute;min-width:48px;height:72px;top:14px;border:1px solid #555;border-radius:0;padding:3px 4px;overflow:hidden;cursor:pointer;text-align:left;color:#fff;background:#7da3ef;box-shadow:none}}
.block:hover{{z-index:50!important;box-shadow:0 3px 9px rgba(0,0,0,.18)}} .block.selected{{outline:2px solid #2878ff;outline-offset:1px;z-index:51!important}}
.block.unassigned{{background:#f2cb5c;border-color:#555;color:#111}}
.location-band{{position:absolute;height:28px;top:0;border-radius:0;background:#8499bd;color:#fff;font-size:14px;line-height:28px;padding:0 6px;text-align:center;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;pointer-events:none;z-index:2}}
.location-band.conflict{{background:#a8bdea;color:#fff;border:0;line-height:28px}}
.destination{{font-size:10px;font-weight:700;line-height:1.15;white-space:nowrap;overflow:visible;text-overflow:clip}} .route{{display:none}} .meta{{margin-top:3px;font-size:9px;font-weight:600;line-height:1.15;color:inherit;white-space:nowrap;overflow:visible;text-overflow:clip}}
.tooltip{{position:fixed;display:none;z-index:9999;max-width:380px;background:#111;color:#fff;border-radius:7px;padding:10px 12px;font-size:11px;line-height:1.5;box-shadow:0 5px 18px rgba(0,0,0,.30);pointer-events:none}} .tooltip b{{font-size:12px}} .tooltip .crew{{color:#52bfff;font-weight:650}}
.detail{{display:none}} .detail h3{{margin:0 0 12px;font-size:18px}}
.detail-block{{border-bottom:1px solid #eee;padding:9px 0}} .detail-block:last-child{{border:0}} .dlabel{{font-size:10px;color:#777;text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px}} .dvalue{{font-size:13px;font-weight:600;overflow-wrap:anywhere}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;color:#666;font-size:11px}} .swatch{{display:inline-block;width:12px;height:12px;border:1px solid #aaa;border-radius:3px;vertical-align:-2px;margin-right:4px}} .fixed-s{{background:#e7f3e8}} .unassigned-s{{background:#eef3fb}}
@media(max-width:950px){{.layout{{grid-template-columns:1fr}}.detail{{position:static;max-height:none;order:-1}}.viewport{{height:560px}}}}
</style>
<div class="toolbar"><div class="toolbar-left"><div class="title">Mission overview</div><div class="hint">UTC · dispatch planning board</div></div><div class="controls"><button class="ctrl" id="startBtn">Start</button><button class="ctrl" id="minus">−</button><span id="zoomTxt">100%</span><button class="ctrl" id="plus">+</button><button class="ctrl" id="fit">Fit</button></div></div>
<div id="missionTooltip" class="tooltip"></div><div class="layout"><div><div class="viewport" id="viewport"><div class="board" id="board"></div></div><div class="legend"><span><i class="swatch fixed-s"></i>Pre-assigned mission</span><span><i class="swatch unassigned-s"></i>Unassigned mission</span><span>🔒 fixed aircraft</span><span><i class="swatch" style="background:#dbe6f7;border-color:#b8c9df"></i>Known aircraft location</span><span><i class="swatch" style="background:#fff0d9;border-color:#d4a14b"></i>Location discontinuity</span></div></div>
<aside class="detail"><h3 id="dTitle">Select a mission</h3><div class="detail-block"><div class="dlabel">Route</div><div class="dvalue" id="dRoute">Click a mission block.</div></div><div class="detail-block"><div class="dlabel">Timing</div><div class="dvalue" id="dTime">—</div></div><div class="detail-block"><div class="dlabel">Passengers</div><div class="dvalue" id="dPax">—</div></div><div class="detail-block"><div class="dlabel">Aircraft</div><div class="dvalue" id="dFixed">—</div></div><div class="detail-block"><div class="dlabel">Arrival</div><div class="dvalue" id="dArrival">—</div></div></aside></div>
<script>(()=>{{
 const data={data_json},board=document.getElementById('board'),viewport=document.getElementById('viewport'),dayMs=86400000,baseDayW=190;let zoom=1;
 const first=new Date(data.dates[0]+'T00:00:00Z'),lastEnd=new Date(data.dates[data.dates.length-1]+'T00:00:00Z').getTime()+dayMs;
 const pct=(iso)=>((new Date(iso+'Z').getTime()-first.getTime())/(lastEnd-first.getTime()))*100; const widthPct=(a,b)=>Math.max(.12,((new Date(b+'Z')-new Date(a+'Z'))/(lastEnd-first.getTime()))*100);
 const pretty=(d)=>new Date(d+'T00:00:00Z').toLocaleDateString('en-GB',{{weekday:'short',day:'2-digit',month:'short',timeZone:'UTC'}}); const time=(iso)=>new Date(iso+'Z').toLocaleString('en-GB',{{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'UTC'}});
 const tip=document.getElementById('missionTooltip'); const esc=(v)=>String(v??'').replace(/[&<>\"']/g,m=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[m])); function showTip(c,e){{const crew=(c.captain||c.fo)?`<span class=\"crew\">CAPT ${{esc(c.captain||'?')}} &nbsp;·&nbsp; FO ${{esc(c.fo||'?')}}</span>`:'Crew not assigned in latest optimizer result';tip.innerHTML=`<b>${{esc(c.id)}} · ${{esc(c.origin)}} → ${{esc(c.destination)}}</b><br>${{time(c.start)}} → ${{time(c.end)}} UTC<br>${{esc(c.pax)}} pax · aircraft ${{esc(c.fixed_aircraft||c.crew_aircraft||'unassigned')}}<br>${{crew}}`;tip.style.display='block';moveTip(e)}} function moveTip(e){{const pad=14,w=tip.offsetWidth,h=tip.offsetHeight;tip.style.left=Math.min(window.innerWidth-w-pad,e.clientX+14)+'px';tip.style.top=Math.min(window.innerHeight-h-pad,e.clientY+14)+'px'}} function hideTip(){{tip.style.display='none'}}
 function detail(c,el){{document.querySelectorAll('.block.selected').forEach(x=>x.classList.remove('selected'));el.classList.add('selected');document.getElementById('dTitle').textContent=(c.fixed_aircraft?'🔒 ':'')+c.id;document.getElementById('dRoute').textContent=c.origin+' → '+c.destination;document.getElementById('dTime').textContent=time(c.start)+' → '+time(c.end)+' UTC';document.getElementById('dPax').textContent=c.pax+' pax';document.getElementById('dFixed').textContent=c.fixed_aircraft||'UNASSIGNED — optimizer may choose aircraft';document.getElementById('dArrival').textContent=c.arrival_source==='manual'?'Manual arrival time':'Estimated from route distance';}}
 function render(){{board.innerHTML='';board.style.setProperty('--days',data.dates.length);const head=document.createElement('div');head.className='header';data.dates.forEach((d,i)=>{{const h=document.createElement('div');h.className='day-head';h.style.left=`calc(${{i}} * var(--day-w))`;h.style.width='var(--day-w)';h.textContent=pretty(d);head.appendChild(h)}});board.appendChild(head);const corner=document.createElement('div');corner.className='corner';corner.textContent='Aircraft';board.appendChild(corner);
 data.rows.forEach(reg=>{{const line=document.createElement('div');line.className='row';const lab=document.createElement('div');lab.className='label-cell';const count=data.missions.filter(c=>c.row===reg).length;lab.innerHTML=`<b>${{reg}}</b><span>${{count}} mission${{count===1?'':'s'}}</span>`;line.appendChild(lab);data.dates.forEach((d,i)=>{{const g=document.createElement('div');g.className='gridline';g.style.left=`calc(${{i}} * var(--day-w))`;line.appendChild(g);const n=document.createElement('div');n.className='noon';n.style.left=`calc(${{i}} * var(--day-w) + var(--day-w)/2)`;line.appendChild(n)}});
 const rowMissions=data.missions.filter(c=>c.row===reg).sort((a,b)=>new Date(a.start+'Z')-new Date(b.start+'Z'));
 rowMissions.forEach((c,j)=>{{
   const el=document.createElement('button');el.type='button';el.className='block'+(reg==='UNASSIGNED'?' unassigned':'');el.style.left=pct(c.start)+'%';el.style.width=`max(48px, ${{widthPct(c.start,c.end)}}%)`;el.style.zIndex=5+(j%10);el.innerHTML=`<div class="destination">${{c.destination}}</div><div class="meta">${{c.id}}</div>`;el.onclick=()=>detail(c,el);el.onmouseenter=(e)=>showTip(c,e);el.onmousemove=moveTip;el.onmouseleave=hideTip;line.appendChild(el);
   if(reg!=='UNASSIGNED'){{
     const next=rowMissions[j+1];
     const bandStart=c.end;
     const bandEnd=next?next.start:null;
     if(bandEnd && new Date(bandEnd+'Z')>new Date(bandStart+'Z')){{
       const band=document.createElement('div');
       const continuous=(c.destination===next.origin);
       band.className='location-band'+(continuous?'':' conflict');
       band.style.left=pct(bandStart)+'%';
       band.style.width=`max(32px, ${{widthPct(bandStart,bandEnd)}}%)`;
       band.textContent=continuous?c.destination:`${{c.destination}} → ${{next.origin}} ?`;
       band.title=continuous?`Aircraft remains at ${{c.destination}} until next mission`:`Aircraft ends at ${{c.destination}}, but next fixed mission starts at ${{next.origin}}. Positioning is not yet planned.`;
       line.appendChild(band);
     }}
   }}
 }});board.appendChild(line)}})}}
 function setZoom(z,anchorX=null){{
   const oldZoom=zoom;
   const labelWidth=parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--label-w'))||0;
   const oldTimelineWidth=Math.max(1,board.scrollWidth-labelWidth);
   const anchorTimelineX=anchorX===null?null:Math.max(0,viewport.scrollLeft+anchorX-labelWidth);
   const anchorRatio=anchorTimelineX===null?null:anchorTimelineX/oldTimelineWidth;
   zoom=Math.max(.08,Math.min(3,z));
   document.documentElement.style.setProperty('--day-w',`${{baseDayW*zoom}}px`);
   document.getElementById('zoomTxt').textContent=Math.round(zoom*100)+'%';
   if(anchorRatio!==null && zoom!==oldZoom){{
     const newTimelineWidth=Math.max(1,board.scrollWidth-labelWidth);
     viewport.scrollLeft=Math.max(0,labelWidth+anchorRatio*newTimelineWidth-anchorX);
   }}
 }}
 document.getElementById('minus').onclick=()=>setZoom(zoom/1.2);document.getElementById('plus').onclick=()=>setZoom(zoom*1.2);document.getElementById('startBtn').onclick=()=>viewport.scrollTo({{left:0,behavior:'smooth'}});document.getElementById('fit').onclick=()=>{{const labelWidth=parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--label-w'))||0;const avail=Math.max(1,viewport.clientWidth-labelWidth);setZoom(avail/(data.dates.length*baseDayW));viewport.scrollLeft=0}};viewport.addEventListener('wheel',e=>{{if(e.ctrlKey||e.metaKey){{e.preventDefault();const rect=viewport.getBoundingClientRect();setZoom(zoom*(e.deltaY<0?1.1:.9),e.clientX-rect.left);}}else if(e.shiftKey){{e.preventDefault();viewport.scrollLeft+=e.deltaY+e.deltaX}}}},{{passive:false}});render();setZoom(1);
}})();</script>
"""
    components.html(component_html, height=760, scrolling=False)

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

    timeline_tab, xls_tab, table_tab = st.tabs(
        ["Timeline", "Load XLS", "Table"]
    )

    with timeline_tab:
        mission_solution = None
        results_stale = optimizer_results_are_stale()
        try:
            mission_gallery = [] if results_stale else viz.load_gallery()
        except Exception:
            mission_gallery = []

        if mission_gallery:
            # Mission Planning uses the actual saved solution number directly.
            if "mission_timeline_solution_number" not in st.session_state:
                st.session_state.mission_timeline_solution_number = 1
            if int(st.session_state.mission_timeline_solution_number) > len(mission_gallery):
                st.session_state.mission_timeline_solution_number = 1

            selected_number = st.number_input(
                "Solution number",
                min_value=1,
                max_value=len(mission_gallery),
                step=1,
                key="mission_timeline_solution_number",
            )
            selected_idx = int(selected_number) - 1
            mission_solution = mission_gallery[selected_idx]

            # Synchronize the Results board with the same saved solution.
            st.session_state.timeline_gallery_index = selected_idx
            st.session_state.timeline_gallery_jump = selected_number
            st.session_state.timeline_browsing_gallery = True
            st.session_state.timeline_mode = "ALTERNATIVE"

            split_ids = []
            for movements in (mission_solution.get("aircraft_movements", {}) or {}).values():
                for movement in movements or []:
                    if movement.get("split_duty"):
                        mid = str(movement.get("mission_id", movement.get("id", "")) or "")
                        if mid and mid not in split_ids:
                            split_ids.append(mid)
            if split_ids:
                st.warning(
                    "⚠️ Split duty is used in this solution: " + ", ".join(split_ids) +
                    ". See the solution planning board for the detailed FDP/split-duty information."
                )
        else:
            if results_stale:
                st.info(
                    "This mission set has changed and has no current solutions yet. "
                    "Run the optimizer first to calculate solutions for this set."
                )
            else:
                st.info(
                    "No optimizer solutions are available yet. "
                    "Run the optimizer first to calculate solutions."
                )

        render_mission_demand_timeline(missions, mission_solution)

    with xls_tab:
        st.markdown("#### Load mission XLS")
        st.caption(
            "Load a flight-export Excel file. It is converted automatically to the mission format and validated first. "
            "If it contains airports that are not yet in data/airports.csv, the app can fetch their "
            "name and coordinates from the public OurAirports reference dataset and add them automatically."
        )
        st.info(f"Current mission source: **{mission_source_name()}** · {len(missions)} missions")

        uploaded = st.file_uploader(
            "Flight-export Excel file",
            type=["xlsx", "xls"],
            key="mission_xls_upload",
            help="The standard flight export is automatically converted to the internal mission format.",
        )
        preview = None
        preview_errors = []
        missing_airports = []
        if uploaded is not None:
            try:
                preview = load_flight_export_excel(uploaded.getvalue())
                missing_airports = unknown_airport_codes(preview)
                if missing_airports:
                    st.warning(
                        f"This mission file uses {len(missing_airports)} airport(s) not yet in the local "
                        f"airport database: {', '.join(missing_airports)}"
                    )
                    if st.button("Add missing airports automatically", key="resolve_missing_airports"):
                        try:
                            added, unresolved = resolve_airports_from_ourairports(missing_airports)
                            if added:
                                st.success(f"Added {len(added)} airport(s) to data/airports.csv.")
                            if unresolved:
                                st.error("Could not resolve: " + ", ".join(unresolved))
                            else:
                                st.rerun()
                        except Exception as exc:
                            st.error(
                                "Automatic airport lookup failed. Check the server's internet connection. "
                                f"No mission data was changed. Details: {exc}"
                            )
                preview_errors = validate_missions(preview)
                if preview_errors:
                    non_airport_errors = [e for e in preview_errors if "unknown origin" not in e and "unknown destination" not in e]
                    if non_airport_errors:
                        st.error("This file cannot be activated yet:\n" + "\n".join(non_airport_errors[:20]))
                if not preview_errors:
                    st.success(f"Valid mission file: {len(preview)} missions.")
                st.dataframe(preview.head(12), hide_index=True, width="stretch")


            except Exception as exc:
                st.error(f"Could not read mission Excel file: {exc}")

        c_load, c_default = st.columns(2)
        if c_load.button(
            "Use uploaded XLS",
            type="primary",
            disabled=(preview is None or bool(preview_errors)),
            key="activate_mission_xls",
        ):
            save_missions(preview, source_name=uploaded.name)
            st.session_state["missions_changed_notice"] = True
            st.session_state.pop("mission_calendar_editor", None)
            _reset_mission_calendar_component()
            st.rerun()

        if c_default.button("Restore standard missions", key="restore_default_missions"):
            try:
                default_df = normalize_missions(load_csv(DEFAULT_MISSIONS_CSV))
                errors = validate_missions(default_df)
                if errors:
                    st.error("Default mission CSV is invalid:\n" + "\n".join(errors[:20]))
                else:
                    save_missions(default_df, source_name=DEFAULT_MISSIONS_CSV.name)
                    st.session_state["missions_changed_notice"] = True
                    st.session_state.pop("mission_calendar_editor", None)
                    _reset_mission_calendar_component()
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not restore the standard mission CSV: {exc}")

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
            key=f"mission_table_{int(st.session_state.get('mission_table_revision', 0))}",
        )
        c1, c2 = st.columns([1, 4])
        if c1.button("Save missions", type="primary", key="save_mission_table"):
            normalized = normalize_missions(edited)
            errors = validate_missions(normalized)
            if errors:
                st.error("\n".join(errors[:15]))
            else:
                save_missions(normalized)
                st.session_state["missions_changed_notice"] = True
                st.success("missions.csv saved.")
                st.rerun()
        c2.caption(f"{len(edited)} missions currently in the table.")


def _ensure_pilot_planning_files() -> None:
    # Pilot planning only needs the availability file.
    if not PILOT_AVAILABILITY.exists():
        save_csv(pd.DataFrame(columns=["pilot_id", "date", "status", "note"]), PILOT_AVAILABILITY)


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
        note = str(row.get("note", "")).strip()
        exceptions[day] = (status, note)

    start_day, end_day = _pilot_calendar_date_range()
    events: list[dict] = []
    for stamp in pd.date_range(start_day, end_day, freq="D"):
        day = stamp.date().isoformat()
        status, note = exceptions.get(day, ("AVAILABLE", ""))
        palette = PILOT_AVAILABILITY_COLORS.get(
            status, PILOT_AVAILABILITY_COLORS["UNAVAILABLE"]
        )
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


def _gallery_index_for_solution(gallery: list[dict], solution: dict | None) -> int:
    """Return the exact 0-based gallery index for a selected solution."""
    if not gallery or solution is None:
        return 0

    # _four_solution_choices() returns an object taken directly from gallery.
    # Prefer identity/equality over schedule_signature because older optimizer
    # outputs can have missing or duplicate signatures.
    for i, candidate in enumerate(gallery):
        if candidate is solution:
            return i
    for i, candidate in enumerate(gallery):
        if candidate == solution:
            return i

    sig = solution.get("schedule_signature")
    if sig is not None:
        for i, candidate in enumerate(gallery):
            if candidate.get("schedule_signature") == sig:
                return i
    return 0


def _four_solution_choices(gallery: list[dict]) -> dict[str, dict]:
    """The four operational solution shortcuts used in Missions and Results."""
    if not gallery:
        return {}

    def obj(sol, key, default=float("inf")):
        try:
            return float(sol.get("objectives", {}).get(key, default))
        except (TypeError, ValueError):
            return default

    def metric(sol, key, default=float("inf")):
        try:
            return float(sol.get("metrics", {}).get(key, default))
        except (TypeError, ValueError):
            return default

    return {
        "CHEAPEST": min(
            gallery,
            key=lambda s: (
                obj(s, "total_operational_cost_eur"),
                obj(s, "complexity_score"),
            ),
        ),
        "MIN COMPLEXITY": min(
            gallery,
            key=lambda s: (
                obj(s, "complexity_score"),
                obj(s, "total_operational_cost_eur"),
            ),
        ),
        "MIN OUTSTATION SWAPS": min(
            gallery,
            key=lambda s: (
                len(actual_outstation_swap_events(s)),
                actual_outstation_changed_pilots(s),
                obj(s, "complexity_score"),
                metric(s, "charged_pilot_days"),
            ),
        ),
        "MIN PILOT DAYS": min(
            gallery,
            key=lambda s: (
                metric(s, "charged_pilot_days"),
                obj(s, "complexity_score"),
            ),
        ),
    }


def _pilot_solution_choices() -> dict[str, dict | None]:
    """Return selectable optimizer solutions for the pilot calendar overlay.

    Named Pareto views come first. Additional gallery schedules are included
    only when their schedule signature is not already represented.
    """
    choices: dict[str, dict | None] = {"Availability only": None}
    try:
        pareto = viz.load_pareto()
    except Exception:
        return choices

    try:
        named = viz.named_solutions(pareto)
    except Exception:
        named = {}

    seen = set()
    preferred = [
        "CHEAPEST",
        "BALANCED",
        "MIN EMPTY LEGS",
        "MIN PILOT DAYS",
        "MIN AIRCRAFT PARKING",
        "MIN COMPLEXITY",
    ]
    for label in preferred:
        solution = named.get(label)
        if not solution:
            continue
        signature = str(solution.get("schedule_signature", ""))
        seen.add(signature or f"named:{label}")
        cost = float(solution.get("objectives", {}).get("total_operational_cost_eur", 0) or 0)
        choices[f"{label} · €{cost:,.0f}"] = solution

    try:
        gallery = viz.load_gallery()
    except Exception:
        gallery = []
    alt_number = 0
    for solution in gallery:
        signature = str(solution.get("schedule_signature", ""))
        marker = signature or repr(solution.get("crew_assignments", {}))
        if marker in seen:
            continue
        seen.add(marker)
        alt_number += 1
        cost = float(solution.get("objectives", {}).get("total_operational_cost_eur", 0) or 0)
        empty = int(solution.get("metrics", {}).get("empty_legs", solution.get("objectives", {}).get("empty_legs", 0)) or 0)
        choices[f"Alternative {alt_number:02d} · €{cost:,.0f} · {empty} empty"] = solution
    return choices


def _solution_pilot_days(solution: dict, pilot_id: str) -> set[str]:
    """Exact charged/active pilot days, with a legacy movement fallback."""
    explicit = solution.get("pilot_days_by_pilot", {}) or {}
    days = {str(day)[:10] for day in explicit.get(str(pilot_id), []) if str(day).strip()}
    if days:
        return days

    # Older result files pre-date pilot_days_by_pilot. Their movement days can
    # still be visualised, although pure away/parking days cannot be recovered.
    for movements in (solution.get("aircraft_movements", {}) or {}).values():
        for movement in movements:
            if str(pilot_id) not in {str(movement.get("captain", "")), str(movement.get("fo", ""))}:
                continue
            try:
                start = pd.to_datetime(movement.get("start"))
                end = pd.to_datetime(movement.get("end"))
                for stamp in pd.date_range(start.normalize(), end.normalize(), freq="D"):
                    days.add(stamp.date().isoformat())
            except Exception:
                continue
    return days


def _pilot_solution_events(solution: dict | None, pilot_id: str, availability: pd.DataFrame) -> list[dict]:
    """Overlay a selected optimizer solution on one pilot's availability calendar.

    Purple pills are charged/planned pilot days. Movement days include aircraft
    and route information; charged days without a movement are shown as AWAY /
    ATTACHED because the pilot remains chargeable in the optimizer cost model.
    """
    if not solution:
        return []

    planned_days = _solution_pilot_days(solution, pilot_id)
    movements_by_day: dict[str, list[str]] = {}
    for reg, movements in (solution.get("aircraft_movements", {}) or {}).items():
        for movement in movements:
            if str(pilot_id) not in {str(movement.get("captain", "")), str(movement.get("fo", ""))}:
                continue
            try:
                start = pd.to_datetime(movement.get("start"))
                end = pd.to_datetime(movement.get("end"))
            except Exception:
                continue
            route = f"{movement.get('from', '?')}→{movement.get('to', '?')}"
            mission = str(movement.get("mission") or movement.get("target_mission") or "").strip()
            kind = "MISSION" if str(movement.get("type", "")).upper() == "MISSION" else "EMPTY"
            detail = f"{reg} {kind} {route}" + (f" {mission}" if mission else "")
            for stamp in pd.date_range(start.normalize(), end.normalize(), freq="D"):
                movements_by_day.setdefault(stamp.date().isoformat(), []).append(detail)

    exceptions = {}
    if not availability.empty:
        work = availability[availability.get("pilot_id", pd.Series(dtype=str)).astype(str) == str(pilot_id)]
        for _, row in work.iterrows():
            exceptions[str(row.get("date", ""))[:10]] = str(row.get("status", "AVAILABLE")).upper()

    events = []
    for day in sorted(planned_days):
        details = movements_by_day.get(day, [])
        conflict = exceptions.get(day) in {"UNAVAILABLE", "LEAVE", "TRAINING"}
        if details:
            regs = sorted({item.split()[0] for item in details if item})
            title = f"{'⚠ ' if conflict else ''}PLANNED · {'/'.join(regs)}"
            if len(details) == 1:
                title += f" · {details[0].split(' ', 1)[1]}"
            elif len(details) > 1:
                title += f" · {len(details)} movements"
        else:
            title = f"{'⚠ ' if conflict else ''}PLANNED · AWAY / ATTACHED"

        color = "#B91C1C" if conflict else "#6D28D9"
        events.append({
            "id": f"solution:{pilot_id}:{day}",
            "title": title,
            "start": day,
            "allDay": True,
            "editable": False,
            "backgroundColor": color,
            "borderColor": color,
            "textColor": "#FFFFFF",
            "extendedProps": {
                "pilot_id": pilot_id,
                "solution_overlay": True,
                "movement_details": details,
                "availability_conflict": conflict,
            },
        })
    return events



def _pilot_solution_day_details(solution: dict | None, pilot_id: str, day: date) -> dict:
    """Return display-ready details for one pilot/day in a selected solution.

    A planned/charged day can exist without a flight movement (for example an
    away/attached day). Movement details are returned separately so the GUI can
    explain exactly why the day is charged.
    """
    result = {
        "planned": False,
        "movements": [],
        "away_or_attached": False,
    }
    if not solution:
        return result

    day_text = day.isoformat()
    planned_days = _solution_pilot_days(solution, pilot_id)
    if day_text not in planned_days:
        return result

    result["planned"] = True
    day_start = pd.Timestamp(day_text)
    day_end = day_start + pd.Timedelta(days=1)

    movements = []
    for reg, aircraft_movements in (solution.get("aircraft_movements", {}) or {}).items():
        for movement in aircraft_movements:
            captain = str(movement.get("captain", ""))
            fo = str(movement.get("fo", ""))
            if str(pilot_id) not in {captain, fo}:
                continue
            try:
                start = pd.to_datetime(movement.get("start"))
                end = pd.to_datetime(movement.get("end"))
            except Exception:
                continue
            # Include any movement touching the selected UTC planning day.
            if not (start < day_end and end >= day_start):
                continue

            role = "CAPT" if captain == str(pilot_id) else "FO"
            movement_type = str(movement.get("type", "")).upper() or "MOVEMENT"
            purpose = str(movement.get("purpose", "")).strip().upper()
            mission = str(movement.get("mission") or movement.get("target_mission") or "").strip()
            movements.append({
                "aircraft": str(reg),
                "role": role,
                "type": movement_type,
                "purpose": purpose,
                "mission": mission,
                "from": str(movement.get("from", "?")),
                "to": str(movement.get("to", "?")),
                "start": start,
                "end": end,
            })

    movements.sort(key=lambda item: item["start"])
    result["movements"] = movements
    result["away_or_attached"] = not bool(movements)
    return result

def parse_pilot_availability_excel(raw: bytes, pilots_df: pd.DataFrame):
    """Parse crew agenda XLSX to pilot_availability.csv format."""
    from io import BytesIO
    agenda = pd.read_excel(BytesIO(raw), header=1, engine="openpyxl")
    agenda.columns = [str(c).strip() for c in agenda.columns]
    required = ["Resource", "Category", "Start (z)", "End (z)", "Remarks"]
    missing = [c for c in required if c not in agenda.columns]
    if missing:
        raise ValueError("Missing Excel columns: " + ", ".join(missing))
    if "id" not in pilots_df.columns or "trigram" not in pilots_df.columns:
        raise ValueError("pilots.csv must contain 'id' and 'trigram' columns.")
    trigram_to_id = {
        str(r["trigram"]).strip().upper(): str(r["id"]).strip()
        for _, r in pilots_df.fillna("").iterrows()
        if str(r.get("trigram", "")).strip() and str(r.get("id", "")).strip()
    }
    unknown, rows = set(), []
    for _, r in agenda.fillna("").iterrows():
        resource = str(r.get("Resource", "")).strip()
        if not resource:
            continue
        trigram = resource[:3].upper()
        pilot_id = trigram_to_id.get(trigram)
        if not pilot_id:
            unknown.add(trigram)
            continue
        start = pd.to_datetime(r.get("Start (z)"), dayfirst=True, errors="coerce")
        end = pd.to_datetime(r.get("End (z)"), dayfirst=True, errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        if end < start:
            start, end = end, start
        category = str(r.get("Category", "")).strip()
        category_upper = category.upper()

        # CAT is a flight activity, not an availability restriction.
        # It must therefore create no blocked pilot-days.
        if category_upper == "CAT":
            continue

        # Every other agenda entry is, for now, a full-day hard block.
        # Preserve the original category + remarks in note for GUI context,
        # while keeping optimizer-facing status deliberately simple.
        remarks = str(r.get("Remarks", "")).strip()
        note = category
        if remarks:
            note = f"{category} - {remarks}" if category else remarks

        for day in pd.date_range(start.normalize(), end.normalize(), freq="D"):
            rows.append({
                "pilot_id": pilot_id,
                "date": day.date().isoformat(),
                "status": "UNAVAILABLE",
                "note": note,
            })
    out = pd.DataFrame(rows, columns=["pilot_id", "date", "status", "note"])
    if not out.empty:
        out = out.drop_duplicates(["pilot_id", "date"], keep="last").sort_values(["pilot_id", "date"]).reset_index(drop=True)
    return out, sorted(unknown)


def pilot_planning_page():
    _ensure_pilot_planning_files()
    st.subheader("Pilot planning")
    st.caption(
        "Pilot availability is used as a hard planning constraint. "
        "Unavailable / Leave / Training block the full calendar day."
    )

    pilots = load_csv(PILOTS)
    if pilots.empty:
        st.warning("No pilots found in data/pilots.csv.")
        return

    display = _pilot_display_lookup()
    pilot_ids = [pid for pid in pilots["id"].astype(str).tolist() if pid]
    timeline_tab, load_tab, table_tab, detailed_tab = st.tabs([
        "Timeline",
        "Load XLS",
        "Table",
        "Detailed view per pilot",
    ])

    with timeline_tab:
        availability = load_csv(PILOT_AVAILABILITY)
        results_stale = optimizer_results_are_stale()
        try:
            pilot_gallery = [] if results_stale else viz.load_gallery()
        except Exception:
            pilot_gallery = []

        if pilot_gallery:
            # Use the same direct saved-solution selector as Mission Planning.
            # Both pages share the selected gallery index so moving between them
            # keeps the same solution active.
            # Initialise the pilot selector only once.  Do NOT copy the Mission
            # Planning value into the widget key on every Streamlit rerun: doing
            # that overwrites the value the user has just selected and makes the
            # control appear to jump back (usually to solution 1).
            if "pilot_timeline_solution_number" not in st.session_state:
                current_number = int(st.session_state.get("mission_timeline_solution_number", 1) or 1)
                st.session_state["pilot_timeline_solution_number"] = max(
                    1, min(current_number, len(pilot_gallery))
                )
            elif int(st.session_state["pilot_timeline_solution_number"]) > len(pilot_gallery):
                st.session_state["pilot_timeline_solution_number"] = 1

            selected_number = st.number_input(
                "Solution number",
                min_value=1,
                max_value=len(pilot_gallery),
                step=1,
                key="pilot_timeline_solution_number",
            )
            selected_idx = int(selected_number) - 1
            selected_solution = pilot_gallery[selected_idx]

            # Synchronize Mission Planning and Results with this solution.
            st.session_state["mission_timeline_solution_number"] = int(selected_number)
            st.session_state.timeline_gallery_index = selected_idx
            st.session_state.timeline_gallery_jump = int(selected_number)
            st.session_state.timeline_browsing_gallery = True
            st.session_state.timeline_mode = "ALTERNATIVE"

            # Reuse the exact crew-planning renderer from the Results mission
            # planning board. This keeps row sizing, colours, timed mission
            # blocks, availability blocks, tooltips, zoom and visible-flight-day
            # counting identical in both places.
            missions_for_timeline = normalize_missions(load_csv(MISSIONS))
            render_operations_board_component(
                selected_solution,
                missions_for_timeline,
                "Pilot planning",
                pilots_only=True,
            )
        elif results_stale:
            st.warning(
                "Planning or pilot data changed — run the optimizer first to calculate solutions for the current data."
            )
        else:
            st.info("No optimizer solutions found yet. Run the optimizer first to calculate solutions.")

    with load_tab:
        st.caption("Load a crew agenda Excel file. The first 3 letters of Resource are matched to the pilot trigram. CAT entries are ignored; all other entries block the full calendar day.")
        uploaded_availability = st.file_uploader(
            "Crew agenda Excel", type=["xlsx"], key="pilot_availability_xls_upload"
        )
        if uploaded_availability is not None:
            try:
                pilots_for_import = pd.read_csv(PILOTS, dtype=str, keep_default_na=False)
                uploaded_df, unknown_trigrams = parse_pilot_availability_excel(uploaded_availability.getvalue(), pilots_for_import)
                if unknown_trigrams:
                    st.warning("Ignored unknown crew trigrams: " + ", ".join(unknown_trigrams))
                active_upload = uploaded_df[uploaded_df["status"] != "AVAILABLE"].copy()
                st.success(f"Valid crew agenda: {len(active_upload)} blocked pilot-days for {active_upload['pilot_id'].nunique()} pilots.")
                st.dataframe(active_upload.head(30), width="stretch", hide_index=True)
                import hashlib
                upload_fingerprint = hashlib.sha256(uploaded_availability.getvalue()).hexdigest()
                if st.session_state.get("last_pilot_agenda_import") != upload_fingerprint:
                    save_csv(active_upload, PILOT_AVAILABILITY)
                    st.session_state["last_pilot_agenda_import"] = upload_fingerprint
                    st.session_state["pilot_availability_table_revision"] = int(st.session_state.get("pilot_availability_table_revision", 0)) + 1
                    _reset_pilot_availability_calendar()
                    st.session_state["pilot_availability_upload_notice"] = f"Loaded {len(active_upload)} blocked pilot-days from {uploaded_availability.name}. CAT entries were ignored."
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not read this crew agenda: {exc}")
        upload_notice = st.session_state.pop("pilot_availability_upload_notice", None)
        if upload_notice:
            st.success(upload_notice)

    with detailed_tab:
        selected = st.selectbox(
            "Pilot", pilot_ids, format_func=lambda pid: display.get(pid, pid),
            key="pilot_planning_selected",
        )
        availability = load_csv(PILOT_AVAILABILITY)

        # Reload after a CSV replacement so calendar and editor always use the active file.
        availability = load_csv(PILOT_AVAILABILITY)

        solution_choices = _pilot_solution_choices()
        solution_labels = list(solution_choices.keys())
        default_solution_index = 1 if len(solution_labels) > 1 else 0
        selected_solution_label = st.selectbox(
            "Optimizer solution overlay",
            solution_labels,
            index=default_solution_index,
            key="pilot_calendar_solution_overlay",
            help=(
                "Choose a discovered optimizer solution to show this pilot's planned/charged days "
                "on top of the availability calendar. Choose Availability only to hide the overlay."
            ),
        )
        selected_solution = solution_choices.get(selected_solution_label)

        events = _pilot_availability_events(availability, selected)
        events.extend(_pilot_solution_events(selected_solution, selected, availability))

        st.caption(
            "Click any day to edit it. Every day has a visible state: green = Available, "
            "red = Unavailable, orange = Leave, blue = Training. Purple = planned/charged in the selected optimizer solution."
        )
        st.markdown(
            "<div style='display:flex;gap:14px;flex-wrap:wrap;margin:2px 0 10px 0'>"
            "<span>🟩 Available</span><span>🟥 Unavailable</span>"
            "<span>🟧 Leave</span><span>🟦 Training</span>"
            "<span>🟪 Planned / charged</span></div>",
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
                    key=f"pilot_availability_calendar_{selected}_{calendar_revision}_{selected_solution_label}",
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

            # Optimizer overlay details for the selected pilot/day. This is
            # derived from the selected solution rather than only from the
            # clicked event, so both clicking the purple PLANNED pill and
            # clicking the day cell itself show the same information.
            if selected_solution:
                plan = _pilot_solution_day_details(selected_solution, selected, edit_day)
                st.markdown("#### Optimizer plan")
                if plan["planned"]:
                    conflict = status in {"UNAVAILABLE", "LEAVE", "TRAINING"}
                    if conflict:
                        st.error(
                            f"⚠ This pilot is planned/charged on {edit_day.isoformat()} "
                            f"but availability is {status}."
                        )
                    else:
                        st.success(
                            f"Planned / charged on {edit_day.isoformat()} in "
                            f"{selected_solution_label}."
                        )

                    movements = plan["movements"]
                    if movements:
                        for i, movement in enumerate(movements, start=1):
                            start_txt = movement["start"].strftime("%H:%M")
                            end_txt = movement["end"].strftime("%H:%M")
                            kind = movement["type"]
                            purpose = movement["purpose"]
                            mission = movement["mission"]
                            heading = (
                                f"{start_txt}–{end_txt} · {movement['aircraft']} · "
                                f"{movement['role']} · {kind}"
                            )
                            st.markdown(f"**{heading}**")
                            st.write(f"{movement['from']} → {movement['to']}")
                            meta = []
                            if mission:
                                meta.append(f"Mission: {mission}")
                            if purpose:
                                meta.append(f"Purpose: {purpose.replace('_', ' ')}")
                            if meta:
                                st.caption(" · ".join(meta))
                    else:
                        st.info(
                            "No aircraft movement is recorded for this pilot on this day. "
                            "The optimizer still charges the day because the pilot remains "
                            "away / attached to an aircraft or crew rotation."
                        )
                else:
                    st.caption(
                        f"Not planned or charged on {edit_day.isoformat()} in the selected solution."
                    )

    with table_tab:
        st.markdown("#### Pilot availability")
        av = load_csv(PILOT_AVAILABILITY)
        av_revision = int(st.session_state.get("pilot_availability_table_revision", 0))
        av_edit = st.data_editor(
            av, width="stretch", hide_index=True, num_rows="dynamic",
            key=f"pilot_availability_table_{av_revision}"
        )
        if st.button("Save availability table", key="save_pilot_availability_table"):
            save_csv(av_edit, PILOT_AVAILABILITY)
            st.success("pilot_availability.csv saved.")



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

    # The terminal fragment is the part of the page that polls every second.
    # When it observes the optimizer transition from running to finished, do
    # one full app rerun so the status badge and Start/Stop buttons outside
    # this fragment are refreshed as well.
    was_running = st.session_state.get("_optimizer_seen_running", False)
    if running:
        st.session_state["_optimizer_seen_running"] = True
    elif was_running:
        st.session_state["_optimizer_seen_running"] = False
        st.rerun(scope="app")

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

    running, pid = optimizer_status()

    basic_tab, advanced_tab = st.tabs(["Basic", "Advanced"])

    with basic_tab:
        st.caption("Quick run: 2 independent runs · population 100 · 10 generations")

        if st.button(
            "Start optimization",
            type="primary",
            disabled=running or bool(errors),
            key="basic_start_optimization",
        ):
            try:
                # Basic mode is deliberately fixed and keeps terminal output verbose.
                start_optimizer(
                    2,      # runs
                    100,    # population
                    10,     # generations
                    42,     # seed
                    False,  # separate live Pareto window
                    2,      # live update interval (unused while live=False)
                    True,   # verbose terminal logging
                    False,  # debug rejection examples
                )
                st.success("Optimizer started.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with advanced_tab:
        col1, col2, col3, col4 = st.columns(4)

        runs = col1.number_input(
            "Independent runs", min_value=1, max_value=50, value=5, step=1,
            key="advanced_runs",
        )
        population = col2.number_input(
            "Population", min_value=10, max_value=2000, value=200, step=10,
            key="advanced_population",
        )
        generations = col3.number_input(
            "Generations", min_value=1, max_value=5000, value=180, step=10,
            key="advanced_generations",
        )
        seed = col4.number_input(
            "Random seed", min_value=0, max_value=10_000_000, value=42, step=1,
            key="advanced_seed",
        )

        live = st.checkbox(
            "Open separate live Pareto window", value=False, key="advanced_live"
        )
        live_every = st.number_input(
            "Live update every N generations", min_value=1, max_value=100,
            value=2, step=1, disabled=not live, key="advanced_live_every",
        )

        log_cols = st.columns(2)
        verbose = log_cols[0].checkbox(
            "Verbose progress logging", value=True, key="advanced_verbose",
            help="Show every generation, feasibility rate, rejection categories, best KPIs and timings in the live terminal.",
        )
        debug = log_cols[1].checkbox(
            "Debug rejection examples", value=False, key="advanced_debug",
            help="Also show a few concrete invalid-reason examples per generation. This automatically enables verbose logging.",
        )
        if debug:
            verbose = True

        if st.button(
            "Start optimization",
            type="primary",
            disabled=running or bool(errors),
            key="advanced_start_optimization",
        ):
            try:
                start_optimizer(
                    int(runs), int(population), int(generations), int(seed),
                    bool(live), int(live_every), bool(verbose), bool(debug),
                )
                st.success("Optimizer started.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    # Keep run status / stop control outside the tabs so it is always visible.
    running, pid = optimizer_status()
    status_cols = st.columns([1, 1, 3])
    if running:
        status_cols[0].success(f"Running · PID {pid}")
    else:
        status_cols[0].info("Optimizer idle")

    if status_cols[1].button("Stop optimizer", disabled=not running):
        stop_optimizer()
        st.warning(
            "Graceful stop requested. The optimizer will save the best solutions found so far before exiting."
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

    # Load the estimated file for calculated timing fields, but keep the
    # currently active missions.csv authoritative for user-entered metadata
    # such as pax. missions_with_estimates.csv can be stale after a new CSV/XLSX
    # import and was the reason the Results board could show "0 pax" while the
    # Missions tab showed the correct value.
    missions = viz.load_csv_dict(
        (
            estimated_missions_path
            if estimated_missions_path.exists()
            else raw_missions_path
        ),
        "id",
    )

    if raw_missions_path.exists():
        raw_missions = viz.load_csv_dict(raw_missions_path, "id")
        for mid, raw_row in raw_missions.items():
            if mid not in missions:
                missions[mid] = dict(raw_row)
                continue

            # These fields belong to the active mission input and must never be
            # taken from an older estimates file.
            for key in (
                "pax",
                "origin",
                "destination",
                "fixed_aircraft",
                "source_mission_number",
            ):
                if key in raw_row:
                    missions[mid][key] = raw_row[key]

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




def actual_outstation_swap_events(solution, home_bases=("EBAW", "EBLG")):
    """Count real crew-change events occurring away from a home base.

    One event = the crew attached to an aircraft changes between two consecutive
    movements at the same physical handover airport.  Changing both CAPT and FO
    still counts as one swap event.
    """
    events = []
    bases = {str(x).strip().upper() for x in home_bases}

    for reg, raw_movements in solution.get("aircraft_movements", {}).items():
        movements = sorted(
            raw_movements,
            key=lambda x: pd.to_datetime(x.get("start")),
        )
        previous = None

        for movement in movements:
            if previous is not None:
                prev_to = str(previous.get("to", "")).strip().upper()
                cur_from = str(movement.get("from", "")).strip().upper()
                prev_crew = (
                    str(previous.get("captain", "")).strip(),
                    str(previous.get("fo", "")).strip(),
                )
                cur_crew = (
                    str(movement.get("captain", "")).strip(),
                    str(movement.get("fo", "")).strip(),
                )

                # Only count a genuine physical handover: same aircraft,
                # continuous airport, known crews, and crew actually changed.
                continuous_location = bool(prev_to and prev_to == cur_from)
                known_crews = all(prev_crew) and all(cur_crew)
                crew_changed = prev_crew != cur_crew
                outstation = cur_from not in bases

                if continuous_location and known_crews and crew_changed and outstation:
                    events.append({
                        "aircraft": reg,
                        "airport": cur_from,
                        "previous_crew": prev_crew,
                        "new_crew": cur_crew,
                        "previous_mission": previous.get("mission"),
                        "next_mission": movement.get("mission"),
                        "time": movement.get("start"),
                    })

            previous = movement

    return events



def actual_outstation_changed_pilots(solution, home_bases=("EBAW", "EBLG")):
    """Count individual pilot changes at genuine physical outstation swaps."""
    total = 0
    for event in actual_outstation_swap_events(solution, home_bases):
        prev_c, prev_f = event["previous_crew"]
        new_c, new_f = event["new_crew"]
        total += int(prev_c != new_c) + int(prev_f != new_f)
    return total


def _mission_lookup(missions) -> dict:
    """Return mission data keyed by mission id, regardless of stored shape."""
    if missions is None:
        return {}

    if isinstance(missions, pd.DataFrame):
        if "id" not in missions.columns:
            return {}
        return {
            str(row["id"]): row.to_dict()
            for _, row in missions.iterrows()
        }

    if isinstance(missions, list):
        return {
            str(row.get("id")): row
            for row in missions
            if isinstance(row, dict) and row.get("id") is not None
        }

    if isinstance(missions, dict):
        # Optimizer snapshots may either already be keyed by mission id or be
        # a dict-like collection of mission records.
        if all(isinstance(v, dict) for v in missions.values()):
            return {
                str(v.get("id", k)): v
                for k, v in missions.items()
            }
        return {str(k): v for k, v in missions.items()}

    return {}


def operations_board_payload(
    solution,
    missions,
):
    """
    Convert a validated solution into a browser-friendly operations-board model.

    Prefer explicit aircraft_movements so the web board is always identical to
    the schedule that passed the hard feasibility checks.
    """
    # The solution snapshot is authoritative for the solved schedule, but older
    # optimizer snapshots may not contain the current mission metadata (notably
    # pax). Keep both lookups: use the snapshot for schedule-related fields and
    # enrich display metadata from the active Missions dataset by mission id.
    active_mission_by_id = _mission_lookup(missions)

    # GUI pax is always authoritative from the currently active missions.csv.
    active_pax_by_id = {}
    try:
        _pax_df = pd.read_csv(MISSIONS, dtype=str).fillna("")
        if "id" in _pax_df.columns and "pax" in _pax_df.columns:
            active_pax_by_id = {str(row["id"]).strip(): row["pax"] for _, row in _pax_df.iterrows()}
    except Exception:
        active_pax_by_id = {}
    snapshot = solution.get("mission_snapshot")
    mission_by_id = _mission_lookup(snapshot) if snapshot else active_mission_by_id

    if snapshot:
        for mid, active_row in active_mission_by_id.items():
            if mid not in mission_by_id:
                mission_by_id[mid] = dict(active_row)
                continue
            snap_row = mission_by_id[mid]
            if isinstance(snap_row, dict) and isinstance(active_row, dict):
                # GUI metadata should reflect the mission CSV currently loaded.
                # Do not alter the solution's timing/aircraft assignment.
                for key in ("pax", "source_mission_number"):
                    if key in active_row and str(active_row.get(key, "")).strip() != "":
                        snap_row[key] = active_row[key]

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
            # GUI display only: keep C01/F01 internally, show trigrams everywhere.
            captain_display = pilot_label(captain) if captain != "?" else "?"
            fo_display = pilot_label(fo) if fo != "?" else "?"
            crew = (
                f"{captain_display}/{fo_display}"
            )

            if movement[
                "type"
            ] == "MISSION":
                mid = movement[
                    "mission"
                ]
                # Do not assume mission_snapshot is a dict. Older/newer
                # optimizer outputs can store it as a list, while the active
                # CSV may also differ from the solution that is being viewed.
                mission = mission_by_id.get(str(mid), {})
                action = actions.get(
                    mid,
                    {},
                )

                movement_action = str(movement.get("crew_action", "") or "")
                action_name = str(action.get("action", "") or movement_action)
                swap = (
                    ("SWAP" in action_name)
                    or int(movement.get("changed_pilots", 0) or 0) > 0
                    or int(action.get("changed_pilots", 0) or 0) > 0
                ) and not str(action.get("transition_mode", "")).startswith("IDLE_RETURN_")

                nonhome = int(
                    movement.get(
                        "outstation_swap",
                        action.get("nonhome_swap_event", 0),
                    ) or 0
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
                    "start_iso": start.isoformat(),
                    "end_iso": end.isoformat(),
                    "crew": crew,
                    "extra": (
                        f'{int(float(active_pax_by_id.get(str(mid), 0) or 0))} pax'
                    ),
                    "fixed": bool(str(mission.get("fixed_aircraft", "")).strip()),
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
                    "split_duty": bool(movement.get("split_duty", False)),
                    "split_duty_info": movement.get("split_duty_info") or {},
                    "crew_swap": bool(swap),
                    "crew_swap_reason": movement.get("crew_change_reason") or ("FTL_REQUIRED" if swap and nonhome else ""),
                    "handover_airport": movement.get("handover_airport") or action.get("handover_airport"),
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
                    "start_iso": start.isoformat(),
                    "end_iso": end.isoformat(),
                    "crew": crew,
                    "extra": (
                        f'{float(movement.get("nm",0)):.0f} NM'
                    ),
                    "fixed": False,
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
                    "start_iso": park_day + "T00:00:00",
                    "end_iso": park_day + "T23:59:59",
                    "crew": parking_crew,
                    "fixed": False,
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

    sorted_dates = sorted(all_dates)
    if sorted_dates:
        first_day = pd.Timestamp(sorted_dates[0]).date()
        last_day = pd.Timestamp(sorted_dates[-1]).date()
        continuous_dates = [
            stamp.date().isoformat()
            for stamp in pd.date_range(first_day, last_day, freq="D")
        ]
    else:
        continuous_dates = []

    # Crew rows for the same operations board. Assigned missions are shown as
    # grey timed blocks; availability exceptions are full-day coloured blocks.
    pilot_rows = []
    try:
        pilots_df = load_csv(PILOTS)
        availability_df = load_csv(PILOT_AVAILABILITY)
    except Exception:
        pilots_df = pd.DataFrame()
        availability_df = pd.DataFrame()

    assignment_cards = {}
    for reg, movements in explicit.items():
        for movement in movements or []:
            if str(movement.get("type", "")).upper() != "MISSION":
                continue
            start = pd.to_datetime(movement.get("start"))
            end = pd.to_datetime(movement.get("end"))
            mid = str(movement.get("mission", ""))
            route = f'{movement.get("from", "")} → {movement.get("to", "")}'
            for role_key, role_label in (("captain", "CAPT"), ("fo", "FO")):
                pid = str(movement.get(role_key, "") or "").strip()
                if not pid or pid == "?":
                    continue
                assignment_cards.setdefault(pid, []).append({
                    "kind": "crewmission", "label": mid, "route": route,
                    "start_iso": start.isoformat(), "end_iso": end.isoformat(),
                    "day": start.strftime("%Y-%m-%d"), "start": start.strftime("%H:%M"),
                    "end": end.strftime("%H:%M"), "aircraft": reg, "role": role_label,
                })

    if not pilots_df.empty and "id" in pilots_df.columns:
        for _, prow in pilots_df.iterrows():
            pid = str(prow.get("id", "")).strip()
            if not pid:
                continue
            role = str(prow.get("role", "")).strip().upper()
            name = str(prow.get("name", "")).strip()
            cards = list(assignment_cards.get(pid, []))
            if not availability_df.empty and "pilot_id" in availability_df.columns:
                work = availability_df[availability_df["pilot_id"].astype(str) == pid]
                for _, arow in work.iterrows():
                    status = str(arow.get("status", "AVAILABLE")).strip().upper() or "AVAILABLE"
                    if status == "AVAILABLE":
                        continue
                    day = str(arow.get("date", "")).strip()
                    try:
                        start = pd.Timestamp(day)
                    except Exception:
                        continue
                    cards.append({
                        "kind": "availability", "status": status,
                        "label": status.title(), "note": str(arow.get("note", "") or "").strip(),
                        "start_iso": start.isoformat(),
                        "end_iso": (start + pd.Timedelta(days=1)).isoformat(),
                        "day": day,
                    })
            cards.sort(key=lambda x: x.get("start_iso", ""))
            pilot_rows.append({"id": pid, "display": pilot_label(pid), "name": name, "role": role, "cards": cards})

    role_order = {"CAPT": 0, "CAPTAIN": 0, "FO": 1, "F/O": 1, "FIRST OFFICER": 1}
    pilot_rows.sort(key=lambda r: (role_order.get(r["role"], 9), r["name"] or r["id"]))

    return {
        "aircraft": aircraft_rows,
        "pilots": pilot_rows,
        "dates": continuous_dates,
    }


def render_operations_board_component(
    solution,
    missions,
    title,
    pilots_only=False,
):
    """Render a continuous UTC aircraft timeline, one horizontal lane per aircraft."""
    payload = operations_board_payload(solution, missions)

    if not payload["dates"]:
        st.info("No aircraft movements available.")
        return

    import json as _json

    data_json = _json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    objectives = solution.get("objectives", {})
    metrics = solution.get("metrics", {})
    header = {
        "title": title,
        "cost": f'€{objectives.get("total_operational_cost_eur",0):,.0f}',
        "complexity": int(objectives.get("complexity_score", 0)),
        "empty": int(metrics.get("empty_legs", 0)),
        "pilot_days": int(metrics.get("charged_pilot_days", 0)),
        "swaps": len(actual_outstation_swap_events(solution)),
    }
    header_json = _json.dumps(header, ensure_ascii=False)
    pilots_only_json = "true" if pilots_only else "false"

    # New optimizer solutions contain an exact reserve-coverage diagnostic.
    # Keep it compact by default, but make the dates/roles immediately inspectable.
    backup_issues = metrics.get("backup_coverage_issues", []) or []
    backup_shortage = int(objectives.get("backup_shortage_days", metrics.get("backup_shortage_days", 0)) or 0)
    if backup_shortage:
        with st.expander(f"⚠ Backup coverage · {backup_shortage} shortage unit{'s' if backup_shortage != 1 else ''}", expanded=False):
            if backup_issues:
                labels = _pilot_display_map()
                rows = []
                for issue in backup_issues:
                    cap_short = int(issue.get("captain_shortage", 0) or 0)
                    fo_short = int(issue.get("fo_shortage", 0) or 0)
                    missing = []
                    if cap_short: missing.append("CAPT")
                    if fo_short: missing.append("FO")
                    rows.append({
                        "Date": issue.get("date", ""),
                        "Missing backup": " + ".join(missing),
                        "CAPT reserve": len(issue.get("reserve_captains", []) or []),
                        "FO reserve": len(issue.get("reserve_fos", []) or []),
                        "CAPT flying": int(issue.get("active_captains", 0) or 0),
                        "FO flying": int(issue.get("active_fos", 0) or 0),
                        "Missions": ", ".join(issue.get("missions", []) or []),
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                st.caption("Target: at least 1 available, non-flying backup CAPT and 1 backup FO on every mission day. A shortage is a reserve warning, not an uncovered flight.")
            else:
                st.info("This solution was generated before detailed backup diagnostics were stored. Run the optimizer again to see the exact dates and crew pool causing the shortage.")

    component_html = f"""
<div id="aircraft-timeline-root">
<style>
* {{ box-sizing:border-box; }}
:root {{ --label-w:145px; --day-w:190px; --row-h:36px; --crew-row-h:28px; --head-h:36px; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#202124; background:transparent; overflow:hidden; }}
.kpis {{ display:flex; flex-wrap:wrap; gap:8px; margin:0 0 10px; }}
.kpi {{ border:1px solid #ddd; border-radius:12px; padding:7px 11px; background:#fff; min-width:100px; }}
.kpi b {{ display:block; font-size:16px; }} .kpi span {{ font-size:10px; color:#70757a; }}
.toolbar {{ display:flex; justify-content:space-between; gap:10px; align-items:center; margin:0 0 9px; flex-wrap:wrap; }}
.toolbar-left {{ display:flex; gap:10px; align-items:baseline; }} .title {{ font-size:18px; font-weight:750; }} .hint {{ color:#777; font-size:11px; }}
.controls {{ display:flex; gap:5px; align-items:center; }}
button.ctrl {{ border:1px solid #d5d5d5; background:#fff; padding:6px 9px; border-radius:8px; cursor:pointer; }}
.layout {{ display:block; }}
.viewport {{ height:760px; overflow:auto; border:1px solid #d2d2d2; border-radius:0; background:#fff; scrollbar-gutter:stable; }}
.board {{ position:relative; width:calc(var(--label-w) + var(--days) * var(--day-w)); min-width:max-content; }}
.header {{ position:sticky; top:0; z-index:30; height:var(--head-h); margin-left:var(--label-w); background:#f3f3f3; border-bottom:1px solid #d2d2d2; }}
.day-head {{ position:absolute; top:0; height:var(--head-h); display:flex; align-items:center; justify-content:center; border-left:1px solid #dedede; color:#333; font-size:12px; white-space:nowrap; }}
.corner {{ position:sticky; left:0; top:0; z-index:50; width:var(--label-w); height:var(--head-h); margin-top:calc(-1 * var(--head-h)); padding:5px 6px; background:#fafafa; border-right:1px solid #ddd; border-bottom:1px solid #ddd; font-size:10px; font-weight:700; }}
.row {{ position:relative; height:var(--row-h); margin-left:var(--label-w); border-bottom:1px solid #e2e2e2; background:#fff; }}
.row:nth-of-type(even) {{ background:#fcfcfc; }}
.label-cell {{ position:sticky; left:0; z-index:20; width:var(--label-w); height:var(--row-h); margin-left:calc(-1 * var(--label-w)); padding:3px 6px; background:#fbfbfb; border-right:1px solid #ddd; }}
.label-cell b {{ display:block; font-size:12px; line-height:1.05; }} .label-cell span {{ display:block; margin-top:2px; font-size:9px; line-height:1; color:#777; }}
.gridline {{ position:absolute; top:0; bottom:0; border-left:1px solid #ededed; pointer-events:none; }}
.noon {{ position:absolute; top:0; bottom:0; border-left:1px dotted #f0f0f0; pointer-events:none; }}
.block {{ position:absolute; min-width:48px; height:28px; top:4px; border:1px solid #555; border-radius:0; padding:3px 4px; overflow:hidden; cursor:pointer; text-align:left; color:#fff; background:#7da3ef; box-shadow:none; }}
.block:hover {{ z-index:12!important; box-shadow:0 3px 9px rgba(0,0,0,.15); }} .block.selected {{ outline:2px solid #2878ff; outline-offset:1px; z-index:13!important; }}
.block.empty {{ background:#fafafa; border-style:dashed; border-color:#999; color:#555; top:4px; height:26px; }}
.block.returnhome {{ background:#eef4fb; border-color:#9eb5cf; color:#334a62; }}
.block.parking {{ top:29px; height:6px; padding:2px 6px; border-radius:5px; background:#f0e9f8; border-color:#cab9db; color:#5d5366; font-size:10px; }}
.route {{ font-size:8px; font-weight:700; line-height:1.15; white-space:nowrap; overflow:visible; text-overflow:clip; }}
.meta {{ margin-top:1px; font-size:7px; font-weight:600; color:inherit; white-space:nowrap; overflow:visible; text-overflow:clip; }}
.swap {{ color:#2878ff; font-weight:900; margin-right:4px; }} .fixed {{ margin-right:4px; }}
.swap-badge {{ position:absolute; left:2px; top:2px; padding:1px 3px; background:#d9ecff; color:#075985; border:1px solid #5aa7d9; font-size:6px; font-weight:900; line-height:1.1; letter-spacing:.03em; z-index:2; }}
.split-badge {{ position:absolute; right:2px; top:2px; padding:1px 3px; background:#fff4c7; color:#6b5200; border:1px solid #d4aa22; font-size:6px; font-weight:900; line-height:1.1; letter-spacing:.03em; z-index:2; }}
.detail {{ position:sticky; top:0; border:1px solid #ddd; border-radius:12px; padding:14px; background:#fff; max-height:650px; overflow:auto; }}
.detail h3 {{ margin:0 0 12px; font-size:18px; }} .detail-block {{ border-bottom:1px solid #eee; padding:9px 0; }} .detail-block:last-child {{ border:0; }}
.dlabel {{ font-size:10px; color:#777; text-transform:uppercase; letter-spacing:.07em; margin-bottom:4px; }} .dvalue {{ font-size:13px; font-weight:600; overflow-wrap:anywhere; }}
.legend {{ display:none; gap:14px; flex-wrap:wrap; margin-top:8px; color:#666; font-size:11px; }}
.swatch {{ display:inline-block; width:12px; height:12px; border:1px solid #aaa; border-radius:3px; vertical-align:-2px; margin-right:4px; }}
.mission-s {{ background:#e7f3e8; }} .empty-s {{ background:#fafafa; border-style:dashed; }} .park-s {{ background:#f0e9f8; }}
.section-row {{ position:relative; height:24px; margin-left:var(--label-w); background:#ededed; border-top:2px solid #999; border-bottom:1px solid #ccc; }}
.section-label {{ position:sticky; left:0; z-index:21; width:var(--label-w); height:22px; margin-left:calc(-1 * var(--label-w)); padding:3px 6px; background:#e6e6e6; border-right:1px solid #ccc; font-size:12px; font-weight:750; }}
.crew-row {{ height:var(--crew-row-h); }} .crew-row .label-cell {{ height:var(--crew-row-h); padding:3px 6px; }}
.crew-row .label-cell b {{ font-size:10px; color:#2d6d9d; }} .crew-row .label-cell span {{ margin-top:1px; font-size:8px; }}
.crew-block {{ position:absolute; top:2px; height:23px; min-width:7px; border:1px solid #555; border-radius:0; padding:2px 3px; overflow:hidden; color:#fff; font-size:7px; line-height:1.05; z-index:6; }}
.crew-block.mission {{ background:#777; }} .crew-block.unavailable {{ background:#ff3030; }} .crew-block.leave {{ background:#f59e42; }} .crew-block.training {{ background:#3b93d1; }} .crew-block.other {{ background:#d9a4ef; color:#111; }}
.crew-block .cb-title {{ font-size:7px; font-weight:700; white-space:nowrap; }} .crew-block .cb-meta {{ font-size:6px; margin-top:1px; white-space:nowrap; }}
.tooltip {{ position:fixed; display:none; z-index:9999; max-width:380px; background:#111; color:#fff; border-radius:7px; padding:10px 12px; font-size:11px; line-height:1.5; box-shadow:0 5px 18px rgba(0,0,0,.30); pointer-events:none; }}
@media(max-width:950px) {{ .viewport {{ height:720px; }} }}
</style>
<div class="kpis" id="kpis"></div>
<div class="toolbar">
  <div class="toolbar-left"><div class="title" id="title"></div><div class="hint">UTC · one row per aircraft · horizontal position = actual time</div></div>
  <div class="controls"><button class="ctrl" id="startBtn">Start</button><button class="ctrl" id="minus">−</button><span id="zoomTxt">100%</span><button class="ctrl" id="plus">+</button><button class="ctrl" id="fit">Fit</button></div>
</div>
<div id="resultTooltip" class="tooltip"></div>
<div class="layout">
 <div>
  <div class="viewport" id="viewport"><div class="board" id="board"></div></div>
  <div class="legend"><span><i class="swatch mission-s"></i>Mission</span><span><i class="swatch empty-s"></i>Empty leg</span><span><i class="swatch park-s"></i>Parking</span><span>🔒 fixed aircraft</span><span><i class="swatch" style="background:#dbe6f7;border-color:#b8c9df"></i>Known aircraft location</span><span><i class="swatch" style="background:#fff0d9;border-color:#d4a14b"></i>Location discontinuity</span><span><b style="color:#2878ff">●</b> crew handover</span></div>
 </div>
 <div style="display:none"><span id="dTitle"></span><span id="dRoute"></span><span id="dTime"></span><span id="dAircraft"></span><span id="dCrew"></span><span id="dExtra"></span></div>
</div>
<script>
(() => {{
 const data={data_json}, header={header_json}, pilotsOnly={pilots_only_json};
 const board=document.getElementById('board'), viewport=document.getElementById('viewport');
 const dayMs=86400000, baseDayW=112;
 const zoomStorageKey='airline_optimizer_results_board_zoom_v1';
 let zoom=1;
 try {{
   const savedZoom=Number(window.localStorage.getItem(zoomStorageKey));
   if(Number.isFinite(savedZoom) && savedZoom>=.35 && savedZoom<=3) zoom=savedZoom;
 }} catch (_) {{}}
 const first=new Date(data.dates[0]+'T00:00:00Z');
 const lastEnd=new Date(data.dates[data.dates.length-1]+'T00:00:00Z').getTime()+dayMs;
 document.getElementById('title').textContent=header.title;
 if(pilotsOnly){{
   document.getElementById('kpis').style.display='none';
   document.querySelector('.hint').textContent='UTC · one row per pilot · horizontal position = actual time';
 }} else {{
   [[header.cost,'Total cost'],[header.complexity,'Complexity'],[header.empty,'Empty legs'],[header.pilot_days,'Pilot-days'],[header.swaps,'Outstation swaps']].forEach(([v,l])=>{{const e=document.createElement('div');e.className='kpi';e.innerHTML=`<b>${{v}}</b><span>${{l}}</span>`;document.getElementById('kpis').appendChild(e);}});
 }}
 const pct=(iso)=>((new Date(iso+'Z').getTime()-first.getTime())/(lastEnd-first.getTime()))*100;
 const widthPct=(a,b)=>Math.max(.12,((new Date(b+'Z')-new Date(a+'Z'))/(lastEnd-first.getTime()))*100);
 function pretty(d){{return new Date(d+'T00:00:00Z').toLocaleDateString('en-GB',{{weekday:'short',day:'2-digit',month:'short',timeZone:'UTC'}})}}
 function detail(c,reg,el){{document.querySelectorAll('.block.selected').forEach(x=>x.classList.remove('selected'));el.classList.add('selected');document.getElementById('dTitle').textContent=(c.fixed?'🔒 ':'')+(c.kind==='mission'?c.label:c.label);document.getElementById('dRoute').textContent=c.route||'—';document.getElementById('dTime').textContent=c.start&&c.end?`${{c.day}} · ${{c.start}} → ${{c.end}} UTC`:c.day;document.getElementById('dAircraft').textContent=reg;document.getElementById('dCrew').textContent=c.crew||'—';document.getElementById('dExtra').textContent=(c.extra||'—')+(c.swap?` · ${{c.swap_type}}`: '')+(c.fixed?' · pre-assigned aircraft':'');}}
 function esc(v){{return String(v??'').replace(/[&<>\"']/g,m=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[m]));}}
 const tip=document.getElementById('resultTooltip');
 function moveTip(e){{const pad=14,w=tip.offsetWidth,h=tip.offsetHeight;tip.style.left=Math.min(window.innerWidth-w-pad,e.clientX+14)+'px';tip.style.top=Math.min(window.innerHeight-h-pad,e.clientY+14)+'px';}}
 function showTip(html,e){{tip.innerHTML=html;tip.style.display='block';moveTip(e);}} function hideTip(){{tip.style.display='none';}}
 function addGrid(line){{data.dates.forEach((d,i)=>{{const g=document.createElement('div');g.className='gridline';g.style.left=`calc(${{i}} * var(--day-w))`;line.appendChild(g);const n=document.createElement('div');n.className='noon';n.style.left=`calc(${{i}} * var(--day-w) + var(--day-w)/2)`;line.appendChild(n);}});}}
 function render(){{
  board.innerHTML=''; board.style.setProperty('--days',data.dates.length);
  const head=document.createElement('div');head.className='header';
  data.dates.forEach((d,i)=>{{const h=document.createElement('div');h.className='day-head';h.style.left=`calc(${{i}} * var(--day-w))`;h.style.width='var(--day-w)';h.textContent=pretty(d);head.appendChild(h);}});board.appendChild(head);
  const corner=document.createElement('div');corner.className='corner';corner.textContent=pilotsOnly?'Pilots':'Aircraft';board.appendChild(corner);
  if(!pilotsOnly) data.aircraft.forEach(row=>{{
   const line=document.createElement('div');line.className='row';
   const lab=document.createElement('div');lab.className='label-cell';lab.innerHTML=`<b>${{row.registration}}</b><span>${{row.crew||'—'}}</span>`;line.appendChild(lab);addGrid(line);
   row.cards.forEach(c=>{{if(!c.start_iso||!c.end_iso)return;const el=document.createElement('button');el.type='button';el.className=`block ${{c.kind}}`;el.style.left=`${{pct(c.start_iso)}}%`;el.style.width=`max(48px, ${{widthPct(c.start_iso,c.end_iso)}}%)`;el.style.zIndex=c.kind==='parking'?2:5;
     const destination=(c.route||'').split('→').pop().trim(); const splitBadge=c.split_duty?'<span class="split-badge">SPLIT</span>':''; el.innerHTML=c.kind==='mission'?`${{splitBadge}}<div class="route">${{esc(destination)}}</div><div class="meta">${{esc(c.label)}}</div>`:`<div class="route">${{esc(c.label)}}</div><div class="meta">${{esc(c.route||'')}}</div>`;
     el.addEventListener('click',()=>detail(c,row.registration,el)); el.onmouseenter=(e)=>{{const si=c.split_duty_info||{{}};const hm=(v)=>{{v=Number(v||0);return `${{String(Math.floor(v/60)).padStart(2,'0')}}:${{String(v%60).padStart(2,'0')}}`;}};const split=c.split_duty?`<br><span style="color:#ffd75e;font-weight:800">SPLIT DUTY · ${{esc(si.station||'')}}</span><br>Ground interval ${{hm(si.ground_break_min)}} · protected break ${{hm(si.protected_break_min)}}<br>FDP extension +${{hm(si.extension_min)}} · basic max ${{hm(si.basic_max_fdp_min)}} → adjusted ${{hm(si.adjusted_max_fdp_min)}} · actual ${{hm(si.actual_fdp_min)}}`:'';showTip(`<b>${{esc(c.label)}} · ${{esc(c.route||'')}}</b><br>${{esc(c.day)}} · ${{esc(c.start)}} → ${{esc(c.end)}} UTC<br>Aircraft ${{esc(row.registration)}}<br><span style="color:#52bfff">Crew ${{esc(c.crew||'—')}}</span><br>${{esc(c.extra||'')}}${{split}}`,e);}};el.onmousemove=moveTip;el.onmouseleave=hideTip;line.appendChild(el);}});
   board.appendChild(line);
  }});
  if((data.pilots||[]).length){{
    if(!pilotsOnly){{ const sep=document.createElement('div');sep.className='section-row';const sl=document.createElement('div');sl.className='section-label';sl.textContent='Crew planning';sep.appendChild(sl);board.appendChild(sep); }}
    data.pilots.forEach(p=>{{const line=document.createElement('div');line.className='row crew-row';const lab=document.createElement('div');lab.className='label-cell';lab.innerHTML=`<b>${{esc(p.display||p.id)}}</b><span>${{esc(p.role)}} · <strong class="flight-days" data-pilot="${{esc(p.id)}}">0 flight days</strong></span>`;line.appendChild(lab);addGrid(line);
      (p.cards||[]).forEach(c=>{{if(!c.start_iso||!c.end_iso)return;const el=document.createElement('div');let cls='other';if(c.kind==='crewmission')cls='mission';else if(c.status==='UNAVAILABLE')cls='unavailable';else if(c.status==='LEAVE')cls='leave';else if(c.status==='TRAINING')cls='training';el.className=`crew-block ${{cls}}`;el.style.left=`${{pct(c.start_iso)}}%`;el.style.width=c.kind==='availability'?`${{widthPct(c.start_iso,c.end_iso)}}%`:`max(7px, ${{widthPct(c.start_iso,c.end_iso)}}%)`;
        if(c.kind==='crewmission')el.innerHTML=`<div class="cb-title">${{esc(c.label)}} · ${{esc(c.aircraft)}}</div><div class="cb-meta">${{esc(c.route)}}</div>`;else el.innerHTML=`<div class="cb-title">${{esc(c.label)}}</div><div class="cb-meta">${{esc(c.note||'')}}</div>`;
        el.onmouseenter=(e)=>showTip(c.kind==='crewmission'?`<b>${{esc(p.display||p.id)}} · ${{esc(c.label)}}</b><br>${{esc(c.route)}}<br>${{esc(c.start)}} → ${{esc(c.end)}} UTC<br>Aircraft ${{esc(c.aircraft)}} · ${{esc(c.role)}}`:`<b>${{esc(p.display||p.id)}} · ${{esc(c.status)}}</b><br>${{esc(c.day)}}<br>${{esc(c.note||'')}}`,e);el.onmousemove=moveTip;el.onmouseleave=hideTip;line.appendChild(el);}});board.appendChild(line);}});
  }}
 }}
 function updateVisibleFlightDays(){{
   const boardWidth=board.scrollWidth||1;
   const viewLeft=viewport.scrollLeft;
   const viewRight=viewLeft+viewport.clientWidth;
   const labelWidth=parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--label-w'))||160;
   const timelineWidth=Math.max(1,boardWidth-labelWidth);
   const timelineLeft=Math.max(0,viewLeft-labelWidth);
   const timelineRight=Math.max(0,viewRight-labelWidth);
   const rangeStart=first.getTime()+(timelineLeft/timelineWidth)*(lastEnd-first.getTime());
   const rangeEnd=first.getTime()+(timelineRight/timelineWidth)*(lastEnd-first.getTime());

   (data.pilots||[]).forEach(p=>{{
     const days=new Set();
     (p.cards||[]).forEach(c=>{{
       if(c.kind!=='crewmission'||!c.start_iso||!c.end_iso)return;
       const s=new Date(c.start_iso+'Z').getTime(), e=new Date(c.end_iso+'Z').getTime();
       if(e<rangeStart||s>rangeEnd)return;
       // Count the UTC calendar day(s) on which an assigned mission overlaps
       // the currently visible timeline range. Multiple missions on one day
       // still count as one flight day.
       const a=new Date(Math.max(s,rangeStart)), b=new Date(Math.min(e,rangeEnd));
       let d=Date.UTC(a.getUTCFullYear(),a.getUTCMonth(),a.getUTCDate());
       const last=Date.UTC(b.getUTCFullYear(),b.getUTCMonth(),b.getUTCDate());
       while(d<=last){{days.add(new Date(d).toISOString().slice(0,10));d+=dayMs;}}
     }});
     const el=document.querySelector(`.flight-days[data-pilot="${{CSS.escape(String(p.id))}}"]`);
     if(el)el.textContent=`${{days.size}} flight day${{days.size===1?'':'s'}}`;
   }});
 }}
 function setZoom(z, anchorClientX=null){{
   const oldZoom=zoom;
   const oldScrollWidth=board.scrollWidth||1;
   const labelWidth=parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--label-w'))||160;
   const rect=viewport.getBoundingClientRect();
   const anchorX=anchorClientX===null ? null : Math.max(labelWidth,Math.min(viewport.clientWidth,anchorClientX-rect.left));
   const oldTimelineWidth=Math.max(1,oldScrollWidth-labelWidth);
   const anchorTimelineX=anchorX===null ? null : Math.max(0,viewport.scrollLeft+anchorX-labelWidth);
   const anchorRatio=anchorTimelineX===null ? null : anchorTimelineX/oldTimelineWidth;

   zoom=Math.max(.35,Math.min(3,z));
   document.documentElement.style.setProperty('--day-w',`${{baseDayW*zoom}}px`);
   document.getElementById('zoomTxt').textContent=`${{Math.round(zoom*100)}}%`;
   try {{ window.localStorage.setItem(zoomStorageKey,String(zoom)); }} catch (_) {{}}

   // Keep the instant underneath the mouse pointer fixed while zooming.
   if(anchorRatio!==null && zoom!==oldZoom){{
     const newTimelineWidth=Math.max(1,board.scrollWidth-labelWidth);
     viewport.scrollLeft=Math.max(0,labelWidth+anchorRatio*newTimelineWidth-anchorX);
   }}
   requestAnimationFrame(updateVisibleFlightDays);
 }}
 document.getElementById('minus').onclick=()=>setZoom(zoom/1.2);document.getElementById('plus').onclick=()=>setZoom(zoom*1.2);document.getElementById('startBtn').onclick=()=>{{viewport.scrollTo({{left:0,behavior:'smooth'}});requestAnimationFrame(updateVisibleFlightDays);setTimeout(updateVisibleFlightDays,400);}};document.getElementById('fit').onclick=()=>{{const avail=Math.max(300,viewport.clientWidth-120);setZoom(Math.max(.35,Math.min(1.2,avail/(data.dates.length*baseDayW))));viewport.scrollLeft=0;scheduleFlightDaysUpdate();}};
 viewport.addEventListener('wheel',e=>{{if(e.ctrlKey||e.metaKey){{e.preventDefault();setZoom(zoom*(e.deltaY<0?1.1:.9),e.clientX);}}else if(e.shiftKey){{e.preventDefault();viewport.scrollLeft+=e.deltaY+e.deltaX;requestAnimationFrame(updateVisibleFlightDays);}}}},{{passive:false}});

 // IMPORTANT: this listener belongs to the Results timeline's own viewport.
 // Recalculate on every horizontal scroll, including scrollbar dragging,
 // trackpad scrolling and programmatic scrolling.
 let resultsFlightDaysRAF=null;
 function scheduleFlightDaysUpdate(){{
   if(resultsFlightDaysRAF!==null) cancelAnimationFrame(resultsFlightDaysRAF);
   resultsFlightDaysRAF=requestAnimationFrame(()=>{{
     resultsFlightDaysRAF=null;
     updateVisibleFlightDays();
   }});
 }}
 viewport.addEventListener('scroll', scheduleFlightDaysUpdate, {{passive:true}});
 window.addEventListener('resize', scheduleFlightDaysUpdate);

 render();
 // Preserve the user's board zoom when Streamlit rerenders after switching solution.
 setZoom(zoom);
 requestAnimationFrame(updateVisibleFlightDays);
}})();
</script>
"""
    components.html(component_html, height=980, scrolling=False)



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


def render_solution_metrics(solution):
    m = solution.get("metrics", {})
    o = solution.get("objectives", {})

    cols = st.columns(5)

    cols[0].metric("Total cost", f'€{o.get("total_operational_cost_eur", 0):,.0f}')
    cols[1].metric("Complexity", int(o.get("complexity_score", 0)))
    cols[2].metric("Pilot-days", int(m.get("charged_pilot_days", o.get("charged_pilot_days", 0))))
    cols[3].metric("Outstation swaps", len(actual_outstation_swap_events(solution)))
    cols[4].metric("Pilots changed outstation", actual_outstation_changed_pilots(solution))

def _set_gallery_index(index: int, gallery_size: int) -> None:
    if gallery_size <= 0:
        return
    index = int(index) % gallery_size
    st.session_state.timeline_gallery_index = index
    # This is the exact same 1-based number shown in the solution number box.
    st.session_state.timeline_gallery_jump = index + 1
    st.session_state.timeline_browsing_gallery = True
    st.session_state.timeline_mode = "ALTERNATIVE"


def _gallery_jump_changed() -> None:
    value = st.session_state.get("timeline_gallery_jump")
    if value is None:
        return
    st.session_state.timeline_gallery_index = int(value) - 1
    st.session_state.timeline_browsing_gallery = True
    st.session_state.timeline_mode = "ALTERNATIVE"


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

    # Results deliberately exposes only the four operational shortcuts.
    # Recompute them from the gallery so MIN OUTSTATION SWAPS uses the corrected
    # physical handover count rather than the legacy stored metric.
    named = _four_solution_choices(gallery)
    named_options = [
        "CHEAPEST",
        "MIN COMPLEXITY",
        "MIN OUTSTATION SWAPS",
        "MIN PILOT DAYS",
    ]

    # These are real action buttons, not merely labels/filters.
    # Every click resolves the criterion to one concrete gallery solution and
    # makes that gallery index the single source of truth for the board and
    # solution-number box.
    selector_cols = st.columns([1.0, 1.35, 1.85, 1.35])

    for col, label in zip(selector_cols, named_options):
        if col.button(
            label,
            key=f"results_solution_{label}",
            use_container_width=True,
        ):
            target = named[label]
            target_idx = _gallery_index_for_solution(gallery, target)
            st.session_state.timeline_last_named_selector = label
            _set_gallery_index(target_idx, len(gallery))
            st.rerun()

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
        _set_gallery_index(
            st.session_state.timeline_gallery_index - 1,
            len(gallery),
        )
        st.rerun()

    if navigation[1].button(
        "Next solution ▶",
        use_container_width=True,
        disabled=not bool(gallery),
    ):
        _set_gallery_index(
            st.session_state.timeline_gallery_index + 1,
            len(gallery),
        )
        st.rerun()

    if gallery:
        # Initialize the widget before creation. Afterwards its on_change callback
        # is the only place that copies widget state back to the solution index.
        desired_alt = (
            st.session_state.timeline_gallery_index % len(gallery)
        ) + 1
        if "timeline_gallery_jump" not in st.session_state:
            st.session_state.timeline_gallery_jump = desired_alt

        navigation[2].number_input(
            "Alternative",
            min_value=1,
            max_value=len(gallery),
            step=1,
            key="timeline_gallery_jump",
            on_change=_gallery_jump_changed,
            label_visibility="collapsed",
        )

        current_gallery_solution = gallery[
            st.session_state.timeline_gallery_index % len(gallery)
        ]
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

    checkpoint_path = OUTPUT / "checkpoint_status.json"
    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("status") == "partial":
                run = checkpoint.get("run")
                generation = checkpoint.get("generation")
                schedules = checkpoint.get("unique_aircraft_schedules", 0)
                st.info(
                    f"Partial optimizer results · run {run or '?'} · generation {generation if generation is not None else '?'} · "
                    f"{schedules} saved schedules. These are usable preliminary results and will be replaced by newer checkpoints."
                )
        except Exception:
            pass

    # Primary Results view: start with the interactive mission planning board.
    render_timeline_browser()

    # Keep the secondary analytics out of the way by default. The planning
    # board remains the primary Results view; users can expand the detailed
    # Pareto information when they need it.
    with st.expander("Detailed Pareto info", expanded=False):
        render_results()


st.title(
    "✈️ Airline Scheduling Optimizer"
)

st.caption(
    "Graphical wrapper around the existing optimizer. "
    "All underlying CSV files and Python scripts remain directly usable from Terminal."
)

# Sidebar navigation.
# Streamlit completely hides sidebar widgets when collapsed, so add a slim
# fixed icon rail that becomes visible only in the collapsed state.
_NAV_ITEMS = [
    ("Mission planning board", "▤"),
    ("Missions", "✈"),
    ("Pilot planning", "♟"),
    ("Fleet & crew", "▦"),
    ("Optimizer", "⚙"),
]

_NAV_LABELS = [label for label, _ in _NAV_ITEMS]
_query_page = st.query_params.get("page")
if _query_page == "Results":
    _query_page = "Mission planning board"
if _query_page in _NAV_LABELS and st.session_state.get("navigation_radio") != _query_page:
    st.session_state["navigation_radio"] = _query_page

def _sidebar_navigation_changed():
    selected = st.session_state.get("navigation_radio", _NAV_LABELS[0])
    st.query_params["page"] = selected

page = st.sidebar.radio(
    "Navigation",
    _NAV_LABELS,
    key="navigation_radio",
    on_change=_sidebar_navigation_changed,
)

st.markdown(
    """
    <style>
    /* Compact navigation rail shown when Streamlit's sidebar is collapsed. */
    .collapsed-nav-rail {
        display: none;
        position: fixed;
        left: 7px;
        top: 112px;
        z-index: 999990;
        width: 38px;
        padding: 5px 3px;
        border: 1px solid rgba(49,51,63,.16);
        border-radius: 9px;
        background: rgba(255,255,255,.96);
        box-shadow: 0 1px 4px rgba(0,0,0,.08);
    }
    .collapsed-nav-rail a {
        display: flex;
        width: 30px;
        height: 30px;
        margin: 2px auto;
        align-items: center;
        justify-content: center;
        border-radius: 6px;
        color: #31333f;
        text-decoration: none !important;
        font-size: 16px;
        line-height: 1;
    }
    .collapsed-nav-rail a:hover {
        background: rgba(151,166,195,.18);
    }
    .collapsed-nav-rail a.active {
        background: #ff4b4b;
        color: white;
    }

    /* Streamlit puts aria-expanded on the sidebar itself.  The previous
       version looked for it on the collapse button, which is why the rail
       never became visible.  Keep several selectors for Streamlit versions. */
    body:has(section[data-testid="stSidebar"][aria-expanded="false"])
        .collapsed-nav-rail,
    body:has([data-testid="stSidebar"][aria-expanded="false"])
        .collapsed-nav-rail,
    body:has([data-testid="stSidebarCollapsedControl"])
        .collapsed-nav-rail,
    body:has([data-testid="collapsedControl"])
        .collapsed-nav-rail {
        display: block !important;
    }

    /* Give the collapsed rail a little room without sacrificing screen area. */
    body:has(section[data-testid="stSidebar"][aria-expanded="false"])
        [data-testid="stMainBlockContainer"] {
        padding-left: 3.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar radio and collapsed icon rail share the same ?page= state.

_icon_links = []
for _label, _icon in _NAV_ITEMS:
    _active = " active" if page == _label else ""
    _href = "?page=" + _label.replace(" ", "%20").replace("&", "%26")
    _icon_links.append(
        f'<a class="{_active.strip()}" href="{_href}" target="_self" '
        f'title="{_label}" aria-label="{_label}">{_icon}</a>'
    )

st.markdown(
    '<nav class="collapsed-nav-rail" aria-label="Collapsed navigation">'
    + "".join(_icon_links)
    + "</nav>",
    unsafe_allow_html=True,
)

if page == "Mission planning board":
    results_page()

elif page == "Missions":
    mission_editor()

elif page == "Pilot planning":
    pilot_planning_page()

elif page == "Fleet & crew":
    data_editor_page()

elif page == "Optimizer":
    optimizer_page()

