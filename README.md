# Airline Scheduling Optimizer

A local scheduling and crew-planning application built around a
Streamlit GUI and a Python genetic optimizer.

The current optimizer is a **fixed-aircraft crew optimizer**: every
customer mission must already contain a `fixed_aircraft`. The search
does not choose aircraft registrations; it assigns legal captain/FO
coverage to the fixed aircraft mission tours while minimizing
operational crew complexity and improving crew balance and reserve
coverage.

The GUI remains a wrapper around the same CSV files and Python scripts,
so the application can be used from the browser or directly from
Terminal.

## Current optimization priorities

Candidate schedules are compared lexicographically in this order:

1.  **Outstation handover events** --- minimize crew changes away from
    EBAW/EBLG.
2.  **Crew balance** --- distribute mission workload across the pilot
    pool.
3.  **Backup shortage** --- preserve reserve CAPT/FO coverage on mission
    days.
4.  **Base handovers** --- reduce additional crew changes at EBAW/EBLG.
5.  **Backup capacity** --- when the preceding criteria are equal,
    prefer more remaining backup slots.

The optimizer keeps multiple discovered schedule signatures and writes
them to the `output/` folder for browsing in the GUI.

## Hard planning rules

A candidate is only accepted when the relevant pilot assignment is
legal.

### Fixed aircraft

-   Every mission in `data/missions.csv` must have a `fixed_aircraft`.
-   Historical/empty positioning rows are ignored by the current lean
    crew optimizer.
-   Missions are grouped into aircraft tours.
-   A tour normally ends when the aircraft returns to EBAW or EBLG.
-   Crew stays attached to the aircraft while it is outstation unless a
    legal relief is required.

### Pilot roles and availability

-   Every operated segment requires one **CAPT** and one **FO**.
-   Pilot roles come from `data/pilots.csv`.
-   `UNAVAILABLE`, `LEAVE`, and `TRAINING` in
    `data/pilot_availability.csv` block the full calendar day.
-   A pilot cannot be assigned to overlapping duties.

### Flight-time limitations (FTL)

The current code performs roster-wide FTL validation rather than a
simple UTC-calendar-day duty check.

Implemented checks include:

-   45 minutes report time before the first sector;
-   30 minutes release time after the final sector;
-   maximum FDP based on report time and number of sectors;
-   legal split-duty extension when the protected break qualifies;
-   minimum rest between duties;
-   rolling duty limits of **60 h / 7 days**, **110 h / 14 days**, and
    **190 h / 28 days**;
-   rolling flight-time limit of **100 h / 28 days**.

Split duty uses the protected ground break between sectors. One hour is
removed for post-flight/travel/pre-flight activities; the remaining
protected break must be at least 3 hours. At most 6 hours of protected
break earns credit, and the FDP extension is 50% of the creditable
break.

The input data does not currently model WOCL encroachment or
accommodation availability, so the optimizer does not infer those
conditions.

If a complete aircraft tour cannot be covered legally by one crew pair,
the optimizer can use a slower relief search to find the minimum legal
segmentation needed. Already assigned segments are included immediately
in each pilot's roster history.

## Backup coverage

Backup shortage is a **reserve-coverage warning**, not an
uncovered-flight count.

On each day with missions, the target is to retain at least:

-   1 available, non-flying backup captain; and
-   1 available, non-flying backup first officer.

A `backup_shortage` value therefore reports missing reserve slots across
mission days.

Current solution files also store `backup_coverage_issues`, allowing the
GUI to show the exact date, missing role(s), flying crew, remaining
backup crew, and missions involved.

## Cost and displayed metrics

The current lean optimizer's monetary objective is based on charged
pilot-days:

-   **EUR 600 per charged pilot-day**

`total_operational_cost_eur` and `pilot_cost_eur` are therefore the
charged pilot-day total multiplied by EUR 600.

The current lean optimizer does **not** add aircraft flight cost,
commercial deadhead cost, empty-flight cost, or aircraft parking cost to
this monetary objective. Those fields may still exist in solution/output
structures for compatibility with the wider application.

The current complexity score is the number of crew handover events:

-   base handovers + outstation handovers.

Additional stored metrics include:

-   outstation handover events;
-   outstation changed pilots;
-   base handovers;
-   crew balance score;
-   backup shortage;
-   backup slots;
-   mission count by pilot;
-   charged pilot-days;
-   split-duty metadata;
-   FTL validation flags.

## Installation

Python 3 with a virtual environment is recommended.

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with the appropriate
`.venv\Scripts\activate` command instead.

Main Python dependencies are pandas, openpyxl, Streamlit, Plotly,
Matplotlib, and streamlit-calendar.

## Start the graphical interface

The normal entry point is:

``` bash
python launch_gui.py
```

You can also start Streamlit directly:

``` bash
streamlit run app.py
```

The navigation order in the current GUI is:

1.  **Mission planning board**
2.  **Missions**
3.  **Pilot planning**
4.  **Fleet & crew**
5.  **Optimizer**

## Mission planning board

The Mission planning board is the primary solution browser.

It provides:

-   previous/next solution navigation;
-   direct saved-solution number selection;
-   operational shortcuts for **CHEAPEST**, **MIN COMPLEXITY**, **MIN
    OUTSTATION SWAPS**, and **MIN PILOT DAYS**;
-   aircraft and pilot planning in the same interactive board;
-   mission, crew, handover, availability, and operational details;
-   horizontal/vertical scrolling;
-   Cmd/Ctrl + mouse-wheel zoom around the cursor;
-   Start, zoom −/+, and true Fit controls;
-   split-duty information where applicable;
-   backup-coverage diagnostics when a solution has reserve shortages.

The board keeps its zoom state while moving between solutions.

Below the planning board, the secondary result analytics are hidden by
default under **Detailed Pareto info**.

## Missions page

The Missions page has three tabs:

### Timeline

Shows the mission demand timeline and, when valid optimizer output
exists, allows direct selection by **Solution number**.

The selected solution is synchronized with the other solution views.

If missions have changed since the last optimization, the GUI does not
silently reuse the old results; it indicates that the optimizer must be
run again.

Split-duty use in the selected solution is also surfaced as a warning.

### Load XLS

Designed for normal user import from Excel rather than direct CSV
handling.

### Table

Provides spreadsheet-style access to the active mission data.

The underlying optimizer still reads `data/missions.csv`.

## Pilot planning page

The Pilot planning page follows the same structure as Mission planning
and contains:

### Timeline

Shows all pilots together using the same planning-board renderer as the
pilot section of the main Mission planning board.

It supports direct **Solution number** selection and shares the active
solution with Missions and Results.

### Load XLS

Imports a crew agenda Excel file.

The importer matches the first three letters of the Excel `Resource`
value to the pilot trigram. `CAT` entries are ignored; other imported
entries block the full calendar day.

### Table

Provides editable access to `pilot_availability.csv`.

### Detailed view per pilot

Shows one pilot at a time with the availability calendar and an optional
optimizer-solution overlay.

Availability states are:

-   Available
-   Unavailable
-   Leave
-   Training

Planned/charged days from the selected optimizer solution are shown as
an overlay.

Legacy **Contract rules** and **Monthly carry-in** controls are no
longer part of the GUI or active optimizer workflow.

## Fleet & crew page

Provides editable tables for:

-   `data/aircraft.csv`
-   `data/pilots.csv`
-   `data/airports.csv`

## Optimizer page

The Optimizer page has **Basic** and **Advanced** tabs.

### Basic

Basic mode exposes only one Start button and always runs:

-   2 independent runs;
-   population 100;
-   10 generations;
-   seed 42;
-   verbose terminal logging enabled;
-   separate live Pareto window disabled.

### Advanced

Advanced mode allows configuration of:

-   independent runs;
-   population;
-   generations;
-   random seed;
-   separate live Pareto window;
-   live update interval;
-   verbose progress logging;
-   debug rejection examples.

The page also contains:

-   optimizer running/idle status;
-   graceful **Stop optimizer** control;
-   live stdout/stderr terminal output;
-   automatic status refresh when the optimizer finishes.

The GUI launches `optimizer.py` as a separate unbuffered Python process,
so progress appears promptly in the live terminal.

## Running from Terminal

A normal run uses the optimizer defaults:

``` bash
python optimizer.py
```

Current command-line defaults are:

-   runs: 2
-   population: 40
-   generations: 30
-   seed: 42

A larger run can be started explicitly, for example:

``` bash
python optimizer.py --runs 5 --population 200 --generations 180 --seed 42
```

Useful options:

``` text
--runs N
--population N
--generations N
--seed N
--verbose
--debug
--live
--live-every N
```

`--verbose` enables detailed progress logging. `--debug` adds concrete
rejection examples. `--live` starts the separate live Pareto monitor,
and `--live-every` controls its update interval in generations.

The GUI's Basic preset is intentionally different from the raw Terminal
defaults.

## Optimizer progress output

Typical progress looks like:

``` text
run=2/2 gen=10/10 outstation=5 balance=130.616 backup_shortage=4 base_handovers=20 elapsed=344.8s
```

Meaning:

-   `outstation`: number of crew handover events away from EBAW/EBLG;
-   `balance`: current crew-balance score;
-   `backup_shortage`: missing reserve CAPT/FO slots across mission
    days;
-   `base_handovers`: handovers at EBAW/EBLG;
-   `elapsed`: total elapsed optimizer time.

A completed run prints a `DONE` line and writes the discovered solutions
to `output/`.

## Output files

The optimizer writes the main results to `output/`, including:

-   `pareto_000.json`, `pareto_001.json`, ... --- individual saved
    solutions;
-   `pareto.csv` --- summary table of saved solutions;
-   `diverse_solutions.json` --- solution gallery used by the GUI;
-   `cheapest.json` --- cheapest discovered solution;
-   `run_summary.json` --- run/evaluation summary;
-   `gui_optimizer.log` --- stdout/stderr when launched from the GUI.

Solution JSON contains the mission assignment, crew actions, metrics,
objectives, validation flags, split-duty metadata, and backup
diagnostics used by the current GUI.

## Data files used by the current workflow

The core active files are:

-   `data/missions.csv`
-   `data/missions_with_estimates.csv` when estimated arrival times are
    needed
-   `data/pilots.csv`
-   `data/pilot_availability.csv`
-   `data/aircraft.csv`
-   `data/airports.csv`

`missions.csv` is authoritative for the active mission set. If a mission
has no arrival time, the optimizer first looks for an estimated arrival
in `missions_with_estimates.csv`; if none is available, the optimizer
currently falls back to departure + 2 hours.

The repository may contain older/sample mission datasets and historical
files. They are not automatically the active optimization input unless
copied/imported into the active files above.

## Stale-result protection

Optimizer results are tied to the data used to generate them.

The GUI checks whether planning data has changed and warns when existing
solutions are stale. Mission and pilot timeline pages then require a new
optimization instead of presenting old solutions as if they applied to
the new data.

## Standalone tools

The repository still includes:

``` bash
python compare.py
python visualize.py
python live_pareto.py
```

These remain available for Terminal-based inspection/visualization. The
Streamlit GUI is the primary interface for the current workflow.

## Notes on the current architecture

-   `app.py` --- Streamlit GUI, import/edit workflows, planning boards,
    optimizer process control, and result browsing.
-   `optimizer.py` --- fixed-aircraft crew genetic optimizer,
    FTL/availability validation, scoring, and solution export.
-   `launch_gui.py` --- convenience launcher for the Streamlit
    application.
-   `visualize.py` --- standalone visualization support also reused by
    parts of the GUI.
-   `live_pareto.py` --- optional separate live optimization monitor.
-   `compare.py` --- Terminal comparison utility.
-   `data/` --- active and sample input data.
-   `output/` --- generated optimizer results and logs.

## Important current limitation

This version should not be described as an aircraft-assignment
optimizer. The optimizer requires `fixed_aircraft` on every mission and
optimizes the **crew plan around those fixed aircraft tours**.

Aircraft reassignment, inferred WOCL/accommodation rules, and other
operational rules not represented in the current input model are outside
the implemented optimizer logic.
