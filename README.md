# Airline Scheduling Optimizer

Standalone genetic scheduling prototype for the July mission dataset.

## What the search does

The optimizer uses a multi-run genetic algorithm with NSGA-II-style Pareto
selection. It deliberately maintains a population of different schedules rather
than greedily collapsing every candidate into the same aircraft rotation.

A chromosome contains:

- a full aircraft preference order for every mission;
- an aircraft transition preference for every mission:
  stay, via EBAW, or via EBLG;
- a crew-alternative selection gene for every mission;
- end-of-horizon aircraft return/stay policy;
- end-of-horizon crew return/stay policy.

Feasibility repair is deliberately minimal: the decoder only moves to the next
aircraft/transition preference when the preferred one is physically impossible.

## Cost model

- Aircraft flying: EUR 6,000 per block hour
- Pilot active or away from personal home base: EUR 600 per pilot-day
- Inbound crew deadhead: EUR 600 per pilot
- Crew return deadhead: EUR 600 per pilot
- Non-homebase crew swap: EUR 600 per event
- Aircraft parking away from EBAW/EBLG: EUR 1,000 per day
- EBAW and EBLG are valid bases for every aircraft
- PHJSK is unavailable

## Operational Complexity Score

Complexity is kept separate from money:

- +1 per aircraft empty positioning leg
- +2 per inbound deadheading pilot
- +4 per non-homebase crew swap
- +1 per aircraft away-parking day
- +1 per crew return deadhead

## Running

Create/activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Run a normal search:

    python optimizer.py

Then:

    python compare.py
    python visualize.py

The GUI contains:

- CHEAPEST
- BALANCED
- MIN EMPTY LEGS
- MIN PILOT DAYS
- MIN AIRCRAFT PARKING
- MIN COMPLEXITY
- ACTUAL
- PREV ALTERNATIVE / NEXT ALTERNATIVE

The alternative buttons browse genuinely different aircraft schedule signatures.

## Larger searches

The defaults are intentionally moderate. For a much larger search:

    python optimizer.py --runs 8 --population 250 --generations 250 --seed 42

Different seeds explore different regions:

    python optimizer.py --runs 5 --population 200 --generations 180 --seed 1001

The output summary explicitly reports:

- total evaluations
- unique aircraft schedules discovered
- Pareto solution count
- unique aircraft schedules on the Pareto front

This makes it easy to verify whether a larger/randomized search is actually
discovering new schedule structures.


## GUI performance

The visualizer uses a debounced synchronous redraw loop and a 20-schedule LRU
cache. For fast browsing, prefer the LEFT and RIGHT arrow keys instead of rapidly
clicking the navigation buttons.

Only long timeline blocks are labeled while browsing, which keeps Matplotlib
responsive on macOS even when the optimizer has discovered thousands of unique
schedules.


## Cost vs complexity Pareto window

`python visualize.py` opens two windows:

1. the aircraft schedule browser;
2. a Total Operational Cost vs Operational Complexity Pareto plot.

The currently selected optimized schedule is highlighted on the Pareto plot.
Changing a named strategy or browsing alternatives updates the highlight.
ACTUAL is not highlighted because it is a historical benchmark rather than an
optimizer Pareto solution.


### 2D Pareto filtering

The cost/complexity window recomputes its own Pareto front using only Total
Operational Cost and Operational Complexity. Solutions that are useful in the
optimizer's broader multi-objective Pareto set but are dominated on these two
specific axes are not shown.

The plotted line therefore represents the actual efficient cost-versus-complexity
trade-off boundary.


## Crew attachment invariant

Crew is attached to an aircraft instead of independently assigned per mission.

- Every used aircraft has exactly one attached captain and one attached FO.
- That crew follows customer flights, empty positioning and outstation parking.
- A pilot cannot be attached to two aircraft simultaneously.
- Crew changes only through an explicit handover at the aircraft location.
- A non-homebase handover can incur inbound deadhead, outgoing return deadhead
  and non-homebase swap costs.
- An aircraft that finishes away from EBAW/EBLG keeps its crew with it through
  the end of the planning horizon.

The current max-duty check remains only a basic daily report-to-release limit.
Full duty/rest/FTL modelling is intentionally left for the later extension.


## Live Pareto monitoring during optimization

Run:

    python optimizer.py --live

The optimizer starts `live_pareto.py` as a separate process. The live window
updates while the genetic algorithm runs and shows:

- all discovered schedules as a downsampled gray background cloud;
- the true current 2D Pareto front for Total Operational Cost vs Complexity;
- the current cheapest discovered schedule;
- run, generation, number of unique schedules, and evaluation count.

Because the monitor is a separate process, closing or freezing the plot does not
stop the optimizer.

For large searches, reduce GUI/file update overhead with:

    python optimizer.py --live --live-every 2 --runs 8 --population 250 --generations 250

`--live-every 2` updates after every second generation instead of every generation.


## Graphical interface

The graphical interface is deliberately a wrapper around the existing files and
scripts. Nothing prevents continued Terminal use.

Install once:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Start the GUI:

    python launch_gui.py

or directly:

    streamlit run app.py

The GUI opens in your local browser and includes:

- **Missions**
  - add one mission with airport dropdowns, date, UTC departure and pax;
  - paste mission blocks in the same human-readable format used during planning;
  - edit/delete all missions in a spreadsheet-like table;
  - blank arrival times are supported and estimated by optimizer.py.

- **Fleet & crew**
  - edit aircraft.csv, pilots.csv and airports.csv.

- **Optimizer**
  - select runs, population, generations and random seed;
  - optionally launch the existing live Pareto monitor;
  - start/stop optimizer.py as a separate process;
  - inspect the optimizer log without leaving the browser.

- **Results**
  - summary statistics and cost-vs-complexity chart;
  - launch the existing timeline visualizer.

Terminal operation remains unchanged, for example:

    python optimizer.py --runs 8 --population 250 --generations 250 --seed 42
    python compare.py
    python visualize.py


### Live terminal panel

The Optimizer page now includes a live terminal panel. It displays the exact
stdout/stderr generated by `optimizer.py` and refreshes automatically every
second while the GUI remains open. No manual refresh button is required.

The optimizer still runs as a separate process and writes the same output to:

    output/gui_optimizer.log

Running `optimizer.py` directly from Terminal remains unchanged.


### Immediate optimizer progress output

The GUI launches the optimizer with Python's unbuffered `-u` mode. This makes
generation progress lines appear in `output/gui_optimizer.log` immediately
instead of arriving in delayed blocks due to stdout buffering.


### Integrated timeline browser

The Results page now embeds the existing aircraft timeline directly inside the
Streamlit GUI.

Available views:

- CHEAPEST
- BALANCED
- MIN EMPTY LEGS
- MIN PILOT DAYS
- MIN AIRCRAFT PARKING
- MIN COMPLEXITY
- ACTUAL
- Previous / Next alternative

The alternative browser cycles through the saved unique schedules from
`output/diverse_solutions.json`, using the same reconstruction and timeline
drawing logic as `visualize.py`.

The standalone `visualize.py` remains available and can still be launched from
Terminal.


### Native Streamlit timeline

The embedded Results timeline is now web-native using Plotly rather than a
Matplotlib screenshot.

It supports:
- hover details for missions, empty legs, parking and crew;
- pan and scroll-wheel zoom;
- a date range slider and 1d / 3d / 7d / All controls;
- explicit crew-handover markers;
- native Streamlit strategy selection;
- Previous / Next and direct alternative-number navigation;
- the same optimizer output files as the Terminal tools.

`visualize.py` is preserved unchanged for standalone Matplotlib use.


## Hard feasibility invariants

Optimizer candidates are now rejected outright when any of these conditions fail:

- no two aircraft flight movements may overlap;
- every flight-to-flight transition has the required turnaround time;
- aircraft airport continuity must be exact;
- every customer mission occurs exactly once;
- every empty positioning flight has an attached captain and FO;
- initial crew is attached before the aircraft's first movement;
- a pilot cannot operate overlapping aircraft movements;
- end-of-horizon empty returns are explicit crewed movements and must fit inside
  the planning horizon.

Each solution JSON now includes:
- `mission_snapshot`: the exact mission data used during that optimization;
- `dataset_fingerprint`: a hash identifying that mission dataset;
- `aircraft_movements`: the exact validated mission and empty-flight timeline,
  including captain/FO on every movement;
- `validation`: explicit flags showing the hard validators passed.

The GUI and standalone visualizer prefer this explicit movement log. This prevents
an old solution file from being accidentally plotted against a newer missions.csv.


### Streamlit operations board

The Results page now uses the same planning-board concept as the standalone
visualizer:

- CHEAPEST is the default solution when Results is first opened.
- Five calendar days are shown at once.
- Aircraft remain fixed as rows.
- Customer missions are green cards.
- Empty positioning legs are dashed cards and always show attached crew.
- Away parking is shown as a muted card.
- Crew handovers are marked on the affected mission.
- Clicking a card opens route, timing, aircraft, attached crew and detail in a
  right-hand panel.
- Previous / Next solution continues to browse `diverse_solutions.json`.
- The board has its own Previous 5 days / Next 5 days controls.

The standalone `visualize.py` remains available and independent.
