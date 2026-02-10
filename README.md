# Football Match Tactical Analytics Portfolio

A professional, reproducible football analytics project built on **StatsBomb Open Data** to evaluate team tactics at match level.

## Objective

This project is designed as a portfolio piece for football recruitment contexts (club analyst roles, consultancy, and performance departments). The aim is to show a practical workflow for tactical analysis:

- identify tactical patterns in possession and out of possession,
- quantify those patterns with event/360-derived metrics,
- communicate results through both technical visuals (Python) and stakeholder dashboards (Power BI).

The project is intentionally focused on method and process. It does **not** claim any precomputed findings in this repository.

## Tactical Focus

The analysis framework is structured around four tactical themes:

- `Build-up structure`: progression routes, pass lane usage, verticality, and zone occupation.
- `Chance creation`: shot context, assist zones, cutbacks, through balls, and set-play profiles.
- `Defensive organization`: pressing actions, duel locations, regain zones, and defensive line behavior (where data permits).
- `Transitions`: attacking and defensive transition speed after regains/losses.

These themes are mapped to clear, reproducible metrics so coaches and recruiters can interpret style, not just output volume.

## Dataset

Source: **StatsBomb Open Data** (JSON files included under `data/`).

Repository dataset components:

- `data/competitions.json`
- `data/matches/<competition_id>/<season_id>.json`
- `data/events/<match_id>.json`
- `data/lineups/<match_id>.json`
- `data/three-sixty/<match_id>.json` (for selected matches)

Reference docs are provided in `doc/`.

If this work is published externally, include proper attribution per StatsBomb terms.

## Project Structure

```text
Football_portfolio/
|-- config/
|   `-- analysis_config.example.yaml
|-- data/
|   |-- competitions.json
|   |-- matches/
|   |-- events/
|   |-- lineups/
|   `-- three-sixty/
|-- doc/
|-- notebooks/
|-- outputs/
|   |-- python_figures/
|   `-- tables/
|-- powerbi/
|   `-- README.md
|-- scripts/
|   `-- run_analysis.py
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_loader.py
|   |-- features/
|   |   `-- tactical_metrics.py
|   `-- visualization/
|       `-- plots.py
|-- tests/
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Methodology

1. **Scope selection**
- Choose competition(s), season(s), and match sample.
- Optionally narrow to a target team for opposition/scouting style reports.

2. **Data ingestion and validation**
- Load matches, events, lineups, and optional 360 freeze-frames.
- Validate IDs, timestamps, and key fields before metric calculation.

3. **Feature engineering (tactical metrics)**
- Build possession chains and classify phases.
- Calculate territorial, progression, chance-creation, pressing, and transition indicators.
- Keep definitions explicit and versioned for repeatability.

4. **Analytical outputs**
- Python: tactical plots and structured summary tables.
- Power BI: recruiter-friendly and coach-facing dashboard pages.

5. **Interpretation layer**
- Explain tactical tendencies with transparent metric definitions.
- Separate descriptive evidence from subjective tactical judgment.

## Outputs

### Python figures
Saved to `outputs/python_figures/`.

Planned visual families include:

- pass maps and network views,
- territory/zone control maps,
- shot maps by phase and assist type,
- regain and pressure location maps,
- transition timing distributions.

### Power BI
Power BI assets live under `powerbi/`.

Typical dashboard pages:

- Team Tactical Identity,
- In Possession,
- Out of Possession,
- Transitions,
- Match Comparison.

`powerbi/README.md` documents expected input tables from Python.

## How To Run

### 1) Create environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure scope

Copy and edit the config:

```bash
copy config\analysis_config.example.yaml config\analysis_config.yaml
```

Set selected competition, season, team filters, and output options.

### 3) Execute pipeline

```bash
python scripts/run_analysis.py --config config/analysis_config.yaml
```

### 4) Review outputs

- Python charts: `outputs/python_figures/`
- Exported tables for BI: `outputs/tables/`
- Build/report dashboard in Power BI using the exported tables.

## Limitations

- StatsBomb Open Data does not cover all competitions/seasons uniformly.
- 360 freeze-frame data exists only for selected matches.
- Event data captures actions, not full continuous tracking for all players at all times.
- Tactical inference is sensitive to sample size, game state, and opponent strength.
- This repository currently provides framework and reproducible workflow; no claim is made here about universal team-quality conclusions.

## Recruiter Notes

This project is intended to demonstrate:

- structured tactical thinking,
- robust data handling and reproducible code,
- communication across technical and non-technical audiences,
- end-to-end delivery from raw event data to decision-ready visuals.
