# PL Match Predictor

Premier League match outcome predictor.

## Prerequisites

- Poetry
- Python 3.13

## Getting Started

1. Clone the repository:
   - `git clone https://github.com/MarwannAhmed/pl-match-predictor.git`

2. Install dependencies:
   - `poetry install --no-root`

3. Collect data:
   - `poetry run python src/data/collect/collect.py`

## Structure

```
├── .github/
│   └── workflows/
│       └── ci.yml               # CI pipeline
├── data/
│  ├── matches
│  │  └── raw/                   # Raw match data
│  │     ├── 2015-16.csv
│  │     ├── ...
│  │     ├── 2024-25.csv
│  │     └── Notes.txt           # Data key
│  └── ELO                       # Historical ELO data
│     ├── Arsenal.csv
│     ├── ...
│     └── Wolves.csv
├── src/
│  └── data/                     # Data processing scripts
│     └── collect/               # Data collection scripts
│        ├── collect.py          # Main data collection script
│        ├── combine_seasons.py  # Combine all seasons into a single CSV
│        ├── get_xg.py           # Fetch xG data
│        └── join_elo.py         # Join ELO data with match data
├── .gitignore
├── poetry.lock                  # Dependency lock file
├── pyproject.toml               # Project configuration
└── README.md
```

## Contributing

- Commit message format: `type(scope): summary`
   - Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `data`
   - Scope: short area name like `data`, `model`, or `pipeline`
   - Summary: present tense, lowercase start, no period
   - Example: `feat(model): add xgboost baseline`
- Branch naming: `type/short-description`
   - Use the same `type` list as above
   - Use hyphenated words, no spaces
   - Example: `data/add-2024-25-season`
