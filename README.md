# PL Match Predictor

Premier League match outcome predictor.

## Prerequisites

- Poetry
- Python 3.13

## Getting Started

1. Install dependencies:
   - `poetry install --no-root`

2. Get xG data:
   - `poetry run python src/data/get_xg.py`

## Structure

```
├── .github/
│   └── workflows/
│       └── ci.yml         # CI pipeline
├── data/
│   └── raw/               # Raw match data
│       ├── 2015-16.csv
│       ├── 2016-17.csv
│       ├── 2017-18.csv
│       ├── 2018-19.csv
│       ├── 2019-20.csv
│       ├── 2020-21.csv
│       ├── 2021-22.csv
│       ├── 2022-23.csv
│       ├── 2023-24.csv
│       ├── 2024-25.csv
│       └── Notes.txt      # Data key
└── src/
    └── data/              # Data processing scripts
        └── get_xg.py      # Script to fetch xG data
├── .gitignore
├── poetry.lock            # Dependency lock file
├── pyproject.toml         # Project configuration
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
