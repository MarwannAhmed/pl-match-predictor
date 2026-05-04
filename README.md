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

4. Validate data:
   - `poetry run python src/data/collect/validate.py`

5. Engineer features:
   - `poetry run python src/features/engineer/engineer.py`

## Structure

```
├── .github/
│   └── workflows/
│       └── ci.yml               # CI pipeline
├── data/
│  ├── matches
│  │  └── raw/                   # Raw match data
│  └── ELO                       # Historical ELO data
├── src/
│  ├── data/                     # Data processing scripts
│  │  ├── collect/               # Data collection scripts
│  │  └── validate/              # Data validation scripts
│  └── features/                 # Feature processing scripts
│     └── engineer/              # Feature engineering scripts
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
