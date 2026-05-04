# PL Match Predictor

This is the source code for a Premier League match outcome predictor that classifies matches into 'H', 'D', or 'A', i.e., home win, draw, or away win.

## Prerequisites

- Poetry
- Python 3.12
- Make

## Getting Started

1. Clone the repository:
   - `git clone https://github.com/MarwannAhmed/pl-match-predictor.git`

2. Install dependencies:
   - make install

3. Run tests:
   - make test

4. Run the pipeline:
   - make pipeline

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
│  ├── features/                 # Feature processing scripts
│  │  ├── engineer/              # Feature engineering scripts
│  │  ├── preprocess/            # Feature preprocessing scripts
│  │  ├── select/                # Feature selection scripts
│  │  └── visualize/             # Feature visualization scripts
│  ├── models/                   # Model training and evaluation scripts
│  └── pipeline.py               # End-to-end pipeline runner
├── tests/
│  ├── integration/              # Integration tests
│  └── unit/                     # Unit tests
├── .gitignore
├── Makefile
├── poetry.lock                  # Dependency lock file
├── pyproject.toml               # Project configuration
└── README.md
```

## Results

- Model outputs are saved under `results/<model>/` (metrics and confusion matrices).
- Summary metrics are saved in `results/model_results.csv`.
- Feature selection artifacts are saved in `results/feature_selection/`.
- Visualizations are saved in `results/visualizations/`.

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

## Data

Data is sourced from:

1. General match data (date, teams, result, stats, etc.): https://www.football-data.co.uk/englandm.php
2. xG values: understat.com API
3. ELO ratings: clubelo.com API