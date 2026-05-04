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

3. Run the pipeline:
   - `poetry run python src/pipeline.py`

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
│  │  ├── select/0               # Feature selection scripts
│  │  └── visualize/             # Feature visualization scripts
│  ├── models/                   # Model training and evaluation scripts
│  └── pipeline.py               # End-to-end pipeline runner
├── tests/
│  ├── integration/              # Integration tests
│  └── unit/                     # Unit tests
├── .gitignore
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
