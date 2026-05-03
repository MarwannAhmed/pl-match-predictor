import csv
from datetime import datetime
from pathlib import Path

import join_elo


def test_parse_elo_history_filters_old_rows_and_sorts_results(tmp_path: Path) -> None:
    elo_path = tmp_path / "Arsenal.csv"
    with elo_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["From", "To", "Elo"])
        writer.writerow(["2015-08-01", "2015-08-06", "1500"])
        writer.writerow(["2015-08-07", "2015-08-20", "1510"])
        writer.writerow(["2015-09-01", "2015-09-10", "1520"])

    history = join_elo.parse_elo_history(elo_path)

    assert history == [
        (datetime.strptime("2015-08-07", "%Y-%m-%d").date(), datetime.strptime("2015-08-20", "%Y-%m-%d").date(), 1510.0),
        (datetime.strptime("2015-09-01", "%Y-%m-%d").date(), datetime.strptime("2015-09-10", "%Y-%m-%d").date(), 1520.0),
    ]


def test_lookup_elo_prefers_matching_range_and_latest_previous_value() -> None:
    history = [
        (
            datetime.strptime("2015-08-07", "%Y-%m-%d").date(),
            datetime.strptime("2015-08-20", "%Y-%m-%d").date(),
            1510.0,
        ),
        (
            datetime.strptime("2015-09-01", "%Y-%m-%d").date(),
            datetime.strptime("2015-09-10", "%Y-%m-%d").date(),
            1520.0,
        ),
    ]

    assert join_elo.lookup_elo(history, datetime.strptime("2015-08-10", "%Y-%m-%d").date()) == 1510.0
    assert join_elo.lookup_elo(history, datetime.strptime("2015-08-25", "%Y-%m-%d").date()) == 1510.0
    assert join_elo.lookup_elo(history, datetime.strptime("2015-09-05", "%Y-%m-%d").date()) == 1520.0


def test_join_season_adds_elo_columns_and_normalizes_dates(tmp_path: Path) -> None:
    matches_dir = tmp_path / "with_xg"
    elo_dir = tmp_path / "elo"
    out_dir = tmp_path / "with_xg_and_elo"
    matches_dir.mkdir()
    elo_dir.mkdir()

    with (matches_dir / "2024-25.csv").open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["Date", "HomeTeam", "AwayTeam", "Referee", "HxG", "AxG"])
        writer.writerow(["10/8/24", "Arsenal", "Chelsea", "Ref A", "1.40", "1.10"])

    with (elo_dir / "Arsenal.csv").open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["From", "To", "Elo"])
        writer.writerow(["2024-08-01", "2024-08-31", "1601.5"])

    with (elo_dir / "Chelsea.csv").open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["From", "To", "Elo"])
        writer.writerow(["2024-08-01", "2024-08-31", "1588.25"])

    join_elo.join_season("2024-25", matches_dir, elo_dir, out_dir)

    with (out_dir / "2024-25.csv").open("r", newline="", encoding="utf-8") as file_handle:
        rows = list(csv.DictReader(file_handle))

    assert rows == [
        {
            "Date": "10/08/2024",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "Referee": "Ref A",
            "HxG": "1.40",
            "AxG": "1.10",
            "H_ELO": "1601.50",
            "A_ELO": "1588.25",
        }
    ]