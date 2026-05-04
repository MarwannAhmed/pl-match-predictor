import csv
from pathlib import Path

import combine_seasons
import get_xg
import join_elo


def test_temp_pipeline_produces_collected_output(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    with_xg_dir = tmp_path / "with_xg"
    with_xg_and_elo_dir = tmp_path / "with_xg_and_elo"
    raw_dir.mkdir()
    with_xg_dir.mkdir()
    with_xg_and_elo_dir.mkdir()

    with (raw_dir / "2024-25.csv").open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow([
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR",
            "HS",
            "AS",
            "HST",
            "AST",
            "HC",
            "AC",
            "HY",
            "AY",
            "HR",
            "AR",
            "B365H",
            "B365D",
            "B365A",
            "Referee",
        ])
        writer.writerow([
            "10/08/2024",
            "Spurs",
            "Man City",
            "2",
            "1",
            "H",
            "10",
            "9",
            "5",
            "4",
            "6",
            "3",
            "1",
            "2",
            "0",
            "0",
            "2.10",
            "3.50",
            "3.25",
            "Ref A",
        ])

    monkeypatch.setattr(
        get_xg,
        "fetch_understat_matches",
        lambda season_label: [
            {
                "h": {"title": "Tottenham"},
                "a": {"title": "Manchester City"},
                "xG": {"h": "1.42", "a": "1.08"},
            }
        ],
    )

    with (tmp_path / "Spurs.csv").open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["From", "To", "Elo"])
        writer.writerow(["2024-08-01", "2024-08-31", "1600"])

    with (tmp_path / "Man_City.csv").open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["From", "To", "Elo"])
        writer.writerow(["2024-08-01", "2024-08-31", "1700"])

    get_xg.join_season("2024-25", raw_dir, with_xg_dir)
    join_elo.join_season("2024-25", with_xg_dir, tmp_path, with_xg_and_elo_dir)

    output_file = tmp_path / "collected.csv"
    combine_seasons.combine_seasons(with_xg_and_elo_dir, output_file)

    with output_file.open("r", newline="", encoding="utf-8") as file_handle:
        rows = list(csv.DictReader(file_handle))

    assert rows == [
        {
            "ID": "1",
            "Season": "2024-25",
            "Date": "10/08/2024",
            "HomeTeam": "Spurs",
            "AwayTeam": "Man City",
            "FTHG": "2",
            "FTAG": "1",
            "FTR": "H",
            "HS": "10",
            "AS": "9",
            "HST": "5",
            "AST": "4",
            "B365H": "2.10",
            "B365D": "3.50",
            "B365A": "3.25",
            "HxG": "1.42",
            "AxG": "1.08",
            "H_ELO": "1600.00",
            "A_ELO": "1700.00",
        }
    ]