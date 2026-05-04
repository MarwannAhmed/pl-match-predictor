import csv

import pandas as pd

import validate

import collect


def test_collect_and_validate_pipeline(tmp_path, monkeypatch, capsys) -> None:
    raw_dir = tmp_path / "data" / "matches" / "raw"
    elo_dir = tmp_path / "data" / "elo"
    raw_dir.mkdir(parents=True)
    elo_dir.mkdir(parents=True)

    with (raw_dir / "2024-25.csv").open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(
            [
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
            ]
        )
        writer.writerow(
            [
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
            ]
        )

    with (elo_dir / "Spurs.csv").open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["From", "To", "Elo"])
        writer.writerow(["2024-08-01", "2024-08-31", "1600"])

    with (elo_dir / "Man_City.csv").open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["From", "To", "Elo"])
        writer.writerow(["2024-08-01", "2024-08-31", "1700"])

    monkeypatch.setattr(collect.get_xg, "SEASONS", ["2024-25"])
    monkeypatch.setattr(collect.join_elo, "SEASONS", ["2024-25"])
    monkeypatch.setattr(
        collect.get_xg,
        "fetch_understat_matches",
        lambda season_label: [
            {
                "h": {"title": "Tottenham"},
                "a": {"title": "Manchester City"},
                "xG": {"h": "1.42", "a": "1.08"},
            }
        ],
    )

    monkeypatch.chdir(tmp_path)

    collect.main()

    collected_path = tmp_path / "data" / "matches" / "collected.csv"
    assert collected_path.exists()

    df = pd.read_csv(collected_path)
    validate.main(df)

    captured = capsys.readouterr().out
    assert "Type checking completed." in captured
    assert "Missing value checking completed." in captured
    assert "Duplicate row checking completed." in captured
    assert "Season validation completed." in captured
    assert "Date validation completed." in captured
    assert "Team validation completed." in captured
    assert "Result validation completed." in captured
    assert "Stats validation completed." in captured
