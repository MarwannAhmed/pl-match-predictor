import csv

import pandas as pd

import validate


def test_validate_pipeline_runs_on_sample_csv(tmp_path, monkeypatch, capsys) -> None:
    data_dir = tmp_path / "data" / "matches"
    data_dir.mkdir(parents=True)
    csv_path = data_dir / "collected.csv"

    headers = [
        "ID",
        "Season",
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
        "HxG",
        "AxG",
        "H_ELO",
        "A_ELO",
    ]

    row = [
        "1",
        "2024-25",
        "10/08/2024",
        "Arsenal",
        "Chelsea",
        "2",
        "1",
        "H",
        "10",
        "8",
        "5",
        "3",
        "6",
        "4",
        "1",
        "2",
        "0",
        "0",
        "2.10",
        "3.50",
        "3.25",
        "1.42",
        "1.08",
        "1600.00",
        "1580.00",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerow(row)

    monkeypatch.chdir(tmp_path)

    df = pd.read_csv(csv_path)
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
