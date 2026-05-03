import csv
from pathlib import Path

import combine_seasons


def test_combine_seasons_assigns_ids_and_season_from_filename(tmp_path: Path) -> None:
    input_dir = tmp_path / "with_xg_and_elo"
    output_file = tmp_path / "collected.csv"
    input_dir.mkdir()

    with (input_dir / "2023-24.csv").open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["Date", "HomeTeam", "AwayTeam", "HxG", "AxG", "H_ELO", "A_ELO"])
        writer.writerow(["01/01/2024", "Arsenal", "Chelsea", "1.20", "0.90", "1600.00", "1580.00"])

    with (input_dir / "2024-25.csv").open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["Date", "HomeTeam", "AwayTeam", "HxG", "AxG", "H_ELO", "A_ELO"])
        writer.writerow(["02/08/2024", "Liverpool", "Everton", "1.80", "0.70", "1650.00", "1550.00"])

    combine_seasons.combine_seasons(input_dir, output_file)

    with output_file.open("r", newline="", encoding="utf-8") as file_handle:
        rows = list(csv.DictReader(file_handle))

    assert rows == [
        {
            "ID": "1",
            "Season": "2023-24",
            "Date": "01/01/2024",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "FTHG": "",
            "FTAG": "",
            "FTR": "",
            "HS": "",
            "AS": "",
            "HST": "",
            "AST": "",
            "HC": "",
            "AC": "",
            "HY": "",
            "AY": "",
            "HR": "",
            "AR": "",
            "B365H": "",
            "B365D": "",
            "B365A": "",
            "HxG": "1.20",
            "AxG": "0.90",
            "H_ELO": "1600.00",
            "A_ELO": "1580.00",
        },
        {
            "ID": "2",
            "Season": "2024-25",
            "Date": "02/08/2024",
            "HomeTeam": "Liverpool",
            "AwayTeam": "Everton",
            "FTHG": "",
            "FTAG": "",
            "FTR": "",
            "HS": "",
            "AS": "",
            "HST": "",
            "AST": "",
            "HC": "",
            "AC": "",
            "HY": "",
            "AY": "",
            "HR": "",
            "AR": "",
            "B365H": "",
            "B365D": "",
            "B365A": "",
            "HxG": "1.80",
            "AxG": "0.70",
            "H_ELO": "1650.00",
            "A_ELO": "1550.00",
        },
    ]