import csv
from pathlib import Path

import get_xg


def test_normalize_team_name_maps_known_aliases_and_trims_whitespace() -> None:
    assert get_xg.normalize_team_name(" Spurs ") == "Tottenham"
    assert get_xg.normalize_team_name("Man City") == "Manchester City"


def test_build_xg_lookup_skips_incomplete_matches() -> None:
    matches = [
        {
            "h": {"title": "Arsenal"},
            "a": {"title": "Chelsea"},
            "xG": {"h": "1.25", "a": "0.80"},
        },
        {
            "h": {"title": ""},
            "a": {"title": "Liverpool"},
            "xG": {"h": "0.40", "a": "1.90"},
        },
    ]

    lookup = get_xg.build_xg_lookup(matches)

    assert lookup == {("Arsenal", "Chelsea"): ("1.25", "0.80")}


def test_join_season_writes_xg_columns_for_matching_fixture(monkeypatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "with_xg"
    raw_dir.mkdir()

    raw_path = raw_dir / "2024-25.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["Date", "HomeTeam", "AwayTeam", "Referee", "B365H"])
        writer.writerow(["10/08/2024", "Spurs", "Man City", "Ref A", "2.10"])

    monkeypatch.setattr(
        get_xg,
        "fetch_understat_matches",
        lambda season_label: [
            {
                "h": {"title": "Tottenham"},
                "a": {"title": "Manchester City"},
                "xG": {"h": "1.40", "a": "1.10"},
            }
        ],
    )

    get_xg.join_season("2024-25", raw_dir, out_dir)

    out_path = out_dir / "2024-25.csv"
    with out_path.open("r", newline="", encoding="utf-8") as file_handle:
        rows = list(csv.DictReader(file_handle))

    assert rows == [
        {
            "Date": "10/08/2024",
            "HomeTeam": "Spurs",
            "AwayTeam": "Man City",
            "Referee": "Ref A",
            "HxG": "1.40",
            "AxG": "1.10",
            "B365H": "2.10",
        }
    ]