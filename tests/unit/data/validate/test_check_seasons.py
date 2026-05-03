import pandas as pd

import check_seasons


def test_check_season_length_reports_incorrect_match_counts(capsys) -> None:
    df = pd.DataFrame(
        {
            "Season": ["2024-25", "2024-25"],
            "HomeTeam": ["Arsenal", "Chelsea"],
            "AwayTeam": ["Liverpool", "Arsenal"],
        }
    )

    check_seasons.check_season_length(df)

    captured = capsys.readouterr().out
    assert "expected 380" in captured


def test_check_season_teams_reports_wrong_team_count(capsys) -> None:
    df = pd.DataFrame(
        {
            "Season": ["2024-25", "2024-25"],
            "HomeTeam": ["Arsenal", "Chelsea"],
            "AwayTeam": ["Liverpool", "Arsenal"],
        }
    )

    check_seasons.check_season_teams(df)

    captured = capsys.readouterr().out
    assert "expected 20" in captured


def test_check_fixture_consistency_reports_missing_fixtures(capsys) -> None:
    df = pd.DataFrame(
        {
            "Season": ["2024-25"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
        }
    )

    check_seasons.check_fixture_consistency(df)

    captured = capsys.readouterr().out
    assert "expected 2" in captured
