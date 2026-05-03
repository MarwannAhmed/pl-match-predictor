import pandas as pd

import check_teams


def test_check_home_away_teams_reports_same_team(capsys) -> None:
    df = pd.DataFrame(
        {
            "Season": ["2024-25"],
            "Date": ["10/08/2024"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Arsenal"],
        }
    )

    check_teams.check_home_away_teams(df)

    captured = capsys.readouterr().out
    assert "HomeTeam and AwayTeam are the same" in captured


def test_check_team_duplicates_reports_multiple_games(capsys) -> None:
    df = pd.DataFrame(
        {
            "Season": ["2024-25", "2024-25"],
            "Date": ["10/08/2024", "10/08/2024"],
            "HomeTeam": ["Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Liverpool"],
        }
    )

    check_teams.check_team_duplicates(df)

    captured = capsys.readouterr().out
    assert "teams play more than once" in captured
