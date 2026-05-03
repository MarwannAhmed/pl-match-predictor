import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests

BASE_URL = "https://understat.com/getLeagueData/EPL/{season}"
HEADERS = {
    "x-requested-with": "XMLHttpRequest",
    "referer": "https://understat.com/league/EPL",
    "user-agent": "Mozilla/5.0",
}

SEASONS = [
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

TEAM_NAME_MAP = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Spurs": "Tottenham",
    "Wolves": "Wolverhampton Wanderers",
    "Sheffield United": "Sheffield Utd",
    "Sheffield Utd": "Sheffield Utd",
    "Nott'm Forest": "Nottingham Forest",
    "Newcastle": "Newcastle United",
    "Brighton": "Brighton",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "West Brom": "West Bromwich Albion",
    "West Bromwich": "West Bromwich Albion",
}


def normalize_team_name(name: str) -> str:
    trimmed = name.strip()
    return TEAM_NAME_MAP.get(trimmed, trimmed)


def season_start_year(season_label: str) -> int:
    return int(season_label.split("-")[0])


def fetch_understat_matches(season_label: str) -> List[dict]:
    start_year = season_start_year(season_label)
    url = BASE_URL.format(season=start_year)
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    payload = response.json()

    matches = payload.get("dates", [])
    if not isinstance(matches, list):
        raise ValueError(f"Unexpected JSON structure for season {season_label}")
    return matches


def build_xg_lookup(matches: Iterable[dict]) -> Dict[Tuple[str, str], Tuple[str, str]]:
    lookup: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for match in matches:
        home_team = match.get("h", {}).get("title", "").strip()
        away_team = match.get("a", {}).get("title", "").strip()
        if not home_team or not away_team:
            continue

        hxg = match.get("xG", {}).get("h", "")
        axg = match.get("xG", {}).get("a", "")
        lookup[(home_team, away_team)] = (hxg, axg)
    return lookup


def join_season(season_label: str, raw_dir: Path, out_dir: Path) -> None:
    raw_path = raw_dir / f"{season_label}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw data: {raw_path}")

    matches = fetch_understat_matches(season_label)
    xg_lookup = build_xg_lookup(matches)

    out_path = out_dir / f"{season_label}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    with raw_path.open("r", encoding="utf-8") as raw_file:
        reader = csv.DictReader(raw_file)
        if not reader.fieldnames:
            raise ValueError(f"Missing header in {raw_path}")

        fieldnames = list(reader.fieldnames)
        try:
            referee_index = fieldnames.index("Referee") + 1
        except ValueError:
            referee_index = len(fieldnames)

        fieldnames[referee_index:referee_index] = ["HxG", "AxG"]
        with out_path.open("w", newline="", encoding="utf-8") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                home_raw = row.get("HomeTeam", "")
                away_raw = row.get("AwayTeam", "")
                home = normalize_team_name(home_raw)
                away = normalize_team_name(away_raw)
                hxg, axg = xg_lookup.get((home, away), ("", ""))
                row["HxG"] = hxg
                row["AxG"] = axg
                writer.writerow(row)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw"
    out_dir = root / "data" / "with_xg"

    for season in SEASONS:
        join_season(season, raw_dir, out_dir)
        print(f"Wrote {season} to {out_dir}")


if __name__ == "__main__":
    main()
