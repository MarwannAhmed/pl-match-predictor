import csv
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

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

MIN_TO_DATE = datetime.strptime("2015-08-07", "%Y-%m-%d").date()

def parse_elo_history(path: Path) -> List[Tuple[datetime, datetime, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                from_d = datetime.strptime((r.get("From") or "").strip(), "%Y-%m-%d").date()
                to_d = datetime.strptime((r.get("To") or "").strip(), "%Y-%m-%d").date()
                elo = float((r.get("Elo") or "").strip())
            except Exception:
                continue
            # skip rows entirely before cutoff
            if to_d < MIN_TO_DATE:
                continue
            rows.append((from_d, to_d, elo))
    # assume file sorted by date but sort to be safe
    rows.sort(key=lambda t: (t[0], t[1]))
    return rows


def lookup_elo(history: List[Tuple[datetime, datetime, float]], match_date: datetime.date) -> Optional[float]:
    for from_d, to_d, elo in history:
        if from_d <= match_date <= to_d:
            return elo
    # fallback: find latest from_date <= match_date
    latest = None
    for from_d, to_d, elo in history:
        if from_d <= match_date:
            latest = elo
        else:
            break
    return latest


def join_season(season: str, matches_dir: Path, elo_dir: Path, out_dir: Path) -> None:
    src = matches_dir / f"{season}.csv"
    if not src.exists():
        print(f"Source season missing, skipping: {src}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{season}.csv"

    print(f"Joining ELO for season {season}")
    with src.open("r", encoding="utf-8") as inf, dst.open("w", newline="", encoding="utf-8") as outf:
        reader = csv.DictReader(inf)
        fieldnames = list(reader.fieldnames or [])
        try:
            insert_at = fieldnames.index("AwayTeam") + 1
        except ValueError:
            insert_at = len(fieldnames)
        fieldnames[insert_at:insert_at] = ["H_ELO", "A_ELO"]

        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            raw_date = (row.get("Date") or "").strip()
            match_date = None
            for fmt in ("%d/%m/%Y", "%d/%m/%y"):
                try:
                    match_date = datetime.strptime(raw_date, fmt).date()
                    break
                except Exception:
                    continue
            if match_date is None:
                # unable to parse date — write row unchanged
                writer.writerow(row)
                continue
            # normalize date output to DD/MM/YYYY
            row["Date"] = match_date.strftime("%d/%m/%Y")

            home = row.get("HomeTeam", "")
            away = row.get("AwayTeam", "")

            home_file = Path(f"{elo_dir}/{home.replace(' ', '_')}.csv")
            away_file = Path(f"{elo_dir}/{away.replace(' ', '_')}.csv")

            home_elo = None
            away_elo = None

            if home_file:
                history = parse_elo_history(home_file)
                home_elo = lookup_elo(history, match_date)
            else:
                print(f"  no elo file for home team {home}")

            if away_file:
                history = parse_elo_history(away_file)
                away_elo = lookup_elo(history, match_date)
            else:
                print(f"  no elo file for away team {away}")

            row["H_ELO"] = "" if home_elo is None else f"{home_elo:.2f}"
            row["A_ELO"] = "" if away_elo is None else f"{away_elo:.2f}"
            writer.writerow(row)


def main() -> None:
    matches_dir = Path("data/matches/with_xg")
    elo_dir = Path("data/elo")
    out_dir = Path("data/matches/with_xg_and_elo")

    print(f"Joining ELO for seasons into {out_dir}")
    for season in SEASONS:
        join_season(season, matches_dir, elo_dir, out_dir)
        print(f"Wrote season {season}")


if __name__ == "__main__":
    main()
