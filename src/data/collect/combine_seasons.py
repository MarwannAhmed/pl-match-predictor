import csv
from pathlib import Path

COLUMNS = [
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
	"B365H",
	"B365D",
	"B365A",
	"HxG",
	"AxG",
	"H_ELO",
	"A_ELO",
]


def combine_seasons(input_dir: Path, output_file: Path) -> None:
	season_files = sorted(
		path for path in input_dir.glob("*.csv") if path.name != output_file.name
	)

	output_file.parent.mkdir(parents=True, exist_ok=True)

	with output_file.open("w", newline="", encoding="utf-8") as outf:
		writer = csv.DictWriter(outf, fieldnames=COLUMNS)
		writer.writeheader()

		row_id = 1
		for season_file in season_files:
			season = season_file.stem
			with season_file.open("r", newline="", encoding="utf-8") as inf:
				reader = csv.DictReader(inf)
				for row in reader:
					combined_row = {column: row.get(column, "") for column in COLUMNS}
					combined_row["ID"] = row_id
					combined_row["Season"] = season
					writer.writerow(combined_row)
					row_id += 1


def main() -> None:
	input_dir = Path("data/matches/with_xg_and_elo")
	output_file = Path("data/matches/collected.csv")
	combine_seasons(input_dir, output_file)


if __name__ == "__main__":
	main()
