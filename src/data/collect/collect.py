import get_xg as get_xg
import join_elo as join_elo
import combine_seasons as combine_seasons

def main() -> None:
    get_xg.main()
    join_elo.main()
    combine_seasons.main()

if __name__ == "__main__":
    main()