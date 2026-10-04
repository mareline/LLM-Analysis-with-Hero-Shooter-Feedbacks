"""Single-game version of review_compare.py: theme analysis for one Steam game.

Uses the same fetching, LLM classification, validation and integrity checks.

Usage:
    python review_themes.py --game "Marvel Rivals" --n 200
    python review_themes.py --appid 2357570 --name "Overwatch 2" --n 200
"""

import argparse
import sys

import review_compare as rc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", choices=list(rc.GAMES), help="one of the built-in games")
    p.add_argument("--appid", type=int, help="any Steam app id (use with --name)")
    p.add_argument("--name", help="display name for --appid")
    args, rest = p.parse_known_args()

    if args.game:
        games = {args.game: rc.GAMES[args.game]}
    elif args.appid:
        games = {args.name or str(args.appid): args.appid}
    else:
        p.error("give --game or --appid")

    slug = next(iter(games)).lower().replace(" ", "_")
    opts = rc.parse_args(rest, default_out=f"output_{slug}")
    return rc.run(games, opts)


if __name__ == "__main__":
    sys.exit(main())
