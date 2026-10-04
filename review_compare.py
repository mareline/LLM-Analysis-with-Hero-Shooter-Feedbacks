"""Compare Steam review themes for Overwatch 2 vs Marvel Rivals.

Pulls recent English Steam reviews for each game, asks an Azure OpenAI model to
label each review's sentiment and themes (from a fixed, hand-defined taxonomy),
then checks the labels before reporting anything:

  * output validation  - every LLM answer must be valid JSON with an allowed
                         sentiment and only allowed themes, or it is retried
                         and finally recorded as a failure (never guessed)
  * failure rate       - share of reviews the LLM could not label; the run is
                         flagged if it exceeds --max-failure-rate
  * LLM-vs-Steam check - the LLM's positive/negative call is compared with the
                         reviewer's own thumbs up/down on Steam

It also writes a blind audit sample (review text only) for hand labeling;
score_audit.py turns those labels into an accuracy figure.

Requires env vars AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY and
AZURE_OPENAI_DEPLOYMENT (optionally AZURE_OPENAI_API_VERSION).

Usage:
    python review_compare.py --n 200
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests

GAMES = {
    "Overwatch 2": 2357570,
    "Marvel Rivals": 2767030,
}

# Hand-defined theme taxonomy. Each review gets 1-3 of these.
THEMES = {
    "matchmaking_ranked": "matchmaking quality, ranked mode, team/skill balance between players",
    "hero_balance": "character power level, nerfs/buffs, the meta, counters",
    "performance_technical": "bugs, crashes, FPS, optimization, servers, lag, netcode",
    "monetization": "prices, skins, battle pass, shop, microtransactions, value for money",
    "content_variety": "number of heroes/maps/modes, pace and amount of new content",
    "toxicity_cheating": "toxic players, harassment, cheaters/hackers, community behavior",
    "core_gameplay": "how the game feels to play: fun, gunplay, abilities, teamwork, pacing",
    "developer_trust": "developer/publisher decisions, communication, broken promises, company",
    "other": "anything that fits none of the above",
}
SENTIMENTS = ("positive", "negative", "mixed")

STEAM_URL = "https://store.steampowered.com/appreviews/{appid}"
MAX_REVIEW_CHARS = 2000

SYSTEM_PROMPT = (
    "You classify Steam reviews of hero shooter games. Reply with JSON only, shaped "
    'exactly like {"sentiment": "...", "themes": ["...", ...]}.\n'
    "sentiment: the reviewer's overall attitude to the game - one of "
    + ", ".join(SENTIMENTS)
    + ". Use mixed only when praise and criticism are roughly balanced.\n"
    "themes: 1 to 3 themes the review actually discusses, most prominent first, "
    "chosen ONLY from this list:\n"
    + "\n".join(f"- {name}: {desc}" for name, desc in THEMES.items())
)


# ---------------------------------------------------------------- Steam

def fetch_reviews(appid, n, min_chars, sleep=0.5):
    """Return up to n recent English reviews with at least min_chars of text."""
    rows, seen, cursor = [], set(), "*"
    while len(rows) < n:
        resp = requests.get(
            STEAM_URL.format(appid=appid),
            params={
                "json": 1,
                "filter": "recent",
                "language": "english",
                "purchase_type": "all",
                "num_per_page": 100,
                "cursor": cursor,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("success") != 1:
            raise RuntimeError(f"Steam API returned success={data.get('success')} for app {appid}")
        batch = data.get("reviews", [])
        new = 0
        for r in batch:
            rid = r["recommendationid"]
            text = (r.get("review") or "").strip()
            if rid in seen or len(text) < min_chars:
                continue
            seen.add(rid)
            new += 1
            author = r.get("author", {})
            rows.append({
                "recommendationid": rid,
                "review_text": text,
                "steam_voted_up": bool(r.get("voted_up")),
                "created_utc": datetime.fromtimestamp(r["timestamp_created"], timezone.utc),
                "playtime_at_review_hrs": round(author.get("playtime_at_review", 0) / 60, 1),
            })
            if len(rows) >= n:
                break
        next_cursor = data.get("cursor")
        if not batch or not next_cursor or next_cursor == cursor or new == 0 and len(batch) < 100:
            break
        cursor = next_cursor
        time.sleep(sleep)
    return rows


# ---------------------------------------------------------------- LLM

def make_client():
    missing = [v for v in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT")
               if not os.environ.get(v)]
    if missing:
        sys.exit(f"Missing environment variable(s): {', '.join(missing)}")
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        max_retries=5,
    )
    return client, os.environ["AZURE_OPENAI_DEPLOYMENT"]


def validate(raw):
    """Parse and check one LLM answer. Returns (sentiment, themes) or raises ValueError."""
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("answer is not a JSON object")
    sentiment = str(obj.get("sentiment", "")).strip().lower()
    if sentiment not in SENTIMENTS:
        raise ValueError(f"bad sentiment {sentiment!r}")
    themes = obj.get("themes")
    if not isinstance(themes, list) or not 1 <= len(themes) <= 3:
        raise ValueError("themes must be a list of 1-3 items")
    themes = [str(t).strip().lower() for t in themes]
    bad = [t for t in themes if t not in THEMES]
    if bad:
        raise ValueError(f"themes not in taxonomy: {bad}")
    return sentiment, list(dict.fromkeys(themes))


class Classifier:
    def __init__(self, client, deployment, attempts=3):
        self.client, self.deployment, self.attempts = client, deployment, attempts
        # Some newer Azure models reject temperature; we drop it once if so.
        self.use_temperature = True

    def _call(self, text):
        kwargs = {
            "model": self.deployment,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text[:MAX_REVIEW_CHARS]},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.use_temperature:
            kwargs["temperature"] = 0
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            if self.use_temperature and "temperature" in str(e).lower():
                self.use_temperature = False
                return self._call(text)
            raise
        return resp.choices[0].message.content or ""

    def classify(self, text):
        """Returns dict with llm_sentiment, llm_themes, llm_error (one of them empty)."""
        error = ""
        for _ in range(self.attempts):
            try:
                sentiment, themes = validate(self._call(text))
                return {"llm_sentiment": sentiment, "llm_themes": "|".join(themes), "llm_error": ""}
            except Exception as e:  # invalid output or API error: retry, then record
                error = f"{type(e).__name__}: {e}"[:300]
        return {"llm_sentiment": "", "llm_themes": "", "llm_error": error}


# ---------------------------------------------------------------- analysis

def summarize(df):
    """Per-game integrity and summary stats."""
    out = {}
    for game, g in df.groupby("game", sort=False):
        ok = g[g["llm_error"] == ""]
        polar = ok[ok["llm_sentiment"].isin(["positive", "negative"])]
        agree = (polar["llm_sentiment"] == "positive") == polar["steam_voted_up"]
        out[game] = {
            "reviews_fetched": len(g),
            "reviews_classified": len(ok),
            "failure_rate": round(1 - len(ok) / len(g), 4) if len(g) else None,
            "review_window_utc": [g["created_utc"].min().strftime("%Y-%m-%d"),
                                  g["created_utc"].max().strftime("%Y-%m-%d")],
            "median_playtime_at_review_hrs": float(g["playtime_at_review_hrs"].median()),
            "steam_pct_positive": round(g["steam_voted_up"].mean(), 4),
            "llm_sentiment_counts": ok["llm_sentiment"].value_counts().to_dict(),
            "llm_vs_steam_agreement": round(agree.mean(), 4) if len(polar) else None,
            "llm_vs_steam_n": len(polar),
        }
    return out


def theme_table(df):
    """One row per (game, theme): share of classified reviews mentioning it and
    the share of those reviews the LLM called negative."""
    ok = df[df["llm_error"] == ""].copy()
    ok["theme"] = ok["llm_themes"].str.split("|")
    long = ok.explode("theme")
    totals = ok.groupby("game").size()
    t = long.groupby(["game", "theme"]).agg(
        reviews=("recommendationid", "size"),
        pct_negative=("llm_sentiment", lambda s: round((s == "negative").mean(), 4)),
    ).reset_index()
    t["share_of_reviews"] = (t["reviews"] / t["game"].map(totals)).round(4)
    return t.sort_values(["game", "share_of_reviews"], ascending=[True, False])


def plot_themes(themes, games, path):
    colors = ["#2a78d6", "#eb6834"]  # validated categorical slots 1-2 (dataviz palette)
    pivot = (themes.pivot(index="theme", columns="game", values="share_of_reviews")
             .reindex(columns=games).fillna(0))
    pivot = pivot.loc[pivot.max(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(pivot) + 1.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    h = 0.38
    y = range(len(pivot))
    for i, (game, color) in enumerate(zip(games, colors)):
        pos = [v + (0.5 - i) * (h + 0.02) for v in y]  # first game on top
        ax.barh(pos, pivot[game] * 100, height=h, color=color, label=game)
    ax.set_yticks(list(y), [t.replace("_", " ") for t in pivot.index], color="#3d3d3a")
    ax.set_xlabel("% of classified reviews mentioning theme", color="#3d3d3a")
    ax.set_title("Steam review themes: " + " vs ".join(games), loc="left", color="#1a1a19")
    ax.grid(axis="x", color="#e5e4df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#6b6a63", length=0)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def audit_sample(df, per_game, seed):
    """Random classified reviews per game, text only, so the human label is blind
    to both the LLM's answer and the reviewer's Steam thumbs."""
    ok = df[df["llm_error"] == ""]
    parts = [g.sample(min(per_game, len(g)), random_state=seed) for _, g in ok.groupby("game", sort=False)]
    s = pd.concat(parts)[["recommendationid", "game", "review_text"]].copy()
    s["human_sentiment"] = ""
    return s.sample(frac=1, random_state=seed)  # shuffle games together


# ---------------------------------------------------------------- main

def run(games, args):
    os.makedirs(args.out, exist_ok=True)
    if not args.fetch_only:
        client, deployment = make_client()  # fail fast on missing keys, before downloading

    frames = []
    for game, appid in games.items():
        print(f"Fetching up to {args.n} reviews for {game} (app {appid})...")
        rows = fetch_reviews(appid, args.n, args.min_chars)
        print(f"  got {len(rows)}")
        if len(rows) < args.n:
            print(f"  WARNING: fewer reviews than requested for {game}")
        frames.append(pd.DataFrame(rows).assign(game=game))
    df = pd.concat(frames, ignore_index=True)

    if args.fetch_only:
        df.to_csv(os.path.join(args.out, "reviews_raw.csv"), index=False)
        print(f"--fetch-only: wrote {len(df)} reviews to {args.out}/reviews_raw.csv")
        return 0

    clf = Classifier(client, deployment)
    print(f"Classifying {len(df)} reviews with Azure OpenAI ({args.workers} workers)...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(clf.classify, df["review_text"]))
    df = pd.concat([df, pd.DataFrame(results)], axis=1)

    stats = summarize(df)
    themes = theme_table(df)
    overall_fail = round((df["llm_error"] != "").mean(), 4)

    df.to_csv(os.path.join(args.out, "reviews_classified.csv"), index=False)
    themes.to_csv(os.path.join(args.out, "theme_summary.csv"), index=False)
    if len(themes):
        plot_themes(themes, list(games), os.path.join(args.out, "theme_comparison.png"))
    audit_sample(df, args.audit_per_game, args.seed).to_csv(
        os.path.join(args.out, "audit_sample.csv"), index=False)
    with open(os.path.join(args.out, "run_summary.json"), "w") as f:
        json.dump({
            "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_requested_per_game": args.n,
            "min_review_chars": args.min_chars,
            "overall_failure_rate": overall_fail,
            "games": stats,
        }, f, indent=2, default=str)

    # ---- printed summary
    print("\n=================== SUMMARY ===================")
    for game, s in stats.items():
        print(f"\n{game}")
        print(f"  reviews fetched / classified : {s['reviews_fetched']} / {s['reviews_classified']}"
              f"  (failure rate {s['failure_rate']:.1%})")
        print(f"  review window (UTC)          : {s['review_window_utc'][0]} to {s['review_window_utc'][1]}")
        print(f"  median playtime at review    : {s['median_playtime_at_review_hrs']} hrs")
        print(f"  Steam thumbs-up rate         : {s['steam_pct_positive']:.1%}")
        print(f"  LLM sentiment counts         : {s['llm_sentiment_counts']}")
        if s["llm_vs_steam_agreement"] is not None:
            print(f"  LLM vs Steam agreement       : {s['llm_vs_steam_agreement']:.1%}"
                  f"  (n={s['llm_vs_steam_n']}, 'mixed' excluded)")
        top = themes[themes["game"] == game].head(5)
        print("  top themes (share of reviews, % negative within theme):")
        for _, r in top.iterrows():
            print(f"    {r['theme']:<22} {r['share_of_reviews']:>6.1%}   {r['pct_negative']:>6.1%} neg")

    print(f"\nOverall failure rate: {overall_fail:.1%}")
    errors = df.loc[df["llm_error"] != "", "llm_error"]
    if len(errors):
        print("Most common errors:")
        for msg, count in errors.str.slice(0, 120).value_counts().head(3).items():
            print(f"  {count}x {msg}")
    print(f"Files written to {args.out}/")

    if overall_fail > args.max_failure_rate:
        print(f"\nDATA INTEGRITY FAIL: failure rate {overall_fail:.1%} exceeds "
              f"{args.max_failure_rate:.0%}. Do not report these results.")
        return 1
    return 0


def parse_args(argv=None, default_out="output_compare"):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200, help="reviews per game (default 200)")
    p.add_argument("--out", default=default_out, help=f"output folder (default {default_out})")
    p.add_argument("--min-chars", type=int, default=40,
                   help="skip reviews shorter than this; too short to theme (default 40)")
    p.add_argument("--workers", type=int, default=4, help="parallel LLM calls (default 4)")
    p.add_argument("--audit-per-game", type=int, default=25,
                   help="reviews per game in the hand-label sample (default 25)")
    p.add_argument("--max-failure-rate", type=float, default=0.05,
                   help="flag the run if more than this share of reviews fail (default 0.05)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fetch-only", action="store_true",
                   help="only download reviews (no LLM calls); for testing the Steam side")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(GAMES, parse_args()))
