"""Compare Ollama models on the same Steam reviews.

Every model labels the exact same reviews with the same prompt and validation as
review_compare.py, so differences come from the model alone. For each model it
reports:

  * failure rate       - share of reviews it could not label validly
  * first-try valid    - share labeled validly without needing self-correction
  * speed              - seconds per review on this machine
  * LLM-vs-Steam       - positive/negative call vs the reviewer's thumbs up/down
  * human accuracy     - sentiment vs your blind hand labels (once you fill them in)
  * model agreement    - how often each pair of models gives the same sentiment

Reviews are fetched once and saved to <out>/reviews.csv; later runs reuse that
file. Each model's labels are cached in <out>/labels/, so adding a model only
labels with the new one. After hand-filling human_sentiment in
<out>/model_audit.csv, rerun the same command to get human accuracy.

Usage:
    python compare_models.py --models llama3.2:3b llama3.1:8b qwen2.5:7b gemma3:12b --n 100
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from math import sqrt

import pandas as pd
import requests

import review_compare as rc


def wilson(k, n, z=1.96):
    """95% confidence interval for a proportion k/n (Wilson score)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def fmt_rate(k, n):
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k / n:.0%} [{lo:.0%}-{hi:.0%}] n={n}"


def ollama(endpoint, **body):
    requests.post(f"{rc.OLLAMA_URL}/api/{endpoint}", json=body, timeout=600).raise_for_status()


def load_reviews(args):
    path = os.path.join(args.out, "reviews.csv")
    if os.path.exists(path) and not args.refetch:
        df = pd.read_csv(path, dtype={"recommendationid": str})
        print(f"Reusing {len(df)} reviews from {path} (--refetch to download new ones)")
        return df
    frames = []
    for game, appid in rc.GAMES.items():
        print(f"Fetching up to {args.n} reviews for {game}...")
        frames.append(pd.DataFrame(rc.fetch_reviews(appid, args.n, args.min_chars)).assign(game=game))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(path, index=False)
    return pd.read_csv(path, dtype={"recommendationid": str})


def make_audit(reviews, args):
    """Blind hand-label sample, drawn from all reviews (not from any model's
    successes) so every model is graded on the same rows. Never overwritten."""
    path = os.path.join(args.out, "model_audit.csv")
    if not os.path.exists(path):
        parts = [g.sample(min(args.audit_per_game, len(g)), random_state=args.seed)
                 for _, g in reviews.groupby("game", sort=False)]
        s = pd.concat(parts)[["recommendationid", "game", "review_text"]].assign(human_sentiment="")
        s.sample(frac=1, random_state=args.seed).to_csv(path, index=False)
        print(f"Wrote blind audit sample to {path}")
    audit = pd.read_csv(path, dtype={"recommendationid": str})
    if not set(audit["recommendationid"]) <= set(reviews["recommendationid"]):
        sys.exit(f"{path} was drawn from a different set of reviews (after --refetch?). "
                 "Rename it to keep your labels, then rerun to draw a new sample.")
    audit["human_sentiment"] = audit["human_sentiment"].fillna("").astype(str).str.strip().str.lower()
    invalid = (audit["human_sentiment"] != "") & ~audit["human_sentiment"].isin(rc.SENTIMENTS)
    if invalid.any():
        sys.exit(f"{path} has labels other than positive/negative/mixed:\n"
                 + audit.loc[invalid, ["recommendationid", "human_sentiment"]].to_string(index=False))
    return audit[audit["human_sentiment"] != ""]


def label_with(model, reviews, args, meta):
    """Label every review with one model, or load the cached labels."""
    path = os.path.join(args.out, "labels", model.replace(":", "_").replace("/", "_") + ".csv")
    if os.path.exists(path) and not args.relabel and model in meta:
        labels = pd.read_csv(path, dtype={"recommendationid": str}, keep_default_na=False)
        if set(labels["recommendationid"]) == set(reviews["recommendationid"]):
            print(f"{model}: using cached labels")
            return labels
    rc.check_ollama(model)
    print(f"{model}: loading...", end=" ", flush=True)
    ollama("generate", model=model)  # load into memory so timing excludes load time
    print(f"labeling {len(reviews)} reviews...", end=" ", flush=True)
    clf = rc.Classifier(model)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(clf.classify, reviews["review_text"]))
    seconds = time.perf_counter() - start
    ollama("generate", model=model, keep_alive=0)  # unload so the next model has the GPU to itself
    print(f"done in {seconds:.0f}s")

    labels = pd.concat([reviews[["recommendationid"]], pd.DataFrame(results)], axis=1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    labels.to_csv(path, index=False)
    meta[model] = {"seconds": round(seconds, 1), "n": len(reviews), "workers": args.workers}
    return labels


def score(model, labels, reviews, human, meta):
    df = reviews.merge(labels, on="recommendationid", validate="1:1")
    ok = df[df["llm_error"] == ""]
    polar = ok[ok["llm_sentiment"].isin(["positive", "negative"])]
    steam_k = int(((polar["llm_sentiment"] == "positive") == polar["steam_voted_up"]).sum())
    h = human.merge(df[["recommendationid", "llm_sentiment"]], on="recommendationid", how="left")
    human_k = int((h["human_sentiment"] == h["llm_sentiment"]).sum())  # a failure counts as wrong
    themes = ok["llm_themes"].str.split("|")
    return {
        "model": model,
        "reviews": len(df),
        "failure_rate": round(1 - len(ok) / len(df), 4),
        "first_try_valid": round(((df["llm_error"] == "") & (df["llm_attempts"] == 1)).mean(), 4),
        "sec_per_review": round(meta[model]["seconds"] / meta[model]["n"], 2),
        "pct_mixed": round((ok["llm_sentiment"] == "mixed").mean(), 4),
        "avg_themes": round(themes.str.len().mean(), 2),
        "pct_other_theme": round(themes.apply(lambda t: "other" in t).mean(), 4),
        "steam_agreement": round(steam_k / len(polar), 4) if len(polar) else None,
        "steam_agreement_ci": fmt_rate(steam_k, len(polar)),
        "human_accuracy": round(human_k / len(h), 4) if len(h) else None,
        "human_accuracy_ci": fmt_rate(human_k, len(h)),
    }


def pairwise_agreement(all_labels):
    """Share of reviews where two models (both valid) give the same sentiment."""
    names = list(all_labels)
    m = pd.DataFrame(1.0, index=names, columns=names)
    for a, b in combinations(names, 2):
        j = all_labels[a].merge(all_labels[b], on="recommendationid", suffixes=("_a", "_b"))
        j = j[(j["llm_error_a"] == "") & (j["llm_error_b"] == "")]
        m.loc[a, b] = m.loc[b, a] = round((j["llm_sentiment_a"] == j["llm_sentiment_b"]).mean(), 3)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="Ollama models to compare")
    p.add_argument("--n", type=int, default=100, help="reviews per game (default 100)")
    p.add_argument("--out", default="output_models", help="output folder (default output_models)")
    p.add_argument("--min-chars", type=int, default=40)
    p.add_argument("--workers", type=int, default=4, help="parallel requests to Ollama (default 4)")
    p.add_argument("--audit-per-game", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refetch", action="store_true", help="download new reviews (clears cached labels)")
    p.add_argument("--relabel", action="store_true", help="ignore cached labels")
    args = p.parse_args()
    if args.refetch:
        args.relabel = True

    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, "models_meta.json")
    meta = {}
    if os.path.exists(meta_path) and not args.relabel:
        with open(meta_path) as f:
            meta = json.load(f)

    reviews = load_reviews(args)
    human = make_audit(reviews, args)

    all_labels, rows = {}, []
    for model in args.models:
        all_labels[model] = label_with(model, reviews, args, meta)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        rows.append(score(model, all_labels[model], reviews, human, meta))

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out, "model_comparison.csv"), index=False)
    agree = pairwise_agreement(all_labels)
    agree.to_csv(os.path.join(args.out, "model_agreement.csv"))

    print(f"\n================ MODEL COMPARISON ({len(reviews)} reviews) ================")
    for r in rows:
        print(f"\n{r['model']}")
        print(f"  failure rate        : {r['failure_rate']:.1%}   (valid on first try: {r['first_try_valid']:.1%})")
        print(f"  speed               : {r['sec_per_review']} s/review")
        print(f"  sentiment mixed     : {r['pct_mixed']:.1%}")
        print(f"  themes per review   : {r['avg_themes']}   ('other' in {r['pct_other_theme']:.1%})")
        print(f"  agrees with Steam   : {r['steam_agreement_ci']}   ('mixed' excluded)")
        print(f"  accuracy vs human   : {r['human_accuracy_ci']}")
    print("\nSentiment agreement between models:")
    print(agree.to_string())
    if human.empty:
        print(f"\nNo human labels yet: fill human_sentiment in {args.out}/model_audit.csv, then rerun "
              "this command (labels are cached, so it finishes instantly).")
    print("\nRanges in [brackets] are 95% confidence intervals. Overlapping ranges mean the "
          "difference between models may be noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
