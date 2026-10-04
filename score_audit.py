"""Score the LLM's sentiment labels against your hand labels.

1. Open output_compare/audit_sample.csv and fill the human_sentiment column with
   positive, negative or mixed, judging from the review text alone.
2. Run:  python score_audit.py

The audit file deliberately hides the LLM's answer and the Steam thumbs so your
label is independent. This script joins your labels back to the LLM's answers
in reviews_classified.csv by recommendationid.
"""

import argparse
import os
import sys

import pandas as pd

LABELS = ["positive", "negative", "mixed"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="output_compare", help="folder with the run outputs")
    args = p.parse_args()

    audit = pd.read_csv(os.path.join(args.dir, "audit_sample.csv"), dtype={"recommendationid": str})
    llm = pd.read_csv(os.path.join(args.dir, "reviews_classified.csv"), dtype={"recommendationid": str},
                      usecols=["recommendationid", "game", "llm_sentiment"])

    audit["human_sentiment"] = audit["human_sentiment"].fillna("").astype(str).str.strip().str.lower()
    unlabeled = audit["human_sentiment"] == ""
    invalid = ~unlabeled & ~audit["human_sentiment"].isin(LABELS)
    if invalid.any():
        print("These rows have labels other than positive/negative/mixed - fix them and re-run:")
        print(audit.loc[invalid, ["recommendationid", "human_sentiment"]].to_string(index=False))
        return 1
    labeled = audit[~unlabeled]
    if labeled.empty:
        print(f"No labels yet. Fill human_sentiment in {args.dir}/audit_sample.csv first.")
        return 1
    if unlabeled.any():
        print(f"Note: {unlabeled.sum()} of {len(audit)} rows are unlabeled and are skipped.\n")

    df = labeled.merge(llm.drop(columns="game"), on="recommendationid", how="left", validate="1:1")
    if df["llm_sentiment"].isna().any():
        sys.exit("Some audit rows have no matching LLM label - was reviews_classified.csv regenerated "
                 "after the audit sample was made?")
    df["correct"] = df["human_sentiment"] == df["llm_sentiment"]

    print(f"LLM sentiment accuracy vs human labels: {df['correct'].mean():.1%}"
          f"  ({df['correct'].sum()}/{len(df)})")
    for game, g in df.groupby("game", sort=False):
        print(f"  {game:<15} {g['correct'].mean():.1%}  ({g['correct'].sum()}/{len(g)})")

    cm = pd.crosstab(
        pd.Categorical(df["human_sentiment"], LABELS),
        pd.Categorical(df["llm_sentiment"], LABELS),
        rownames=["human"], colnames=["LLM"], dropna=False,
    )
    print("\nConfusion matrix (rows = your label, columns = LLM label):")
    print(cm.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
