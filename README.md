# What Are Hero Shooter Players Complaining About? Overwatch 2 vs Marvel Rivals

An LLM pipeline that reads recent Steam reviews for **Overwatch 2** and **Marvel Rivals**, tags each one with a sentiment and up to three themes (matchmaking, monetization, hero balance and so on), and compares what players talk about in each game. The pipeline checks its own output before reporting anything, and its accuracy is measured against hand labels.

**Product question:** Two free-to-play hero shooters compete for the same players. Where does each one lose goodwill, and which pain points are shared across the genre versus specific to one game?

## How it works

1. **Collect.** The script pulls the *N* most recent English Steam reviews per game from Steam's public review API. Reviews under 40 characters ("good", "mid") are skipped because they're too short to tag with a theme.
2. **Classify.** An Azure OpenAI model labels each review with:
   - `sentiment`: positive, negative, or mixed
   - `themes`: 1–3 themes from a fixed taxonomy: matchmaking_ranked, hero_balance, performance_technical, monetization, content_variety, toxicity_cheating, core_gameplay, developer_trust, other
3. **Check the data before trusting it.**
   - **Output validation:** every model answer must be valid JSON with an allowed sentiment and only allowed themes. Bad answers are retried up to 3 times, then recorded as failures. They are never guessed or patched.
   - **Failure rate:** the run is flagged `DATA INTEGRITY FAIL` if more than 5% of reviews can't be labeled.
   - **LLM vs Steam agreement:** each reviewer's own thumbs up/down acts as a free sanity check on the model's positive/negative call.
   - **Human audit:** a random, blind sample of 25 reviews per game is hand-labeled. The labeler sees only the review text, with no model answer and no Steam thumbs. `score_audit.py` then reports accuracy and a confusion matrix.
4. **Compare.** For each game and theme, it reports the share of reviews that mention the theme and how negative those reviews are.

## Findings

*Run date: [date]. [N] reviews per game.*

| | Overwatch 2 | Marvel Rivals |
|---|---|---|
| Reviews analyzed | [n] | [n] |
| Review window | [start – end] | [start – end] |
| Median playtime at review | [x] hrs | [x] hrs |
| Steam thumbs-up rate | [x]% | [x]% |
| LLM labeling failure rate | [x]% | [x]% |
| LLM vs Steam agreement | [x]% | [x]% |
| LLM accuracy vs human labels | [x]% | [x]% |

**Top themes** (share of reviews; % of those reviews that are negative):

| Theme | Overwatch 2 | Marvel Rivals |
|---|---|---|
| [theme] | [x]% ([y]% neg) | [x]% ([y]% neg) |

![Theme comparison](output_compare/theme_comparison.png)

**Takeaways:** [2–3 product insights written from the real numbers above]

## Limitations

- **Different review windows.** "Most recent N reviews" covers a different stretch of time for each game, depending on how many reviews each one gets per day. A short window can be dominated by one patch, season launch, or controversy. Check the review window row above before comparing the games directly.
- **Different player tenure.** Overwatch 2's audience includes players who have been around since 2016 (the original Overwatch). Marvel Rivals launched in December 2024. Playtime at review differs, and long-time players complain about different things than newcomers. The two groups of reviewers are not like-for-like.
- **Small sample.** A few hundred reviews per game, and a 50-review human audit, can show broad patterns. Differences of a few percentage points between games may be noise.
- **Hand-defined theme taxonomy.** I chose the themes. Topics that don't fit them fall into `other`, and a different taxonomy would split the results differently.
- **Steam reviewers aren't all players.** Console players (a large share of both games) and players who never write reviews aren't represented. Steam reviews skew toward strong opinions.
- **LLM labels are imperfect.** That's the reason for the human audit. Theme labels are not hand-audited, only sentiment is.

## Run it yourself

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install requests pandas matplotlib openai
```

Set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_DEPLOYMENT` as environment variables (never in a file in this repo). Then:

```bash
python review_compare.py --n 20       # quick test
python review_compare.py --n 200      # full run -> output_compare/
# hand-fill human_sentiment in output_compare/audit_sample.csv, then:
python score_audit.py
```

`review_themes.py` runs the same analysis for one game, e.g. `python review_themes.py --game "Marvel Rivals" --n 200`.

| Output file | What's in it |
|---|---|
| `reviews_classified.csv` | every review with the Steam thumbs, playtime, LLM sentiment, themes, and any error |
| `theme_summary.csv` | per game and theme: share of reviews, % negative |
| `theme_comparison.png` | the chart above |
| `run_summary.json` | run metadata and integrity stats |
| `audit_sample.csv` | blind sample for hand labeling |
