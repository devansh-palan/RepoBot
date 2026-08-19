"""Eval CLI: `python -m src.eval --validate` / `--score`.

Scoring is retrieval-only and needs no LLM, so it runs in seconds.

The default report slices by split, style, and repo rather than printing one
number, because one number hides the two ways a retrieval "improvement" lies:
scoring well only on the split it was tuned against, and scoring well only on
the question style the set happens to be full of.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pathlib import Path

from src import config
from src.retrieve import RETRIEVERS

from .ablation import CONFIGS, report, run_ablation
from .gold import GoldQuestion, GoldSet, load_gold, summary, validate
from .metrics import mean, recall_at_k, reciprocal_rank

# @1 is "did it nail it", @8 is what the agent actually receives (FINAL_TOP_K),
# @20 is the ceiling fusion has to work with.
CUTOFFS = (1, 5, config.FINAL_TOP_K, 20)
DEPTH = max(CUTOFFS)


def score(
    gold: GoldSet,
    retriever: str,
    questions: Sequence[GoldQuestion],
) -> tuple[dict[int, float], float]:
    """Recall at each cutoff plus MRR, over the questions given."""
    search = RETRIEVERS[retriever]
    rows = [(q, search(gold.repo_path(q), q.question, DEPTH)) for q in questions]
    recalls = {k: mean(recall_at_k(r, q.answers, k) for q, r in rows) for k in CUTOFFS}
    return recalls, mean(reciprocal_rank(r, q.answers) for q, r in rows)


def _table(gold: GoldSet, names: list[str], slices: list[tuple[str, list[GoldQuestion]]]) -> None:
    head = "  ".join(f"R@{k}".rjust(6) for k in CUTOFFS)
    for label, questions in slices:
        if not questions:
            continue
        print(f"\n{label}  (n={len(questions)})")
        print(f"  {'retriever':<10} {head}  {'MRR':>6}")
        for name in names:
            recalls, mrr = score(gold, name, questions)
            cells = "  ".join(f"{recalls[k]:6.3f}" for k in CUTOFFS)
            print(f"  {name:<10} {cells}  {mrr:6.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.eval", description=__doc__)
    parser.add_argument("--validate", action="store_true", help="check every gold label")
    parser.add_argument("--score", action="store_true", help="score the retrievers")
    parser.add_argument("--retriever", choices=[*sorted(RETRIEVERS), "all"], default="all")
    parser.add_argument("--per-question", action="store_true", help="show every question")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="run the four-config ablation and print a markdown report",
    )
    parser.add_argument(
        "--configs",
        default=",".join(CONFIGS),
        help=f"comma-separated subset of {','.join(CONFIGS)}",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="questions per config; the answering configs are minutes each on a local model",
    )
    parser.add_argument("--balanced", action="store_true",
                        help="with --limit, take an equal number of each kind")
    parser.add_argument("--out", type=Path, default=None, help="write the report here too")
    parser.add_argument("--no-cache", action="store_true", help="ignore cached runs")
    args = parser.parse_args()

    gold = load_gold()
    print(summary(gold))

    if args.validate or not args.score:
        problems = validate(gold)
        print(f"\nlabel check: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
        if problems:
            raise SystemExit(1)

    if args.ablate:
        _ablate(gold, args)
        return

    if not args.score:
        return

    names = sorted(RETRIEVERS) if args.retriever == "all" else [args.retriever]
    splits = ("dev", "test") if args.split == "all" else (args.split,)

    slices: list[tuple[str, list[GoldQuestion]]] = []
    for split in splits:
        slices.append((f"[{split}] all", gold.where(split=split)))
        for style in ("semantic", "lexical"):
            slices.append((f"[{split}] {style}", gold.where(split=split, style=style)))
        for repo in sorted(gold.repos):
            slices.append((f"[{split}] repo={repo}", gold.where(split=split, repo=repo)))

    _table(gold, names, slices)

    if args.per_question:
        for name in names:
            print(f"\nper question — {name}")
            rows = [
                (q, RETRIEVERS[name](gold.repo_path(q), q.question, DEPTH))
                for split in splits
                for q in gold.where(split=split)
            ]
            for question, results in sorted(
                rows, key=lambda row: recall_at_k(row[1], row[0].answers, config.FINAL_TOP_K)
            ):
                recall = recall_at_k(results, question.answers, config.FINAL_TOP_K)
                flag = "  <-- check label" if recall == 0.0 else ""
                print(
                    f"  {question.id:<5} {question.split:<4} {question.style:<8} "
                    f"R@{config.FINAL_TOP_K}={recall:.2f} "
                    f"RR={reciprocal_rank(results, question.answers):.2f}  "
                    f"{question.question[:52]}{flag}"
                )


def _select(gold: GoldSet, args) -> list[GoldQuestion]:
    """Pick the questions to run, optionally balanced across kinds.

    Balancing matters when limiting: factual questions are graded by citation
    and explanatory ones by an LLM judge, so an unbalanced sample silently
    changes which instrument produced the headline number.
    """
    pool = list(gold.questions if args.split == "all" else gold.where(split=args.split))
    if args.limit is None:
        return pool
    if not args.balanced:
        return pool[: args.limit]

    # Stratify across kind *and* style. Taking the first N of each kind draws
    # entirely from the dev split, which is 100% semantic, so the style
    # breakdown comes back empty and the report has nothing to say about the
    # case BM25 exists for.
    per_cell = max(1, args.limit // 4)
    chosen: list[GoldQuestion] = []
    for kind in ("factual", "explanatory"):
        for style in ("semantic", "lexical"):
            chosen += [q for q in pool if q.kind == kind and q.style == style][:per_cell]
    return chosen


def _ablate(gold: GoldSet, args) -> None:
    questions = _select(gold, args)
    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown config(s) {unknown}; known: {list(CONFIGS)}")

    cache = None if args.no_cache else config.EVAL_DIR / "runs"
    print(f"\nablation: {len(questions)} questions x {len(names)} configs", flush=True)

    results = run_ablation(
        gold,
        questions,
        names,
        cache=cache,
        on_progress=lambda name: print(f"  running {name}...", flush=True),
    )

    text = report(results, gold, questions)
    print("\n" + text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
