#!/usr/bin/env bash
#
# reproduce.sh — reproduce the paper.
#
#   ./reproduce.sh verify     recompute every reported number from the released
#                             predictions and compare against the paper  (~1 min, CPU)
#   ./reproduce.sh figures    regenerate all five figures                 (~1 min, CPU)
#   ./reproduce.sh tfidf      re-run the classical baseline from scratch  (~1 min, CPU)
#   ./reproduce.sh corpus     rebuild corpus from source, check byte-identity (~1 min, CPU)
#   ./reproduce.sh train      re-run all 175 transformer training runs    (~25 h, GPU)
#   ./reproduce.sh all        verify + figures + tfidf + corpus
#
# The default target is `verify`. Nothing but `train` needs a GPU, and `train` is
# the only target that takes longer than a couple of minutes.

set -euo pipefail
cd "$(dirname "$0")"

TARGET="${1:-verify}"

banner() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

verify() {
    banner "Verifying every reported number against the released predictions"
    python code/verify_paper.py
}

figures() {
    banner "Regenerating figures"
    python code/make_figures.py --preds predictions/pooled --corpus corpus \
                                --results results --out figures
}

tfidf() {
    banner "Re-running the classical baseline"
    python code/run_tfidf.py --corpus corpus/nfr_corpus.csv \
                             --folds corpus/folds.csv --out predictions/pooled
}

corpus() {
    banner "Rebuilding the corpus from source spreadsheets, then checking byte-identity"
    if ! ls sources/*.xlsx >/dev/null 2>&1; then
        echo "No spreadsheets in sources/. See sources/SOURCES.md."
        exit 1
    fi
    TMP=$(mktemp -d)
    cp -r sources "$TMP/"
    cp code/build_corpus.py "$TMP/"
    ( cd "$TMP" && python build_corpus.py --base sources --build \
                          --min-class 40 --cv 5 --repeats 5 >/dev/null )

    FAIL=0
    for f in nfr_corpus.csv folds.csv; do
        if cmp -s "$TMP/corpus/$f" "corpus/$f"; then
            echo "  IDENTICAL  corpus/$f"
        else
            echo "  DIFFERS    corpus/$f"
            FAIL=1
        fi
    done
    rm -rf "$TMP"
    if [ "$FAIL" -eq 0 ]; then
        echo ""
        echo "The released corpus is exactly what this code produces from these sources."
    else
        echo ""
        echo "Rebuild does not match the released corpus. Report this as a bug."
        exit 1
    fi
}

train() {
    banner "Re-running all transformer training runs (GPU, ~25 hours)"
    echo "This overwrites predictions/folds/. Interrupt and re-run to resume."
    mkdir -p predictions/folds
    for V in roberta_only full no_gcn no_xl no_contrastive no_meta; do
        for REP in 0 1 2 3 4; do
            for FOLD in 0 1 2 3 4; do
                python code/train.py --variant "$V" --seed 42 --rep "$REP" --fold "$FOLD" \
                    --corpus corpus/nfr_corpus.csv --folds corpus/folds.csv \
                    --out predictions/folds/
            done
        done
    done
    for REP in 0 1 2 3 4; do
        for FOLD in 0 1 2 3 4; do
            python code/run_norbert.py --corpus corpus/nfr_corpus.csv \
                --folds corpus/folds.csv --rep "$REP" --fold "$FOLD" \
                --out predictions/folds/ --seeds 42
        done
    done
    banner "Pooling out-of-fold predictions"
    python code/pool_folds.py --preds predictions/folds --out predictions/pooled
    tfidf
    verify
}

case "$TARGET" in
    verify)  verify ;;
    figures) figures ;;
    tfidf)   tfidf ;;
    corpus)  corpus ;;
    train)   train ;;
    all)     verify; figures; tfidf; corpus ;;
    *)       echo "unknown target: $TARGET"; sed -n '3,16p' "$0"; exit 1 ;;
esac
