# Next experiment: source-aware evaluation of the neural configurations

Closes the Critical reviewer comment that leave-one-source-out currently runs on the
classical pipeline only. The justification in Section 4.4 is circular: architecture can
be irrelevant under random record-level splits and still matter under distribution
shift, which is exactly what structured knowledge and meta-learning are motivated for.

## Cost

18 runs (3 configurations x 6 held-out collections), roughly 2.5 hours on a single T4.
Compare against the 25.1 GPU-hours already spent on the main experiment.

## Run it

```bash
python code/run_source_holdout.py --prepare      # already done; splits/ is in this package
python code/run_source_holdout.py --run-all      # 18 runs, sequential
python code/run_source_holdout.py --score        # scores whatever has finished
```

`--score` is safe to run at any point, so a partial sweep still yields results. If a run
fails the script reports it and continues rather than aborting the sweep.

## Splits

Already generated and shipped in `splits/`, with `manifest.json` recording each holdout.
Verified: zero item overlap between any train and test file, and every held-out condition
retains at least 31 training records in every class, so no configuration is asked to
predict a class it never saw.

| Held out | Train | Val | Test | Classes in test |
|---|---|---|---|---|
| PURE_and_others | 507 | 89 | 1,071 | 1 |
| software_requirements_extended | 1,215 | 215 | 237 | 4 |
| promise_exp | 1,276 | 226 | 165 | 4 |
| requirements_MoSCoW | 1,326 | 233 | 108 | 1 |
| Curated Mendeley | 1,368 | 242 | 57 | 4 |
| SecReq CPN | 1,394 | 246 | 27 | 3 |

The validation split is carved from the training collections only and stratified by
label, so early stopping never sees the held-out source.

## Reading the result

The script prints macro F1 for the four multi-class holdouts and class recall for the two
single-class ones, and does not aggregate the two — the same convention Table 5 uses.

The number to compare against is **0.607**, the classical pipeline's mean over the
multi-class holdouts.

- If the three configurations land within roughly 0.02 of each other, the architectural
  null survives source shift. That is the strongest single result this paper could add,
  and Section 5 should say so.
- If the full composition separates from encoder-only under source shift while matching
  it under record-level cross-validation, the paper's central claim needs narrowing to
  record-level evaluation, and that finding is worth reporting in its own right.

Either outcome is publishable. The experiment is worth running because it can falsify
the paper's main claim, not because it is expected to confirm it.

## After it finishes

Send me `results/source_holdout_neural.csv` and I will write the section, extend
`verify_paper.py` to cover the new numbers, and adjust Sections 4.4, 5 and 7 to match
whichever way it comes out.
