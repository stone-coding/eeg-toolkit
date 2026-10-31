# Analysis: `run` and `report`

## `eegkit run`

```bash
eegkit run -c configs/default.yaml [-s 1,2] [--force]
```

For every subject: load (BIDS if present, else raw EDF) -> preprocess -> epoch -> reject ->
band power, PSD, ERD/ERS, CSP+LDA decoding -> per-subject tables and figures. Then the group
level: mean ERD/ERS maps, mean PSD, geometric-mean band-power topomaps, decoding table.

```
sub-001  ok     epochs=45  dropped=0   acc=0.78 (cached: epochs,analysis)
...
Group: n=10  decoding 0.624 ± 0.169  -> outputs/group
```

**Caching.** There are two stages, each storing a hash of everything it depends on:

1. *epochs* - preprocess config, epoch config, runs, input file names and sizes;
2. *analysis* - the epochs hash plus the analysis config.

Rerunning skips a stage whose hash is unchanged: change only `analysis.bands` and just the
analysis is recomputed; change a filter and everything is. `--force` ignores the cache.

**Failures** are isolated per subject: a subject that fails is recorded in `status.json` with
the error, the rest continue, and the exit code is 1.

## `eegkit report`

```bash
eegkit report --subject 2
```

Writes `outputs/subjects/sub-002/report.md` - quality control (bad channels, dropped epochs),
decoding accuracy, mu-band ERD table, and the three figures.
