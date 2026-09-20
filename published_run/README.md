# Published run

Outputs of the run reported in *Electronics* **2025**, *14*, 4199, kept
unchanged for reference.

| File | Backs |
|---|---|
| `vlm_experiment_results.csv` | Tables 1–5, Figure 6 (126 narratives × 3 judges) |
| `Results_revisions.txt` | Table 6 (§4.8), §4.9 transition scores, Table 7 (§4.10) |
| `vlm_narratives_cache.pkl` | the 126 narratives as evaluated |
| `vlm_evaluations.json` | direct-VLM judge cache from that run |

Read these as the record of that run, not as a target to reproduce — see
"Notes" in the top-level README. The `narrative_maps` rows are repeated
evaluations of one extraction per baseline. The Qwen scores in
`Results_revisions.txt` were parsed from free text before the shared output
schema was in place. `vlm_evaluations.json` predates per-judge cache keys and is
not read by the current code.
