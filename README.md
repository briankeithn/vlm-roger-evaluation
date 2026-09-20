# VLM-as-a-Judge Evaluation for the ROGER Data Set

Code and data for evaluating visual narrative coherence with vision-language
models as judges: a caption-based path (images → GPT-4o captions → LLM judge)
and a direct-vision path (image sequence → VLM judge), both scored against an
embedding-based coherence metric, over 126 narratives from the ROGER archive of
Robert Gerstmann's 1928 Sacambaya Expedition.

Research code, tidied for release but otherwise provided AS-IS. 
The code should reproduce the experiments in the paper, but it is not a library
or a maintained tool.

## Citation

If you use this code or data, please cite the paper:

> Keith, B.; Meneses, C.; Matus, M.; Castro, M.C.; Urrutia, D.
> **VLM-as-a-Judge Approaches for Evaluating Visual Narrative Coherence in
> Historical Photographical Records.**
> *Electronics* **2025**, *14*, 4199.
> https://doi.org/10.3390/electronics14214199

[CITATION.cff](CITATION.cff) carries the same in machine-readable form, so
GitHub's "Cite this repository" will produce BibTeX or APA for you.

Much of what this repository reuses comes from our previous semi-supervised extraction
study. Please cite it too if you build on the extraction or the ROGER data:

> German, F.; Keith, B.; Matus, M.; Urrutia, D.; Meneses, C.
> **Semi-Supervised Image-Based Narrative Extraction: A Case Study with
> Historical Photographic Records.**
> In *Advances in Information Retrieval* (ECIR 2025), Lecture Notes in Computer
> Science; Springer: Cham, 2025; pp. 248–262.
> https://doi.org/10.1007/978-3-031-88711-6_16
> Preprint: arXiv:[2501.09884](https://arxiv.org/abs/2501.09884).
> Code and data: [faustogerman/ROGER-Concept-Narratives](https://github.com/faustogerman/ROGER-Concept-Narratives).

`library/narrative_maps.py` and `data/` derive from that repository; see
[NOTICE](NOTICE). Also relevant:

- **The text-domain counterpart of this study** - Keith, B.
  *LLM-as-a-Judge Approaches as Proxies for Mathematical Coherence in Narrative
  Extraction.* *Electronics* **2025**, *14*(13), 2735.
  https://doi.org/10.3390/electronics14132735
- **The extraction algorithm** - Keith Norambuena, B.F.; Mitra, T.
  *Narrative Maps: An Algorithmic Approach to Represent and Extract Information
  Narratives.* *Proc. ACM Hum.-Comput. Interact.* **2021**, *4*(CSCW3),
  Article 228, pp. 1–33. https://doi.org/10.1145/3432927

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.13
pip install -r requirements.txt
export OPENAI_API_KEY="..."
ollama pull qwen2.5vl:7b        # only for the Qwen comparison (§4.8)
```

## Running

Run `VLM-as-a-Judge.ipynb` top to bottom for the main experiment (Tables 1–5,
Figure 6), then `python vlm_revision_experiments.py` for §4.8–4.10. The original
run was 756 judge calls plus 501 caption calls, about an hour. Cached embeddings
and captions are reused, so a re-run pays only for the judging; delete a cache
to force it.

`python reconstruct_coherence.py` recomputes the coherence columns from the
archived embeddings and paths with no API calls.

## Layout

```
VLM-as-a-Judge.ipynb          Main experiment: extraction, both judges, analysis
vlm_revision_experiments.py   §4.8–4.10: Qwen2.5-VL, breaking points, in-context
reconstruct_coherence.py      Coherence columns, offline
library/narrative_maps.py     Narrative Maps linear program (see NOTICE)
data/                         501 photographs + ground-truth storylines
roger_embeddings.npy          Cached CLIP embeddings (clip-vit-base-patch32)
nm_paths_cache.json           Cached extractions
vlm_cache/captions/           501 cached GPT-4o captions
published_run/                Outputs of the run reported in the paper
```

The photographs are committed in full so the repository is self-contained and
citable.

## Notes

- **Coherence** is Equation (1), `sqrt(S * T)`, on the unrescaled angular
  similarity. The membership matrix has one column, so `T = 1` throughout and
  the metric reduces to `sqrt(S)`. Under this convention Table 1's `Min Coh.`
  column reads **0.855 / 0.851 / 0.839** (human / Narrative Maps / random).
  Table 2's correlations are unchanged. `calculate_coherence_scores` documents
  the reasoning.
- **Replications** repeat one extraction per baseline rather than drawing a new
  one, which balances sample size against the random condition and measures
  judge variance on a fixed stimulus; treat those rows as repeated measures.
  `NarrativeMapsAdapter(vary_replications=True)` samples the extractor instead.
- **Re-running** will not match the published tables line for line: prompts,
  seeds and the metric are all pinned here. The original outputs are kept in
  `published_run/`.

## Attribution

The Narrative Maps implementation, the ROGER photographs and the ground-truth
file come from [faustogerman/ROGER-Concept-Narratives](https://github.com/faustogerman/ROGER-Concept-Narratives)
(MIT, © 2025 Fausto German). Full attribution and the list of modifications are
in [NOTICE](NOTICE).

We thank the *Robert Gerstmann Fonds* at Universidad Católica del Norte for
access to the photographic archive.

## Funding

Agencia Nacional de Investigación y Desarrollo (ANID) - FONDEF ID25I10169 and
FONDECYT de Iniciación 11250039.

## Licence

MIT, see [LICENSE](LICENSE).
