#!/usr/bin/env python3
"""Revision experiments for Sections 4.8-4.10 of the paper.

This is the script that produced `Results_revisions.txt`, and hence Tables 6
and 7 and the transition-score figures reported in the paper. It reuses the
cached GPT-4o captions in `vlm_cache/captions/` and the narratives pickled in
`vlm_narratives_cache.pkl`, so it does not re-extract anything.

Three experiments, in order:

  1. Breaking-point identification (Section 4.9)
     GPT-4o scores every consecutive image-pair transition on a 1-10 scale via
     the `evaluate_transitions` function schema. Sampled over 6 human, 30
     Narrative Maps and 30 random narratives (66 total).

  2. Qwen2.5-VL comparison (Section 4.8, Table 6)
     Runs `qwen2.5vl:7b` locally through Ollama in both caption-based and
     direct-vision modes, over the same 66 narratives. Qwen receives the same
     Figure 3 prompt and the same output schema as the GPT-4o judges, so the
     comparison isolates the model rather than the prompt.

  3. In-context learning ablation (Section 4.10, Table 7)
     GPT-4o with and without a high-quality and a low-quality anchor example.
     Both arms use the same Figure 3 prompt and differ only by the anchor
     block. The two narratives used as examples are excluded from scoring,
     which is why n = 55 rather than 66.

All coherence scoring in this script goes through `build_evaluation_prompt`
and the `evaluate_coherence` schema, so the prompt is defined once.

Requires OPENAI_API_KEY in the environment for experiments 1 and 3, and a
running Ollama daemon with `qwen2.5vl:7b` pulled for experiment 2.

Usage:
    python vlm_revision_experiments.py [data_dir]     # data_dir defaults to "."
"""

import json
import time
import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI
import pickle
import pandas as pd
import ollama

client = OpenAI()

QWEN_MODEL = "qwen2.5vl:7b"

# Figure 4 of the paper: the expedition context prepended to every evaluation.
DATASET_CONTEXT = """These are historical photographs from Robert Gerstmann's 1928 Sacambaya Expedition archive. The expedition was a five-month treasure-hunting venture (March-November 1928) searching for alleged Jesuit treasure in Bolivia's Sacambaya Valley. The 500 photographs document the complete journey including: maritime voyage from Europe to South America, overland travel through Bolivia, and excavation activities at various sites. Images capture expedition members, transportation modes, equipment, landscapes, and the systematic search efforts in the Bolivian mountains."""

# Figure 5 of the paper. Shared by the OpenAI judges (as a tool) and the Qwen
# judges (as an Ollama structured-output schema) so that every model in the
# comparison is constrained to the same output.
COHERENCE_FUNCTION = {
    "name": "evaluate_coherence",
    "description": "Rate narrative coherence from 1-10",
    "parameters": {
        "type": "object",
        "properties": {
            "coherence_score": {
                "type": "integer",
                "description": "Coherence score (1-10)",
                "minimum": 1,
                "maximum": 10,
            }
        },
        "required": ["coherence_score"],
    },
}
COHERENCE_TOOL = {"type": "function", "function": COHERENCE_FUNCTION}
COHERENCE_TOOL_CHOICE = {"type": "function", "function": {"name": COHERENCE_FUNCTION["name"]}}
COHERENCE_SCHEMA = COHERENCE_FUNCTION["parameters"]


def build_evaluation_prompt(captions=None, examples=None):
    """The Figure 3 evaluation prompt.

    Every coherence-scoring call in this script and in VLM-as-a-Judge.ipynb now
    uses this one prompt, so that the GPT-4o and Qwen arms of Section 4.8, and
    the two arms of Section 4.10, differ only in the variable under test.
    Previously each experiment carried its own ad-hoc wording, which confounded
    the model comparison with a prompt comparison.

    Args:
        captions: caption strings to embed, or None for the direct-vision
            variant, where the images are attached to the message instead.
        examples: optional (good_example, bad_example) caption lists, prepended
            as scored anchors for the in-context-learning condition.
    """
    prompt = f"Context: {DATASET_CONTEXT}\n\n"

    if examples is not None:
        good_example, bad_example = examples
        good = chr(10).join(f"{i + 1}. {c}" for i, c in enumerate(good_example[:5]))
        bad = chr(10).join(f"{i + 1}. {c}" for i, c in enumerate(bad_example[:5]))
        prompt += (
            f"Example of a GOOD narrative (Score: 8):\n{good}\n"
            "This shows clear progression with logical transitions.\n\n"
            f"Example of a POOR narrative (Score: 3):\n{bad}\n"
            "This shows disconnected scenes without logical flow.\n\n"
        )

    prompt += """Evaluate how well this sequence of images forms a coherent narrative.

Specifically check for:
- Temporal consistency: Do events follow a logical time sequence?
- Spatial continuity: Do locations transition naturally?
- Causal relationships: Does each image logically follow from the previous?
- Activity coherence: Do the activities shown form a sensible progression?

Specifically deduct points for:
- Abrupt location changes without transition.
- Time sequence violations (e.g., arriving before departing).
- Repeated similar scenes that don't advance the story.
- Missing key narrative steps between major transitions.

Score 1-3: Multiple violations, no discernible story thread.
Score 4-6: Some connections but significant gaps or illogical jumps.
Score 7-9: Mostly coherent with minor issues.
Score 10: Perfect narrative flow with clear progression.

Be critical - most random sequences should score 1-4.
"""

    if captions is not None:
        sequence = chr(10).join(f"Image {i + 1}: {c}" for i, c in enumerate(captions))
        prompt += f"\nSequence:\n{sequence}\n"

    prompt += "\nRate the coherence (1-10):"
    return prompt


_LABELLED_SCORE = re.compile(
    r"(?:score|rating|coherence)\D{0,12}?(10|[1-9])(?!\d)|(10|[1-9])(?!\d)\s*(?:/|out\s+of)\s*10",
    re.IGNORECASE,
)
# "1-10", "1 to 10", "1 through 10": the scale echoed back from the prompt.
_SCALE_ECHO = re.compile(r"\b1\s*(?:-|–|—|to|through)\s*10\b", re.IGNORECASE)


def parse_coherence_score(text):
    """Recover a 1-10 score from an unconstrained reply.

    The previous implementation took the *first* integer anywhere in the text
    via `re.findall(r'\\b([1-9]|10)\\b', text)[0]`. Because the prompt itself
    contains the string "1-10", any reply that echoed the scale ("On a scale of
    1-10, I would say 7") parsed as 1. This strips the scale echo first, then
    prefers an explicitly labelled score, and only then falls back to the last
    standalone integer in range - the last, not the first, because the prompt
    asks for the score at the end of the reply.

    Returns None when no valid score can be recovered, so the caller can retry
    rather than record a spurious 1.
    """
    if not text:
        return None
    cleaned = _SCALE_ECHO.sub(" ", text)

    matches = [m for m in _LABELLED_SCORE.finditer(cleaned)]
    if matches:
        last = matches[-1]
        value = int(last.group(1) or last.group(2))
        if 1 <= value <= 10:
            return float(value)

    # Not preceded by a digit or decimal point, not followed by another digit,
    # and not the integer part of a decimal. A trailing sentence period is fine,
    # so "the answer is 1." still parses.
    standalone = re.findall(r"(?<![\d.])(10|[1-9])(?!\d)(?!\.\d)", cleaned)
    if standalone:
        value = int(standalone[-1])
        if 1 <= value <= 10:
            return float(value)
    return None


def _openai_score(response):
    """Read the coherence score out of a constrained tool call."""
    message = response.choices[0].message
    if not getattr(message, "tool_calls", None):
        raise RuntimeError("model returned no evaluate_coherence tool call")
    return float(json.loads(message.tool_calls[0].function.arguments)["coherence_score"])


def _ollama_score(messages, max_retries=3):
    """Query Qwen through Ollama with structured output, falling back to text.

    Ollama supports a JSON-schema `format` on recent versions; when it is not
    available the call is retried without it and the reply is parsed with
    `parse_coherence_score`.
    """
    for attempt in range(max_retries):
        for use_schema in (True, False):
            try:
                kwargs = {"model": QWEN_MODEL, "messages": messages}
                if use_schema:
                    kwargs["format"] = COHERENCE_SCHEMA
                response = ollama.chat(**kwargs)
            except Exception as exc:
                if use_schema:
                    continue  # server too old for schema-constrained output
                print(f"  Ollama call failed: {exc}")
                break

            text = (response.get("message") or {}).get("content", "").strip()
            if use_schema:
                try:
                    score = float(json.loads(text)["coherence_score"])
                    if 1 <= score <= 10:
                        return score
                except (ValueError, KeyError, TypeError):
                    pass  # fall through to free-text parsing
            score = parse_coherence_score(text)
            if score is not None:
                return score

        if attempt < max_retries - 1:
            time.sleep(1)
    return None


@dataclass
class ImageNarrative:
    """Represents a sequence of images forming a narrative."""
    id: str
    source: str
    image_ids: List[str]
    image_paths: List[str]
    coherence_scores: Optional[Dict[str, float]] = None

def load_captions_from_cache(image_paths: List[str], cache_dir: str = "vlm_cache/captions") -> List[str]:
    """Load captions from cache."""
    captions = []
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        raise FileNotFoundError(f"Caption cache directory not found: {cache_dir}")
    
    for img_path in image_paths:
        img_name = Path(img_path).name
        caption_file = cache_path / f"{img_name}.txt"
        
        if caption_file.exists():
            caption = caption_file.read_text(encoding='utf-8').strip()
            captions.append(caption)
        else:
            raise FileNotFoundError(f"Caption not found for {img_name} at {caption_file}")
    
    return captions

def load_actual_roger_data(data_dir: str = ".") -> Tuple[List[ImageNarrative], pd.DataFrame, Dict[str, List[str]]]:
    """Load actual ROGER dataset."""
    print("Loading actual ROGER dataset...")
    
    def resolve(name):
        """Prefer a local run's output, falling back to the published run."""
        local = Path(data_dir) / name
        if local.exists():
            return local
        archived = Path(data_dir) / "published_run" / name
        if archived.exists():
            print(f"  {name}: using {archived} (run the notebook to produce your own)")
            return archived
        raise FileNotFoundError(f"{name} not found in {data_dir} or {data_dir}/published_run")

    pickle_path = resolve("vlm_narratives_cache.pkl")
    
    with open(pickle_path, 'rb') as f:
        cache_data = pickle.load(f)
    
    narratives = cache_data['narratives']
    
    csv_path = resolve("vlm_experiment_results.csv")
    
    results_df = pd.read_csv(csv_path)
    
    caption_map = {}
    caption_cache_dir = Path(data_dir) / "vlm_cache" / "captions"
    
    print(f"Loading captions from {caption_cache_dir}...")
    
    for narrative in narratives:
        captions = load_captions_from_cache(narrative.image_paths, str(caption_cache_dir))
        caption_map[narrative.id] = captions
    
    print(f"Loaded {len(narratives)} narratives with captions:")
    print(f"  Human: {len([n for n in narratives if n.source == 'human'])}")
    print(f"  Narrative Maps: {len([n for n in narratives if n.source == 'narrative_maps'])}")
    print(f"  Random: {len([n for n in narratives if n.source == 'random'])}")
    
    return narratives, results_df, caption_map

# ============================================================================
# EXPERIMENT 1: Breaking Points (NEW - for reviewers)
# ============================================================================

def identify_breaking_points(narrative_captions: List[str], narrative_id: str) -> Dict:
    """Identify weak transitions using CRITICAL evaluation."""
    
    function_schema = {
        "name": "evaluate_transitions",
        "description": "Evaluate narrative transitions and identify weak points",
        "parameters": {
            "type": "object",
            "properties": {
                "transition_scores": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1, "maximum": 10},
                    "description": "Score for each transition between consecutive images"
                },
                "weakest_index": {
                    "type": "integer",
                    "description": "Index of the first image in the weakest transition (1-based)"
                },
                "weakest_transition": {"type": "string"},
                "reason": {"type": "string"}
            },
            "required": ["transition_scores", "weakest_index", "weakest_transition", "reason"]
        }
    }
    
    prompt = f"""Context: {DATASET_CONTEXT}

Analyze transitions in this narrative sequence:
{chr(10).join([f"{i+1}. {caption}" for i, caption in enumerate(narrative_captions)])}

BE CRITICAL. Score each transition (1-10):
- 8-10: Seamless progression with clear causal/thematic/temporal connection
- 5-7: Adequate connection but missing clear progression or some gaps
- 3-4: Weak connection, thematic relation only, jarring shifts
- 1-2: Disconnected, abrupt changes, breaks narrative flow

Most randomly ordered sequences should score 2-5 per transition.
Only truly coherent narratives should average above 6.

Identify the single most disruptive transition."""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        tools=[{"type": "function", "function": function_schema}],
        tool_choice={"type": "function", "function": {"name": "evaluate_transitions"}},
        temperature=0.3
    )

    result = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    result['narrative_id'] = narrative_id
    return result

# ============================================================================
# EXPERIMENT 2: Qwen Comparison (NEW - for reviewers)
# Uses SIMPLE prompt adapted for Qwen
# ============================================================================

def load_image(image_path: str) -> bytes:
    """Load image bytes."""
    with open(image_path, 'rb') as f:
        return f.read()

def evaluate_qwen_caption_based(narrative_captions: List[str], max_retries: int = 3) -> Optional[float]:
    """Qwen caption-based evaluation, using the same prompt as the GPT-4o judges."""
    prompt = build_evaluation_prompt(captions=narrative_captions)
    return _ollama_score([{'role': 'user', 'content': prompt}], max_retries=max_retries)

def evaluate_qwen_direct_vlm(image_paths: List[str], max_retries: int = 3) -> Optional[float]:
    """Qwen direct-vision evaluation, using the same prompt as the GPT-4o judges."""
    images_bytes = []
    for path in image_paths:
        try:
            images_bytes.append(load_image(path))
        except Exception as exc:
            print(f"Could not load {path}: {exc}")

    if not images_bytes:
        return None

    # captions=None: the images are attached to the message instead of being
    # described in the prompt, exactly as in the direct-vision GPT-4o judge.
    prompt = build_evaluation_prompt(captions=None)
    messages = [{'role': 'user', 'content': prompt, 'images': images_bytes}]
    return _ollama_score(messages, max_retries=max_retries)

# ============================================================================
# EXPERIMENT 3: In-Context Learning (NEW - for reviewers)
# Uses standard evaluation prompt from original paper
# ============================================================================

def evaluate_with_examples(narrative_captions: List[str],
                          good_example: List[str],
                          bad_example: List[str]) -> Optional[float]:
    """In-context condition: the Figure 3 prompt plus two scored anchors."""
    prompt = build_evaluation_prompt(
        captions=narrative_captions,
        examples=(good_example, bad_example),
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        tools=[COHERENCE_TOOL],
        tool_choice=COHERENCE_TOOL_CHOICE,
        temperature=0.7,
    )
    return _openai_score(response)

def evaluate_without_examples(narrative_captions: List[str]) -> Optional[float]:
    """Control condition: the identical Figure 3 prompt, anchors omitted.

    The two conditions now differ *only* by the anchor block, which is the
    comparison Section 4.10 is meant to make.
    """
    prompt = build_evaluation_prompt(captions=narrative_captions)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        tools=[COHERENCE_TOOL],
        tool_choice=COHERENCE_TOOL_CHOICE,
        temperature=0.7,
    )
    return _openai_score(response)

# ============================================================================
# Main Experiment Runner
# ============================================================================

def main(data_dir: str = "."):
    """Run all three new experiments requested by reviewers."""
    
    print("="*60)
    print("VLM PAPER REVISION - EXPERIMENTS")
    print("="*60)
    
    narratives, results_df, caption_map = load_actual_roger_data(data_dir)
    
    # Get all human baselines
    human_narratives = [n for n in narratives if n.source == 'human']
    
    # EXPERIMENT 1: Breaking Points
    print("\n[1/3] Breaking Points Identification")
    print("-"*60)
    
    bp_results = {'human': [], 'narrative_maps': [], 'random': []}
    
    for human_narr in human_narratives:
        human_len = len(human_narr.image_ids)
        baseline_label = human_narr.id  # e.g., "30a", "25b", etc.
        
        # Test human
        captions = caption_map[human_narr.id]
        result = identify_breaking_points(captions, human_narr.id)
        bp_results['human'].append(result)
        print(f"{'human':15} {human_narr.id:20} Scores:{result['transition_scores']}")
        
        # Test 5 NM and 5 random of same length
        # Match by baseline label in ID (nm_<baseline>_rep<n>)
        nm_matching = [n for n in narratives 
                      if n.source == 'narrative_maps' and 
                      n.id.startswith(f'nm_{baseline_label}_')][:5]
        
        # Fallback: if no exact baseline match, try length matching
        if not nm_matching:
            nm_matching = [n for n in narratives 
                          if n.source == 'narrative_maps' and len(n.image_ids) == human_len][:5]
        
        random_matching = [n for n in narratives 
                          if n.source == 'random' and len(n.image_ids) == human_len][:5]
        
        for nm_narr in nm_matching:
            captions = caption_map[nm_narr.id]
            result = identify_breaking_points(captions, nm_narr.id)
            bp_results['narrative_maps'].append(result)
            print(f"{'narrative_maps':15} {nm_narr.id:20} Scores:{result['transition_scores']}")
        
        for rand_narr in random_matching:
            captions = caption_map[rand_narr.id]
            result = identify_breaking_points(captions, rand_narr.id)
            bp_results['random'].append(result)
            print(f"{'random':15} {rand_narr.id:20} Scores:{result['transition_scores']}")
    
    print("\nBreaking Points Summary:")
    for source in ['human', 'narrative_maps', 'random']:
        all_scores = []
        for r in bp_results[source]:
            all_scores.extend(r['transition_scores'])
        if all_scores:
            print(f"  {source:15}: {np.mean(all_scores):.1f} ± {np.std(all_scores):.1f}")
    
    # EXPERIMENT 2: Qwen Comparison
    print("\n[2/3] Qwen Caption vs Direct VLM")
    print("-"*60)
    
    qwen_results = {'caption': {'human': [], 'narrative_maps': [], 'random': []},
                    'direct': {'human': [], 'narrative_maps': [], 'random': []}}
    
    for human_narr in human_narratives:
        human_len = len(human_narr.image_ids)
        baseline_label = human_narr.id
        print(f"\n--- Length {human_len} ({baseline_label}) ---")
        
        # Test human
        print(f"Human: {human_narr.id}")
        captions = caption_map[human_narr.id]
        caption_score = evaluate_qwen_caption_based(captions)
        direct_score = evaluate_qwen_direct_vlm(human_narr.image_paths)
        if caption_score:
            qwen_results['caption']['human'].append(caption_score)
            print(f"  Caption: {caption_score}")
        if direct_score:
            qwen_results['direct']['human'].append(direct_score)
            print(f"  Direct: {direct_score}")
        
        # Test 5 NM and 5 random - match by baseline label
        nm_matching = [n for n in narratives 
                      if n.source == 'narrative_maps' and 
                      n.id.startswith(f'nm_{baseline_label}_')][:5]
        
        # Fallback
        if not nm_matching:
            nm_matching = [n for n in narratives 
                          if n.source == 'narrative_maps' and len(n.image_ids) == human_len][:5]
        
        random_matching = [n for n in narratives 
                          if n.source == 'random' and len(n.image_ids) == human_len][:5]
        
        for nm_narr in nm_matching:
            print(f"NM: {nm_narr.id}")
            captions = caption_map[nm_narr.id]
            caption_score = evaluate_qwen_caption_based(captions)
            direct_score = evaluate_qwen_direct_vlm(nm_narr.image_paths)
            if caption_score:
                qwen_results['caption']['narrative_maps'].append(caption_score)
                print(f"  Caption: {caption_score}")
            if direct_score:
                qwen_results['direct']['narrative_maps'].append(direct_score)
                print(f"  Direct: {direct_score}")
        
        for rand_narr in random_matching:
            print(f"Random: {rand_narr.id}")
            captions = caption_map[rand_narr.id]
            caption_score = evaluate_qwen_caption_based(captions)
            direct_score = evaluate_qwen_direct_vlm(rand_narr.image_paths)
            if caption_score:
                qwen_results['caption']['random'].append(caption_score)
                print(f"  Caption: {caption_score}")
            if direct_score:
                qwen_results['direct']['random'].append(direct_score)
                print(f"  Direct: {direct_score}")
    
    print("\nQwen Summary:")
    for method in ['caption', 'direct']:
        print(f"\n{method.upper()}:")
        for source in ['human', 'narrative_maps', 'random']:
            scores = qwen_results[method][source]
            if scores:
                print(f"  {source:15}: {np.mean(scores):.2f} ± {np.std(scores):.2f} (n={len(scores)})")
    
    # EXPERIMENT 3: In-Context Learning
    print("\n[3/3] In-Context Learning")
    print("-"*60)
    
    example_human = human_narratives[0]
    example_random = [n for n in narratives if n.source == 'random'][-1]
    good_example = caption_map[example_human.id]
    bad_example = caption_map[example_random.id]
    
    print(f"Using examples: Good={example_human.id}, Bad={example_random.id}\n")
    
    icl_results = {'human': {'with': [], 'without': []},
                   'narrative_maps': {'with': [], 'without': []},
                   'random': {'with': [], 'without': []}}
    
    for human_narr in human_narratives:
        if human_narr.id == example_human.id:
            continue
        
        human_len = len(human_narr.image_ids)
        baseline_label = human_narr.id
        
        # Test human
        captions = caption_map[human_narr.id]
        with_ex = evaluate_with_examples(captions, good_example, bad_example)
        without_ex = evaluate_without_examples(captions)
        if with_ex and without_ex:
            icl_results['human']['with'].append(with_ex)
            icl_results['human']['without'].append(without_ex)
            print(f"{'human':15} {human_narr.id:20} With:{with_ex} Without:{without_ex}")
        
        # Test 5 NM and 5 random - match by baseline label
        nm_matching = [n for n in narratives 
                      if n.source == 'narrative_maps' and 
                      n.id.startswith(f'nm_{baseline_label}_')][:5]
        
        # Fallback
        if not nm_matching:
            nm_matching = [n for n in narratives 
                          if n.source == 'narrative_maps' and len(n.image_ids) == human_len][:5]
        
        random_matching = [n for n in narratives 
                          if n.source == 'random' and len(n.image_ids) == human_len 
                          and n.id != example_random.id][:5]
        
        for nm_narr in nm_matching:
            captions = caption_map[nm_narr.id]
            with_ex = evaluate_with_examples(captions, good_example, bad_example)
            without_ex = evaluate_without_examples(captions)
            if with_ex and without_ex:
                icl_results['narrative_maps']['with'].append(with_ex)
                icl_results['narrative_maps']['without'].append(without_ex)
                print(f"{'narrative_maps':15} {nm_narr.id:20} With:{with_ex} Without:{without_ex}")
        
        for rand_narr in random_matching:
            captions = caption_map[rand_narr.id]
            with_ex = evaluate_with_examples(captions, good_example, bad_example)
            without_ex = evaluate_without_examples(captions)
            if with_ex and without_ex:
                icl_results['random']['with'].append(with_ex)
                icl_results['random']['without'].append(without_ex)
                print(f"{'random':15} {rand_narr.id:20} With:{with_ex} Without:{without_ex}")
    
    print("\nIn-Context Learning Summary:")
    for source in ['human', 'narrative_maps', 'random']:
        with_scores = icl_results[source]['with']
        without_scores = icl_results[source]['without']
        if with_scores and without_scores:
            print(f"\n{source.upper()}:")
            print(f"  With:    {np.mean(with_scores):.2f} ± {np.std(with_scores):.2f} (n={len(with_scores)})")
            print(f"  Without: {np.mean(without_scores):.2f} ± {np.std(without_scores):.2f} (n={len(without_scores)})")
            print(f"  STD reduction: {np.std(without_scores) - np.std(with_scores):.3f}")
    
    print("\n" + "="*60)
    print("Experiments complete!")
    print("="*60)

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    main(data_dir)
