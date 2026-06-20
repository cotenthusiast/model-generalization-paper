# Handoff — 2026-06-21

## Current HEAD

`main` @ `7029e5d`, pushed to `origin/main` (`88a3a85..7029e5d`). Two commits
landed this session, in this order:

1. `a4d91e4` — "Fix leading-letter and span-isolation gaps in abcd/text_extraction matching"
2. `7029e5d` — "Redesign additional_option (Eq.6) and cyclic (Eq.1) for paper fidelity"

Full test suite: `PYTHONPATH=src python -m pytest -q` → **391 passed, 0 failed**
as of `7029e5d`. Re-run this yourself before trusting that number if more time
has passed — don't carry it forward as fact across sessions without checking.

---

## 1. additional_option / cyclic redesign — code done, NO run data exists yet

Both methods were rewritten to use `score_options()` instead of
`generate()`+text-matching, to match their cited papers' actual mechanisms
(verified term-by-term against the papers, not from memory):

- **`additional_option`** (`src/modelgen/runners/additional_option.py`):
  now scores the full option set (real options + "I don't know") in one
  `score_options()` call, then `match_options_via_scoring` applies Eq.(6) of
  Choi et al. 2025: argmax restricted to the real options only, with IDK
  structurally excluded from ever being the final answer regardless of its
  probability mass. The old free-text+Jaccard function
  (`match_text_to_options_jaccard`) is kept in the same file, relabeled
  LEGACY, only used by `rematch_additional_option_rows` to re-evaluate rows
  collected before this redesign (those have `raw_text` populated; new rows
  don't).
- **`cyclic`** (`src/modelgen/runners/permutation.py`): now scores each of
  the N cyclic permutations with one `score_options()` call each, then
  combines them via Eq.(1) of Zheng et al. 2024 (probability averaging
  across permutations, un-permuted back to canonical content identity) using
  `equation1_cyclic_debiased_content_probs` in `pride_debias.py` — this
  function already existed (dead code, unused by anything) and was verified
  correct against the rotation formula in `_generate_permutations` before
  wiring it in. The old majority-vote helpers (`_unpermute_choice`,
  `_majority_vote`) are still in the file, just unused by the active
  `run_one`.

**SLURM scripts are ready but nothing has been run.** `slurm/13_redo_aoi_cyclic_qwen7b.sh`
through `slurm/17_redo_aoi_cyclic_llama70b.sh` (GPU-tiered: MIG slice for
7B/8B, single A100 for 32B, 2×A100 for 72B/70B), backed by 10 new
`config/<model>_<benchmark>_redo_aoi_cyclic.yaml` files. **Nothing has been
queued on Kelvin2. No `runs/` data exists for either redesigned method under
the new mechanism.** The next real step is `git pull` on Kelvin2 and
`sbatch` these 5 scripts.

Old majority-vote `cyclic` run data (12 CSVs) is archived, not deleted, at
`runs_archive/cyclic_majority_vote_pre_eq1_redesign_20260621/` — see the
`README.md` there.

Verification performed this session (no real model, no GPU): 12 real
ARC-Challenge questions run through both new mechanisms with hand-designed
`DummyBackend` scores (content-aware and pure-token-bias variants) — see
commit `7029e5d`'s message for the specific results (12/12 correct under
zero-bias scoring for both methods, pure letter-bias debiased to exact
uniform by `cyclic`'s Eq.(1), IDK never selected by `additional_option` even
when it had the highest raw probability of any shown option). This is
algebraic/logic verification, not a real-model accuracy check — that only
happens once the SLURM jobs actually run.

---

## 2. CORRECTION: the "Nowak et al. 2026 is fabricated" conclusion was wrong

Earlier this session (before commit `a4d91e4`), the citation
`"Nowak et al. 2026"` attached to `abcd` was investigated, a WebSearch came
back empty, and the conclusion was: **no such paper exists, it's fabricated**.
Acting on that, `a4d91e4` removed the citation from `abcd.py`/`text_extraction.py`
and reframed `abcd` as "this codebase's own design... not a verified
reproduction of a specific published procedure."

**That conclusion was wrong.** The paper is real:

> Mateusz Nowak, Xavier Cadet, Peter Chin. **"ABCD: All Biases Come
> Disguised"** (Dartmouth). arXiv:2602.17445.

Verified directly against the arXiv API this session (`export.arxiv.org/api/query?search_query=ti:"All Biases Come Disguised"`)
— title, all three author names, and the abstract match exactly: *"a simple
bias-reduced evaluation protocol that replaces the labels of each question
with uniform, unordered labels."* That's precisely what this repo's `abcd`
method does (dash labels instead of A/B/C/D). The arXiv ID prefix (`2602`)
means it was published February 2026 — after this assistant's training
cutoff and not well-indexed by the general web searches run earlier in the
session, which is almost certainly why the first search came back empty and
got treated as "doesn't exist" rather than "couldn't find it."

**This wrong conclusion is now baked into already-pushed commit `a4d91e4`**
(its commit message says the citation "does appear fabricated") and into the
in-code docstrings it touched (`abcd.py`'s class docstring, `text_extraction.py`'s
`_ABCD_EMBEDDING_MODEL` comment and `extract_final_answer_span` docstring all
currently say some variant of "independent design choice... not a verified
reproduction"). None of that has been corrected yet — this HANDOFF entry is
the only place the correction currently lives. A future session should:

- Re-fetch arXiv:2602.17445's actual methodology (full PDF, not just the
  abstract) before changing any code, the same way every other citation in
  this codebase was verified this session — don't just restore the old
  "Nowak et al. 2026" comment text from before `a4d91e4` without checking
  what the paper *actually* specifies, since the original comments were
  written under the same "couldn't verify" uncertainty that turned out to be
  wrong in the other direction too (i.e. don't assume the pre-`a4d91e4`
  comments were accurate either — they may have been a confident-sounding
  guess, not a checked fact).
  Pull the PDF: `https://arxiv.org/pdf/2602.17445`.
- Once verified, restore an accurate citation in `abcd.py`/`text_extraction.py`,
  correcting the over-correction from `a4d91e4`.
- Decide whether `a4d91e4`'s commit message itself needs a follow-up note
  (not a rewrite — don't rewrite pushed history) acknowledging the error, or
  whether this HANDOFF.md entry is sufficient.

---

## 3. Open decision, not resolved: does `abcd` need its own redo?

Given #2, the open question is whether `abcd`'s *current* implementation
(dash labels + free-text generation + `resolve_stage2_answer`'s
leading-letter/cue-letter/embedding-match cascade, from commit `a4d91e4`)
actually matches what arXiv:2602.17445 specifies, or whether — like
`additional_option` and `cyclic` turned out to be — it's measuring a
different manipulation than the paper describes and needs a prompt and/or
extraction redo.

**Not yet investigated.** Don't assume either way.

**Next concrete step (not yet done):** before assuming a full GPU rerun is
needed (expensive, no model cached locally per this session's experience —
see the additional_option/cyclic verification section above, which used
`DummyBackend` specifically because no causal LM was available locally),
scan the existing `abcd` `free_text_response` raw data across the corpus
(`runs/*/*.csv` where `method_name == "abcd"`) for how often it *already*
naturally contains an explicit "the answer is X" / "Answer: X" style phrase
(reuse `_FINAL_ANSWER_CUE_RE` from `text_extraction.py` as a starting point,
or write a fresh regex once the paper's actual extraction method is
confirmed). If that phrasing is already common in the existing free-text
responses, a cheap re-match against already-saved data (same pattern as this
session's `rematch_abcd_rows`) might recover paper-faithful behavior without
new model calls. If it's rare, a full prompt-modification + rerun is the
only path. Don't commit to either path before running this scan.

---

## 4. Still open — not code tasks

- **`paper_sections_draft.md` was audited this session but never edited.**
  The audit happened in conversation (not persisted to any file) and
  produced specific line-number findings — D.6/D.7/D.8's method descriptions
  are stale (don't reflect the `resolve_stage2_answer` cascade or the
  leading-letter precondition), the `abcd` citation framing throughout
  (lines ~325-353, Checklist #7 and #12, Source Map line ~718) needs
  correcting per #2 above, and `master_table.csv`/test-count references are
  now considerably more stale than the draft already flagged (two more
  commits have landed since the draft's own staleness warnings were
  written). None of this has been written into the file. A future session
  should either re-run that audit fresh (cheap, just re-reading the current
  code) or ask whoever has the original line-number findings to supply them
  again — they were not saved anywhere outside the conversation that
  produced them.

- **Two small, disclosure-only notes for whoever writes the methods section**
  (code is correct as-is, no fix needed — these are just things the paper
  text should say plainly):
  - `pride`'s calibration pool (`PriDeRunner`, `run_experiment.py:load_calibration_questions`)
    is deliberately sourced from *outside* the 1000-question eval split
    entirely (the rest of that benchmark's full question pool, minus the
    eval IDs) — not from a held-out subset *of* the eval split the way
    Zheng et al. 2024's Algorithm 1 actually specifies (`D_e ⊆ D`, the same
    set being scored). This is an intentional, documented design choice in
    the code (see `PriDeRunner`'s class docstring), not a bug — it just needs
    to be stated as a deviation in the paper, not implied to be a literal
    reproduction.
  - `calibration`'s content-free prior (`AnswerCalibrationRunner._build_neutral_prompt`)
    uses one probe ("N/A" for the question and every option). Zhao et al.
    2021 average three ("N/A", "[MASK]", empty string). The decision rule
    itself is mathematically equivalent either way (verified this session:
    log-space subtraction of two genuine logprobs vs. the paper's
    probability-ratio-then-softmax select the identical argmax) — only the
    number of content-free probes differs.

---

## Orientation for a fresh session

- Run the suite: `PYTHONPATH=src python -m pytest -q`
- Dry-run any config: `PYTHONPATH=src python scripts/run_experiment.py --config config/<name>.yaml --dry-run`
- The redesigned methods' verification this session used `DummyBackend`
  with custom `score_options()` overrides (see `tests/runners/test_permutation.py`'s
  `ContentAwareScoreBackend`/`TokenBiasedScoreBackend` and the inline scripts
  in this session's transcript) — no real causal LM was cached locally
  (`~/.cache/huggingface/hub` only had embedding models and datasets, not a
  Qwen/Llama causal LM), so nothing in this session's verification used a
  real model's actual weights. Treat all accuracy numbers from this session
  as logic/algebra checks, not real-model evidence.
