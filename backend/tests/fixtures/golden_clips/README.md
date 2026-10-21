# Golden set — prompt / quality-gate regression

Short synthetic transcripts with keep/drop labels. Used by
`tests/unit/test_prompt_regression.py` to assert:

1. Quality gate keeps publishable clips and drops filler / weak openers
2. Brief HARD filter still drops known skip phrases (ex.: balde)
3. Prompt contract stays quality-only (no pad-to-quota language)

Bump `TRANSCRIPT_ANALYSIS_CACHE_VERSION` when changing selection prompts;
if these asserts fail, the change regressed excellence criteria.
