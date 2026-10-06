# BattleNet IEEE conference-paper package

This directory is a self-contained IEEE conference-paper source package for the BattleNet project.

## Contents

- `main.tex` - five-page IEEE conference manuscript source with an expanded methodology figure and a four-panel proposed-model analysis figure
- `references.bib` - bibliography using publication metadata and DOIs for cited work
- `main.pdf` - compiled manuscript, generated during verification

## Build

From this directory, run:

```bash
tectonic -X compile main.tex
```

The manuscript intentionally distinguishes the KIIT-MiTA benchmark evidence from the Bangladesh-facing application discussion. Its reported comparison is scoped to the baselines in Table I; the evaluation is not a claim of Bangladesh-specific or operational readiness.

Before submission, replace the anonymous author block with the correct author names, affiliations, and email addresses required by the target conference.
