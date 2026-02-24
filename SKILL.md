---
name: repomap
description: Generate and refresh a compact repository map (ranked files, key symbols, and dependency links) for fast whole-repo understanding with low token cost. Use when asked to review an entire repo, create/update a repomap, quickly understand high-level architecture before coding, or rebuild context similar to aider's repo map workflow.
---

# RepoMap

## Overview

Build and refresh a compact repository map that prioritizes the most structurally important files and symbols. Use the map to bootstrap context before deeper file reads.

## Workflow

1. Generate a fresh map:

```bash
python3 /Users/saejin/.codex/skills/repomap/scripts/build_repomap.py \
  --root . \
  --out .codex/repomap.txt \
  --max-chars 14000 \
  --max-files 120
```

2. Read the generated map into context:

```bash
sed -n '1,240p' .codex/repomap.txt
```

3. Use the map to guide next reads:
- Start with the top-ranked files.
- Expand only the files/directories relevant to the user request.
- Regenerate the map after major structural changes.
- Trust that ignored files are excluded using `.gitignore` rules when present.

## Tuning

- Use `--include-tests` when test architecture matters for the request.
- Use `--max-chars` to control context cost aggressively.
- Use `--max-files` and `--max-symbols` to trade coverage vs compactness.
- Use `--no-symbol-refs` for very large repos when speed matters more than ranking fidelity.

## Output Contract

- Write the map to `.codex/repomap.txt` unless the user asks for a different path.
- Keep the output compact and ranked.
- Include:
  - Repo metadata (scan size, languages, dominant directories)
  - Ranked file list with key symbols plus definition line (and next line only if it is a `return` statement)
  - Dependency/reference edges summary

## Notes

- This skill approximates aider's repo-map strategy (definitions + references + graph ranking) without requiring aider as a runtime dependency.
- Read `references/aider-parity-notes.md` only when you need implementation details or parity tuning guidance.
