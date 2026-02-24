# Aider Parity Notes

This skill targets practical parity with aider's repo-map concept, not byte-for-byte output parity.

## Shared ideas

- Build a compact representation of repository structure.
- Extract definitions and cross-file signals.
- Rank files by graph importance instead of listing files alphabetically.
- Constrain output to a token/character budget.

## Intentional simplifications

- Use language-aware parsing for common languages plus regex fallbacks.
- Use import resolution and lightweight symbol-reference edges instead of full ctags + tree-sitter coverage.
- Use an internal PageRank implementation (no external graph dependency).
- Keep output in plain text for direct context injection.

## Tuning strategy

- Increase `--max-chars` and `--max-files` for larger context windows.
- Enable `--include-tests` when architecture understanding depends on test harnesses.
- Disable symbol references with `--no-symbol-refs` for very large monorepos.

## Quality checks

After generation, verify:

1. Core entry points appear near the top.
2. Key modules have representative symbols.
3. Dominant directories reflect real architecture.
4. Output fits expected context budget.
