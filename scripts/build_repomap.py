#!/usr/bin/env python3
"""
Build a compact repository map inspired by aider's repo map workflow.

Algorithm:
1) Collect source files (prefer `git ls-files` for deterministic scope).
2) Extract definitions/imports/tokens per file.
3) Build a directed graph from import links + symbol-reference links.
4) Run PageRank to rank architectural importance.
5) Render a compact text map bounded by character and file limits.
"""

from __future__ import annotations

import argparse
import ast
import keyword
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

EXT_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
}

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "target",
    "out",
    ".next",
    ".nuxt",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
}

WORD_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,63}\b")
COMMON_SYMBOLS = set(keyword.kwlist) | {
    "true",
    "false",
    "none",
    "null",
    "self",
    "this",
    "main",
    "index",
    "default",
    "config",
    "setup",
    "test",
    "tests",
}


@dataclass
class FileInfo:
    path: str
    language: str
    lines: int
    definitions: list[str]
    imports: list[str]
    tokens: set[str]


def list_files(root: Path, include_hidden: bool) -> list[str]:
    git_cmd = ["git", "-C", str(root), "ls-files"]
    files: set[str] = set()
    try:
        proc = subprocess.run(git_cmd, capture_output=True, text=True, check=True)
        files.update(line.strip() for line in proc.stdout.splitlines() if line.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Include files discoverable from disk to cover nested repos/submodules and untracked files.
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(root).as_posix()
        files.add(rel)

    filtered: list[str] = []
    for rel in files:
        parts = rel.split("/")
        if any(part in IGNORED_DIRS for part in parts):
            continue
        if not include_hidden and any(part.startswith(".") for part in parts):
            continue
        suffix = Path(rel).suffix.lower()
        if suffix not in EXT_LANGUAGE:
            continue
        filtered.append(rel)
    return sorted(set(filtered))


def is_test_path(path: str) -> bool:
    name = Path(path).name.lower()
    parts = {part.lower() for part in Path(path).parts}
    if "test" in parts or "tests" in parts:
        return True
    if name.startswith("test_") or name.endswith("_test.py") or name.endswith(".test.ts"):
        return True
    if name.endswith(".spec.ts") or name.endswith(".spec.js") or name.endswith("_test.go"):
        return True
    return False


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def extract_python(text: str) -> tuple[list[str], list[str]]:
    defs: list[str] = []
    imports: list[str] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return defs, imports

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            if module:
                imports.append(module)
            else:
                for alias in node.names:
                    imports.append("." * node.level + alias.name)
    return dedupe(defs), dedupe(imports)


def extract_regex(text: str, patterns: list[re.Pattern[str]]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        for match in pattern.findall(text):
            if isinstance(match, tuple):
                out.append(match[-1])
            else:
                out.append(match)
    return dedupe(out)


def extract_js_ts(text: str) -> tuple[list[str], list[str]]:
    def_patterns = [
        re.compile(r"^\s*(?:export\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
        re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
        re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var|type|interface)\s+([A-Za-z_][A-Za-z0-9_]*)",
            re.MULTILINE,
        ),
    ]
    defs = extract_regex(text, def_patterns)

    imports: list[str] = []
    for match in re.findall(r"(?:import|export)\s+[^;]*?\s+from\s+['\"]([^'\"]+)['\"]", text):
        imports.append(match)
    for match in re.findall(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)", text):
        imports.append(match)
    return defs, dedupe(imports)


def extract_go(text: str) -> tuple[list[str], list[str]]:
    def_patterns = [
        re.compile(r"^\s*func\s+(?:\([^)]+\)\s*)?([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
        re.compile(r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
        re.compile(r"^\s*(?:var|const)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
    ]
    defs = extract_regex(text, def_patterns)
    imports = re.findall(r"\"([^\"]+)\"", text)
    return defs, dedupe(imports)


def extract_rust(text: str) -> tuple[list[str], list[str]]:
    def_patterns = [
        re.compile(
            r"^\s*(?:pub\s+)?(?:fn|struct|enum|trait|mod)\s+([A-Za-z_][A-Za-z0-9_]*)",
            re.MULTILINE,
        ),
        re.compile(r"^\s*impl\s+(?:[A-Za-z_][A-Za-z0-9_]*\s+for\s+)?([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
    ]
    defs = extract_regex(text, def_patterns)
    imports = re.findall(r"^\s*use\s+([^;]+);", text, flags=re.MULTILINE)
    return defs, dedupe(imports)


def extract_default(text: str) -> tuple[list[str], list[str]]:
    def_patterns = [
        re.compile(r"^\s*(?:class|interface|struct|enum)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
        re.compile(r"^\s*(?:def|func|function)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
    ]
    defs = extract_regex(text, def_patterns)
    return defs, []


def parse_file(path: Path, rel_path: str) -> FileInfo:
    text = read_text(path)
    language = EXT_LANGUAGE.get(path.suffix.lower(), "unknown")
    lines = text.count("\n") + (1 if text else 0)

    if language == "python":
        defs, imports = extract_python(text)
    elif language in {"javascript", "typescript"}:
        defs, imports = extract_js_ts(text)
    elif language == "go":
        defs, imports = extract_go(text)
    elif language == "rust":
        defs, imports = extract_rust(text)
    else:
        defs, imports = extract_default(text)

    tokens = {w for w in WORD_RE.findall(text) if w.lower() not in COMMON_SYMBOLS}
    if len(tokens) > 6000:
        tokens = set(sorted(tokens)[:6000])

    return FileInfo(
        path=rel_path,
        language=language,
        lines=lines,
        definitions=defs,
        imports=imports,
        tokens=tokens,
    )


def build_py_module_index(paths: set[str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        if not path.endswith(".py"):
            continue
        base = path[:-3]
        if base.endswith("/__init__"):
            base = base[: -len("/__init__")]
        dotted = base.replace("/", ".")
        if dotted:
            index[dotted].append(path)

        # Support src-layout packages.
        if "/src/" in path:
            src_tail = path.split("/src/", 1)[1]
            mod = src_tail[:-3]
            if mod.endswith("/__init__"):
                mod = mod[: -len("/__init__")]
            dotted_src = mod.replace("/", ".")
            if dotted_src:
                index[dotted_src].append(path)
    return {k: dedupe(v) for k, v in index.items()}


def resolve_js_import(src: str, raw_import: str, file_set: set[str]) -> list[str]:
    if not raw_import:
        return []

    if raw_import.startswith("@/"):
        base = raw_import[2:]
        candidates = [base]
    elif raw_import.startswith("/"):
        candidates = [raw_import.lstrip("/")]
    elif raw_import.startswith("."):
        src_dir = Path(src).parent
        candidates = [(src_dir / raw_import).as_posix()]
    else:
        return []

    resolved: list[str] = []
    extensions = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"]
    for cand in candidates:
        path = Path(cand)
        trial = [path.as_posix()]
        if path.suffix == "":
            trial.extend((path.with_suffix(ext).as_posix() for ext in extensions))
            trial.extend(((path / f"index{ext}").as_posix() for ext in extensions))
        for item in trial:
            if item in file_set:
                resolved.append(item)
    return dedupe(resolved)


def resolve_py_import(raw_import: str, src: str, py_index: dict[str, list[str]]) -> list[str]:
    if not raw_import:
        return []

    if raw_import.startswith("."):
        dots = len(raw_import) - len(raw_import.lstrip("."))
        tail = raw_import[dots:]
        src_pkg = Path(src).parent.as_posix().replace("/", ".")
        src_parts = [part for part in src_pkg.split(".") if part]
        up = max(0, dots - 1)
        if up <= len(src_parts):
            src_parts = src_parts[: len(src_parts) - up]
        if tail:
            src_parts.extend(part for part in tail.split(".") if part)
        module = ".".join(src_parts)
    else:
        module = raw_import

    if not module:
        return []

    if module in py_index:
        return py_index[module]

    prefix_hits: list[str] = []
    for known_mod, paths in py_index.items():
        if known_mod.startswith(module + "."):
            prefix_hits.extend(paths)
    return dedupe(prefix_hits)


def build_graph(
    files: dict[str, FileInfo],
    use_symbol_refs: bool,
) -> dict[str, Counter[str]]:
    file_set = set(files.keys())
    py_index = build_py_module_index(file_set)
    edges: dict[str, Counter[str]] = {path: Counter() for path in files}

    for src_path, info in files.items():
        for raw_import in info.imports:
            if info.language == "python":
                targets = resolve_py_import(raw_import, src_path, py_index)
            elif info.language in {"javascript", "typescript"}:
                targets = resolve_js_import(src_path, raw_import, file_set)
            else:
                targets = []
            for dst in targets:
                if dst != src_path:
                    edges[src_path][dst] += 1.0

    if not use_symbol_refs:
        return edges

    def_index: dict[str, list[str]] = defaultdict(list)
    for path, info in files.items():
        for symbol in info.definitions:
            if len(symbol) < 3 or len(symbol) > 64:
                continue
            if symbol.lower() in COMMON_SYMBOLS:
                continue
            def_index[symbol].append(path)

    filtered_defs = {
        symbol: owners
        for symbol, owners in def_index.items()
        if 1 <= len(owners) <= 3
    }
    symbol_set = set(filtered_defs.keys())

    for src_path, info in files.items():
        refs = info.tokens & symbol_set
        ref_count = 0
        for symbol in refs:
            owners = filtered_defs[symbol]
            for dst in owners:
                if dst == src_path:
                    continue
                edges[src_path][dst] += 0.15
                ref_count += 1
            if ref_count >= 2000:
                break

    return edges


def pagerank(edges: dict[str, Counter[str]], damping: float = 0.85, iterations: int = 30) -> dict[str, float]:
    nodes = sorted(edges.keys())
    count = len(nodes)
    if count == 0:
        return {}

    rank = {node: 1.0 / count for node in nodes}
    out_weight = {node: float(sum(edges[node].values())) for node in nodes}

    for _ in range(iterations):
        new_rank = {node: (1.0 - damping) / count for node in nodes}
        sink_total = sum(rank[node] for node in nodes if out_weight[node] == 0.0)
        sink_share = damping * sink_total / count
        for node in nodes:
            new_rank[node] += sink_share

        for src in nodes:
            total = out_weight[src]
            if total <= 0:
                continue
            contribution = damping * rank[src]
            for dst, weight in edges[src].items():
                if dst in new_rank:
                    new_rank[dst] += contribution * (weight / total)
        rank = new_rank

    return rank


def build_dir_summary(ranked_paths: list[str], ranks: dict[str, float], depth: int = 2) -> list[tuple[str, float]]:
    buckets: dict[str, float] = defaultdict(float)
    for path in ranked_paths[:200]:
        parts = Path(path).parts
        if not parts:
            continue
        bucket = "/".join(parts[: min(depth, len(parts))])
        buckets[bucket] += ranks.get(path, 0.0)
    return sorted(buckets.items(), key=lambda kv: (-kv[1], kv[0]))[:12]


def format_map(
    root: Path,
    files: dict[str, FileInfo],
    edges: dict[str, Counter[str]],
    ranks: dict[str, float],
    max_files: int,
    max_symbols: int,
    max_chars: int,
) -> str:
    ranked_paths = sorted(files.keys(), key=lambda p: (-ranks.get(p, 0.0), p))
    language_counts = Counter(info.language for info in files.values())
    dir_summary = build_dir_summary(ranked_paths, ranks)

    lines: list[str] = []
    lines.append(f"# RepoMap: {root.name}")
    lines.append(f"generated_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"files_scanned: {len(files)}")
    langs = ", ".join(f"{k}:{v}" for k, v in sorted(language_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    lines.append(f"languages: {langs}")
    if dir_summary:
        lines.append("top_dirs: " + ", ".join(f"{d}({score:.3f})" for d, score in dir_summary))
    edge_count = sum(len(v) for v in edges.values())
    lines.append(f"graph_edges: {edge_count}")
    lines.append("")
    lines.append("## Ranked files")

    for path in ranked_paths[:max_files]:
        info = files[path]
        score = ranks.get(path, 0.0)
        defs = dedupe(info.definitions)[:max_symbols]
        import_preview = dedupe(info.imports)[:4]

        block = [f"{path} [{info.language}] score={score:.5f} lines={info.lines}"]
        if defs:
            block.append("  defs: " + ", ".join(defs))
        if import_preview:
            block.append("  imports: " + ", ".join(import_preview))

        prospective = "\n".join(lines + block) + "\n"
        if len(prospective) > max_chars:
            lines.append("... truncated by max_chars ...")
            break
        lines.extend(block)

    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a compact repository map.")
    parser.add_argument("--root", default=".", help="Repository root directory")
    parser.add_argument("--out", help="Optional output path for the generated map")
    parser.add_argument("--max-files", type=int, default=120, help="Maximum ranked files to print")
    parser.add_argument("--max-symbols", type=int, default=8, help="Maximum symbols per file in output")
    parser.add_argument("--max-chars", type=int, default=14000, help="Hard character budget for output")
    parser.add_argument("--include-tests", action="store_true", help="Include test files")
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden paths")
    parser.add_argument("--no-symbol-refs", action="store_true", help="Disable symbol-reference edges")
    parser.add_argument("--stats", action="store_true", help="Print generation stats to stderr")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] invalid root: {root}", file=sys.stderr)
        return 1

    all_files = list_files(root, include_hidden=args.include_hidden)
    if not args.include_tests:
        all_files = [path for path in all_files if not is_test_path(path)]

    if not all_files:
        print("[ERROR] no supported source files found", file=sys.stderr)
        return 1

    files: dict[str, FileInfo] = {}
    for rel in all_files:
        abs_path = root / rel
        try:
            files[rel] = parse_file(abs_path, rel)
        except OSError:
            continue

    edges = build_graph(files, use_symbol_refs=not args.no_symbol_refs)
    ranks = pagerank(edges)
    output = format_map(
        root=root,
        files=files,
        edges=edges,
        ranks=ranks,
        max_files=max(1, args.max_files),
        max_symbols=max(1, args.max_symbols),
        max_chars=max(1000, args.max_chars),
    )

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")

    print(output, end="")

    if args.stats:
        edge_total = sum(len(counter) for counter in edges.values())
        print(
            f"[stats] files={len(files)} edges={edge_total} output_chars={len(output)}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
