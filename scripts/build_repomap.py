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
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

EXT_LANGUAGE = {
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
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
    definitions: list[str]
    symbol_snippets: dict[str, list[str]]
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
    filtered = sorted(set(filtered))
    return apply_gitignore_filters(root, filtered)


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


def apply_gitignore_filters(root: Path, candidates: list[str]) -> list[str]:
    if not candidates:
        return candidates

    filtered = _filter_with_git_check_ignore(root, candidates)
    if filtered is not None:
        return filtered
    filtered = _filter_with_ephemeral_git_check_ignore(root, candidates)
    if filtered is not None:
        return filtered
    return _filter_with_discovered_gitignores(root, candidates)


def _filter_with_git_check_ignore(root: Path, candidates: list[str]) -> list[str] | None:
    cmd = ["git", "-C", str(root), "check-ignore", "--stdin"]
    payload = "\n".join(candidates) + "\n"
    try:
        proc = subprocess.run(cmd, input=payload, text=True, capture_output=True)
    except FileNotFoundError:
        return None

    if proc.returncode not in (0, 1):
        return None

    ignored = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    if not ignored:
        return candidates
    return [path for path in candidates if path not in ignored]


def _filter_with_ephemeral_git_check_ignore(root: Path, candidates: list[str]) -> list[str] | None:
    try:
        with tempfile.TemporaryDirectory(prefix="repomap-ignore-") as temp_dir:
            init = subprocess.run(
                ["git", "-C", temp_dir, "init", "--quiet"],
                capture_output=True,
                text=True,
            )
            if init.returncode != 0:
                return None

            payload = "\n".join(candidates) + "\n"
            proc = subprocess.run(
                [
                    "git",
                    f"--git-dir={temp_dir}/.git",
                    f"--work-tree={root}",
                    "check-ignore",
                    "--no-index",
                    "--stdin",
                ],
                input=payload,
                text=True,
                capture_output=True,
            )
    except (FileNotFoundError, OSError):
        return None

    if proc.returncode not in (0, 1):
        return None

    ignored = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    if not ignored:
        return candidates
    return [path for path in candidates if path not in ignored]


def _filter_with_discovered_gitignores(root: Path, candidates: list[str]) -> list[str]:
    rules: list[tuple[str, str, bool]] = []
    gitignores = sorted(
        root.rglob(".gitignore"),
        key=lambda p: (len(p.relative_to(root).parts), p.relative_to(root).as_posix()),
    )

    for gitignore in gitignores:
        try:
            rel_base = gitignore.parent.relative_to(root).as_posix()
            if rel_base == ".":
                rel_base = ""
            content = gitignore.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            negated = line.startswith("!")
            pattern = line[1:] if negated else line
            pattern = pattern.strip()
            if pattern:
                rules.append((rel_base, pattern, negated))

    if not rules:
        return candidates

    kept: list[str] = []
    for rel_path in candidates:
        ignored = False
        for base, pattern, negated in rules:
            if _matches_gitignore_rule(rel_path, base, pattern):
                ignored = not negated
        if not ignored:
            kept.append(rel_path)
    return kept


def _matches_gitignore_rule(rel_path: str, base: str, pattern: str) -> bool:
    # Normalize to POSIX-like relative paths.
    if base and rel_path != base and not rel_path.startswith(base + "/"):
        return False

    subpath = rel_path[len(base) + 1 :] if base else rel_path
    if not subpath:
        return False

    directory_only = pattern.endswith("/")
    anchored = pattern.startswith("/")
    clean = pattern.strip("/")
    if not clean:
        return False

    if directory_only:
        if "/" in clean:
            if anchored:
                return subpath == clean or subpath.startswith(clean + "/")
            return subpath == clean or subpath.startswith(clean + "/") or f"/{clean}/" in f"/{subpath}/"
        return subpath == clean or subpath.startswith(clean + "/") or f"/{clean}/" in f"/{subpath}/"

    if "/" in clean:
        if anchored:
            return _path_match(subpath, clean)
        return _path_match(subpath, clean) or _path_match(subpath, f"*/{clean}")

    return any(_path_match(part, clean) for part in subpath.split("/"))


def _path_match(path: str, pattern: str) -> bool:
    # Gitignore-style wildcard matching for fallback mode.
    regex = re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*").replace(r"\?", "[^/]")
    return re.fullmatch(regex, path) is not None


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


def extract_markdown(text: str) -> tuple[list[str], list[str]]:
    headings = re.findall(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", text, flags=re.MULTILINE)
    defs = []
    for h in headings:
        cleaned = re.sub(r"[^A-Za-z0-9_ ]+", " ", h).strip()
        if cleaned:
            defs.append(cleaned.replace(" ", "_")[:64])
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    refs = [link for link in links if not link.startswith("http://") and not link.startswith("https://")]
    return dedupe(defs), dedupe(refs)


def _definition_patterns(language: str, symbol: str) -> list[re.Pattern[str]]:
    escaped = re.escape(symbol)
    if language == "python":
        return [re.compile(rf"^\s*(?:async\s+def|def|class)\s+{escaped}\b")]
    if language in {"javascript", "typescript"}:
        return [
            re.compile(rf"^\s*(?:export\s+)?(?:async\s+)?function\s+{escaped}\b"),
            re.compile(rf"^\s*(?:export\s+)?class\s+{escaped}\b"),
            re.compile(rf"^\s*(?:export\s+)?(?:const|let|var|type|interface)\s+{escaped}\b"),
        ]
    if language == "go":
        return [
            re.compile(rf"^\s*func\s+(?:\([^)]+\)\s*)?{escaped}\b"),
            re.compile(rf"^\s*(?:type|var|const)\s+{escaped}\b"),
        ]
    if language == "rust":
        return [
            re.compile(rf"^\s*(?:pub\s+)?(?:fn|struct|enum|trait|mod)\s+{escaped}\b"),
            re.compile(rf"^\s*impl\s+{escaped}\b"),
        ]
    if language == "markdown":
        heading = re.escape(symbol.replace("_", " "))
        return [re.compile(rf"^\s{{0,3}}#{{1,6}}\s+{heading}\s*$", re.IGNORECASE)]
    return [re.compile(rf"\b{escaped}\b")]


def _trim_code_line(value: str, max_len: int = 140) -> str:
    line = value.rstrip()
    if len(line) <= max_len:
        return line
    return line[: max_len - 3] + "..."


def extract_symbol_snippets(text: str, language: str, definitions: list[str]) -> dict[str, list[str]]:
    lines = text.splitlines()
    snippets: dict[str, list[str]] = {}
    if not lines or not definitions:
        return snippets

    for symbol in definitions:
        patterns = _definition_patterns(language, symbol)
        line_index = -1
        for idx, raw in enumerate(lines):
            if any(pattern.search(raw) for pattern in patterns):
                line_index = idx
                break
        if line_index < 0:
            continue

        capture: list[str] = []
        if language == "python" and line_index > 0 and lines[line_index - 1].lstrip().startswith("@"):
            capture.append(_trim_code_line(lines[line_index - 1].strip()))

        capture.append(_trim_code_line(lines[line_index].strip()))
        probe = line_index + 1
        while len(capture) < 2 and probe < len(lines):
            candidate = lines[probe].strip()
            if candidate:
                capture.append(_trim_code_line(candidate))
            probe += 1
        snippets[symbol] = capture[:2]

    return snippets


def parse_file(path: Path, rel_path: str) -> FileInfo:
    text = read_text(path)
    language = EXT_LANGUAGE.get(path.suffix.lower(), "unknown")

    if language == "python":
        defs, imports = extract_python(text)
    elif language in {"javascript", "typescript"}:
        defs, imports = extract_js_ts(text)
    elif language == "go":
        defs, imports = extract_go(text)
    elif language == "rust":
        defs, imports = extract_rust(text)
    elif language == "markdown":
        defs, imports = extract_markdown(text)
    else:
        defs, imports = extract_default(text)

    symbol_snippets = extract_symbol_snippets(text, language, defs)
    tokens = {w for w in WORD_RE.findall(text) if w.lower() not in COMMON_SYMBOLS}
    if len(tokens) > 6000:
        tokens = set(sorted(tokens)[:6000])

    return FileInfo(
        path=rel_path,
        language=language,
        definitions=defs,
        symbol_snippets=symbol_snippets,
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


def build_dir_summary(ranked_paths: list[str], depth: int = 2) -> list[tuple[str, int]]:
    buckets: dict[str, int] = defaultdict(int)
    for path in ranked_paths[:200]:
        parts = Path(path).parts
        if not parts:
            continue
        bucket = "/".join(parts[: min(depth, len(parts))])
        buckets[bucket] += 1
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
    dir_summary = build_dir_summary(ranked_paths)

    lines: list[str] = []
    lines.append(f"# RepoMap: {root.name}")
    lines.append(f"generated_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"files_scanned: {len(files)}")
    langs = ", ".join(f"{k}:{v}" for k, v in sorted(language_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    lines.append(f"languages: {langs}")
    if dir_summary:
        lines.append("top_dirs: " + ", ".join(f"{d}({count})" for d, count in dir_summary))
    edge_count = sum(len(v) for v in edges.values())
    lines.append(f"graph_edges: {edge_count}")
    lines.append("")
    lines.append("## Ranked files")

    for rank_index, path in enumerate(ranked_paths[:max_files], start=1):
        info = files[path]
        defs = dedupe(info.definitions)[:max_symbols]
        import_preview = dedupe(info.imports)[:4]

        block = [f"{rank_index}. {path} [{info.language}]"]
        if defs:
            block.append("  symbols:")
            for symbol in defs:
                snippet = info.symbol_snippets.get(symbol, [])
                block.append(f"    - {symbol}")
                for line in snippet[:2]:
                    block.append(f"      | {line}")
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
