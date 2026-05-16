#!/usr/bin/env python3
"""
codepack — Bundle a codebase into a single LLM-readable file.
Usage: python codepack.py [OPTIONS] <directory>
"""

import os
import sys
import argparse
import fnmatch
import re
from pathlib import Path
from datetime import datetime

# ── Default ignore patterns ────────────────────────────────────────────────────
DEFAULT_IGNORE_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache",
    "venv", ".venv", "env", ".env",
    "dist", "build", "out", ".next", ".nuxt", ".output",
    "coverage", ".nyc_output",
    ".idea", ".vscode",
    "target",       # Rust / Java
    "vendor",       # Go / PHP
    "Pods",         # iOS
    ".gradle",
    "eggs", ".eggs", "*.egg-info",
}

DEFAULT_IGNORE_FILES = {
    "*.pyc", "*.pyo", "*.pyd",
    "*.class", "*.jar",
    "*.o", "*.a", "*.so", "*.dylib", "*.dll", "*.exe",
    "*.zip", "*.tar", "*.gz", "*.bz2", "*.xz", "*.rar", "*.7z",
    "*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.ico", "*.svg",
    "*.mp3", "*.mp4", "*.mov", "*.avi", "*.webm",
    "*.pdf", "*.doc", "*.docx", "*.xls", "*.xlsx", "*.ppt", "*.pptx",
    "*.woff", "*.woff2", "*.ttf", "*.eot",
    "*.lock",                    # package-lock.json, Cargo.lock, etc.
    "package-lock.json",
    "yarn.lock", "pnpm-lock.yaml",
    "Pipfile.lock", "poetry.lock",
    "*.min.js", "*.min.css",
    "*.map",
    ".DS_Store", "Thumbs.db",
    ".env", ".env.*",
    "*.sqlite", "*.db",
    "*.log",
}

# ── Code / text extensions we WANT to include ─────────────────────────────────
CODE_EXTENSIONS = {
    # Web
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte",
    ".json", ".jsonc",
    # Backend
    ".py", ".rb", ".php", ".go", ".rs", ".java", ".kt", ".scala",
    ".cs", ".cpp", ".c", ".h", ".hpp",
    ".swift", ".m",
    # Shell / config
    ".sh", ".bash", ".zsh", ".fish",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".env.example",
    # Data / markup
    ".md", ".mdx", ".rst", ".txt",
    ".xml", ".graphql", ".gql",
    ".sql",
    # Misc
    ".dockerfile", ".tf", ".hcl",
    ".r", ".R",
    ".lua", ".pl",
}

ALWAYS_INCLUDE_NAMES = {
    "Dockerfile", "Makefile", "Rakefile", "Procfile",
    "docker-compose.yml", "docker-compose.yaml",
    ".gitignore", ".editorconfig",
    "README", "README.md", "README.rst",
}


# ── Gitignore parser ───────────────────────────────────────────────────────────

def load_gitignore_patterns(root: Path) -> list[str]:
    gitignore = root / ".gitignore"
    patterns = []
    if gitignore.exists():
        for line in gitignore.read_text(errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def matches_gitignore(rel_path: str, patterns: list[str]) -> bool:
    parts = rel_path.replace("\\", "/")
    for pat in patterns:
        pat = pat.lstrip("/")
        if fnmatch.fnmatch(parts, pat):
            return True
        if fnmatch.fnmatch(os.path.basename(parts), pat):
            return True
        # directory pattern
        if pat.endswith("/") and parts.startswith(pat):
            return True
    return False


# ── File filtering ─────────────────────────────────────────────────────────────

def should_ignore_dir(name: str, extra_ignore: set[str]) -> bool:
    return name in DEFAULT_IGNORE_DIRS or name in extra_ignore


def should_include_file(
    filepath: Path,
    rel_path: str,
    extra_ignore_files: set[str],
    gitignore_patterns: list[str],
    use_gitignore: bool,
    extensions: set[str] | None,
) -> bool:
    name = filepath.name

    # Always-include list beats everything
    if name in ALWAYS_INCLUDE_NAMES:
        return True

    # Check gitignore
    if use_gitignore and matches_gitignore(rel_path, gitignore_patterns):
        return False

    # Check default + extra ignore globs
    for pat in DEFAULT_IGNORE_FILES | extra_ignore_files:
        if fnmatch.fnmatch(name, pat):
            return False

    # Check extension whitelist
    suffix = filepath.suffix.lower()
    allowed = extensions if extensions else CODE_EXTENSIONS
    if suffix not in allowed and name not in ALWAYS_INCLUDE_NAMES:
        return False

    return True


def is_binary(path: Path, sample: int = 8192) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(sample)
        return b"\x00" in chunk
    except OSError:
        return True


# ── Content processing ─────────────────────────────────────────────────────────

def compact_content(text: str, max_blank_lines: int = 1) -> str:
    """Collapse consecutive blank lines to max_blank_lines."""
    lines = text.splitlines()
    result, blanks = [], 0
    for line in lines:
        if line.strip() == "":
            blanks += 1
            if blanks <= max_blank_lines:
                result.append(line)
        else:
            blanks = 0
            result.append(line)
    return "\n".join(result)


# ── Tree builder ───────────────────────────────────────────────────────────────

def build_tree(
    root: Path,
    included_files: list[Path],
) -> str:
    """Render a simple ASCII tree of only the included files."""
    # Build a nested dict representing the tree
    tree: dict = {}
    for f in included_files:
        parts = f.relative_to(root).parts
        node = tree
        for part in parts:
            node = node.setdefault(part, {})

    lines = [f"{root.name}/"]

    def _render(node: dict, prefix: str):
        items = sorted(node.keys(), key=lambda x: (bool(node[x]), x))
        for i, key in enumerate(items):
            connector = "└── " if i == len(items) - 1 else "├── "
            lines.append(f"{prefix}{connector}{key}{'/' if node[key] else ''}")
            if node[key]:
                extension = "    " if i == len(items) - 1 else "│   "
                _render(node[key], prefix + extension)

    _render(tree, "")
    return "\n".join(lines)


# ── Token estimator ────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 chars per token."""
    return len(text) // 4


# ── Core packer ───────────────────────────────────────────────────────────────

def pack(
    root: Path,
    output: Path,
    use_gitignore: bool = True,
    extra_ignore_dirs: set[str] = None,
    extra_ignore_files: set[str] = None,
    extensions: set[str] = None,
    max_file_kb: int = 500,
    compact: bool = True,
    max_blank_lines: int = 1,
    include_hidden: bool = False,
):
    extra_ignore_dirs = extra_ignore_dirs or set()
    extra_ignore_files = extra_ignore_files or set()

    gitignore_patterns = load_gitignore_patterns(root) if use_gitignore else []

    collected: list[tuple[Path, str]] = []  # (abs_path, rel_path)
    skipped_binary: list[str] = []
    skipped_large: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)
        rel_dir = dirpath.relative_to(root)

        # Prune ignored directories in-place
        dirnames[:] = [
            d for d in sorted(dirnames)
            if not should_ignore_dir(d, extra_ignore_dirs)
            and (include_hidden or not d.startswith("."))
            and (
                not use_gitignore
                or not matches_gitignore(str(rel_dir / d), gitignore_patterns)
            )
        ]

        for fname in sorted(filenames):
            if not include_hidden and fname.startswith(".") and fname not in ALWAYS_INCLUDE_NAMES:
                continue
            fpath = dirpath / fname
            rel = str(fpath.relative_to(root))

            if not should_include_file(
                fpath, rel, extra_ignore_files, gitignore_patterns,
                use_gitignore, extensions
            ):
                continue

            # Size check
            try:
                size_kb = fpath.stat().st_size / 1024
            except OSError:
                continue

            if size_kb > max_file_kb:
                skipped_large.append(f"{rel} ({size_kb:.0f} KB)")
                continue

            # Binary check
            if is_binary(fpath):
                skipped_binary.append(rel)
                continue

            collected.append((fpath, rel))

    # ── Build output ───────────────────────────────────────────────────────────
    sections: list[str] = []
    included_paths = [p for p, _ in collected]

    tree_str = build_tree(root, included_paths)

    # Header block
    header = (
        f'<codepack generated="{datetime.now().strftime("%Y-%m-%d %H:%M")}" '
        f'root="{root.name}" files="{len(collected)}">\n\n'
        f"<file_tree>\n{tree_str}\n</file_tree>\n"
    )

    if skipped_large or skipped_binary:
        notes = ["<skipped>"]
        if skipped_large:
            notes.append("  Large files (increase --max-file-kb to include):")
            notes.extend(f"    - {f}" for f in skipped_large)
        if skipped_binary:
            notes.append("  Binary files:")
            notes.extend(f"    - {f}" for f in skipped_binary)
        notes.append("</skipped>")
        header += "\n" + "\n".join(notes) + "\n"

    sections.append(header)

    # File blocks
    for fpath, rel in collected:
        try:
            content = fpath.read_text(errors="replace")
        except OSError as e:
            content = f"[ERROR reading file: {e}]"

        if compact:
            content = compact_content(content, max_blank_lines)

        content = content.rstrip()
        lang = fpath.suffix.lstrip(".") or "text"
        sections.append(
            f'\n<file path="{rel}" lang="{lang}">\n{content}\n</file>\n'
        )

    sections.append("\n</codepack>")

    full_output = "".join(sections)

    output.write_text(full_output, encoding="utf-8")

    return {
        "files": len(collected),
        "chars": len(full_output),
        "tokens_est": estimate_tokens(full_output),
        "skipped_large": skipped_large,
        "skipped_binary": skipped_binary,
        "output": output,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pack a codebase into a single LLM-readable .txt file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python codepack.py ./my-project
  python codepack.py ./my-project -o context.txt
  python codepack.py ./my-project --no-gitignore --max-file-kb 200
  python codepack.py ./my-project --ext .py .js .ts
  python codepack.py ./my-project --ignore-dir tests --ignore-dir docs
        """,
    )
    p.add_argument("directory", help="Root directory of the codebase")
    p.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path (default: <dirname>_codepack.txt)",
    )
    p.add_argument(
        "--no-gitignore",
        action="store_true",
        help="Do not use .gitignore to filter files",
    )
    p.add_argument(
        "--ext",
        nargs="+",
        metavar="EXT",
        help="Only include files with these extensions (e.g. .py .js .ts)",
    )
    p.add_argument(
        "--ignore-dir",
        nargs="+",
        default=[],
        metavar="DIR",
        help="Additional directory names to ignore",
    )
    p.add_argument(
        "--ignore-file",
        nargs="+",
        default=[],
        metavar="GLOB",
        help="Additional file glob patterns to ignore (e.g. '*.test.js')",
    )
    p.add_argument(
        "--max-file-kb",
        type=int,
        default=500,
        help="Skip files larger than this size in KB (default: 500)",
    )
    p.add_argument(
        "--no-compact",
        action="store_true",
        help="Do not collapse consecutive blank lines",
    )
    p.add_argument(
        "--max-blank-lines",
        type=int,
        default=1,
        help="Max consecutive blank lines to keep when compacting (default: 1)",
    )
    p.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files and directories (starting with .)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    root = Path(args.directory).resolve()
    if not root.is_dir():
        print(f"[ERROR] Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    output_name = args.output or f"{root.name}_codepack.txt"
    output = Path(output_name).resolve()

    print(f"📦  Packing: {root}")
    print(f"    Output : {output}")
    print()

    result = pack(
        root=root,
        output=output,
        use_gitignore=not args.no_gitignore,
        extra_ignore_dirs=set(args.ignore_dir),
        extra_ignore_files=set(args.ignore_file),
        extensions={e if e.startswith(".") else f".{e}" for e in args.ext} if args.ext else None,
        max_file_kb=args.max_file_kb,
        compact=not args.no_compact,
        max_blank_lines=args.max_blank_lines,
        include_hidden=args.include_hidden,
    )

    size_kb = output.stat().st_size / 1024

    print(f"✅  Done!")
    print(f"    Files packed   : {result['files']}")
    print(f"    Output size    : {size_kb:.1f} KB")
    print(f"    Token estimate : ~{result['tokens_est']:,}")

    if result["skipped_large"]:
        print(f"\n⚠️  Skipped {len(result['skipped_large'])} large file(s):")
        for f in result["skipped_large"]:
            print(f"    - {f}")

    if result["skipped_binary"]:
        print(f"\n⚠️  Skipped {len(result['skipped_binary'])} binary file(s)")

    print(f"\n💡  Upload '{output.name}' to your AI chat and ask your question!")


if __name__ == "__main__":
    main()