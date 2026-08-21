#!/usr/bin/env python3
"""generate_code_graph.py

Generate per-module and per-function Markdown notes inside the Obsidian vault
so Obsidian's graph view can represent files, functions, and their call links.

Usage:
  python generate_code_graph.py --repo-root .. --vault . --append-handoff \
    --agent my-agent --summary "Generated code graph"

This script is best-effort: it parses Python AST to find functions, methods,
and call expressions and then creates notes linking functions where possible.
"""
from pathlib import Path
import ast
import argparse
import datetime
import os
import re
import shutil


IGNORE_DIRS = {
    'Stochpylib-Obsidian-Vault', 'build', 'dist', '__pycache__', '.venv', 'venv', '.git',
    '.pytest_cache',
}

# populated by main() to help resolve import aliases across files
IMPORT_MAP = {}


def is_ignored(path: Path):
    for p in path.parts:
        if p.startswith('.') or p in IGNORE_DIRS or p.endswith('.egg-info'):
            return True
    return False


def find_py_files(root: Path):
    for p in root.rglob('*.py'):
        if is_ignored(p):
            continue
        yield p


class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []  # list of (qualname, lineno, docstring)
        self.calls = {}  # qualname -> set(called_names)
        self.current = None
        self.current_class = None
        self.imports = {}  # alias -> module
        self.from_imports = {}  # alias -> module.object

    def visit_Import(self, node):
        for alias in node.names:
            self.imports[alias.asname or alias.name] = alias.name

    def visit_ImportFrom(self, node):
        module = node.module or ''
        for alias in node.names:
            name = alias.asname or alias.name
            self.from_imports[name] = f"{module}.{alias.name}" if module else alias.name

    def visit_ClassDef(self, node):
        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node):
        qual = node.name
        if self.current_class:
            qual = f"{self.current_class}.{node.name}"
        self.functions.append((qual, node.lineno, ast.get_docstring(node) or ''))
        prev = self.current
        self.current = qual
        self.calls.setdefault(qual, set())
        self.generic_visit(node)
        self.current = prev

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    def visit_Call(self, node):
        name = self.get_name(node.func)
        if name and self.current:
            self.calls.setdefault(self.current, set()).add(name)
        self.generic_visit(node)

    def get_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts = []
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
                parts.reverse()
                return '.'.join(parts)
        return None


def add_parents(node):
    for child in ast.iter_child_nodes(node):
        child.parent = node
        add_parents(child)


def read_source(path: Path):
    try:
        return path.read_text(encoding='utf-8')
    except Exception:
        return ''


def sanitize_name(s: str):
    return re.sub(r'[^0-9A-Za-z_.-]+', '_', s)


def make_wiki_link(source: Path, target: Path):
    try:
        rel = target.relative_to(source.parent)
    except ValueError:
        rel = Path(os.path.relpath(target, source.parent))
    if rel.suffix == '.md':
        rel = rel.with_suffix('')
    return rel.as_posix()


def write_module_note(vault: Path, relpath: Path, functions):
    out_dir = vault / 'code' / relpath.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (relpath.name + '.md')
    lines = [f'# Module: {relpath.as_posix()}\n\n']
    lines.append('Functions:\n\n')
    for fn in functions:
        fn_name = fn[0]
        fn_path = sanitize_name(relpath.as_posix() + '__' + fn_name) + '.md'
        target = vault / 'code' / relpath.parent / fn_path
        link = make_wiki_link(out_file, target)
        lines.append(f'- [[{link}]]\n')
    out_file.write_text(''.join(lines), encoding='utf-8')
    # If this is a package __init__.py, also create/update a folder index in code/folders
    if relpath.name == '__init__.py':
        folders_dir = vault / 'code' / 'folders'
        folders_dir.mkdir(parents=True, exist_ok=True)
        folder_note = folders_dir / (sanitize_name(relpath.parent.as_posix()) + '.md')
        link = make_wiki_link(folder_note, out_file)
        header = f'# Folder: {relpath.parent.as_posix()}\n\n'
        entry = f'- [[{link}]]\n'
        if folder_note.exists():
            txt = folder_note.read_text(encoding='utf-8')
            if link not in txt:
                # append a Contents section if missing
                if 'Contents:' not in txt:
                    txt = txt + '\nContents:\n\n'
                txt = txt + entry
                folder_note.write_text(txt, encoding='utf-8')
        else:
            folder_note.write_text(header + 'Contents:\n\n' + entry, encoding='utf-8')
    return out_file


def resolve_targets(call_name: str, relpath: Path, current_class: str | None, func_map: dict, vault: Path):
    candidates = []
    # exact match
    if call_name in func_map:
        candidates.extend(func_map[call_name])
    # self.method -> class method
    if current_class and call_name.startswith('self.'):
        method = call_name.split('.', 1)[1]
        candidates.extend(func_map.get(f'{current_class}.{method}', []))
        candidates.extend(func_map.get(method, []))
    # imported module or module.function
    if '.' in call_name:
        module, rest = call_name.split('.', 1)
        # direct lookup like module.func or module.Class.method
        candidates.extend(func_map.get(f'{module}.{rest}', []))
        # unqualified function name
        if '.' not in rest:
            candidates.extend(func_map.get(rest, []))
        # try import alias resolution: map alias -> full module name
        full = IMPORT_MAP.get(module)
        if full:
            # treat as external module if not in repo
            # create an external note for the top-level module
            candidates.append(ensure_external_note(vault, module, full))
    # dedupe while preserving order
    seen = set()
    unique = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def ensure_external_note(vault: Path, prefix: str, fullname: str):
    ext_dir = vault / 'code' / 'external'
    ext_dir.mkdir(parents=True, exist_ok=True)
    note = ext_dir / (sanitize_name(prefix) + '.md')
    if not note.exists():
        # Map some known libraries to docs
        docs = {
            'torch': 'https://pytorch.org/docs/stable/',
            'np': 'https://numpy.org/doc/stable/',
            'numpy': 'https://numpy.org/doc/stable/',
            'pd': 'https://pandas.pydata.org/docs/',
            'pandas': 'https://pandas.pydata.org/docs/',
            'sklearn': 'https://scikit-learn.org/stable/',
            'flask': 'https://flask.palletsprojects.com/',
            'requests': 'https://docs.python-requests.org/',
        }
        url = docs.get(prefix, f'https://www.google.com/search?q={fullname}')
        note.write_text(f'# External: {prefix}\n\n- Module: {fullname}\n\n- Docs: {url}\n', encoding='utf-8')
    return ('external', sanitize_name(prefix), fullname)


def write_function_note(vault: Path, relpath: Path, func_name: str, lineno: int, doc: str, calls, current_class: str | None, func_map):
    out_dir = vault / 'code' / relpath.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = sanitize_name(relpath.as_posix() + '__' + func_name) + '.md'
    out_file = out_dir / fname
    lines = [f'# Function: {func_name}\n\n', f'- Module: {relpath.as_posix()}\n', f'- Defined at: line {lineno}\n\n']
    if doc:
        lines.append('## Docstring\n\n')
        lines.append(doc + '\n\n')
    if calls:
        lines.append('## Calls\n\n')
        for c in sorted(calls):
            targets = resolve_targets(c, relpath, current_class, func_map, vault)
            if targets:
                for t in targets:
                    if isinstance(t, tuple) and t[0] == 'external':
                        # ('external', prefix, fullname)
                        ext_name = t[1]
                        ext_target = vault / 'code' / 'external' / (ext_name + '.md')
                        link = make_wiki_link(out_file, ext_target)
                        lines.append(f'- [[{link}]] (external `{c}`)\n')
                    else:
                        t_path = sanitize_name(t[0].as_posix() + '__' + t[1]) + '.md'
                        target = vault / 'code' / t[0].parent / t_path
                        link = make_wiki_link(out_file, target)
                        lines.append(f'- [[{link}]] (from `{c}`)\n')
            else:
                lines.append(f'- {c}\n')
    out_file.write_text(''.join(lines), encoding='utf-8')
    return out_file


def append_handoff(vault: Path, agent: str, summary: str, files: list):
    handoff = vault / 'HANDOFF.MD'
    if not handoff.exists():
        handoff.write_text('# HANDOFF\n', encoding='utf-8')
    ts = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'
    lines = [
        '\n',
        f'- **Timestamp:** {ts}\n',
        f'- **Agent:** {agent}\n',
        f'- **Action:** generated\n',
        f'- **Summary:** {summary}\n',
        f'- **Files changed:** {", ".join(files) if files else "none"}\n',
        '\n'
    ]
    handoff.write_text(handoff.read_text(encoding='utf-8') + '\n'.join(lines), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo-root', default='..')
    p.add_argument('--vault', default='.')
    p.add_argument('--append-handoff', action='store_true')
    p.add_argument('--agent', default='agent')
    p.add_argument('--summary', default='Generated code graph')
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    vault = Path(args.vault).resolve()

    # remove any stale generated notes first
    code_dir = vault / 'code'
    if code_dir.exists() and code_dir.is_dir():
        shutil.rmtree(code_dir)

    # First pass: collect functions
    module_functions = {}  # Path -> list of (name, lineno, doc)
    func_map = {}  # simple name -> list of (module_relpath, func_name)
    import_map_global = {}  # alias -> full module name (aggregate across files)

    for py in find_py_files(repo_root):
        rel = py.relative_to(repo_root)
        src = read_source(py)
        try:
            tree = ast.parse(src)
        except Exception:
            continue
        add_parents(tree)
        vis = CodeVisitor()
        vis.visit(tree)
        if vis.functions:
            module_functions[rel] = vis.functions
            module_key = rel.stem
            for name, lineno, doc in vis.functions:
                simple_name = name.split('.')[-1]
                func_map.setdefault(name, []).append((rel, name))
                func_map.setdefault(simple_name, []).append((rel, name))
                func_map.setdefault(f'{module_key}.{simple_name}', []).append((rel, name))
                if '.' in name:
                    func_map.setdefault(f'{module_key}.{name}', []).append((rel, name))
        # collect imports for cross-file resolution
        for k, v in getattr(vis, 'imports', {}).items():
            import_map_global.setdefault(k, v)
        for k, v in getattr(vis, 'from_imports', {}).items():
            import_map_global.setdefault(k, v)

    # expose collected imports to resolver
    global IMPORT_MAP
    IMPORT_MAP.update(import_map_global)

    # Second pass: collect calls per function and write notes
    for py in find_py_files(repo_root):
        rel = py.relative_to(repo_root)
        src = read_source(py)
        try:
            tree = ast.parse(src)
        except Exception:
            continue
        add_parents(tree)
        vis = CodeVisitor()
        vis.visit(tree)
        functions = module_functions.get(rel, [])
        # write module note
        write_module_note(vault, rel, functions)
        # write function notes
        for name, lineno, doc in functions:
            calls = vis.calls.get(name, set())
            current_class = name.rsplit('.', 1)[0] if '.' in name else None
            write_function_note(vault, rel, name, lineno, doc, calls, current_class, func_map)

    if args.append_handoff:
        append_handoff(vault, args.agent, args.summary, ['generated code notes'])


if __name__ == '__main__':
    main()
