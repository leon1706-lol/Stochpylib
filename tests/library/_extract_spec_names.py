"""Extract per-module spec names from development/Implementation-Checklist.md."""
import io
import json
import re

s = io.open("development/Implementation-Checklist.md", encoding="utf-8").read()
sections = re.split(r"^## ", s, flags=re.M)
out = {}
for sec in sections:
    m = re.match(r"\[([a-z_]+)\]", sec)
    if not m:
        continue
    mod = m.group(1)
    names = []
    for ln in sec.splitlines():
        t = ln.strip()
        if not t.startswith("- [") or "]" not in t:
            continue
        token = t[t.index("]") + 1:].strip()
        if not (token.startswith("`") and "`" in token[1:]):
            continue
        name = token[1:token.index("`", 1)]
        name = name.split("(")[0].rstrip("()")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) and "." not in name:
            names.append(name)
    out[mod] = sorted(set(names))
    print(mod, len(names))

with io.open("tests/library/_spec_names.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("written")
