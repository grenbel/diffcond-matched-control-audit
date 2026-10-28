"""Fetch the two ODGNet files needed by the UpTrans experiments and apply our patch.

The ODGNet repository states no licence, so this repository does not include
models/pointnet.py and models/UpTrans.py. This script downloads them from the
ODGNet commit used in the paper, checks their SHA-256, applies
third_party/odgnet_uptrans.patch (five lines of models/UpTrans.py) and checks
the result against the files used in our runs (the hashes sealed in
data/results/gap_a_provenance/meta/code_attestation.json) before writing
anything to models/. Standard library only.

Usage (from the repository root):
  python third_party/fetch_odgnet.py               # download, check, patch, write
  python third_party/fetch_odgnet.py --check-only  # same checks, writes nothing
  python third_party/fetch_odgnet.py --from-dir D  # use files already downloaded to D
"""
import argparse
import hashlib
import os
import re
import sys
import urllib.request

COMMIT = "032bca98f89cc3ea4733b331ff91468df1bea0f9"
RAW_URL = "https://raw.githubusercontent.com/corecai163/ODGNet/" + COMMIT + "/models/"
PATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "odgnet_uptrans.patch")
# name: (SHA-256 of the ODGNet file, SHA-256 of the file used in our runs, patched?)
FILES = {
    "pointnet.py": ("26f4925535e46702145af8d4e06e30384e189770e78c50aeeb995a03a78e5ab7",
                    "26f4925535e46702145af8d4e06e30384e189770e78c50aeeb995a03a78e5ab7", False),
    "UpTrans.py": ("a72033030560556f1dfef8b1d0a1b686d8fa4367326f61f909d52e5560ce2e34",
                   "0e8baead8cf8c8b12750ebbc236c36847278d2edd3a89c8d2cb49db2e6927a17", True),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def apply_unified_diff(original, patch):
    """Apply a unified diff with context to `original` (str); every context and
    removed line must match exactly, otherwise ValueError."""
    src = original.splitlines(keepends=True)
    lines = patch.replace("\r\n", "\n").splitlines(keepends=True)
    out, pos, i = [], 0, 0
    while i < len(lines) and not lines[i].startswith("@@"):
        i += 1
    while i < len(lines):
        m = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", lines[i])
        if not m:
            raise ValueError("bad hunk header: " + lines[i].rstrip())
        start = int(m.group(1)) - 1
        if start < pos:
            raise ValueError("overlapping hunks at line %d" % (start + 1))
        out.extend(src[pos:start])
        pos = start
        i += 1
        while i < len(lines) and not lines[i].startswith("@@"):
            tag, body = lines[i][:1], lines[i][1:]
            if tag in (" ", "-"):
                if pos >= len(src) or src[pos] != body:
                    raise ValueError("patch does not match the original at line %d" % (pos + 1))
                if tag == " ":
                    out.append(src[pos])
                pos += 1
            elif tag == "+":
                out.append(body)
            else:
                raise ValueError("unsupported patch line: " + lines[i].rstrip())
            i += 1
    out.extend(src[pos:])
    return "".join(out)


def get_original(name, from_dir):
    if from_dir:
        with open(os.path.join(from_dir, name), "rb") as f:
            return f.read()
    with urllib.request.urlopen(RAW_URL + name, timeout=60) as r:
        return r.read()


def prepare(from_dir=None, patch_path=PATCH, files=FILES, getter=get_original):
    """Return {name: bytes} of the files as used in our runs; raise on any mismatch."""
    result = {}
    for name, (sha_orig, sha_final, patched) in files.items():
        data = getter(name, from_dir).replace(b"\r\n", b"\n")
        if sha256(data) != sha_orig:
            raise ValueError("%s: SHA-256 %s differs from ODGNet commit %s (%s)"
                             % (name, sha256(data), COMMIT[:7], sha_orig))
        if patched:
            with open(patch_path, encoding="utf-8") as f:
                data = apply_unified_diff(data.decode("utf-8"), f.read()).encode("utf-8")
        if sha256(data) != sha_final:
            raise ValueError("%s: result SHA-256 %s differs from the file used in our runs (%s)"
                             % (name, sha256(data), sha_final))
        result[name] = data
    return result


def install(prepared, models_dir, force=False):
    """Write the files to models_dir; skip identical files; refuse to overwrite others without force."""
    actions = []
    for name, data in prepared.items():
        target = os.path.join(models_dir, name)
        if os.path.exists(target):
            with open(target, "rb") as f:
                current = f.read()
            if current.replace(b"\r\n", b"\n") == data:
                actions.append((name, "already present"))
                continue
            if not force:
                raise ValueError("%s exists and differs; rerun with --force to overwrite" % target)
        with open(target, "wb") as f:
            f.write(data)
        actions.append((name, "written"))
    return actions


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--from-dir", help="folder holding ODGNet's pointnet.py and UpTrans.py")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite differing files in models/")
    args = ap.parse_args(argv)
    try:
        prepared = prepare(args.from_dir)
        print("checked:", ", ".join("%s %s" % (n, sha256(d)[:12]) for n, d in prepared.items()))
        if args.check_only:
            return 0
        for name, action in install(prepared, os.path.join(args.repo_root, "models"), args.force):
            print(name, action)
    except (ValueError, OSError) as e:
        print("error:", e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
