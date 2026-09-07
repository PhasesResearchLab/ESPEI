"""Copy a rendered site into a versioned directory of the gh-pages checkout.

Usage: deploy_version.py --site docs/_site --pages <gh-pages checkout> --version 0.9.1

The gh-pages branch holds one directory per deployed version: release
numbers such as ``0.9.1`` and ``dev`` for the master branch. After copying
the site, ``switcher.json`` (read by the version switcher in every page)
and the root ``index.html`` (a redirect to the latest release) are
regenerated from the directories present.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

RELEASE_RE = re.compile(r"^\d+(\.\d+)*$")
DEV = "dev"

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={target}/">
<link rel="canonical" href="{target}/">
<title>ESPEI documentation</title>
</head>
<body>
<p>Redirecting to the <a href="{target}/">ESPEI {target} documentation</a>.</p>
</body>
</html>
"""


def release_key(name):
    return tuple(int(part) for part in name.split("."))


def deployed_versions(pages):
    names = [p.name for p in pages.iterdir() if p.is_dir() and not p.name.startswith(".")]
    releases = sorted((n for n in names if RELEASE_RE.match(n)), key=release_key, reverse=True)
    return releases, DEV in names


def write_switcher(pages, releases, has_dev):
    entries = []
    if has_dev:
        entries.append({"name": DEV, "version": DEV, "url": f"{DEV}/"})
    for i, release in enumerate(releases):
        entry = {"name": release, "version": release, "url": f"{release}/"}
        if i == 0:
            entry["name"] = f"{release} (latest)"
            entry["preferred"] = True
        entries.append(entry)
    (pages / "switcher.json").write_text(json.dumps(entries, indent=2) + "\n")
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", required=True, type=Path, help="rendered site (quarto output directory)")
    parser.add_argument("--pages", required=True, type=Path, help="checkout of the gh-pages branch")
    parser.add_argument("--version", required=True, help="directory name for this build, e.g. 0.9.1 or dev")
    args = parser.parse_args()

    if not (RELEASE_RE.match(args.version) or args.version == DEV):
        parser.error(f"version must be a release number or '{DEV}', got {args.version!r}")
    if not (args.site / "index.html").exists():
        parser.error(f"{args.site} does not contain a rendered site")

    target = args.pages / args.version
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(args.site, target)
    (args.pages / ".nojekyll").touch()

    releases, has_dev = deployed_versions(args.pages)
    entries = write_switcher(args.pages, releases, has_dev)
    default = releases[0] if releases else DEV
    (args.pages / "index.html").write_text(INDEX_TEMPLATE.format(target=default))

    print(f"Deployed {args.version} to {target}")
    print("Versions: " + ", ".join(e["name"] for e in entries))
    print(f"Root redirects to {default}/")


if __name__ == "__main__":
    main()
