"""Quarto pre-render script: expose the installed ESPEI version to the site.

Writes ``_variables.yml`` so pages and ``_quarto.yml`` can use
``{{< var version >}}`` and ``{{< var docs_version >}}``, and writes
``_version.html`` so the version switcher knows which deployed version it
belongs to.

``docs_version`` is the name of the directory the site is deployed to on
the gh-pages branch: the release number for a tagged release, otherwise
``dev``. Set ``ESPEI_DOCS_VERSION`` to override it.
"""
import os
import re
from importlib.metadata import version
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent

espei_version = version("espei")
is_release = re.fullmatch(r"\d+(\.\d+)*", espei_version) is not None
docs_version = os.environ.get("ESPEI_DOCS_VERSION") or (espei_version if is_release else "dev")

(DOCS_DIR / "_variables.yml").write_text(
    f'version: "{espei_version}"\n'
    f'docs_version: "{docs_version}"\n'
)
(DOCS_DIR / "_version.html").write_text(
    f'<script>window.ESPEI_DOCS_VERSION = "{docs_version}";</script>\n'
)
print(f"ESPEI {espei_version} (docs version: {docs_version})")
