"""Regression test: deploy_to_homeserver.sh covers every live source file.

Why this exists
---------------

Twice in a row (`svi_live.py` 2026-04-26, `store.py` + 3 subdirs
2026-05-01) we shipped a live engine with stale modules on the
homeserver because the deploy script's ``SOURCE_FILES`` manifest
fell behind the ``deribit_paper/`` tree. Both surfaced as
production-time AttributeError or "method not found" errors that
LOOK like real bugs but are actually deploy-script gaps.

This test parses the manifest from ``scripts/deploy_to_homeserver.sh``
and walks the relevant source trees. Any source ``.py`` file not in
the manifest fails the test with a one-line "add to manifest" hint.

It does NOT enforce the reverse direction — files in the manifest
but missing on disk — because the deploy script's ``sync_one``
helper already prints ``SKIP <path> (not present locally)`` for
those, which is informative rather than wrong.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy_to_homeserver.sh"

# Source trees the manifest is supposed to cover end-to-end. Each
# entry is the directory walked and the package name for error
# messages. Test directories are excluded because tests don't ship.
TRACKED_TREES: list[tuple[str, str]] = [
    ("deribit_paper", "live engine"),
]


def _parse_manifest_paths() -> set[str]:
    """Pull every quoted ``"...py"`` or ``"...sh"`` token out of
    SOURCE_FILES + MODEL_FILES.

    Robust to comments, conditional logic, and trailing whitespace —
    we just regex-match the quoted paths. Strings that don't end in
    ``.py`` / ``.sh`` (data files, JSONs, .zip) are filtered out by
    the caller using the per-tree path prefix.

    H2 (2026-05-06): extended to ``.sh`` so engine-internal shell
    wrappers under ``deribit_paper/scripts/`` are part of the
    completeness contract. The wrapper ``run_engine.sh`` was missing
    on the homeserver because the prior regex only matched ``.py``.
    """
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    return set(re.findall(r'"([^"]+\.(?:py|sh))"', text))


def _walk_source_tree(tree_root: Path) -> set[str]:
    """Return every ``.py`` and (under ``deribit_paper/scripts/``)
    ``.sh`` file under ``tree_root`` as a repo-relative POSIX path.
    Skips ``__pycache__`` and ``tests`` subdirectories (tests have
    their own dev-time gate).

    Shell wrappers are intentionally tracked only inside
    ``deribit_paper/scripts/`` because that's where engine-internal
    operational helpers live (auto-restart wrapper, dry-run
    validator). General ``scripts/`` files are not part of the live
    engine contract.
    """
    out: set[str] = set()
    for pattern in ("*.py", "*.sh"):
        for p in tree_root.rglob(pattern):
            parts = p.relative_to(REPO_ROOT).parts
            if "__pycache__" in parts or "tests" in parts:
                continue
            # Only track .sh under deribit_paper/scripts/
            if p.suffix == ".sh" and "scripts" not in parts:
                continue
            out.add(p.relative_to(REPO_ROOT).as_posix())
    return out


def test_deploy_manifest_covers_all_live_sources():
    """Every ``.py`` under each tracked tree must be in the manifest.

    Stale-deploy gap is the failure mode this test exists for. A
    missing file means the homeserver can run on out-of-date code
    after a deploy, which has caused production AttributeError
    twice in this cycle.
    """
    manifest = _parse_manifest_paths()
    missing_per_tree: dict[str, list[str]] = {}

    for tree_name, label in TRACKED_TREES:
        tree_root = REPO_ROOT / tree_name
        assert tree_root.is_dir(), f"{tree_name} not found"
        on_disk = _walk_source_tree(tree_root)
        missing = sorted(on_disk - manifest)
        if missing:
            missing_per_tree[label] = missing

    if missing_per_tree:
        lines = ["Files present on disk but NOT in deploy manifest:"]
        for label, paths in missing_per_tree.items():
            lines.append(f"\n  {label}:")
            for p in paths:
                lines.append(f'    "{p}"')
        lines.append(
            "\nAdd these lines to scripts/deploy_to_homeserver.sh "
            "SOURCE_FILES so they reach the homeserver on next deploy."
        )
        raise AssertionError("\n".join(lines))


def test_manifest_has_no_dead_python_paths():
    """Every ``.py`` referenced in the manifest exists on disk.

    Catches the inverse drift — the manifest references a file that
    was renamed or removed. Less critical than the forward direction
    (dead manifest entries fail loudly at deploy time as ``SKIP``),
    but worth catching at test time so we don't accumulate cruft.
    """
    manifest = _parse_manifest_paths()
    missing_on_disk = sorted(p for p in manifest if not (REPO_ROOT / p).is_file())
    assert not missing_on_disk, (
        "Manifest references files that don't exist on disk:\n  "
        + "\n  ".join(missing_on_disk)
        + "\nRemove these lines from scripts/deploy_to_homeserver.sh "
        "SOURCE_FILES."
    )
