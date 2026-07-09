#!/usr/bin/env python3
"""Verify that every policy reference in data/sleec_rules.json is real.

For each rule, the `policy_references` map holds the *verbatim* policy language
for each source (anthropic / openai / gemini). This script checks that each of
those quoted strings can actually be found in the corresponding provider's saved
HTML page (data/<source>.html), i.e. that the citation is a genuine link to text
the provider really published -- not a paraphrase or fabrication.

Matching is deliberately robust to HTML noise:
  * all tags are stripped (replaced with spaces),
  * HTML entities are decoded (&quot; -> "),
  * curly quotes / dashes / ellipses are normalised to ASCII,
  * every run of non-alphanumeric characters collapses to a single space,
  * comparison is case-insensitive.

A reference is split into segments on ';' (each bullet the author merged) and on
'...'/'…' (author-elided text). A segment PASSES when every one of its parts is
found as a substring of the normalised page text. This tolerates punctuation,
markup and whitespace differences while still requiring the real wording.

Usage:
    python scripts/verify_policy_references.py            # human-readable report
    python scripts/verify_policy_references.py --json     # machine-readable JSON
    python scripts/verify_policy_references.py --html a.html  # (unused: see --data-dir)
    python scripts/verify_policy_references.py --data-dir data

Exit code is 0 when every reference fully matches, 1 otherwise.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# Unicode punctuation that should be folded to an ASCII equivalent before the
# non-alphanumeric collapse (kept explicit for clarity/debuggability).
_UNICODE_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",   # single quotes
    "“": '"', "”": '"', "„": '"', "‟": '"',   # double quotes
    "–": "-", "—": "-", "―": "-", "−": "-",   # dashes/minus
    "…": "...",                                                 # ellipsis
    " ": " ", " ": " ", " ": " ",                   # spaces
}


def _fold_unicode(text: str) -> str:
    for src, dst in _UNICODE_FOLD.items():
        text = text.replace(src, dst)
    return unicodedata.normalize("NFKC", text)


def loose(text: str) -> str:
    """Fold to a case-insensitive, punctuation-insensitive comparison form."""
    text = _fold_unicode(text).lower()
    return _NON_ALNUM_RE.sub(" ", text).strip()


def html_to_text(raw_html: str) -> str:
    """Strip tags + decode entities, yielding a loose, searchable haystack."""
    no_tags = _TAG_RE.sub(" ", raw_html)
    decoded = html.unescape(no_tags)
    return loose(decoded)


# ---------------------------------------------------------------------------
# Reference segmentation + matching
# ---------------------------------------------------------------------------


def split_segments(reference: str) -> list[str]:
    """Split a reference into independently-checkable clauses (on ';')."""
    return [seg.strip() for seg in reference.split(";") if seg.strip()]


def split_parts(segment: str) -> list[str]:
    """Split a segment on author-elision markers ('...') into required parts."""
    parts = re.split(r"\.\.\.|…", segment)
    return [loose(p) for p in parts if loose(p)]


def match_segment(segment: str, haystack_loose: str) -> bool:
    parts = split_parts(segment)
    return all(part and part in haystack_loose for part in parts)


def word_coverage(segment: str, haystack_loose: str) -> float:
    """Fraction of a failing segment's words that appear at all (diagnostic)."""
    words = [w for w in loose(segment).split() if w]
    if not words:
        return 0.0
    hay_words = set(haystack_loose.split())
    hit = sum(1 for w in words if w in hay_words)
    return hit / len(words)


def longest_verbatim_run(segment: str, haystack_loose: str) -> str:
    """Longest contiguous run of the segment's words that appears verbatim.

    A long run with small gaps signals a light paraphrase of real text; a short
    run signals the quote is largely absent (possible fabrication).
    """
    words = [w for w in loose(segment).split() if w]
    best = ""
    for start in range(len(words)):
        for end in range(len(words), start, -1):
            if (end - start) * 1 <= len(best.split()):
                break  # can't beat current best from here
            phrase = " ".join(words[start:end])
            if phrase in haystack_loose:
                best = phrase
                break
    return best


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(rules_path: Path, data_dir: Path) -> dict:
    doc = json.loads(rules_path.read_text(encoding="utf-8"))

    # Load + normalise each provider's HTML once.
    source_ids = [s["id"] for s in doc.get("sources", [])]
    haystacks: dict[str, str] = {}
    missing_html: list[str] = []
    for sid in source_ids:
        html_path = data_dir / f"{sid}.html"
        if html_path.exists():
            haystacks[sid] = html_to_text(html_path.read_text(encoding="utf-8", errors="replace"))
        else:
            missing_html.append(sid)

    results = []
    for rule in doc.get("rules", []):
        rule_id = rule.get("rule_id", "<no id>")
        refs = rule.get("policy_references", {})
        declared_sources = rule.get("sources", [])
        rule_entry = {
            "rule_id": rule_id,
            "category_id": rule.get("category_id"),
            "sources": [],
        }

        # Consistency: sources[] should line up with policy_references keys.
        ref_keys = set(refs.keys())
        src_set = set(declared_sources)
        rule_entry["sources_mismatch"] = sorted(src_set ^ ref_keys) or None

        for source, reference in refs.items():
            segments = split_segments(reference)
            hay = haystacks.get(source)
            seg_results = []
            for seg in segments:
                if hay is None:
                    matched = False
                    cov = 0.0
                    run = ""
                else:
                    matched = match_segment(seg, hay)
                    cov = 1.0 if matched else round(word_coverage(seg, hay), 2)
                    run = "" if matched else longest_verbatim_run(seg, hay)
                seg_results.append(
                    {"segment": seg, "matched": matched, "coverage": cov, "longest_run": run}
                )

            total = len(seg_results)
            passed = sum(1 for s in seg_results if s["matched"])
            if hay is None:
                status = "NO_HTML"
            elif passed == total:
                status = "PASS"
            elif passed == 0:
                status = "FAIL"
            else:
                status = "PARTIAL"

            rule_entry["sources"].append(
                {
                    "source": source,
                    "status": status,
                    "matched": passed,
                    "total": total,
                    "segments": seg_results,
                }
            )
        results.append(rule_entry)

    return {
        "rules_file": str(rules_path),
        "data_dir": str(data_dir),
        "missing_html": missing_html,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_ICON = {"PASS": "PASS ", "PARTIAL": "PART ", "FAIL": "FAIL ", "NO_HTML": "NO-HTML"}


def print_report(report: dict) -> None:
    W = 78
    print("=" * W)
    print("POLICY REFERENCE VERIFICATION REPORT")
    print("=" * W)
    print(f"Rules file : {report['rules_file']}")
    print(f"HTML dir   : {report['data_dir']}")
    if report["missing_html"]:
        print(f"MISSING HTML for sources: {', '.join(report['missing_html'])}")
    print()

    tot = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "NO_HTML": 0}
    n_refs = 0
    failing_rules = []

    for rule in report["results"]:
        rule_worst_ok = True
        lines = []
        if rule.get("sources_mismatch"):
            lines.append(
                f"    ! sources[] vs policy_references[] mismatch: "
                f"{rule['sources_mismatch']}"
            )
        for s in rule["sources"]:
            n_refs += 1
            tot[s["status"]] += 1
            if s["status"] != "PASS":
                rule_worst_ok = False
            lines.append(
                f"    [{_ICON[s['status']]}] {s['source']:<9} "
                f"{s['matched']}/{s['total']} segments"
            )
            for seg in s["segments"]:
                if not seg["matched"]:
                    excerpt = seg["segment"]
                    if len(excerpt) > 90:
                        excerpt = excerpt[:87] + "..."
                    lines.append(
                        f"          - UNMATCHED (word-coverage {seg['coverage']:.0%}): "
                        f"\"{excerpt}\""
                    )
                    run = seg.get("longest_run", "")
                    if run:
                        if len(run) > 90:
                            run = run[:87] + "..."
                        lines.append(f"            longest verbatim run on page: \"{run}\"")

        header_mark = "OK " if rule_worst_ok else ">> "
        print(f"{header_mark}{rule['rule_id']}  ({rule['category_id']})")
        for ln in lines:
            print(ln)
        if not rule_worst_ok:
            failing_rules.append(rule["rule_id"])
        print()

    print("=" * W)
    print("SUMMARY")
    print("-" * W)
    print(f"  References checked : {n_refs}")
    print(f"  PASS               : {tot['PASS']}")
    print(f"  PARTIAL            : {tot['PARTIAL']}")
    print(f"  FAIL               : {tot['FAIL']}")
    print(f"  NO_HTML            : {tot['NO_HTML']}")
    if failing_rules:
        print()
        print(f"  Rules needing review ({len(failing_rules)}):")
        for rid in failing_rules:
            print(f"    - {rid}")
    else:
        print()
        print("  All policy references were located verbatim in the saved HTML.")
    print("=" * W)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--rules",
        type=Path,
        default=repo_root / "data" / "sleec_rules.json",
        help="Path to sleec_rules.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data",
        help="Directory holding <source>.html files",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a report")
    args = parser.parse_args(argv)

    report = verify(args.rules, args.data_dir)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_report(report)

    # Non-zero exit if anything is not a clean PASS.
    ok = all(
        s["status"] == "PASS"
        for rule in report["results"]
        for s in rule["sources"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
