#!/usr/bin/env python3
"""
Parse Mastercaller matchcenter full_text fields into structured match records.

Reads matchcenter_YYYY_M_D_*.json files from the input directory, parses
the full_text field (captured via document.body.innerText from Mastercaller
matchcenter pages), and extracts structured match records.

Usage:
    python scripts/parse_mastercaller_fulltext.py [--input-dir DIR] [--output-file FILE]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAV_LINES: set[str] = {
    "English",
    "Home",
    "Matchcenter",
    "Tournaments",
    "PDC",
    "WDF",
    "Callers",
    "Players",
    "Nine Darters",
    "Streamed",
    "Other",
    "Rankings",
    "Calendar",
    "Videos",
    "Contact",
    "Privacy Statement",
    "Jump to Today",
    "SEE ALL RESULTS",
    "Stats by iDarts",
}

# Patterns
TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")                  # "14:00", "9:30"
FLOAT_RE = re.compile(r"^\d+\.\d+$")                       # "99.06"
INT_RE = re.compile(r"^\d+$")                               # "10"
DATE_NAV_RE = re.compile(
    r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+\d{1,2}\s+\w+$"
)
NO_MATCHES_RE = re.compile(r"^No Matches Played On\s+")
SLIDE_RE = re.compile(r"^slide\s+\d+")
FILENAME_RE = re.compile(r"matchcenter_(\d{4})_(\d{1,2})_(\d{1,2})_")
FOOTER_LINES: set[str] = {
    "Designed & Developed by Laravel Specialist Arno Poot",
}

# Known round names (case-insensitive matching via normalized lookup).
# We detect rounds dynamically rather than hardcoding an exhaustive list.
ROUND_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(Quarter-finals|Semi-finals|Final|Preliminary Round)$", re.IGNORECASE),
    re.compile(r"^(Last\s+\d+)$", re.IGNORECASE),
    re.compile(r"^(Round\s+\d+)$", re.IGNORECASE),
    re.compile(r"^(Week\s+\d+)$", re.IGNORECASE),
    re.compile(r"^(Group\s+\w+)$", re.IGNORECASE),
    re.compile(r"^(1st Round|2nd Round|3rd Round|4th Round|5th Round|6th Round)$", re.IGNORECASE),
    re.compile(r"^(Phase\s+\d+)$", re.IGNORECASE),
    re.compile(r"^(Board\s+\d+)$", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def is_time(s: str) -> bool:
    return bool(TIME_RE.match(s))


def is_float_val(s: str) -> bool:
    return bool(FLOAT_RE.match(s))


def is_int_val(s: str) -> bool:
    return bool(INT_RE.match(s))


def is_nav_line(s: str) -> bool:
    return s in NAV_LINES


def is_date_nav(s: str) -> bool:
    return bool(DATE_NAV_RE.match(s))


def is_no_matches(s: str) -> bool:
    return bool(NO_MATCHES_RE.match(s))


def is_slide_line(s: str) -> bool:
    return bool(SLIDE_RE.match(s))


def is_footer(s: str) -> bool:
    return s in FOOTER_LINES


def is_round_name(s: str) -> bool:
    for pat in ROUND_PATTERNS:
        if pat.match(s):
            return True
    return False


def is_na(s: str) -> bool:
    return s == "N/A"


def is_skip_line(s: str) -> bool:
    """Return True if this line should be skipped entirely."""
    return (
        not s
        or is_nav_line(s)
        or is_date_nav(s)
        or is_no_matches(s)
        or is_slide_line(s)
        or is_footer(s)
        or s == "SEE ALL RESULTS"
        or s == "Privacy Statement"
        or s == "Jump to Today"
    )


def is_event_name(line: str, next_line: Optional[str]) -> bool:
    """
    Heuristic: a line is an event name if the next non-blank line is
    'SEE ALL RESULTS'. We check this via lookahead in the caller.
    We also accept it if it looks like a multi-word title that is NOT
    a round name, time, number, or nav line.
    """
    # This is called with lookahead -- the caller passes next_line
    if next_line is not None and next_line.strip() == "SEE ALL RESULTS":
        return True
    return False


def parse_date_from_filename(filename: str) -> Optional[str]:
    """Extract date string YYYY-MM-DD from filename like matchcenter_2026_3_8_xxx.json."""
    m = FILENAME_RE.search(filename)
    if not m:
        return None
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return f"{year:04d}-{month:02d}-{day:02d}"


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_full_text(full_text: str, date: str) -> list[dict]:
    """
    Parse a Mastercaller matchcenter full_text into a list of match dicts.

    Uses a line-by-line parser with lookahead to handle the variable
    structure (optional times, optional averages, N/A placeholders).
    """
    raw_lines = full_text.split("\n")
    # Strip each line and collect non-empty, non-skip lines with awareness
    # of event detection (which requires seeing "SEE ALL RESULTS" next).
    # First pass: strip whitespace from all lines.
    stripped: list[str] = [line.strip() for line in raw_lines]

    # Build a filtered line list, but we need raw ordering for event detection.
    # Strategy: iterate stripped lines, skip obvious junk, detect events by
    # lookahead on the *original* stripped list (before filtering).
    #
    # We'll work with indices into `stripped`.
    matches: list[dict] = []
    current_event: Optional[str] = None
    current_round: Optional[str] = None

    i = 0
    n = len(stripped)

    def peek(offset: int = 1) -> Optional[str]:
        """Return stripped[i + offset] if in range, skipping blank lines."""
        j = i + offset
        while j < n and not stripped[j]:
            j += 1
        if j < n:
            return stripped[j]
        return None

    def peek_abs(offset: int = 1) -> Optional[str]:
        """Return stripped[i + offset] exactly (even if blank)."""
        j = i + offset
        if j < n:
            return stripped[j]
        return None

    def next_nonblank_line(start: int) -> Optional[str]:
        """Find next non-blank line from start (inclusive)."""
        j = start
        while j < n:
            if stripped[j]:
                return stripped[j]
            j += 1
        return None

    while i < n:
        line = stripped[i]

        # Skip blank lines
        if not line:
            i += 1
            continue

        # Skip known junk
        if is_nav_line(line) or is_date_nav(line) or is_no_matches(line) or is_slide_line(line) or is_footer(line):
            i += 1
            continue

        if line == "SEE ALL RESULTS":
            i += 1
            continue

        if line == "Privacy Statement" or line == "Jump to Today":
            i += 1
            continue

        # Event detection: if the next non-blank line is "SEE ALL RESULTS",
        # this line is an event name.
        nbl = next_nonblank_line(i + 1)
        if nbl == "SEE ALL RESULTS":
            current_event = line
            current_round = None
            i += 1
            continue

        # Round detection
        if is_round_name(line):
            current_round = line
            i += 1
            continue

        # N/A as a standalone (placeholder for events with no data)
        if is_na(line):
            i += 1
            continue

        # At this point we might be at the start of a match block.
        # Match block structure:
        #   [time]  (optional)
        #   player1_name
        #   [avg1]  (optional float)
        #   score1  (int)
        #   score2  (int)
        #   player2_name
        #   [avg2]  (optional float)
        #
        # Or we might be at a time line, which starts a match.

        # Try to parse a match block starting from current position.
        match_record = _try_parse_match(stripped, i, n, date, current_event, current_round)
        if match_record is not None:
            record, consumed = match_record
            matches.append(record)
            i += consumed
            continue

        # If we can't parse a match, skip the line (unknown content).
        i += 1

    return matches


def _try_parse_match(
    lines: list[str],
    start: int,
    n: int,
    date: str,
    event: Optional[str],
    round_name: Optional[str],
) -> Optional[tuple[dict, int]]:
    """
    Attempt to parse a match block starting at index `start`.

    Returns (record_dict, lines_consumed) or None if not a valid match block.

    Match block variants:
      A) time, name1, avg1, score1, score2, name2, avg2   (7 lines)
      B) time, name1, avg1, score1, score2, name2          (6 lines)
      C) time, name1, score1, score2, name2, avg2          (6 lines)
      D) time, name1, score1, score2, name2                (5 lines)
      E) name1, avg1, score1, score2, name2, avg2          (6 lines, no time)
      F) name1, avg1, score1, score2, name2                (5 lines, no time)
      G) name1, score1, score2, name2, avg2                (5 lines, no time)
      H) name1, score1, score2, name2                      (4 lines, no time)
    """

    def get(idx: int) -> Optional[str]:
        if idx < n:
            s = lines[idx]
            return s if s else None  # treat blank as None
        return None

    # Skip blank lines from start
    pos = start
    while pos < n and not lines[pos]:
        pos += 1
    if pos >= n:
        return None

    # Check for optional time
    time_val: Optional[str] = None
    cur = pos
    line0 = get(cur)
    if line0 is None:
        return None

    if is_time(line0):
        time_val = line0
        cur += 1
        # Skip blanks after time
        while cur < n and not lines[cur]:
            cur += 1
    elif is_na(line0):
        # N/A as time placeholder -- skip
        time_val = None
        cur += 1
        while cur < n and not lines[cur]:
            cur += 1

    # Next must be player1 name -- must NOT be a number, float, time, round, nav, etc.
    p1_line = get(cur)
    if p1_line is None or is_int_val(p1_line) or is_float_val(p1_line) or is_time(p1_line):
        return None
    if is_round_name(p1_line) or is_nav_line(p1_line) or is_na(p1_line):
        return None
    if is_slide_line(p1_line) or is_footer(p1_line) or is_date_nav(p1_line):
        return None
    if p1_line == "SEE ALL RESULTS" or is_no_matches(p1_line):
        return None

    # Check if next non-blank after event name is "SEE ALL RESULTS" -- then this is
    # actually an event name, not a player.
    peek_idx = cur + 1
    while peek_idx < n and not lines[peek_idx]:
        peek_idx += 1
    if peek_idx < n and lines[peek_idx] == "SEE ALL RESULTS":
        return None

    player1 = p1_line
    cur += 1

    # Next: either avg1 (float) or score1 (int)
    next_val = get(cur)
    if next_val is None:
        return None

    avg1: Optional[float] = None
    if is_float_val(next_val):
        avg1 = float(next_val)
        cur += 1
        next_val = get(cur)
        if next_val is None:
            return None

    # score1 (must be int)
    if not is_int_val(next_val):
        return None
    score1 = int(next_val)
    cur += 1

    # score2 (must be int)
    score2_val = get(cur)
    if score2_val is None or not is_int_val(score2_val):
        return None
    score2 = int(score2_val)
    cur += 1

    # player2 name
    p2_line = get(cur)
    if p2_line is None or is_int_val(p2_line) or is_float_val(p2_line) or is_time(p2_line):
        return None
    if is_round_name(p2_line) or is_nav_line(p2_line) or is_na(p2_line):
        return None
    if is_slide_line(p2_line) or is_footer(p2_line) or is_date_nav(p2_line):
        return None
    if p2_line == "SEE ALL RESULTS" or is_no_matches(p2_line):
        return None

    # Check if this is actually an event name (next line is SEE ALL RESULTS)
    peek2 = cur + 1
    while peek2 < n and not lines[peek2]:
        peek2 += 1
    if peek2 < n and lines[peek2] == "SEE ALL RESULTS":
        return None

    player2 = p2_line
    cur += 1

    # Optional avg2 (float)
    avg2: Optional[float] = None
    avg2_val = get(cur)
    if avg2_val is not None and is_float_val(avg2_val):
        avg2 = float(avg2_val)
        cur += 1

    record = {
        "date": date,
        "event": event,
        "round": round_name,
        "time": time_val,
        "player1": player1,
        "avg1": avg1,
        "score1": score1,
        "score2": score2,
        "player2": player2,
        "avg2": avg2,
    }

    consumed = cur - start
    return (record, consumed)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def process_directory(input_dir: Path) -> list[dict]:
    """Read all matchcenter_*.json files and extract match records."""
    all_matches: list[dict] = []
    json_files = sorted(input_dir.glob("matchcenter_*.json"))

    if not json_files:
        print(f"WARNING: No matchcenter_*.json files found in {input_dir}")
        return all_matches

    print(f"Found {len(json_files)} matchcenter files in {input_dir}")

    for fpath in json_files:
        date = parse_date_from_filename(fpath.name)
        if date is None:
            print(f"  SKIP {fpath.name}: cannot parse date from filename")
            continue

        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"  ERROR {fpath.name}: {exc}")
            continue

        full_text = data.get("full_text")
        if not full_text:
            print(f"  SKIP {fpath.name}: no full_text field")
            continue

        matches = parse_full_text(full_text, date)
        print(f"  {fpath.name} -> {date}: {len(matches)} matches extracted")
        all_matches.extend(matches)

    return all_matches


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

MARCH_8_FULL_TEXT = """English
Home
Matchcenter
Tournaments
PDC
WDF
Callers
Players
Nine Darters
Streamed
Other
Rankings
PDC
WDF
Calendar
Videos
Contact
Matchcenter
Saturday, 7 Mar
Sunday, 8 March
Jump to Today
Monday, 9 Mar
UK Open 2026
SEE ALL RESULTS
Quarter-finals
14:00
Josh Rock
99.06
10
7
Krzysztof Ratajski
97.29
15:00
James Wade
91.03
10
9
Rob Cross
89.59
16:00
Luke Littler
104.47
10
6
Danny Noppert
98.66
17:00
Gerwyn Price
98.26
10
8
Jonny Clayton
96.62
Semi-finals
20:00
James Wade
105.53
11
8
Gerwyn Price
101.39
21:00
Luke Littler
100.36
11
9
Josh Rock
94.00
Final
22:15
Luke Littler
99.58
11
7
James Wade
89.49
slide 2 of 4





Isle of Man Open 2026
SEE ALL RESULTS
Final
12:48
Jack Drayton
89.75
6
2
Ben Townley
84.83
Isle of Man Open Women 2026
SEE ALL RESULTS
Final
12:29
Gemma Hayter
80.83
5
1
Steph Clarke
74.67
Isle of Man Classic Women 2026
SEE ALL RESULTS
Final
10:47
Leanne Topper
66.16
5
4
Eve Watson
69.94
Isle of Man Masters Women 2026
SEE ALL RESULTS
Final
11:40
Rhian O'Sullivan
87.29
5
2
Aileen de Graaf
78.70
Isle of Man Masters Men 2026
SEE ALL RESULTS
Final
12:01
Reece Colley
87.97
5
3
Jim McEwan
84.65
Isle of Man Classic Men 2026
SEE ALL RESULTS
Final
11:23
Moreno Blom
83.15
5
1
Daniel Zapata
74.86
PDC Europe NEXT GEN 2026 PDC Europe NEXT GEN 06 Hildesheim
SEE ALL RESULTS
Final
13:23
Daniel Klose
6
5
René Eidams
Isle of Man Open Paradarts 2026 Standing
SEE ALL RESULTS
Final
N/A
Isle of Man Open Paradarts 2026 Wheelchair
SEE ALL RESULTS
Final
N/A
Isle of Man Classic Youth 2026
SEE ALL RESULTS
Final
10:28
Ben Townley
93.81
5
2
Kaya Baysal
94.31
Missouri St. Patrick's Day Open U23 Girls 2026
SEE ALL RESULTS
Final
N/A
Missouri St. Patrick's Day Open U23 Youth 2026
SEE ALL RESULTS
Final
N/A
slide 15 to 18 of 8









Designed & Developed by Laravel Specialist Arno Poot
Stats by iDarts
Privacy Statement"""


def run_self_test() -> bool:
    """Run self-test against known March 8 data. Returns True if all pass."""
    print("\n=== SELF-TEST ===")
    matches = parse_full_text(MARCH_8_FULL_TEXT, "2026-03-08")

    errors: list[str] = []

    # -- Check UK Open matches --
    uk_open = [m for m in matches if m["event"] == "UK Open 2026"]
    qf = [m for m in uk_open if m["round"] == "Quarter-finals"]
    sf = [m for m in uk_open if m["round"] == "Semi-finals"]
    final = [m for m in uk_open if m["round"] == "Final"]

    if len(uk_open) != 7:
        errors.append(f"UK Open: expected 7 matches, got {len(uk_open)}")
    if len(qf) != 4:
        errors.append(f"UK Open QF: expected 4 matches, got {len(qf)}")
    if len(sf) != 2:
        errors.append(f"UK Open SF: expected 2 matches, got {len(sf)}")
    if len(final) != 1:
        errors.append(f"UK Open Final: expected 1 match, got {len(final)}")

    # Luke Littler QF avg
    littler_qf = [m for m in qf if m["player1"] == "Luke Littler"]
    if len(littler_qf) != 1:
        errors.append(f"Littler QF: expected 1 match, got {len(littler_qf)}")
    elif littler_qf[0]["avg1"] != 104.47:
        errors.append(f"Littler QF avg1: expected 104.47, got {littler_qf[0]['avg1']}")

    # Final: Luke Littler 11-7 James Wade
    if final:
        f = final[0]
        if f["player1"] != "Luke Littler":
            errors.append(f"Final player1: expected 'Luke Littler', got '{f['player1']}'")
        if f["player2"] != "James Wade":
            errors.append(f"Final player2: expected 'James Wade', got '{f['player2']}'")
        if f["score1"] != 11 or f["score2"] != 7:
            errors.append(f"Final score: expected 11-7, got {f['score1']}-{f['score2']}")
        if f["avg1"] != 99.58:
            errors.append(f"Final avg1: expected 99.58, got {f['avg1']}")

    # -- Check total matches (UK Open 7 + IOM Open 1 + IOM Open Women 1 +
    #    IOM Classic Women 1 + IOM Masters Women 1 + IOM Masters Men 1 +
    #    IOM Classic Men 1 + PDC Europe NEXT GEN 1 + IOM Classic Youth 1 = 15)
    #    Paradarts and Missouri have N/A with no data -> 0 matches.
    total_expected = 15
    if len(matches) != total_expected:
        errors.append(f"Total matches: expected {total_expected}, got {len(matches)}")

    # -- Check PDC Europe NEXT GEN (no averages) --
    nextgen = [m for m in matches if m["event"] and "NEXT GEN" in m["event"]]
    if len(nextgen) != 1:
        errors.append(f"NEXT GEN: expected 1 match, got {len(nextgen)}")
    elif nextgen[0]["avg1"] is not None or nextgen[0]["avg2"] is not None:
        errors.append(f"NEXT GEN avgs should be None, got {nextgen[0]['avg1']}, {nextgen[0]['avg2']}")

    # -- Print results --
    if errors:
        print("SELF-TEST FAILED:")
        for e in errors:
            print(f"  FAIL: {e}")
        # Debug: print all matches
        print(f"\n  All {len(matches)} parsed matches:")
        for m in matches:
            print(f"    {m['event']} | {m['round']} | {m['time']} | "
                  f"{m['player1']} ({m['avg1']}) {m['score1']}-{m['score2']} "
                  f"{m['player2']} ({m['avg2']})")
        return False
    else:
        print(f"ALL SELF-TESTS PASSED ({len(matches)} matches parsed correctly)")
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Mastercaller matchcenter full_text into structured match records."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("D:/codex/Data/Darts/02_processed/json/mastercaller_playwright/event/"),
        help="Directory containing matchcenter_*.json files",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("D:/codex/Data/Darts/02_processed/json/mastercaller_matches.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run self-test only (no file I/O)",
    )
    args = parser.parse_args()

    # Always run self-test first
    test_ok = run_self_test()
    if args.self_test:
        sys.exit(0 if test_ok else 1)

    if not test_ok:
        print("\nSelf-test failed -- aborting. Fix parser before processing files.")
        sys.exit(1)

    # Process files
    print(f"\nProcessing files from: {args.input_dir}")
    all_matches = process_directory(args.input_dir)

    # Summary by date
    print(f"\n=== SUMMARY ===")
    date_counts: dict[str, int] = {}
    for m in all_matches:
        d = m["date"]
        date_counts[d] = date_counts.get(d, 0) + 1

    for d in sorted(date_counts):
        print(f"  {d}: {date_counts[d]} matches")
    print(f"  TOTAL: {len(all_matches)} matches")

    # Summary by event
    print(f"\n=== BY EVENT ===")
    event_counts: dict[str, int] = {}
    for m in all_matches:
        ev = m["event"] or "(unknown)"
        event_counts[ev] = event_counts.get(ev, 0) + 1
    for ev in sorted(event_counts):
        print(f"  {ev}: {event_counts[ev]} matches")

    # Write output
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_matches, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(all_matches)} matches to {args.output_file}")


if __name__ == "__main__":
    main()
