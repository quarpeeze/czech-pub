from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParseResult:
    label: str | None
    status: str
    matched_text: str | None = None


def parse_mcq_answer(raw_text: str | None, valid_labels: list[str]) -> ParseResult:
    """
    parser for MCQ labels like A/B/C/D

    accepted patterns:
    - exact short answers: "A", "B.", "(C)", "D)"
    - leading label before explanation: "D\n\nVysvětlení: ..."
    - explicit markers: "Odpověď: A", "Answer is B", "Správná odpověď je C"
    """
    if raw_text is None:
        return ParseResult(label=None, status="empty", matched_text=None)

    text = raw_text.strip()
    if not text:
        return ParseResult(label=None, status="empty", matched_text=None)

    valid = {label.upper() for label in valid_labels}

    # case 0: output starts with a standalone label, even if more text follows
    start_match = re.match(
        r"^\s*\(?([A-Za-z])\)?(?:[\.\:\)]|\b)",
        text,
        flags=re.IGNORECASE,
    )
    if start_match:
        label = start_match.group(1).upper()
        if label in valid:
            return ParseResult(label=label, status="ok", matched_text=start_match.group(0))

    # case 1: exact short answers only
    exact_patterns = [
        r"^\s*\(?([A-Za-z])\)?[\.\:\)]?\s*$",
    ]

    for pattern in exact_patterns:
        m = re.match(pattern, text, flags=re.IGNORECASE)
        if m:
            label = m.group(1).upper()
            if label in valid:
                return ParseResult(label=label, status="ok", matched_text=m.group(0))

    # case 2: longer outputs with explicit answer markers
    marker_patterns = [
        r"(?:správná\s+odpověď|spravna\s+odpoved|odpověď|odpoved|answer)(?:\s+je|\s+is)?\s*[:\-]?\s*\(?([A-Za-z])\)?\b",
        r"(?:správná\s+volba|spravna\s+volba|volba|možnost|moznost|varianta|option)\s*[:\-]?\s*\(?([A-Za-z])\)?\b",
        r"(?:písmeno|pismeno)\s*[:\-]?\s*\(?([A-Za-z])\)?\b",
    ]

    candidates: list[tuple[str, str]] = []
    for pattern in marker_patterns:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            label = m.group(1).upper()
            if label in valid:
                candidates.append((label, m.group(0)))

    # deduplicate while preserving order
    seen = set()
    deduped: list[tuple[str, str]] = []
    for label, matched in candidates:
        if label not in seen:
            seen.add(label)
            deduped.append((label, matched))

    if len(deduped) == 1:
        return ParseResult(label=deduped[0][0], status="ok", matched_text=deduped[0][1])

    if len(deduped) > 1:
        return ParseResult(
            label=None,
            status="ambiguous",
            matched_text=", ".join(label for label, _ in deduped),
        )

    return ParseResult(label=None, status="invalid", matched_text=None)