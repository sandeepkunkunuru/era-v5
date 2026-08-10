"""Unicode script identification, for reporting collisions *per script*.

Session 7 asks for a collision count "per script" (§2h of the report), because the whole
argument is that a byte window generous for Latin is not generous for Brahmic scripts.
`unicodedata` exposes no script property, so we carry the Unicode block ranges for the
scripts V5 targets. Ranges are from the Unicode 15 block definitions.
"""
import unicodedata

# (first, last, name) — Unicode blocks, checked in order.
_BLOCKS = [
    (0x0000, 0x007F, "Latin"),          # ASCII
    (0x0080, 0x024F, "Latin"),          # Latin-1 Supplement + Extended-A/B
    (0x0300, 0x036F, "Combining"),
    (0x0400, 0x04FF, "Cyrillic"),
    (0x0600, 0x06FF, "Arabic"),         # Urdu lives here
    (0x0700, 0x074F, "Syriac"),
    (0x0900, 0x097F, "Devanagari"),     # Hindi, Marathi, Sanskrit
    (0x0980, 0x09FF, "Bengali"),        # Bengali, Assamese
    (0x0A00, 0x0A7F, "Gurmukhi"),       # Punjabi
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x1000, 0x109F, "Myanmar"),
    (0x3040, 0x30FF, "Kana"),
    (0x4E00, 0x9FFF, "Han"),
    (0xAC00, 0xD7AF, "Hangul"),
]

# Scripts that UTF-8 encodes in three bytes per character — the ones the 32-byte
# window actually punishes. (Latin is 1, Arabic is 2.)
THREE_BYTE_SCRIPTS = frozenset({
    "Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya", "Tamil",
    "Telugu", "Kannada", "Malayalam", "Sinhala", "Thai", "Myanmar",
    "Han", "Hangul", "Kana",
})

INDIC_SCRIPTS = frozenset({
    "Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya",
    "Tamil", "Telugu", "Kannada", "Malayalam", "Sinhala",
})


def char_script(ch: str) -> str:
    cp = ord(ch)
    for lo, hi, name in _BLOCKS:
        if lo <= cp <= hi:
            return name
    return "Other"


def token_script(text: str) -> str:
    """The dominant script of a token, ignoring punctuation, digits and the
    Metaspace marker. Returns 'Mixed' when no single script holds a majority."""
    counts: dict[str, int] = {}
    for ch in text:
        if ch in "▁ \t\n":
            continue
        cat = unicodedata.category(ch)
        if cat.startswith(("P", "N", "Z", "C")):   # punctuation, numbers, spaces, control
            continue
        s = char_script(ch)
        if s == "Combining":                        # attribute marks to their base script
            continue
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return "Symbol"
    top, n = max(counts.items(), key=lambda kv: kv[1])
    total = sum(counts.values())
    return top if n / total > 0.5 else "Mixed"


def bytes_per_char(script: str) -> int:
    """Nominal UTF-8 cost per character, used to explain the window in characters."""
    if script in THREE_BYTE_SCRIPTS:
        return 3
    if script in ("Arabic", "Cyrillic", "Syriac"):
        return 2
    return 1
