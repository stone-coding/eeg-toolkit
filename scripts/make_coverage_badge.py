"""Write a shields-style SVG badge from a coverage.xml (no third-party dependencies).

Usage: python scripts/make_coverage_badge.py coverage.xml out.svg
"""

import sys
import xml.etree.ElementTree as ET

TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" width="106" height="20" role="img" \
aria-label="coverage: {pct}%">
<linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/>\
<stop offset="1" stop-opacity=".1"/></linearGradient>
<clipPath id="r"><rect width="106" height="20" rx="3" fill="#fff"/></clipPath>
<g clip-path="url(#r)"><rect width="63" height="20" fill="#555"/>\
<rect x="63" width="43" height="20" fill="{color}"/><rect width="106" height="20" fill="url(#s)"/></g>
<g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" \
font-size="11"><text x="32.5" y="14">coverage</text><text x="84.5" y="14">{pct}%</text></g></svg>
"""


def colour(pct: float) -> str:
    return (
        "#4c1" if pct >= 90 else "#97ca00" if pct >= 80 else "#dfb317" if pct >= 70 else "#e05d44"
    )


def main(xml_path: str, out_path: str) -> None:
    rate = float(ET.parse(xml_path).getroot().attrib["line-rate"])
    pct = int(rate * 100 + 0.5)
    with open(out_path, "w") as fh:
        fh.write(TEMPLATE.format(pct=pct, color=colour(pct)))
    print(f"coverage {pct}% -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
