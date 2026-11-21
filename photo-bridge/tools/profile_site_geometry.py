"""Read-only V3-A site geometry profile CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from registration_algorithms.site_geometry import profile_site  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", action="append", required=True,
                        help="NAME:VISIBLE_DSM:THERMAL_DSM")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    output = {}
    for spec in args.station:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError("station must be NAME:VISIBLE_DSM:THERMAL_DSM")
        name, visible, thermal = parts
        output[name] = profile_site(Path(visible), Path(thermal))
    text = json.dumps(output, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
