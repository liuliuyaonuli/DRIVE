#!/usr/bin/env python3
"""Shopping Admin wrapper for the generic site skill generator."""

import sys

try:
    from .site_skill_generator_v3 import main as generic_main
except ImportError:
    from site_skill_generator_v3 import main as generic_main


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if "--site" not in args:
        args = ["--site", "shopping_admin"] + args
    return generic_main(args)


if __name__ == "__main__":
    sys.exit(main())
