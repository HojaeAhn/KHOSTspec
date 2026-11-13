"""Build local HTML documentation while keeping private build caches separate."""

import argparse
from pathlib import Path

from sphinx.cmd.build import build_main


def main(argv=None):
    """Build HTML with warnings treated as errors and return Sphinx's exit code.

    ``--build-dir`` selects the parent of the ``html`` and ``doctrees``
    directories. The default is ``docs/_build`` in the source checkout.
    Only the HTML directory is intended for publication or offline sharing.
    """
    source = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build KHAO pipeline documentation")
    parser.add_argument("--build-dir", type=Path, default=source / "_build",
                        help="Parent directory for generated HTML and build caches")
    args = parser.parse_args(argv)
    build_dir = args.build_dir.expanduser().resolve()
    return build_main([
        "-W", "--keep-going", "-d", str(build_dir / "doctrees"),
        "-b", "html", str(source), str(build_dir / "html"),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
