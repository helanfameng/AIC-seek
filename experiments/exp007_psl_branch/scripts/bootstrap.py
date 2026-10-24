"""Use an existing package installation with a working Python launcher."""
import argparse
from pathlib import Path
import runpy
import site
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--packages")
options, arguments = parser.parse_known_args()
if options.packages:
    package_path = Path(options.packages)
    if not package_path.is_dir():
        raise SystemExit(f"Package directory missing: {package_path}")
    sys.path.insert(0, str(package_path))
    # Editable Pyronear installs expose core/src through a .pth file. Merely
    # extending sys.path does not process that file in a fallback interpreter.
    site.addsitedir(str(package_path))
entry = Path(__file__).with_name("run_experiment.py")
sys.argv = [str(entry), *arguments]
runpy.run_path(str(entry), run_name="__main__")
