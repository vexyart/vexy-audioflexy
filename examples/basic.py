# this_file: examples/basic.py
"""Print installed metadata without audio, models, or network access."""

from vexy_audioflexy import __version__
from vexy_audioflexy.cli import version

assert version() == __version__
print(f"vexy-audioflexy {__version__}")
