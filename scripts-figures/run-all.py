# run all figure script
import subprocess, os
from pathlib import Path

PYTHON = 'C:/Users/alice/miniforge3/envs/mathai/python.exe'
REPO_ROOT = str(Path(__file__).parent.parent)
FIG_SCRIPT_PATH = os.path.join(REPO_ROOT, 'scripts-figures')

fnames = [#"figure-approximation.py", 
          #"figure-heatmaps.py",
          #"figure-lcc-over-f-large.py","figure-lcc-over-f-small.py",
          #"figure-lcc-over-p.py",
          "figure-voronoi.py"
          ]
for fname in fnames:
    print(os.getcwd())
    print("Script:", fname)
    output = subprocess.run([PYTHON, os.path.join(FIG_SCRIPT_PATH, fname)])
    print(output)
