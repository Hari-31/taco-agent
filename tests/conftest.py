import sys
from pathlib import Path

# Let tests import the project's top-level modules (agent, eval, order, tools...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
