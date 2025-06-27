import os
import sys
from pathlib import Path

project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from app import app

if __name__ == "__main__":
    app.run()
