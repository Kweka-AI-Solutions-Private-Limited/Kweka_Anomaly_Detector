import sys
from pathlib import Path
import pytest

if __name__ == "__main__":
    src_dir = Path(__file__).resolve().parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    code = pytest.main(["-v", str(Path(__file__).parent / "test_history.py")])
    sys.exit(code)
