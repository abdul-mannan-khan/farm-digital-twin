"""Run tests/test_offline.py without pytest (Isaac Sim's bundled Python has numpy but not pytest)."""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_offline as T  # noqa: E402

failed = 0
for name in sorted(n for n in dir(T) if n.startswith("test_")):
    try:
        getattr(T, name)()
        print("PASS", name)
    except Exception:
        failed += 1
        print("FAIL", name)
        traceback.print_exc()
print(f"{failed} failed" if failed else "all passed")
sys.exit(1 if failed else 0)
