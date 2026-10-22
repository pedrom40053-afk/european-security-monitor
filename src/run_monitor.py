"""Optional local refresh loop for the educational demo."""
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().with_name("update_data.py")
CHECK_INTERVAL_SECONDS = 15 * 60

if __name__ == "__main__":
    print("Educational metadata refresh loop. Press Ctrl+C to stop.")
    while True:
        try:
            subprocess.run([sys.executable, str(SCRIPT)], check=False)
            time.sleep(CHECK_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("Stopped.")
            break
