import subprocess
import sys
import time
from datetime import datetime


CHECK_INTERVAL_SECONDS = 15 * 60
CHECK_INTERVAL_MINUTES = CHECK_INTERVAL_SECONDS // 60


def run_update():

    print()
    print("=" * 60)

    print(
        "Security Monitor check:",
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    print("=" * 60)

    subprocess.run(
        [
            sys.executable,
            "src/update_data.py"
        ],
        check=False
    )


def main():

    print()
    print(
        "European Security Monitor "
        "- Automatic Update Service"
    )

    print(
        f"Checking GDELT every "
        f"{CHECK_INTERVAL_MINUTES} minutes."
    )

    print(
        "Press CTRL+C to stop."
    )

    while True:

        run_update()

        print()

        print(
            f"Next check in "
            f"{CHECK_INTERVAL_MINUTES} minutes..."
        )

        time.sleep(
            CHECK_INTERVAL_SECONDS
        )


if __name__ == "__main__":
    main()