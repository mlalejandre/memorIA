from __future__ import annotations

import sys

from bucle import run_loop
from servidor import start_remote_server_if_needed, stop_remote_server


def main() -> int:
    try:
        run_loop(start_remote_server_if_needed, stop_remote_server)
    except KeyboardInterrupt:
        print("\nDetenido por el usuario.", flush=True)
        stop_remote_server()
    except RuntimeError as exc:
        print(f"\n❌ {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
