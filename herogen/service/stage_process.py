"""Windows stage entry point: wait for durable registration before creating any descendants."""
import subprocess
import sys

from herogen.pipeline.ledger import BRIDGE
from herogen.service.refine_budget import remote_budget


def main():
    remote_budget('stage_ready', prefix=BRIDGE)
    return subprocess.call(sys.argv[1:])


if __name__ == '__main__':
    raise SystemExit(main())
