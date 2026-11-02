"""Run one shard of the unit tests.

    python3 tools/ci/test_shard.py <index> <count>

The same discovery as `python3 -m unittest discover -s tests -v`, split by test module so class and module fixtures
run once. Modules go largest first to the shard with the fewest tests so far, so every shard gets a similar share;
a module that fails to import is a module of its own and still fails its shard. Output matches `unittest -v`.
"""

from __future__ import annotations

import sys
import unittest


def tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from tests(item)
        else:
            yield item


def module(test) -> str:
    # unittest.loader._FailedTest.<module> is a module that failed to import.
    parts = test.id().split(".")
    return parts[-1] if parts[:2] == ["unittest", "loader"] else parts[0]


def shard(all_tests: list, index: int, count: int) -> list:
    by_module: dict[str, list] = {}
    for test in all_tests:
        by_module.setdefault(module(test), []).append(test)
    loads = [0] * count
    mine = []
    for name in sorted(by_module, key=lambda m: (-len(by_module[m]), m)):
        target = loads.index(min(loads))
        loads[target] += len(by_module[name])
        if target == index:
            mine += by_module[name]
    return mine


def main() -> int:
    index, count = int(sys.argv[1]), int(sys.argv[2])
    if not 0 <= index < count:
        sys.exit(f"shard index {index} is outside 0..{count - 1}")
    suite = unittest.TestSuite(shard(list(tests(unittest.defaultTestLoader.discover("tests"))), index, count))
    print(f"shard {index + 1}/{count}: {suite.countTestCases()} tests", file=sys.stderr)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
