import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from herogen.blender.remote import client as remote_client
from herogen.pipeline import events
from herogen.pipeline.fast import tracked


class _Log:
    def __init__(self):
        self.calls = []


class _Remote:
    """A fast-lane client whose GPU call records the cost the remote client gives a GPU with no list price."""
    fast = True

    def __init__(self):
        self.log = _Log()

    @tracked("sam-3d-objects", "gpu", isolate=True)
    def objects(self):
        self.log.calls.append({"model": "sam-3d-objects", "kind": "gpu", "cached": False, "cost_usd": None})
        return "mesh"


class UnpricedGpuTest(unittest.TestCase):
    def test_a_gpu_without_a_list_price_does_not_stop_the_stage(self):
        remote = _Remote()
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "events.jsonl"
            with mock.patch.dict("os.environ", {events.ENV_FILE: str(log)}), events.step("3"):
                self.assertEqual(remote.objects(), "mesh")
            lines = [json.loads(line) for line in log.read_text().splitlines()]
        end = next(e for e in lines if e["kind"] == "gpu_end")
        self.assertEqual(end["usd"], 0.0)
        self.assertEqual([events.validate(e) for e in lines], lines)
        self.assertIsNone(remote.log.calls[0]["cost_usd"])   # the record still says the cost is unknown

    def test_h100_has_a_list_price(self):
        self.assertAlmostEqual(remote_client.PRICE_PER_SECOND["H100"] * 3600, 3.95)


if __name__ == "__main__":
    unittest.main()
