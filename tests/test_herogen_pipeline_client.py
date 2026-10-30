"""The model client keeps a paid SAM response even when decoding it fails, and asks Muse Spark about images with a
strict schema, checking every answer against it."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.pipeline import retry
from fake_model_server import FakeModelServer, error_body, no_wait_policy
from test_herogen_pipeline_flow import DUMMY_KEY
from herogen.pipeline import client as client_mod  # noqa: E402
from herogen.pipeline.client import ApiError, ClientError, ModelClient, ReplayMiss, schema_errors  # noqa: E402

_RESPONSE = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "<masks>"}]}]}


class SamPaidResponseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "cache"
        self.image = Image.new("RGB", (16, 16), "white")

    def tearDown(self):
        self.tmp.cleanup()

    def test_failed_decode_keeps_the_response_and_bills_once(self):
        posts = []

        def fake_post(_client, endpoint, body, **_):
            posts.append(endpoint)
            return _RESPONSE

        def broken_decode(text, size):
            raise ClientError("no parser")

        with mock.patch.object(ModelClient, "_post", fake_post), \
                mock.patch.object(client_mod, "decode_sam_output", broken_decode):
            c = ModelClient(self.cache, mode="live")
            with self.assertRaises(ClientError):
                c.segment(self.image, "character")
            with self.assertRaises(ClientError):
                c.segment(self.image, "character")  # decodes the saved text again, no second POST
        self.assertEqual(posts, ["responses"])
        self.assertEqual([call["cached"] for call in c.log.calls], [False, True])
        self.assertAlmostEqual(c.log.cost_usd(), client_mod.PRICE_PER_IMAGE_USD[client_mod.SAM_MODEL])

        with mock.patch.object(client_mod, "decode_sam_output", lambda text, size: []):
            replay = ModelClient(self.cache, mode="replay")
            self.assertEqual(replay.segment(self.image, "character"), [])  # recovered from the cache offline


SCHEMA = {"name": "verdict", "schema": {
    "type": "object",
    "properties": {"score": {"type": "number"}, "kind": {"type": "string", "enum": ["a", "b"]},
                   "tags": {"type": "array", "items": {"type": "string"}}},
    "required": ["score", "kind", "tags"], "additionalProperties": False}}


def _answer(text: str, status: str = "completed") -> dict:
    return {"status": status, "usage": {"input_tokens": 1000, "output_tokens": 200},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}


class CompareImagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "cache"
        self.image = Image.new("RGB", (2048, 1024), "white")

    def tearDown(self):
        self.tmp.cleanup()

    def _live(self, resp):
        posts = []

        def fake_post(_client, endpoint, body, **kw):
            posts.append((endpoint, body, kw))
            return resp

        with mock.patch.object(ModelClient, "_post", fake_post):
            c = ModelClient(self.cache, mode="live")
            try:
                return c.compare_images("prompt", [self.image], schema=SCHEMA), c, posts
            except ClientError as e:
                return e, c, posts

    def test_answer_is_recorded_priced_and_replayed(self):
        out, c, posts = self._live(_answer('{"score": 0.25, "kind": "a", "tags": ["x"]}'))
        self.assertEqual(out, {"score": 0.25, "kind": "a", "tags": ["x"]})
        endpoint, body, kw = posts[0]
        self.assertEqual((endpoint, kw["model"], kw["kind"]), ("responses", client_mod.TEXT_MODEL, "compare"))
        self.assertEqual(body["text"]["format"]["strict"], True)
        self.assertTrue(body["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertAlmostEqual(c.log.cost_usd(), (1000 * 1.25 + 200 * 4.25) / 1e6)
        replay = ModelClient(self.cache, mode="replay")
        self.assertEqual(replay.compare_images("prompt", [self.image], schema=SCHEMA), out)
        self.assertEqual(replay.log.cost_usd(), 0)
        with self.assertRaises(ReplayMiss):
            replay.compare_images("another prompt", [self.image], schema=SCHEMA)

    def test_a_non_conforming_answer_is_never_guessed(self):
        for text, why in (('{"score": 0.2, "kind": "c", "tags": []}', "not one of"),
                          ('{"score": 0.2, "tags": []}', "kind is missing"),
                          ('{"score": "high", "kind": "a", "tags": []}', "should be number"),
                          ('{"score": 1, "kind": "a", "tags": [], "extra": 1}', "extra is not in the schema"),
                          ("not json", "is not JSON"), ("", "no output text")):
            self.tmp.cleanup()
            self.tmp = tempfile.TemporaryDirectory()
            self.cache = Path(self.tmp.name) / "cache"
            err, _, _ = self._live(_answer(text))
            self.assertIsInstance(err, ClientError, text)
            self.assertIn(why, str(err))

    def test_an_incomplete_answer_is_a_failure_and_not_cached(self):
        err, c, _ = self._live(_answer('{"score": 0.2, "kind": "a", "tags": []}', status="incomplete"))
        self.assertIsInstance(err, ApiError)
        self.assertGreater(c.log.cost_usd(), 0)  # answered, so billed
        self.assertFalse(list(self.cache.glob("*.json")))

    def test_schema_errors(self):
        self.assertEqual(schema_errors({"score": 1, "kind": "b", "tags": ["t"]}, SCHEMA["schema"]), [])
        self.assertEqual(schema_errors(True, {"type": "number"}), ["answer should be number, got bool"])
        self.assertIn("can't be checked", schema_errors(1, {"type": "number", "minimum": 0})[0])


def _png_b64(colour: str) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


SAMPLE_SCHEMA = {"name": "verdict", "schema": {"type": "object", "properties": {"same": {"type": "string"}},
                                        "required": ["same"], "additionalProperties": False}}


class GenerateAttemptTest(unittest.TestCase):
    def test_attempt_is_a_new_sample_with_its_own_key_and_same_body(self):
        bodies = []

        def fake_post(_client, endpoint, body, **_):
            bodies.append(body)
            return {"data": [{"b64_json": _png_b64("white" if len(bodies) == 1 else "black")}]}

        with tempfile.TemporaryDirectory() as d, mock.patch.object(ModelClient, "_post", fake_post):
            c = ModelClient(d, mode="live")
            a = c.generate_image("p", size="64x64")
            b = c.generate_image("p", size="64x64", attempt=1)
            self.assertNotEqual(a.getpixel((0, 0)), b.getpixel((0, 0)))
            self.assertEqual(bodies[0], bodies[1])  # the API takes no seed; only the cache key differs
            self.assertNotIn("seed", bodies[0])
            self.assertEqual(c.generate_image("p", size="64x64").getpixel((0, 0)), a.getpixel((0, 0)))  # cached
            self.assertEqual(len(bodies), 2)


class SampleCompareImagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.images = [Image.new("RGB", (2000, 1000), "white"), Image.new("RGB", (300, 600), "grey")]

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _resp(text, usage=None):
        return {"status": "completed", "usage": usage or {"input_tokens": 2000, "output_tokens": 1000},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}

    def test_request_shape_cost_and_replay(self):
        seen = []

        def fake_post(_client, endpoint, body, **kw):
            seen.append((endpoint, body, kw))
            return self._resp('{"same": "yes"}')

        with mock.patch.object(ModelClient, "_post", fake_post):
            c = ModelClient(self.tmp.name, mode="live")
            out = c.compare_images("question", self.images, schema=SAMPLE_SCHEMA, image_max_side=1024)
        self.assertEqual(out, {"same": "yes"})
        endpoint, body, kw = seen[0]
        self.assertEqual(endpoint, "responses")
        # Through the one request path, with the cost an attempt of unknown outcome is counted at: the output cap.
        self.assertEqual(kw, {"model": client_mod.TEXT_MODEL, "kind": "compare", "key": c.log.calls[0]["key"],
                              "unknown_cost_usd": 3000 * 4.25 / 1e6})
        self.assertEqual(body["model"], client_mod.TEXT_MODEL)
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertEqual(body["text"]["format"]["strict"], True)
        content = body["input"][0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "question"})
        first = Image.open(io.BytesIO(base64.b64decode(content[1]["image_url"].split(",", 1)[1])))
        self.assertEqual((first.format, first.size), ("JPEG", (1024, 512)))
        want = (2000 * 1.25 + 1000 * 4.25) / 1e6
        self.assertAlmostEqual(c.log.cost_usd(), want)
        self.assertEqual(c.log.calls[0]["cost_usd"], round(want, 6))

        def no_network(*_a, **_k):
            raise AssertionError("replay must not call the network")

        with mock.patch.object(ModelClient, "_post", no_network):
            r = ModelClient(self.tmp.name, mode="replay")
            self.assertEqual(r.compare_images("question", self.images, schema=SAMPLE_SCHEMA, image_max_side=1024),
                             {"same": "yes"})
            self.assertEqual(r.log.cost_usd(), 0.0)
            with self.assertRaises(client_mod.ReplayMiss):
                r.compare_images("question", self.images, schema=SAMPLE_SCHEMA, image_max_side=1024, attempt=1)

    def test_bad_answers_raise(self):
        for text in ("", "not json", "[1, 2]"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as d, \
                    mock.patch.object(ModelClient, "_post", lambda c, e, b, t=text, **_: self._resp(t)):
                with self.assertRaises(ClientError):
                    ModelClient(d, mode="live").compare_images("q", self.images, schema=SAMPLE_SCHEMA)
                saved = [json.loads(p.read_text()) for p in Path(d).glob("*.json")]
                self.assertEqual(len(saved), 1)  # the paid answer is kept for inspection

    def test_incomplete_answer_is_an_api_error_and_not_cached(self):
        resp = {**self._resp('{"same": "yes"}'), "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"}}
        with mock.patch.object(ModelClient, "_post", lambda c, e, b, **_: resp):
            c = ModelClient(self.tmp.name, mode="live")
            with self.assertRaisesRegex(client_mod.ApiError, "not 'completed'"):
                c.compare_images("q", self.images, schema=SAMPLE_SCHEMA)
        self.assertEqual(list(Path(self.tmp.name).glob("*.json")), [])
        self.assertGreater(c.log.cost_usd(), 0)  # answered, so billed and logged

    def test_non_conforming_answers_raise_live_and_on_replay(self):
        with mock.patch.object(ModelClient, "_post", lambda c, e, b, **_: self._resp('{"same": 1}')):
            with self.assertRaisesRegex(ClientError, "answer.same should be string"):
                ModelClient(self.tmp.name, mode="live").compare_images("q", self.images, schema=SAMPLE_SCHEMA)
        with self.assertRaisesRegex(ClientError, "does not match schema 'verdict'"):
            ModelClient(self.tmp.name, mode="replay").compare_images("q", self.images, schema=SAMPLE_SCHEMA)

    def test_unpriced_model_is_an_error(self):
        with self.assertRaises(ClientError):
            client_mod.token_cost_usd("no-such-model", {"input_tokens": 1})


class CompareThroughServerTest(unittest.TestCase):
    """compare_images goes through the real request path: the retry policy, the shared limiter and the ledger."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "cache"
        self.limiter = retry.SharedLimiter(Path(self.tmp.name) / "limiter", "k")
        self.images = [Image.new("RGB", (32, 32), "white"), Image.new("RGB", (32, 32), "grey")]
        self.ok = (200, {}, SampleCompareImagesTest._resp('{"same": "yes"}'))

    def tearDown(self):
        self.tmp.cleanup()

    def _compare(self, *answers):
        queue = list(answers)
        server = FakeModelServer(lambda e, b: queue.pop(0) if len(queue) > 1 else queue[0], DUMMY_KEY)
        c = ModelClient(self.cache, mode="live", policy=no_wait_policy(), limiter=self.limiter)
        with server.serving(self.tmp.name + "/limiter"), mock.patch.dict(os.environ, {"MODEL_API_KEY": DUMMY_KEY}), \
                self.assertLogs("herogen.pipeline.client", "WARNING"):
            out = c.compare_images("q", self.images, schema=SAMPLE_SCHEMA)
        rows = [json.loads(x) for x in (self.cache / client_mod.LEDGER).read_text().splitlines()]
        return out, c, server, rows

    def test_a_429_is_retried_and_lowers_the_shared_limit(self):
        out, c, server, rows = self._compare((429, {}, error_body(429)), self.ok)
        self.assertEqual(out, {"same": "yes"})
        self.assertEqual(server.count("responses"), 2)
        self.assertEqual([(r["model"], r["attempt"], r["status"], r["outcome"]) for r in rows],
                         [(client_mod.TEXT_MODEL, 1, 429, "failed"), (client_mod.TEXT_MODEL, 2, 200, "ok")])
        self.assertEqual(json.loads(self.limiter.state_path.read_text())["limit"], retry.LIMIT_START // 2)
        self.assertEqual([x["cached"] for x in c.log.calls], [False])  # a 429 is not billed
        self.assertAlmostEqual(c.log.cost_usd(), (2000 * 1.25 + 1000 * 4.25) / 1e6)

    def test_an_attempt_of_unknown_outcome_is_counted_at_the_output_cap(self):
        out, c, _, rows = self._compare((0, {}, None), self.ok)  # the first connection is dropped unanswered
        self.assertEqual(out, {"same": "yes"})
        self.assertEqual([r["outcome"] for r in rows], ["unknown", "ok"])
        self.assertEqual([x.get("outcome") for x in c.log.calls], ["unknown", None])
        self.assertEqual(c.log.calls[0]["cost_usd"], round(3000 * 4.25 / 1e6, 6))
        self.assertAlmostEqual(c.log.cost_usd(), (3000 * 4.25 + 2000 * 1.25 + 1000 * 4.25) / 1e6)

    def test_a_token_priced_call_needs_its_unknown_outcome_cost(self):
        c = ModelClient(self.cache, mode="live", policy=no_wait_policy(), limiter=self.limiter)
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": DUMMY_KEY}), \
                self.assertRaisesRegex(ClientError, "unknown outcome"):
            c._post("responses", {}, model=client_mod.TEXT_MODEL, kind="compare", key="k")
        self.assertFalse((self.cache / client_mod.LEDGER).exists())  # refused before any attempt


class SchemaErrorsTest(unittest.TestCase):
    S = {"type": "object", "additionalProperties": False, "required": ["v", "xs"],
         "properties": {"v": {"type": "string", "enum": ["yes", "no"]}, "xs": {"type": "array", "items": {"type": "string"}},
                        "n": {"type": "integer"}}}

    def test_valid(self):
        self.assertEqual(client_mod.schema_errors({"v": "yes", "xs": ["a"], "n": 2}, self.S), [])

    def test_each_kind_of_error(self):
        errs = client_mod.schema_errors({"v": "maybe", "xs": ["a", 1], "n": True, "z": 0}, self.S)
        self.assertEqual(errs, ["answer.z is not in the schema", "answer.v is 'maybe', not one of ['yes', 'no']",
                                "answer.xs[1] should be string, got int", "answer.n should be integer, got bool"])
        self.assertEqual(client_mod.schema_errors({}, self.S), ["answer.v is missing", "answer.xs is missing"])
        self.assertEqual(client_mod.schema_errors([], self.S), ["answer should be object, got list"])

    def test_unsupported_keywords_are_errors_not_skipped(self):
        self.assertIn("can't be checked", client_mod.schema_errors("x", {"type": "string", "pattern": "^x$"})[0])
        self.assertIn("can't be checked", client_mod.schema_errors("x", {"type": "date"})[0])


if __name__ == "__main__":
    unittest.main()
