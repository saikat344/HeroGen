"""Suggested hero ideas: the frozen Muse Spark prompt and reply checks (herogen/pipeline/idea.py), and
`POST /api/ideas` (session, CSRF, rate limit, generation switch, spend), with synthetic model answers only."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import unittest
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from test_herogen_service_support import HAVE_WEB, web_test  # noqa: E402

from herogen import prompts  # noqa: E402
from herogen.pipeline import idea  # noqa: E402
from herogen.pipeline.client import ApiError, CallLog, ClientError  # noqa: E402

IDEA = ("A cheerful beekeeper robot with a honey-gold enamel body, a white mesh veil hat and a striped canvas apron, "
        "carrying a small wicker hive. Full body, standing, plain background.")
SECRET = "model said: Iron Man, the user's secret brief"   # text that must never reach a person


class FakeClient:
    """Answers in turn: a dict is the parsed answer, an exception is raised. Every call costs $0.006."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.log = CallLog()
        self.sent: list[tuple[str, dict]] = []

    def compare_images(self, prompt, images, **kw):
        assert images == []
        self.sent.append((prompt, kw))
        self.log.add(kw["model"], "compare", f"k{len(self.sent)}", cached=False, cost_usd=0.006)
        a = self.answers.pop(0)
        if isinstance(a, BaseException):
            raise a
        return a


class Prompt(unittest.TestCase):
    def test_frozen_version_asks_for_one_original_family_friendly_full_body_hero(self):
        v = prompts.selected("hero_idea")
        step = v.step("idea")
        self.assertEqual((v.name, step["model"]), ("hero-idea-v2", "muse-spark-1.3"))
        self.assertEqual(v.data["limits"], {"min_chars": 20, "max_chars": 300})
        text = idea.request_prompt(v, "frog knight", "brave")
        self.assertIn("a brave frog knight", text)
        self.assertNotIn("{theme}", text)
        self.assertNotIn("{mood}", text)
        for rule in ("ONE playful, original hero", "Exactly one character, shown full body and standing",
                     "clear colours and materials for the outfit and worn accessories", "with empty hands",
                     "no companions, pets", "Nothing held or carried", "Worn clothing and accessories are fine", "brand", "franchise",
                     "trademarked character", "real-person", "no gore", "anything sexual", "at most 240 characters",
                     "Full body, standing, plain background.", "Reply with JSON only."):
            with self.subTest(rule=rule):
                self.assertIn(rule, text)

    def test_seeds_vary_the_request(self):
        v = prompts.selected("hero_idea")
        self.assertGreaterEqual(len(v.data["seed"]["themes"]) * len(v.data["seed"]["moods"]), 400)
        rng = random.Random(7)
        asked = {idea.request_prompt(v, *idea.pick_seed(v, rng)) for _ in range(20)}
        self.assertGreater(len(asked), 15, "twenty requests are almost never the same request")


class Check(unittest.TestCase):
    def test_strips_quotes_markdown_and_labels(self):
        for raw in (IDEA, f'"{IDEA}"', f"**{IDEA}**", f"“{IDEA}”", f"Idea: {IDEA}", f'Description: "{IDEA}"',
                    f"  {IDEA}\n", f"- {IDEA}", f"`{IDEA}`", f"# {IDEA}"):
            with self.subTest(raw=raw[:16]):
                self.assertEqual(idea.check({"description": raw}), IDEA)

    def test_refuses_short_long_empty_and_refusals(self):
        for raw, why in (("A hero.", "at least 20"), ("x " * 200, "can't be used"), ("", "can't be used"),
                         ("   ", "can't be used"), (None, "no description"),
                         ("I'm sorry, but I can't help with creating that character description.", "refusal"),
                         ("I cannot create that. As an AI, I must decline this request politely.", "refusal"),
                         ("Sorry, that request goes against my guidelines for characters here.", "refusal")):
            with self.subTest(raw=str(raw)[:20]), self.assertRaisesRegex(idea.IdeaRejected, why):
                idea.check({"description": raw})
        with self.assertRaises(idea.IdeaRejected):
            idea.check("not a dict")

    def test_a_hero_who_cannot_fly_is_not_a_refusal(self):
        text = "A grumpy penguin pilot who can't fly, in a red leather jacket and brass goggles. Full body, standing."
        self.assertEqual(idea.check({"description": text}), text)


class Suggest(unittest.TestCase):
    def test_success_is_one_call_with_its_cost(self):
        client = FakeClient({"description": f'"{IDEA}"'})
        rec = idea.suggest(client, rng=random.Random(1))
        self.assertEqual((rec["status"], rec["idea"], rec["attempts"], rec["cost_usd"]), ("ok", IDEA, 1, 0.006))
        self.assertEqual(rec["model"], "muse-spark-1.3")
        self.assertEqual(rec["prompt_version"]["name"], "hero-idea-v2")
        prompt, kw = client.sent[0]
        self.assertEqual((kw["model"], kw["reasoning_effort"], kw["max_output_tokens"]), ("muse-spark-1.3", "low", 3000))
        self.assertEqual(kw["schema"]["name"], "hero_idea")
        self.assertIn(f"a {rec['seed']['mood']} {rec['seed']['theme']}", prompt)

    def test_an_unusable_reply_is_asked_once_more_with_a_new_seed(self):
        for first in ({"description": "I'm sorry, I can't help with that request at all."}, {"description": "Hi"},
                      ClientError("the answer does not fit the schema"),
                      ApiError("responses: the response status is 'incomplete'")):
            with self.subTest(first=str(first)[:30]):
                client = FakeClient(first, {"description": IDEA})
                rec = idea.suggest(client, rng=random.Random(3))
                self.assertEqual((rec["status"], rec["idea"], rec["attempts"], rec["cost_usd"]), ("ok", IDEA, 2, 0.012))

    def test_two_unusable_replies_fail_without_keeping_model_text(self):
        client = FakeClient({"description": SECRET[:12]}, {"description": f"I'm sorry, I can't: {SECRET}"})
        rec = idea.suggest(client)
        self.assertEqual((rec["status"], rec["error"], rec["attempts"], rec["cost_usd"]), ("failed", "unusable", 2, 0.012))
        self.assertNotIn("idea", rec)
        self.assertNotIn("Iron", json.dumps(rec))

    def test_a_content_policy_refusal_or_failed_call_ends_at_once(self):
        for exc, error in ((ApiError(SECRET, http_status=400, error_code="content_policy_violation"), "refused"),
                           (ApiError(SECRET, http_status=500), "model_error"),
                           (ApiError(SECRET, network_error="TimeoutError"), "model_error")):
            with self.subTest(error=error):
                client = FakeClient(exc, {"description": IDEA})
                rec = idea.suggest(client)
                self.assertEqual((rec["status"], rec["error"], rec["attempts"]), ("failed", error, 1))
                self.assertEqual(len(client.sent), 1)
                self.assertNotIn("Iron", json.dumps(rec))


class ServiceSource(unittest.TestCase):
    def test_live_without_a_key_calls_nothing(self):
        from unittest import mock

        from herogen.service import ideas

        with mock.patch.dict(os.environ, {"MODEL_API_KEY": ""}), \
                mock.patch("herogen.pipeline.client.ModelClient", side_effect=AssertionError("no call")):
            self.assertEqual(ideas.live(), {"status": "failed", "error": "unavailable", "cost_usd": 0.0})

    def test_transportable_keeps_only_shown_and_counted_fields(self):
        from herogen.service import ideas

        rec = idea.suggest(FakeClient({"description": IDEA}))
        out = ideas.transportable(rec)
        self.assertEqual(set(out), {"status", "idea", "cost_usd", "model", "attempts", "seconds", "prompt_version"})
        self.assertEqual(out["prompt_version"], "hero-idea-v2")

    def test_unknown_outcomes_count_every_attempt_at_its_cap(self):
        from herogen.service import ideas

        self.assertAlmostEqual(ideas.unknown_cost_usd(), 2 * 3000 * 4.25 / 1e6)


if HAVE_WEB:
    from test_herogen_service_web import Browser, WebCase

    from herogen.service.web import create_app
else:  # pragma: no cover
    WebCase = unittest.TestCase


@web_test
class IdeasHTTP(WebCase):
    def setUp(self):
        super().setUp()
        self.calls = 0
        self.answer = {"status": "ok", "idea": IDEA, "cost_usd": 0.0061, "model": "muse-spark-1.3",
                       "prompt_version": "hero-idea-v2", "attempts": 1}
        self.svc.ideas = self.ideas
        self.app = create_app(self.svc)

    def ideas(self):
        self.calls += 1
        if isinstance(self.answer, BaseException):
            raise self.answer
        return dict(self.answer)

    def guest(self) -> Browser:
        b = Browser(self.app)
        b.session()
        r = b.post("/api/demo/access", json={})
        self.assertEqual(r.status_code, 200, r.text)
        b.csrf = r.json()["csrf"]
        return b

    def spend(self):
        return [dict(r) for r in self.store.db.execute("SELECT user_id, purpose, model, outcome, cost_usd FROM model_spend")]

    def test_success_returns_the_idea_and_counts_its_cost_in_the_spend(self):
        b = self.guest()
        before = self.store.generation_get()["spent_usd"]
        r = b.post("/api/ideas", json={})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), {"idea": IDEA, "prompt_version": "hero-idea-v2"})
        rows = self.spend()
        self.assertEqual([(x["purpose"], x["model"], x["outcome"], x["cost_usd"]) for x in rows],
                         [("idea", "muse-spark-1.3", "ok", 0.0061)])
        status = self.store.generation_get()
        self.assertAlmostEqual(status["spent_usd"], before + 0.0061)
        self.assertEqual((status["side_calls"], status["side_spend_usd"]), (1, 0.0061))

    def test_needs_a_session_and_csrf(self):
        anonymous = Browser(self.app)
        anonymous.session()
        self.assertEqual(anonymous.post("/api/ideas", json={}).status_code, 401)
        b = self.guest()
        self.assertEqual(b.post("/api/ideas", json={}, csrf=False).status_code, 403)
        self.assertEqual(b.post("/api/ideas", json={}, headers={"X-CSRF-Token": "x" * 43}).status_code, 403)
        self.assertEqual(b.post("/api/ideas", json={}, origin=False, headers={"Origin": "https://evil.example"})
                         .status_code, 403)
        self.assertEqual(self.calls, 0)
        self.assertEqual(self.spend(), [])

    def test_rate_limit_per_user(self):
        self.svc.settings = replace(self.s, rate_ideas_user=(2, 60))
        self.app = create_app(self.svc)
        b, other = self.guest(), self.guest()
        self.assertEqual([b.post("/api/ideas", json={}).status_code for _ in range(3)], [200, 200, 429])
        r = b.post("/api/ideas", json={})
        self.assertEqual(r.json()["error"]["code"], "rate_limited")
        self.assertIn("retry-after", r.headers)
        self.assertEqual(other.post("/api/ideas", json={}).status_code, 200, "the limit is per user")
        self.assertEqual(self.calls, 3)
        self.clock.advance(61)
        self.assertEqual(b.post("/api/ideas", json={}).status_code, 200)

    def test_a_refusal_or_model_error_is_a_fixed_sentence_and_still_counted(self):
        b = self.guest()
        for rec, status, code in (({"status": "failed", "error": "refused", "cost_usd": 0.002}, 422, "idea_refused"),
                                  ({"status": "failed", "error": "model_error", "cost_usd": 0.0}, 502, "idea_failed"),
                                  ({"status": "failed", "error": "unusable", "cost_usd": 0.012}, 502, "idea_failed"),
                                  ({"status": "ok", "idea": "", "cost_usd": 0.004}, 502, "idea_failed")):
            with self.subTest(code=code, error=rec.get("error")):
                self.answer = {**rec, "message": SECRET}
                r = b.post("/api/ideas", json={})
                self.assertEqual((r.status_code, r.json()["error"]["code"]), (status, code))
                self.assertNotIn("Iron", r.text)
        self.assertEqual([x["outcome"] for x in self.spend()], ["refused", "model_error", "unusable", "model_error"])
        self.assertAlmostEqual(self.store.generation_get()["side_spend_usd"], 0.018)

    def test_an_exception_is_counted_at_the_bound_and_never_echoed(self):
        from herogen.service.ideas import unknown_cost_usd

        b = self.guest()
        self.answer = RuntimeError(SECRET)
        with self.assertLogs("herogen.service", level=logging.ERROR) as logs:
            r = b.post("/api/ideas", json={})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (502, "idea_failed"))
        self.assertNotIn("Iron", r.text)
        self.assertNotIn("Iron", "\n".join(logs.output))
        self.assertEqual([(x["outcome"], x["cost_usd"]) for x in self.spend()], [("unknown", unknown_cost_usd())])

    def test_no_key_says_unavailable_without_counting(self):
        b = self.guest()
        self.answer = {"status": "failed", "error": "unavailable", "cost_usd": 0.0}
        r = b.post("/api/ideas", json={})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (503, "ideas_unavailable"))
        self.assertEqual(self.spend(), [])

    def test_a_server_without_ideas_says_unavailable(self):
        self.svc.ideas = None
        self.app = create_app(self.svc)
        r = self.guest().post("/api/ideas", json={})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (503, "ideas_unavailable"))

    def test_paused_or_spent_generation_refuses_before_any_call(self):
        b = self.guest()
        self.store.generation_update(actor="fixture", enabled=False)
        r = b.post("/api/ideas", json={})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (503, "generation_paused"))
        self.store.generation_update(actor="fixture", enabled=True, spend_limits_enabled=True, limit_usd=0)
        r = b.post("/api/ideas", json={})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (503, "generation_budget_exhausted"))
        self.assertEqual(self.calls, 0)

    def test_idea_spend_can_exhaust_the_budget(self):
        b = self.guest()
        self.store.generation_update(actor="fixture", spend_limits_enabled=True, limit_usd=1.51, running_estimate_usd=1.5)
        self.answer = {**self.answer, "cost_usd": 0.02}
        self.assertEqual(b.post("/api/ideas", json={}).status_code, 200)
        self.assertEqual(self.store.generation_get()["reason"], "generation_budget_exhausted")
        self.assertEqual(b.post("/api/ideas", json={}).status_code, 503)


class Store(unittest.TestCase):
    def test_spend_record_refuses_bad_values(self):
        import tempfile
        from pathlib import Path

        from herogen.service.store import Store as MetaStore

        with tempfile.TemporaryDirectory() as d:
            st = MetaStore(Path(d) / "m.db")
            try:
                for args in (("u", "other", "m", "ok", 0.1), ("u", "idea", "m", "bogus", 0.1),
                             ("u", "idea", "m", "ok", -1), ("u", "idea", "m", "ok", float("nan")),
                             ("u", "idea", "m", "ok", True)):
                    with self.subTest(args=args), self.assertRaises(ValueError):
                        st.spend_record(*args)
                st.spend_record("u", "idea", "muse-spark-1.3", "ok", 0.0061)
                self.assertEqual(st.generation_get()["side_calls"], 1)
                self.assertIn("spend_record", MetaStore.PUBLIC)
            finally:
                st.close()


if __name__ == "__main__":
    unittest.main()
