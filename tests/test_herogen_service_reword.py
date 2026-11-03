"""The fast lane's one reword-and-retry after a content-policy refusal (herogen/pipeline/reword.py and
`herogen.service.runner.reword_and_retry`), with synthetic Model API answers only."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from tests import test_herogen_service_jobs as jobs_base
from tests import test_herogen_service_prompt as prompt_base

from herogen import prompts
from herogen.pipeline import events, reword
from herogen.pipeline.client import ApiError, CallLog
from herogen.service.fakes import FAKE_REWORD, FakeRunner
from herogen.service.runner import PLAN, Outcome, StageContext, reword_due

BRIEF = "A transformer style robot, with red armor, Ironman inspired"
REWORDED = "An original armoured robot in red plating with a plain glowing chest light"
REMOVED = ["Ironman", "transformer"]
FAST = {"HEROGEN_FAST_LANE": "1"}


def policy(text="refused"):
    return ApiError(text, http_status=400, error_code="content_policy_violation")


def answer(reworded=REWORDED, removed=REMOVED):
    return {"status": "completed", "usage": {"input_tokens": 400, "output_tokens": 120},
            "output": [{"type": "message", "content": [{"type": "output_text",
                                                         "text": json.dumps({"reworded": reworded,
                                                                             "removed_names": removed})}]}]}


def read_events(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / events.FILE).read_text().splitlines()] if (run / events.FILE).exists() else []


class RewriteRules(unittest.TestCase):
    def test_guidance_asks_for_an_original_character_without_names_or_logos(self):
        version = prompts.selected("brief_reword")
        self.assertEqual((version.name, version.step("reword")["model"]), ("brief-reword-v1", "muse-spark-1.3"))
        text = reword.request_prompt(version, BRIEF)
        self.assertEqual(text.count(BRIEF), 1)
        for rule in ("ORIGINAL character, not a lookalike", "brand", "franchise", "trademarked character",
                     "real-person", "logos, emblems and signature trademark costume details", "arc reactor",
                     "plain glowing chest light is fine", "general archetype", "'armoured robot'",
                     "colours, materials, gear, props and silhouette", "Do not mention any brand, name or logo at all",
                     "removed_names"):
            with self.subTest(rule=rule):
                self.assertIn(rule, text)

    def test_check_passes_a_clean_rewrite_and_refuses_one_that_still_names_a_removed_term(self):
        self.assertEqual(reword.check(BRIEF, {"reworded": f"  {REWORDED} ", "removed_names": REMOVED}),
                         (REWORDED, REMOVED))
        for bad, why in ((f"{REWORDED}, Iron-Man style", "still names"), ("A TRANSFORMER robot", "still names"),
                         (BRIEF, "same as the brief"), ("", "can't be used"), ("x" * 400, "can't be used")):
            with self.subTest(bad=bad[:20]), self.assertRaisesRegex(reword.RewordRejected, why):
                reword.check(BRIEF, {"reworded": bad, "removed_names": REMOVED})

    def test_rewrite_sends_the_brief_once_and_passes_the_canned_rewrite_through(self):
        sent = []

        class Client:
            log = CallLog()

            def compare_images(self, prompt, images, **kw):
                sent.append((prompt, images, kw))
                self.log.add("muse-spark-1.3", "compare", "k", cached=False, cost_usd=0.001)
                return {"reworded": REWORDED, "removed_names": REMOVED}

        client = Client()
        rec = reword.rewrite(client, BRIEF)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][1], [])
        self.assertIn(f"«{BRIEF}»", sent[0][0])
        self.assertEqual(sent[0][2]["model"], "muse-spark-1.3")
        self.assertEqual(client.purpose, "reword brief")
        self.assertEqual((rec["status"], rec["original"], rec["reworded"], rec["removed_names"], rec["reason"]),
                         ("used", BRIEF, REWORDED, REMOVED, "content_policy"))
        self.assertEqual(rec["cost_usd"], 0.001)

    def test_only_a_used_rewrite_within_the_limit_is_shown(self):
        self.assertEqual(reword.shown({"status": "used", "reworded": f"  {REWORDED} "}), REWORDED)
        self.assertEqual(reword.shown({"status": "refused_again", "reworded": REWORDED}), REWORDED)
        for data in ({"status": "rewrite_failed", "reworded": REWORDED}, {"status": "used", "reworded": "x" * 301},
                     {"status": "used", "reworded": 7}, {"status": "used"}, None):
            with self.subTest(data=str(data)[:40]):
                self.assertIsNone(reword.shown(data))

    def test_only_a_used_rewrite_is_the_brief(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(reword.brief(Path(tmp)))
            for status, expected in (("used", REWORDED), ("refused_again", None), ("rewrite_failed", None)):
                reword.write(Path(tmp), {"status": status, "reworded": REWORDED})
                self.assertEqual(reword.brief(Path(tmp)), expected)


class RewordDue(unittest.TestCase):
    def ctx(self, tmp, kind="prompt", prompt=BRIEF):
        return StageContext(job_id="j_x", kind=kind, profile="roblox-ingame", name="", prompt=prompt,
                            run_dir=Path(tmp), input_path=Path(tmp) / "in.png", cache_dir=Path(tmp),
                            remote_cache_dir=Path(tmp))

    def test_only_a_fast_lane_policy_refusal_of_a_step_that_sent_the_brief_once(self):
        import tempfile

        refused = Outcome("rejected", code="rejected")
        spec = {s.key: s for s in PLAN}
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "stage-1.json").write_text(json.dumps({"details": {"brief_slot_used": True}}))
            (run / "stage-2.json").write_text(json.dumps({"details": {"prompt": BRIEF}}))
            ctx = self.ctx(tmp)
            with mock.patch.dict(os.environ, FAST):
                self.assertTrue(reword_due(spec["p"], ctx, refused, 4))
                self.assertTrue(reword_due(spec["1"], ctx, refused, 4))
                # The turnaround prompt holds no brief, so a rewrite could not change its request.
                self.assertFalse(reword_due(spec["2"], ctx, refused, 4))
                self.assertFalse(reword_due(spec["p"], ctx, refused, 6), "a gate failure is not a refusal")
                self.assertFalse(reword_due(spec["p"], ctx, refused, 5), "a transient error is not a refusal")
                self.assertFalse(reword_due(spec["p"], ctx, Outcome("failed"), 4))
                self.assertFalse(reword_due(spec["3"], ctx, refused, 4))
                self.assertFalse(reword_due(spec["1"], self.ctx(tmp, kind="sketch", prompt=""), refused, 4))
                reword.write(run, {"status": "used", "reworded": REWORDED})
                self.assertFalse(reword_due(spec["p"], ctx, refused, 4), "at most once per job")
                (run / reword.FILE).write_text("{not json")
                self.assertIsNone(reword.read(run))
                self.assertFalse(reword_due(spec["p"], ctx, refused, 4), "a corrupt record still means reworded")
            (run / reword.FILE).unlink()
            with mock.patch.dict(os.environ, {"HEROGEN_FAST_LANE": ""}):
                self.assertFalse(reword_due(spec["p"], ctx, refused, 4), "the full lane never rewords")


class RewordJobTest(prompt_base.PromptJobTest):
    """The real runner, pipestage and pipeline stages in process, with the Model API answered by fakes."""

    def setUp(self):
        super().setUp()
        mock.patch.dict(os.environ, FAST).start()
        self.description = BRIEF
        self.refusals = {"images/generations": 1}
        self.reword_answers = [answer()]
        self.reword_prompts = []
        self.generation_prompts = []
        self.edit_prompts = []

    def post(self, client, endpoint, body, **kw):
        if endpoint == "responses" and body.get("text", {}).get("format", {}).get("name") == "brief_reword":
            self.reword_prompts.append(body["input"][0]["content"][0]["text"])
            got = self.reword_answers[min(len(self.reword_prompts), len(self.reword_answers)) - 1]
            if isinstance(got, Exception):
                raise got
            return got
        if endpoint == "images/generations":
            self.generation_prompts.append(body["prompt"])
        if endpoint == "images/edits" and self.active_stage == "1":
            self.edit_prompts.append(body["prompt"])
        key = endpoint if self.active_stage == "p" or endpoint != "images/edits" else "cover"
        if self.refusals.get(key):
            self.refusals[key] -= 1
            raise policy(f"refused: {body['prompt']}")
        return super().post(client, endpoint, body, **kw)

    def subprocess(self, argv, **kw):
        """Route each model call by the stage command running it, so a sketch redrawn inside stage 1's retry
        is answered as stage p."""
        before = self.active_stage
        if argv[2] == "herogen.service.pipestage" and argv[3] in ("p", "1", "2"):
            self.active_stage = argv[3]
        try:
            return super().subprocess(argv, **kw)
        finally:
            self.active_stage = before

    def run_dir(self, job):
        return self.storage.run_dir(self.user["id"], job["id"])

    def view(self, job):
        from herogen.service.web import App

        app = App.__new__(App)
        app.svc = mock.Mock(storage=self.storage, store=self.store)
        return app._files(self.user["id"], self.store.job_get(job["id"]))

    def test_refused_sketch_is_reworded_once_and_the_retry_draws_the_rewrite(self):
        job = self.prompt_job()
        with self.assertLogs("herogen.service", "INFO") as logs:
            self.assertEqual(self.go(job), "succeeded")
        run = self.run_dir(job)
        self.assertEqual(len(self.reword_prompts), 1)
        self.assertIn(f"«{BRIEF}»", self.reword_prompts[0])
        self.assertEqual(len(self.generation_prompts), 2)
        self.assertIn("Ironman", self.generation_prompts[0])
        self.assertIn(REWORDED, self.generation_prompts[1])
        self.assertNotIn("Ironman", self.generation_prompts[1])
        data = reword.read(run)
        self.assertEqual((data["status"], data["stage"], data["original"], data["reworded"]), ("used", "p", BRIEF, REWORDED))
        rec = self.record(job, "p")
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["details"]["user_text"], REWORDED)
        self.assertEqual((rec["reword"]["original"], rec["reword"]["reworded"], rec["reword"]["reason"]),
                         (BRIEF, REWORDED, "content_policy"))
        self.assertGreater(rec["reword"]["cost_usd"], 0)
        self.assertEqual(rec["reword"]["refused_try"]["status"], "content_policy")
        stage = self.stages(job)["p"]
        self.assertEqual(stage["state"], "passed")
        self.assertAlmostEqual(stage["cost_usd"], rec["cost_usd"], places=6)
        kinds = [(e["kind"], e.get("who"), e.get("what"), e.get("outcome"), e.get("ok")) for e in read_events(run)
                 if e.get("step") == "p"]
        self.assertIn(("call_end", "muse-image", "sketch_design", "failed", None), kinds)
        self.assertIn(("call_end", "muse-spark", "reword_brief", "ok", None), kinds)
        self.assertIn(("decision", None, "brief_reworded", None, True), kinds)
        self.assertLess(kinds.index(("decision", None, "brief_reworded", None, True)),
                        kinds.index(("call_end", "muse-image", "sketch_design", "ok", None)))
        self.assertEqual(self.view(job)["reword"], {"status": "used", "stage": "p", "text": REWORDED})
        self.assertNotIn(BRIEF, "\n".join(logs.output))
        self.assertNotIn(REWORDED, "\n".join(logs.output))

    def test_rewrite_call_events_keep_the_refused_step_after_earlier_work_left_a_step(self):
        # Stages run in process here, as they would after any earlier work in the same thread: a step left in the
        # context must not relabel the rewrite's call, which belongs to the refused step the runner names.
        self.addCleanup(events.reset_step, events.set_step("3f"))
        job = self.prompt_job()
        with self.assertLogs("herogen.service", "INFO"):
            self.assertEqual(self.go(job), "succeeded")
        spark = [(e["kind"], e.get("step")) for e in read_events(self.run_dir(job)) if e.get("who") == "muse-spark"]
        self.assertEqual(spark, [("call_start", "p"), ("call_end", "p")])

    def test_second_refusal_stops_with_the_clearer_message_and_keeps_the_rewrite_for_editing(self):
        self.refusals = {"images/generations": 2}
        job = self.prompt_job()
        with self.assertLogs("herogen.service", "INFO"):
            self.assertEqual(self.go(job), "rejected")
        saved = self.store.job_get(job["id"])
        self.assertEqual((saved["error_stage"], saved["error_code"], saved["attempts"]), ("p", "rejected", 1))
        self.assertIn("even after HeroGen reworded it once", saved["error_message"])
        self.assertIn("Brand, franchise, character or real-person names usually cause this", saved["error_message"])
        self.assertNotIn(REWORDED, saved["error_message"])
        self.assertEqual((len(self.generation_prompts), len(self.reword_prompts)), (2, 1))
        self.assertEqual(reword.read(self.run_dir(job))["status"], "refused_again")
        self.assertEqual(self.record(job, "p")["reword"]["status"], "refused_again")
        self.assertEqual(self.view(job)["reword"], {"status": "refused_again", "stage": "p", "text": REWORDED})
        whats = [e.get("what") for e in read_events(self.run_dir(job)) if e["kind"] == "decision"]
        self.assertEqual(whats, ["brief_reworded", "reworded_brief_refused"])

    def test_failed_rewrite_stops_without_a_retry(self):
        for case, got in (("api", ApiError("HTTP 503", http_status=503, transient=True)),
                          ("names", answer(f"{REWORDED}, like Ironman"))):
            with self.subTest(case=case):
                self.refusals = {"images/generations": 1}
                self.reword_answers = [got]
                self.reword_prompts, self.generation_prompts = [], []
                job = self.prompt_job()
                with self.assertLogs("herogen.service", "INFO"):
                    self.assertEqual(self.go(job), "rejected")
                saved = self.store.job_get(job["id"])
                self.assertIn("could not reword it automatically", saved["error_message"])
                self.assertIn("Brand, franchise", saved["error_message"])
                self.assertEqual((len(self.generation_prompts), len(self.reword_prompts)), (1, 1))
                data = reword.read(self.run_dir(job))
                self.assertEqual(data["status"], "rewrite_failed")
                # A rewrite that failed its checks is never kept or shown; the box gets the user's own brief.
                self.assertNotIn("reworded", data)
                self.assertIsNone(self.view(job)["reword"]["text"])
                whats = [e.get("what") for e in read_events(self.run_dir(job)) if e["kind"] == "decision"]
                self.assertEqual(whats, ["brief_reword_failed"])
                self.store.job_cancel(job["id"], self.user["id"])

    def test_non_policy_errors_and_the_full_lane_are_never_reworded(self):
        for case in ("http_400", "http_503", "full_lane"):
            with self.subTest(case=case):
                self.refusals = {}
                err = (ApiError("bad request", http_status=400) if case == "http_400" else
                       ApiError("HTTP 503", http_status=503, transient=True) if case == "http_503" else policy())
                self.reword_prompts, self.generation_prompts = [], []

                def failing(client, endpoint, body, **kw):
                    if endpoint == "images/generations":
                        self.generation_prompts.append(body["prompt"])
                        raise err
                    raise AssertionError(f"unexpected call {endpoint}")

                job = self.prompt_job()
                env = {"HEROGEN_FAST_LANE": ""} if case == "full_lane" else FAST
                with mock.patch.object(self, "post", side_effect=failing), mock.patch.dict(os.environ, env), \
                        self.assertLogs("herogen.service", "INFO"):
                    self.go(job)
                self.assertEqual(len(self.generation_prompts), 1)
                self.assertEqual(self.reword_prompts, [])
                self.assertIsNone(reword.read(self.run_dir(job)))
                saved = self.store.job_get(job["id"])
                self.assertNotIn("reworded", saved["error_message"] or "")
                if case == "full_lane":
                    self.assertEqual(saved["error_message"], "The image model refused this description under its "
                                     "content policy. Try rewording the description.")
                self.assertIsNone(self.view(job)["reword"])
                self.store.job_cancel(job["id"], self.user["id"])

    def test_refused_cover_is_reworded_and_retried_with_the_rewrite_in_its_brief(self):
        self.real_stages.add("1")
        self.refusals = {"cover": 1}
        job = self.prompt_job()
        with self.assertLogs("herogen.service", "INFO"):
            self.go(job)
        self.assertEqual(len(self.reword_prompts), 1)
        covers = [p for p in self.edit_prompts if "The brief between « and »" in p]
        self.assertEqual(len(covers), 2)
        self.assertIn(f"«{BRIEF}»", covers[0])
        self.assertIn(f"«{REWORDED}»", covers[1])
        self.assertNotIn("Ironman", covers[1])
        # The sketch is drawn again from the rewrite (stage p, then stage 0) before the cover is retried.
        stages = [c[3] if c[2] == "herogen.service.pipestage" else c[c.index("--stages") + 1] for c in self.commands]
        self.assertEqual(stages, ["p", "0", "1", "reword", "p", "0", "1"])
        self.assertEqual(len(self.generation_prompts), 2)
        self.assertIn("Ironman", self.generation_prompts[0])
        for name in REMOVED:
            self.assertNotIn(name.casefold(), self.generation_prompts[1].casefold())
        self.assertIn(REWORDED, self.generation_prompts[1])
        self.assertEqual(self.record(job, "p")["details"]["user_text"], REWORDED)
        data = reword.read(self.run_dir(job))
        self.assertEqual((data["stage"], data["redraw"]["status"], data["redraw"]["stages"]), ("1", "passed", ["p", "0"]))
        self.assertGreater(data["redraw"]["cost_usd"], 0)
        cover = self.record(job, "1")
        self.assertTrue(cover["details"]["brief_reworded"])
        self.assertTrue(cover["details"]["input"]["generated_sketch"])
        self.assertEqual(cover["reword"]["reworded"], REWORDED)
        self.assertEqual(cover["reword"]["redraw"]["status"], "passed")
        whats = [e.get("what") for e in read_events(self.run_dir(job)) if e["kind"] == "decision"]
        self.assertEqual([w for w in whats if w in ("brief_reworded", "sketch_redrawn")],
                         ["brief_reworded", "sketch_redrawn"])
        self.assertEqual(self.stages(job)["1"]["state"], "passed")
        self.assertEqual(self.store.job_get(job["id"])["state"], "succeeded")


class RedrawRules(unittest.TestCase):
    """`reword_and_retry`'s sketch redraw, with the steps stubbed."""

    def run_case(self, spec_key, redraw_result=None, redo=False):
        import tempfile

        from herogen.service.runner import reword_and_retry

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        run = Path(tmp.name)
        if redo:
            (run / "redo.json").write_text("{}")
        ctx = RewordDue().ctx(tmp.name)
        spec = next(s for s in PLAN if s.key == spec_key)
        calls = []

        def rewrite():
            reword.write(run, {"status": "used", "reworded": REWORDED, "removed_names": REMOVED, "cost_usd": 0.004})

        def redraw():
            calls.append("redraw")
            return redraw_result or (Outcome("passed", 1.0, 0.02), 0)

        def retry():
            calls.append("retry")
            return Outcome("passed", 1.0, 0.01), 0

        out = reword_and_retry(spec, ctx, Outcome("rejected", 1.0, 0.005), rewrite, retry, redraw)
        return out, calls, reword.read(run)

    def test_only_a_first_version_cover_refusal_redraws(self):
        out, calls, data = self.run_case("1")
        self.assertEqual((out.state, calls, data["redraw"]["status"]), ("passed", ["redraw", "retry"], "passed"))
        self.assertAlmostEqual(out.cost_usd, 0.005 + 0.004 + 0.02 + 0.01)
        self.assertEqual(self.run_case("p")[1], ["retry"])
        self.assertEqual(self.run_case("1", redo=True)[1], ["retry"], "a redo keeps its parent's sketch")

    def test_a_retry_that_writes_no_record_does_not_count_the_refused_try_twice(self):
        import tempfile

        from herogen.service.runner import reword_and_retry, write_record

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_record(run, {"stage": "p", "ok": False, "status": "error", "cost_usd": 0.005})
            ctx = RewordDue().ctx(tmp)
            spec = next(s for s in PLAN if s.key == "p")
            out = reword_and_retry(
                spec, ctx, Outcome("rejected", 1.0, 0.005),
                lambda: reword.write(run, {"status": "used", "reworded": REWORDED, "cost_usd": 0.004}),
                lambda: (Outcome("failed", 1.0, 0.0, code="stage_error"), 1))
            rec = json.loads((run / "stage-p.json").read_text())
            self.assertAlmostEqual(rec["cost_usd"], 0.009)
            self.assertAlmostEqual(out.cost_usd, 0.009)
            self.assertEqual(out.state, "failed")

    def test_a_refused_or_failed_redraw_stops_without_the_retry(self):
        out, calls, data = self.run_case("1", (Outcome("rejected", 1.0, 0.0, code="rejected"), 4))
        self.assertEqual((out.state, calls, data["status"]), ("rejected", ["redraw"], "refused_again"))
        self.assertIn("even after HeroGen reworded it once", out.message)
        out, calls, data = self.run_case("1", (Outcome("failed", 1.0, 0.0, "busy", "model_unavailable", True), 5))
        self.assertEqual((out.state, out.code, out.transient, calls, data["status"]),
                         ("failed", "model_unavailable", True, ["redraw"], "used"))


# Only this file's own tests run on RewordJobTest, not the inherited prompt-job tests.
for _name in dir(prompt_base.PromptJobTest):
    if _name.startswith("test_") and _name not in RewordJobTest.__dict__:
        setattr(RewordJobTest, _name, None)


class RedoKeepsTheRewrite(unittest.TestCase):
    def test_a_cover_redo_copies_the_used_rewrite(self):
        import tempfile

        from herogen.pipeline.redo import fork_run
        from herogen.pipeline.runfolder import RunFolder
        from herogen.pipeline.stages import cover_prompt
        from tests.test_herogen_service_jobs import png

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            parent = root / "parent"
            parent.mkdir(parents=True)
            (parent / "00-sketch.png").write_bytes(png(64, 64))
            base = {"ok": True, "outputs": [], "checks": []}
            (parent / "stage-p.json").write_text(json.dumps({**base, "stage": "p", "details": {"user_text": BRIEF}}))
            (parent / "stage-0.json").write_text(json.dumps({**base, "stage": 0, "outputs": ["00-sketch.png"],
                                                            "details": {"origin": {"stage": "p"}}}))
            reword.write(parent, {"status": "used", "reworded": REWORDED})
            child = fork_run(root, "child", parent, 1, "")
            self.assertEqual(reword.brief(child.path), REWORDED)
            prompt, used = cover_prompt(RunFolder(root, "child"))
            self.assertTrue(used)
            self.assertIn(f"«{REWORDED}»", prompt)
            self.assertNotIn(BRIEF, prompt)


class FakeRunnerScenarios(jobs_base.JobCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(os.environ, FAST)
        patcher.start()
        self.addCleanup(patcher.stop)

    def job(self, text):
        job = self.new_job("prompt")
        self.store.db.execute("UPDATE jobs SET prompt=? WHERE id=?", (text, job["id"]))
        return self.store.job_get(job["id"])

    def test_marker_words_pick_each_scenario_with_no_spend(self):
        for word, state, status, decisions in (
                ("fakerefuse", "succeeded", "used", ["brief_reworded"]),
                ("fakerefusecover", "succeeded", "used", ["brief_reworded", "sketch_redrawn"]),
                ("fakerefusetwice", "rejected", "refused_again", ["brief_reworded", "reworded_brief_refused"]),
                ("fakerewordfail", "rejected", "rewrite_failed", ["brief_reword_failed"]),
                ("plain", "succeeded", None, [])):
            with self.subTest(word=word):
                job = self.job(f"A red armoured robot {word}")
                self.assertEqual(self.go(job), state)
                run = self.storage.run_dir(self.user["id"], job["id"])
                data = reword.read(run)
                self.assertEqual(data and data["status"], status)
                evs = read_events(run)
                self.assertEqual([e["what"] for e in evs if e["kind"] == "decision"
                                  and e["what"].startswith(("brief", "reworded", "sketch"))], decisions)
                if status == "used":
                    self.assertEqual((data["reworded"], data["stage"]), (FAKE_REWORD, "1" if "cover" in word else "p"))
                    self.assertEqual(json.loads((run / "stage-p.json").read_text())["details"]["user_text"],
                                     FAKE_REWORD)
                if word == "plain":
                    self.assertFalse(any(e.get("who") == "muse-spark" for e in evs))

    def test_a_refused_cover_reruns_p_and_0_before_the_cover(self):
        self.runner = FakeRunner(fail={"1": "policy"})
        job = self.job("A red armoured robot")
        self.assertEqual(self.go(job), "succeeded")
        self.assertEqual(self.runner.calls[:6], ["p", "0", "1", "p", "0", "2"])
        run = self.storage.run_dir(self.user["id"], job["id"])
        self.assertEqual(json.loads((run / "stage-p.json").read_text())["details"]["user_text"], FAKE_REWORD)
        self.assertEqual(reword.read(run)["redraw"]["status"], "passed")

    def test_fail_modes_and_the_full_lane(self):
        self.runner = FakeRunner(fail={"p": "policy"})
        with mock.patch.dict(os.environ, {"HEROGEN_FAST_LANE": ""}):
            job = self.job("A red armoured robot")
            self.assertEqual(self.go(job), "rejected")
            self.assertIsNone(reword.read(self.storage.run_dir(self.user["id"], job["id"])))
            self.assertIn("Try rewording", self.store.job_get(job["id"])["error_message"])
        job = self.job("A red armoured robot")
        self.assertEqual(self.go(job), "succeeded")
        self.assertEqual(reword.read(self.storage.run_dir(self.user["id"], job["id"]))["status"], "used")


if __name__ == "__main__":
    unittest.main()
