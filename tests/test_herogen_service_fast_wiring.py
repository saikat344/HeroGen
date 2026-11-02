"""Per-job texture instructions and compatible fast refinement CLI selection."""

import json
import os
import subprocess
from unittest import mock

from test_herogen_service_support import TempDirCase
from herogen.service.runner import PLAN, CliRunner, StageContext


class TestFastRefineCommand(TempDirCase):
    def setUp(self):
        super().setUp()
        self.ctx = StageContext(job_id='texture-job', kind='sketch', profile='roblox-ingame', name='', prompt='',
                                run_dir=self.tmp / 'runs' / 'texture-job', input_path=self.tmp / 'input.png',
                                cache_dir=self.tmp / 'cache', remote_cache_dir=self.tmp / 'remote')
        self.ctx.run_dir.mkdir(parents=True)
        self.runner = CliRunner(python='py')
        self.runner._flags['herogen.blender'] = '--profile --refine-profile --refine-note'
        self.spec = next(s for s in PLAN if s.key == '3f')

    def command(self, env=None):
        return self.runner.command(self.spec, self.ctx, env or {}, self.tmp)[0]

    def test_fast_lane_refines_fast_and_settings_profile_is_the_source(self):
        for fast, configured, expected in (('1', 'best', 'fast'), ('1', 'fast', 'fast'),
                                            ('0', 'fast', 'fast'), ('0', 'best', None)):
            self.ctx.refine_profile = configured  # Settings.refine_profile, via the worker's context
            with self.subTest(fast=fast, configured=configured), mock.patch.dict(os.environ, {}, clear=True):
                argv = self.command({'HEROGEN_FAST_LANE': fast})
                if expected:
                    self.assertEqual(argv[argv.index('--refine-profile') + 1], expected)
                else:
                    self.assertNotIn('--refine-profile', argv)
                self.assertNotIn('--refine-note', argv)
                self.assertEqual(argv[argv.index('--profile') + 1], 'roblox-ingame')

    def test_older_cli_omits_unsupported_flags_independently(self):
        (self.ctx.run_dir / 'redo.json').write_text(json.dumps({'from_stage': '3f', 'note': 'Blue coat'}))
        for flags in ('--profile', '--profile --refine-profile', '--profile --refine-note'):
            with self.subTest(flags=flags), mock.patch.dict(os.environ, {}, clear=True):
                self.runner._flags['herogen.blender'] = flags
                argv = self.command({'HEROGEN_FAST_LANE': '1'})
                self.assertEqual('--refine-profile' in argv, '--refine-profile' in flags)
                self.assertEqual('--refine-note' in argv, '--refine-note' in flags)

    def test_lineage_note_only_reaches_texture_stage(self):
        path = self.ctx.run_dir / 'redo.json'
        for stage, note in (('3f', 'Blue coat'), (2, 'Other stage')):
            path.write_text(json.dumps({'from_stage': stage, 'note': note}))
            argv = self.command()
            self.assertEqual('--refine-note' in argv, stage == '3f')
            if stage == '3f':
                self.assertEqual(argv[argv.index('--refine-note') + 1], note)
            for key in ('3', '4'):
                argv, _ = self.runner.command(next(s for s in PLAN if s.key == key), self.ctx, {}, self.tmp)
                self.assertNotIn('--refine-note', argv)
                self.assertNotIn('--refine-profile', argv)

    def test_note_is_one_literal_subprocess_argument_and_never_environment(self):
        note = 'Use "blue" and \'gold\'\n雪; $(touch should-not-exist) `literal`'
        with mock.patch('herogen.pipeline.redo.note_for', return_value=note) as read_note, \
                mock.patch.dict(os.environ, {'HEROGEN_FAST_LANE': '1', 'HEROGEN_REFINE_NOTE': 'wrong-job'}, clear=True), \
                mock.patch('herogen.service.runner.subprocess.run',
                           return_value=subprocess.CompletedProcess([], 1, '', '')) as execute:
            self.runner.run(self.spec, self.ctx)
        read_note.assert_called_once()
        self.assertEqual(read_note.call_args.args[0].path, self.ctx.run_dir)
        self.assertEqual(read_note.call_args.args[1], '3f')
        argv = execute.call_args.args[0]
        self.assertEqual(argv[-2:], ['--refine-note', note])
        self.assertEqual(argv.count(note), 1)
        self.assertFalse(execute.call_args.kwargs.get('shell', False))
        env = execute.call_args.kwargs['env']
        self.assertNotIn(note, env.values())
        self.assertNotIn('HEROGEN_REFINE_NOTE', env)
        self.assertNotIn('wrong-job', env.values())
        self.assertNotIn('--refine-note', self.command())


class TestFastLaneReachesDeployedWorkers(TempDirCase):
    """The full chain: the release's deploy env, the image env the Modal app bakes, a worker's stage env, argv."""

    def test_deployed_worker_runs_the_fast_lane_with_its_budgets(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "release"))
        try:
            import release
        finally:
            sys.path.pop(0)
        from herogen.service.config import env_for_deploy
        from herogen.service.runner import base_env

        github = {"MODAL_TOKEN_ID": "id", "MODAL_TOKEN_SECRET": "secret", "HEROGEN_SERVICE_APP": "app",
                  "HEROGEN_SERVICE_BASE_URL": "https://hero.example.org",
                  "HEROGEN_SERVICE_EMAIL_FROM": "HeroGen <signin@hero.example.org>",
                  "HEROGEN_SERVICE_TERMS_VERSION": "1", "HEROGEN_SERVICE_DOMAIN": "hero.example.org",
                  "HEROGEN_FAST_LANE": "1", "HEROGEN_MAX_IN_FLIGHT": "32", "HEROGEN_RPM_MUSE_IMAGE_1_0": "140",
                  "HEROGEN_RPM_SAM_3_1": "60", "HEROGEN_RPM_MUSE_SPARK_1_3": "60",
                  "HEROGEN_RPM_SECRET_TOKEN": "never-forwarded"}
        image = env_for_deploy(release.deploy_env(github))  # what modal_app bakes into every image
        with mock.patch.dict(os.environ, image, clear=True):
            stage_env = base_env({"HOME": str(self.tmp)})  # what a worker container's stage subprocess gets
        for name in ("HEROGEN_FAST_LANE", "HEROGEN_MAX_IN_FLIGHT", "HEROGEN_RPM_MUSE_IMAGE_1_0",
                     "HEROGEN_RPM_SAM_3_1", "HEROGEN_RPM_MUSE_SPARK_1_3"):
            self.assertEqual(stage_env[name], github[name], name)
        self.assertNotIn("HEROGEN_RPM_SECRET_TOKEN", image)
        self.assertNotIn("HEROGEN_RPM_SECRET_TOKEN", stage_env)
        self.assertNotIn("MODAL_TOKEN_SECRET", stage_env)
        ctx = StageContext(job_id='deployed', kind='sketch', profile='best', name='', prompt='',
                           run_dir=self.tmp / 'runs' / 'deployed', input_path=self.tmp / 'input.png',
                           cache_dir=self.tmp / 'cache', remote_cache_dir=self.tmp / 'remote')
        ctx.run_dir.mkdir(parents=True)
        runner = CliRunner(python='py')
        runner._flags['herogen.blender'] = '--profile --refine-profile --refine-note'
        argv = runner.command(next(s for s in PLAN if s.key == '3f'), ctx, stage_env, self.tmp)[0]
        self.assertEqual(argv[argv.index('--refine-profile') + 1], 'fast')

    def test_beauty_off_switches_reach_deployed_stage_subprocesses(self):
        from herogen.service.config import env_for_deploy
        from herogen.service.runner import base_env

        switches = {"HEROGEN_SIDE_BEAUTY": "0", "HEROGEN_REFINE_BEAUTY": "0",
                    "HEROGEN_REFINE_BEAUTY_DIR": "/never/forwarded", "HEROGEN_REFINE_REGION_PALETTE": "1"}
        image = env_for_deploy(switches)
        with mock.patch.dict(os.environ, image, clear=True):
            stage_env = base_env({"HOME": str(self.tmp)})
        for name in ("HEROGEN_SIDE_BEAUTY", "HEROGEN_REFINE_BEAUTY"):
            self.assertEqual(stage_env[name], "0", name)
        # Saved development references and the region palette never reach the product lane.
        for name in ("HEROGEN_REFINE_BEAUTY_DIR", "HEROGEN_REFINE_REGION_PALETTE"):
            self.assertNotIn(name, image)
            self.assertNotIn(name, stage_env)
