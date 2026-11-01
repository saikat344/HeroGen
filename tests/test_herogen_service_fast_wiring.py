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
