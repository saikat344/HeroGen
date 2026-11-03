import test from 'node:test';
import assert from 'node:assert/strict';
import { CLIPS, LIMITS, R15, axisAngle, blend, createPlayer, fromTo, mul, period, pose, rotate, skeleton, slerp }
  from '../herogen/service/static/animate.js';

// An A-posed hero in the GLB frame (+Y up, faces +Z, the character's left is +X), pivots in studs.
const PIVOTS = {
  LowerTorso: [0, 2.88, -0.05], UpperTorso: [0, 2.94, -0.07], Head: [0.01, 4.85, 0],
  LeftUpperArm: [1.0, 3.78, -0.11], LeftLowerArm: [1.11, 3.67, -0.1], LeftHand: [1.66, 3.08, 0.04],
  RightUpperArm: [-0.98, 3.81, -0.15], RightLowerArm: [-1.12, 3.66, -0.14], RightHand: [-1.65, 3.08, -0.03],
  LeftUpperLeg: [0.46, 2.36, -0.04], LeftLowerLeg: [0.57, 1.62, 0.04], LeftFoot: [0.66, 0.41, -0.05],
  RightUpperLeg: [-0.46, 2.36, -0.07], RightLowerLeg: [-0.57, 1.6, 0.02], RightFoot: [-0.66, 0.4, -0.06],
};
const sk = skeleton(PIVOTS);
const PARTS = R15.map(([part]) => part);
const near = (a, b, eps = 1e-6) => a.every((v, i) => Math.abs(v - b[i]) < eps);
// Where a part's bone points after the pose, in the hero's frame: compose the turns down the R15 tree.
function world(p, part) {
  let q = p.q[part];
  for (let parent = R15.find(([n]) => n === part)[1]; parent; parent = R15.find(([n]) => n === parent)[1]) q = mul(p.q[parent], q);
  return rotate(q, sk.dirs[part]);
}
const degrees = (a, b) => Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))) * 180 / Math.PI;

test('quaternion helpers turn the way the clips assume', () => {
  assert.ok(near(rotate(axisAngle([0, 1, 0], 90), [1, 0, 0]), [0, 0, -1]));
  assert.ok(near(rotate(fromTo([0, -1, 0], [0, 0, 1]), [0, -1, 0]), [0, 0, 1]));
  assert.ok(near(rotate(fromTo([1, 0, 0], [-1, 0, 0]), [1, 0, 0]), [-1, 0, 0]));
  assert.ok(near(slerp([0, 0, 0, 1], axisAngle([1, 0, 0], 90), 0.5), axisAngle([1, 0, 0], 45)));
});

test('the rest pose and unknown clips leave every part where it is', () => {
  for (const name of ['rest', 'dance', undefined]) {
    const p = pose(name, 1.3, sk);
    assert.deepEqual(Object.keys(p.q).sort(), [...PARTS].sort());
    for (const part of PARTS) assert.deepEqual(p.q[part], [0, 0, 0, 1]);
    assert.equal(p.lift, 0);
  }
});

test('every clip is finite, unit length and loops seamlessly', () => {
  for (const name of CLIPS) {
    for (let t = 0; t < 2 * period(name); t += 0.037) {
      const p = pose(name, t, sk);
      for (const part of PARTS) {
        const q = p.q[part];
        assert.ok(q.every(Number.isFinite), `${name} ${part} at ${t}`);
        assert.ok(Math.abs(Math.hypot(...q) - 1) < 1e-9, `${name} ${part} is a unit quaternion`);
      }
      assert.ok(p.lift >= 0 && p.lift <= 0.2, `${name} lift ${p.lift}`);
    }
    const a = pose(name, 0.25, sk), b = pose(name, 0.25 + period(name), sk);
    for (const part of PARTS) assert.ok(near(a.q[part], b.q[part], 1e-9) || near(a.q[part], b.q[part].map(v => -v), 1e-9), `${name} ${part} loops`);
  }
});

test('walking swings the legs against each other and the arms against the legs', () => {
  const quarter = pose('walk', period('walk') / 4, sk);   // left leg furthest forward
  assert.ok(world(quarter, 'LeftUpperLeg')[2] > 0.3, 'left thigh forward');
  assert.ok(world(quarter, 'RightUpperLeg')[2] < -0.3, 'right thigh back');
  assert.ok(world(quarter, 'LeftUpperArm')[2] < -0.2, 'left arm back');
  assert.ok(world(quarter, 'RightUpperArm')[2] > 0.2, 'right arm forward');
  const later = pose('walk', 3 * period('walk') / 4, sk);
  assert.ok(world(later, 'RightUpperLeg')[2] > 0.3 && world(later, 'LeftUpperLeg')[2] < -0.3, 'then the other way round');
});

test('knees and elbows only fold the natural way, within their limits', () => {
  for (const name of CLIPS) {
    for (let t = 0; t < period(name); t += 0.05) {
      const p = pose(name, t, sk);
      for (const side of ['Left', 'Right']) {
        const thigh = world(p, `${side}UpperLeg`), shin = world(p, `${side}LowerLeg`);
        const bend = degrees(thigh, shin);
        assert.ok(bend <= LIMITS.knee[1] + 25, `${name} ${side} knee ${bend}`);
        // A folded knee puts the shin behind the thigh, never in front of it.
        if (bend > 15) assert.ok(shin[2] < thigh[2] + 0.05, `${name} ${side} knee folds backwards at ${t}`);
      }
    }
  }
});

test('the wave raises the right hand above the shoulder and leaves the left arm down', () => {
  for (let t = 0; t < period('wave'); t += 0.1) {
    const p = pose('wave', t, sk);
    assert.ok(world(p, 'RightLowerArm')[1] > 0.7, 'right forearm points up');
    assert.ok(world(p, 'RightUpperArm')[1] > 0.3, 'right upper arm raised');
    assert.ok(world(p, 'LeftUpperArm')[1] < -0.5, 'left arm hangs');
  }
});

test('the jump crouches, then leaves the floor with the arms up, then lands', () => {
  const at = u => pose('jump', u * period('jump'), sk);
  assert.equal(at(0.28).lift, 0, 'crouching on the floor');
  assert.ok(degrees(world(at(0.28), 'LeftUpperLeg'), world(at(0.28), 'LeftLowerLeg')) > 60, 'knees bent in the crouch');
  assert.ok(at(0.56).lift > 0.08, 'in the air');
  assert.ok(world(at(0.56), 'LeftUpperArm')[1] > 0.5 && world(at(0.56), 'RightUpperArm')[1] > 0.5, 'arms up');
  assert.equal(at(0.95).lift, 0, 'landed');
});

test('the player blends between clips instead of jumping', () => {
  const player = createPlayer(sk, {blendMs: 250});
  player.play('walk', 0);
  const walk = pose('walk', 1, sk);
  player.sample(1000);
  player.play('wave', 1000);
  const start = player.sample(1000), mid = player.sample(1125), end = player.sample(1300);
  assert.ok(near(start.q.LeftUpperLeg, walk.q.LeftUpperLeg, 1e-9), 'a switch starts from the pose on screen');
  const wave = pose('wave', 0.3, sk);
  assert.ok(near(end.q.RightUpperArm, wave.q.RightUpperArm, 1e-9), 'and arrives at the new clip');
  assert.ok(!near(mid.q.RightUpperArm, start.q.RightUpperArm, 1e-3) && !near(mid.q.RightUpperArm, wave.q.RightUpperArm, 1e-3));
  assert.equal(player.busy(1300), true);
  player.play('rest', 2000);
  assert.equal(player.busy(2100), true, 'still easing back');
  assert.equal(player.busy(2300), false, 'at rest');
  for (const part of PARTS) assert.deepEqual(player.sample(2300).q[part], [0, 0, 0, 1]);
});

test('a blend of two poses keeps every part', () => {
  const b = blend(pose('idle', 0.5, sk), pose('jump', 0.5, sk), 0.5);
  assert.deepEqual(Object.keys(b.q).sort(), [...PARTS].sort());
});

test('a limb with a missing or zero-length pivot still has a usable direction', () => {
  const broken = skeleton({...PIVOTS, LeftHand: PIVOTS.LeftUpperArm, RightFoot: undefined});
  for (const dir of Object.values(broken.dirs)) assert.ok(Math.abs(Math.hypot(...dir) - 1) < 1e-9);
  for (const name of CLIPS) for (const q of Object.values(pose(name, 0.4, broken).q)) assert.ok(q.every(Number.isFinite));
});
