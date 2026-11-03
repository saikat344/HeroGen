// Procedural R15 animations for the 3D viewer: Idle, Walk, Wave and Jump on the Roblox-ready body.
// The body GLB holds the 15 R15 parts as flat, unrigged meshes; the service sends each part's joint pivot (the
// pivots the Roblox rig's Motor6Ds turn about). `createAnimator` re-parents the parts under pivot groups in the R15
// tree and turns them rigidly, as a Roblox avatar moves. Pose math is plain arrays (quaternions are [x, y, z, w]),
// so it runs without three.js; only `createAnimator` takes the THREE namespace.
// GLB frame: +Y up, the hero faces +Z, the character's left is +X.

export const CLIPS = ['idle', 'walk', 'wave', 'jump'];
// Root first; every part after its parent.
export const R15 = [['LowerTorso', null], ['UpperTorso', 'LowerTorso'], ['Head', 'UpperTorso'],
  ['LeftUpperArm', 'UpperTorso'], ['LeftLowerArm', 'LeftUpperArm'], ['LeftHand', 'LeftLowerArm'],
  ['RightUpperArm', 'UpperTorso'], ['RightLowerArm', 'RightUpperArm'], ['RightHand', 'RightLowerArm'],
  ['LeftUpperLeg', 'LowerTorso'], ['LeftLowerLeg', 'LeftUpperLeg'], ['LeftFoot', 'LeftLowerLeg'],
  ['RightUpperLeg', 'LowerTorso'], ['RightLowerLeg', 'RightUpperLeg'], ['RightFoot', 'RightLowerLeg']];
const PARTS = R15.map(([part]) => part);
// Degrees each joint may bend by in the clips: knees and elbows only fold one way.
export const LIMITS = {knee: [0, 135], elbow: [0, 140], hip: [-45, 95], shoulder: [-60, 175], wrist: [-60, 60],
  ankle: [-45, 45], neck: [-45, 45], waist: [-25, 25]};
export const BLEND_MS = 250;
export const JUMP_LIFT = 0.1;                  // the jump's peak, as a share of the hero's height
const UP = [0, 1, 0], DOWN = [0, -1, 0], FWD = [0, 0, 1], BACK = [0, 0, -1], X = [1, 0, 0], Z = [0, 0, 1];
const IDENTITY = [0, 0, 0, 1];
const TAU = 2 * Math.PI, RAD = Math.PI / 180;

// ---- vector and quaternion helpers -------------------------------------------------------------------------
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const length = a => Math.hypot(a[0], a[1], a[2]);
function unit(a, fallback = DOWN) {
  const n = length(a);
  return n > 1e-6 && Number.isFinite(n) ? [a[0] / n, a[1] / n, a[2] / n] : fallback;
}
export function axisAngle(axis, degrees) {
  const a = unit(axis, X), h = degrees * RAD / 2, s = Math.sin(h);
  return [a[0] * s, a[1] * s, a[2] * s, Math.cos(h)];
}
export function mul(a, b) {
  return [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
}
const inverse = q => [-q[0], -q[1], -q[2], q[3]];
export function rotate(q, v) {
  const r = mul(mul(q, [v[0], v[1], v[2], 0]), inverse(q));
  return [r[0], r[1], r[2]];
}
// The shortest turn taking direction a onto direction b.
export function fromTo(a, b) {
  const u = unit(a), v = unit(b), d = dot(u, v);
  if (d > 1 - 1e-9) return IDENTITY;
  if (d < -1 + 1e-9) return axisAngle(Math.abs(u[0]) < 0.9 ? cross(u, X) : cross(u, Z), 180);
  const c = cross(u, v), q = [c[0], c[1], c[2], 1 + d], n = Math.hypot(...q);
  return q.map(x => x / n);
}
export function slerp(a, b, t) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const s = d < 0 ? -1 : 1; d *= s;
  if (d > 0.9995) {
    const q = a.map((x, i) => x + (s * b[i] - x) * t), n = Math.hypot(...q);
    return q.map(x => x / n);
  }
  const th = Math.acos(Math.min(1, d)), sin = Math.sin(th), wa = Math.sin((1 - t) * th) / sin, wb = s * Math.sin(t * th) / sin;
  return a.map((x, i) => x * wa + b[i] * wb);
}
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const limit = (joint, degrees) => clamp(degrees, ...LIMITS[joint]);
const smooth = t => { const x = clamp(t, 0, 1); return x * x * (3 - 2 * x); };
const ease = t => { const x = clamp(t, 0, 1); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };

// ---- the hero's limbs ----------------------------------------------------------------------------------------
// Each part's bone direction at rest, from the pivots: an arm or leg points from its root joint to the wrist or
// ankle (the whole limb, since a cut can leave an upper arm only a shoulder pad long); a torso and head point up.
export function skeleton(pivots) {
  const p = name => pivots[name], dirs = {};
  const limb = (from, to, fallback) => unit(p(to) && p(from) ? sub(p(to), p(from)) : fallback, fallback);
  for (const side of ['Left', 'Right']) {
    const out = side === 'Left' ? 1 : -1;
    const arm = limb(`${side}UpperArm`, `${side}Hand`, [0.6 * out, -0.8, 0]);
    dirs[`${side}UpperArm`] = arm;
    dirs[`${side}LowerArm`] = dirs[`${side}Hand`] = limb(`${side}LowerArm`, `${side}Hand`, arm);
    const leg = limb(`${side}UpperLeg`, `${side}Foot`, DOWN);
    dirs[`${side}UpperLeg`] = leg;
    dirs[`${side}LowerLeg`] = limb(`${side}LowerLeg`, `${side}Foot`, leg);
    dirs[`${side}Foot`] = FWD;
  }
  dirs.LowerTorso = dirs.UpperTorso = dirs.Head = UP;
  // How far each arm hangs out from straight down, in degrees (an A-pose is about 40).
  const spread = side => Math.acos(clamp(dot(dirs[`${side}UpperArm`], DOWN), -1, 1)) / RAD;
  return {dirs, spread: {Left: spread('Left'), Right: spread('Right')}};
}
// Turn a bone towards a direction by some degrees (negative turns it away), about the axis square to both.
function swing(dir, target, degrees) {
  let axis = cross(dir, target);
  if (length(axis) < 1e-6) axis = cross(dir, X);
  return axisAngle(axis, degrees);
}
const outward = side => (side === 'Left' ? X : [-1, 0, 0]);
const turnX = degrees => axisAngle(X, degrees);
const turnY = degrees => axisAngle(UP, degrees);
const turnZ = degrees => axisAngle(Z, degrees);
// Arms in an A-pose hang a little closer to the body while relaxed, never into it.
function relaxed(sk, side) {
  return swing(sk.dirs[`${side}UpperArm`], DOWN, clamp(sk.spread[side] - 30, 0, 8));
}
// The upper arm pointing at `upper` and the forearm at `fore` (both in the chest's frame) as local turns.
function aim(sk, side, upper, fore) {
  const shoulder = fromTo(sk.dirs[`${side}UpperArm`], upper);
  return {[`${side}UpperArm`]: shoulder, [`${side}LowerArm`]: mul(inverse(shoulder), fromTo(sk.dirs[`${side}LowerArm`], fore))};
}

// ---- the clips: (time in seconds, skeleton) -> {q: {part: quaternion}, lift: share of the hero's height} --------
const PERIOD = {idle: 7.2, walk: 1.1, wave: 0.9, jump: 1.6};
export function period(name) { return PERIOD[name] || 1; }

function idle(t, sk) {
  // Two breaths, one glance round and one sway per loop.
  const loop = TAU * t / PERIOD.idle, breath = Math.sin(2 * loop), look = Math.sin(loop), sway = Math.sin(loop + 1.3);
  const q = {
    LowerTorso: turnZ(0.8 * sway),
    LeftUpperLeg: turnZ(-0.8 * sway), RightUpperLeg: turnZ(-0.8 * sway),
    UpperTorso: mul(turnY(limit('waist', 2.5 * look)), swing(UP, BACK, 1.2 * breath)),
    Head: mul(turnY(limit('neck', 7 * Math.sin(loop + 0.6))), swing(UP, FWD, 1.5 * breath)),
  };
  for (const side of ['Left', 'Right']) {
    const s = sk.dirs[`${side}UpperArm`];
    q[`${side}UpperArm`] = mul(swing(s, outward(side), 1.5 + 1.5 * breath), relaxed(sk, side));
    q[`${side}LowerArm`] = swing(sk.dirs[`${side}LowerArm`], FWD, limit('elbow', 7 + 2 * breath));
  }
  return {q, lift: 0};
}

// Walking in place: thighs swing, the swinging leg's knee folds to clear the floor, the feet stay level and the
// arms swing against the legs; the hips' bob comes from keeping the lower foot on the floor.
function walk(t, sk) {
  const p = TAU * t / PERIOD.walk, s = Math.sin(p), c = Math.cos(p), q = {};
  for (const [side, sign] of [['Left', 1], ['Right', -1]]) {
    const hip = limit('hip', 26 * sign * s), forward = Math.max(0, sign * c);
    const knee = limit('knee', 6 + 52 * Math.pow(forward, 1.4) + 6 * Math.max(0, -sign * s));
    q[`${side}UpperLeg`] = swing(sk.dirs[`${side}UpperLeg`], FWD, hip);
    q[`${side}LowerLeg`] = swing(sk.dirs[`${side}LowerLeg`], BACK, knee);
    q[`${side}Foot`] = turnX(limit('ankle', 0.8 * (hip - knee) + 4 * forward));
    const arm = limit('shoulder', -22 * sign * s);
    q[`${side}UpperArm`] = mul(swing(sk.dirs[`${side}UpperArm`], FWD, arm), relaxed(sk, side));
    q[`${side}LowerArm`] = swing(sk.dirs[`${side}LowerArm`], FWD, limit('elbow', 14 + 20 * Math.max(0, -sign * s)));
  }
  q.LowerTorso = turnY(-3 * s);
  q.UpperTorso = mul(turnY(limit('waist', 7 * s)), swing(UP, FWD, 3));
  q.Head = turnY(limit('neck', -4 * s));
  return {q, lift: 0};
}

// The right arm up beside the head, the forearm waving from the elbow; the body leans a little away from it.
function wave(t, sk) {
  const w = Math.sin(TAU * t / PERIOD.wave);
  const q = {
    UpperTorso: turnZ(-3), Head: mul(turnZ(4), turnY(limit('neck', -6))),
    ...aim(sk, 'Right', [-0.75, 0.55, 0.28], [-0.22 - 0.4 * w, 1, 0.2]),
    RightHand: swing(sk.dirs.RightHand, [-1, 0, 0], limit('wrist', 10 * w)),
    LeftUpperArm: mul(swing(sk.dirs.LeftUpperArm, FWD, 4), relaxed(sk, 'Left')),
    LeftLowerArm: swing(sk.dirs.LeftLowerArm, FWD, limit('elbow', 12)),
  };
  return {q, lift: 0};
}

// Crouch, spring up with both arms thrown overhead, land soft, stand.
function jump(t, sk) {
  const u = (t % PERIOD.jump) / PERIOD.jump;
  const crouch = u < 0.28 ? ease(u / 0.28) : u < 0.38 ? 1 - smooth((u - 0.28) / 0.1)
    : u < 0.74 ? 0 : u < 0.84 ? 0.7 * smooth((u - 0.74) / 0.1) : 0.7 * (1 - ease((u - 0.84) / 0.16));
  const air = u >= 0.36 && u <= 0.76 ? Math.sin(Math.PI * (u - 0.36) / 0.4) : 0;
  const reach = u < 0.28 ? 0 : u < 0.42 ? smooth((u - 0.28) / 0.14) : u < 0.74 ? 1 : 1 - smooth((u - 0.74) / 0.2);
  const tuck = 26 * air;
  const q = {
    UpperTorso: swing(UP, FWD, limit('waist', 16 * crouch - 4 * air)),
    Head: swing(UP, BACK, limit('neck', 10 * crouch + 6 * air)),
  };
  for (const side of ['Left', 'Right']) {
    const out = side === 'Left' ? 1 : -1;
    const hip = limit('hip', 55 * crouch + 0.7 * tuck), knee = limit('knee', 95 * crouch + tuck);
    q[`${side}UpperLeg`] = swing(sk.dirs[`${side}UpperLeg`], FWD, hip);
    q[`${side}LowerLeg`] = swing(sk.dirs[`${side}LowerLeg`], BACK, knee);
    q[`${side}Foot`] = turnX(limit('ankle', hip - knee + 12 * air));
    const back = mul(swing(sk.dirs[`${side}UpperArm`], BACK, limit('shoulder', 28 * crouch)), relaxed(sk, side));
    const up = aim(sk, side, [0.5 * out, 1, 0.12], [0.3 * out, 1, 0.05]);
    q[`${side}UpperArm`] = slerp(back, up[`${side}UpperArm`], reach);
    q[`${side}LowerArm`] = slerp(swing(sk.dirs[`${side}LowerArm`], FWD, limit('elbow', 18 * crouch)), up[`${side}LowerArm`], reach);
  }
  return {q, lift: JUMP_LIFT * air};
}
const POSES = {idle, walk, wave, jump};

/** The pose of clip `name` at `t` seconds; `rest` (or anything unknown) is the rest pose. Every part is listed. */
export function pose(name, t, sk) {
  const got = POSES[name] ? POSES[name](Math.max(0, t), sk) : {q: {}, lift: 0};
  const q = {};
  for (const part of PARTS) {
    const v = got.q[part];
    q[part] = v && v.every(Number.isFinite) ? v : IDENTITY;
  }
  return {q, lift: Number.isFinite(got.lift) ? Math.max(0, got.lift) : 0};
}
export function blend(a, b, t) {
  const q = {};
  for (const part of PARTS) q[part] = slerp(a.q[part], b.q[part], t);
  return {q, lift: a.lift + (b.lift - a.lift) * t};
}

/**
 * A clip player: `play(name)` blends from whatever pose is showing to the new clip over BLEND_MS (`rest` blends
 * back to the rest pose), `sample(ms)` is the pose to show at that time.
 */
export function createPlayer(sk, {blendMs = BLEND_MS} = {}) {
  let clip = 'rest', start = 0, from = null, fromAt = 0, last = pose('rest', 0, sk);
  return {
    get clip() { return clip; },
    play(name, ms) {
      const next = POSES[name] ? name : 'rest';
      if (next === clip) return;
      from = last; fromAt = ms; clip = next; start = ms;
    },
    // True while blending into the current clip or playing one.
    busy(ms) { return clip !== 'rest' || (from !== null && ms - fromAt < blendMs); },
    sample(ms) {
      const live = pose(clip, (ms - start) / 1000, sk);
      const k = from && blendMs > 0 ? (ms - fromAt) / blendMs : 1;
      if (k >= 1) from = null;
      last = from ? blend(from, live, ease(k)) : live;
      return last;
    },
  };
}

/**
 * Re-parent the body GLB's 15 part nodes under pivot groups in the R15 tree.
 * `scene` is the loaded GLB scene; `parts` the service's [{part, parent, pivot}] in the GLB frame (studs).
 * Returns {root, apply(pose), ground, height, dispose} or throws when the body is not the 15 R15 parts.
 */
export function createAnimator(THREE, scene, parts) {
  const pivots = {};
  for (const item of Array.isArray(parts) ? parts : []) {
    if (item && PARTS.includes(item.part) && Array.isArray(item.pivot) && item.pivot.length === 3 && item.pivot.every(Number.isFinite))
      pivots[item.part] = item.pivot;
  }
  const nodes = {};
  scene.traverse(node => { if (PARTS.includes(node.name) && !nodes[node.name]) nodes[node.name] = node; });
  const missing = PARTS.filter(part => !nodes[part] || !pivots[part]);
  if (missing.length) throw new Error(`The body is missing ${missing.join(', ')}`);
  const sk = skeleton(pivots);
  const root = new THREE.Group(); root.name = 'hg-rig';
  scene.add(root); scene.updateMatrixWorld(true);
  const joints = {};
  for (const [part, parent] of R15) {
    const joint = new THREE.Group(), at = pivots[part], base = parent ? pivots[parent] : [0, 0, 0];
    joint.name = `hg-joint-${part}`;
    joint.position.set(at[0] - base[0], at[1] - base[1], at[2] - base[2]);
    (parent ? joints[parent] : root).add(joint);
    joints[part] = joint;
  }
  root.updateMatrixWorld(true);
  for (const part of PARTS) joints[part].attach(nodes[part]);
  // Points on the soles: the feet's vertices (at most a few hundred each), in their joint's frame, so the lowest
  // one can be kept on the floor in any pose.
  const soles = [];
  for (const foot of ['LeftFoot', 'RightFoot']) {
    const joint = joints[foot], toJoint = new THREE.Matrix4().copy(joint.matrixWorld).invert(), points = [];
    nodes[foot].traverse(mesh => {
      const position = mesh.isMesh && mesh.geometry?.attributes?.position;
      if (!position) return;
      const step = Math.max(1, Math.floor(position.count / 400));
      for (let i = 0; i < position.count; i += step)
        points.push(new THREE.Vector3().fromBufferAttribute(position, i).applyMatrix4(mesh.matrixWorld).applyMatrix4(toJoint));
    });
    soles.push([joint, points]);
  }
  const rest = joints.LowerTorso.position.clone();
  const toRoot = new THREE.Matrix4(), point = new THREE.Vector3();
  function lowest() {
    root.updateMatrixWorld(true);
    const rootInverse = new THREE.Matrix4().copy(root.matrixWorld).invert();
    let low = Infinity;
    for (const [joint, points] of soles) {
      toRoot.multiplyMatrices(rootInverse, joint.matrixWorld);
      for (const p of points) low = Math.min(low, point.copy(p).applyMatrix4(toRoot).y);
    }
    return low;
  }
  const ground = lowest();
  const box = new THREE.Box3().setFromObject(root), height = Math.max(0.01, box.max.y - box.min.y);
  const restBox = {min: box.min.clone(), max: box.max.clone()};
  if (!Number.isFinite(ground)) throw new Error('The body has no feet to stand on');
  function apply(p) {
    for (const part of PARTS) joints[part].quaternion.set(...p.q[part]);
    joints.LowerTorso.position.copy(rest);
    // Keep the lowest sole on the floor, then lift (jumping) above it.
    const drop = ground - lowest();
    joints.LowerTorso.position.y = rest.y + (Number.isFinite(drop) ? drop : 0) + p.lift * height;
    root.updateMatrixWorld(true);
  }
  return {root, joints, skeleton: sk, ground, height, restBox, apply, maxLift: JUMP_LIFT,
    player: options => createPlayer(sk, options)};
}
