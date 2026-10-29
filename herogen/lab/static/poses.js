// The pose test of herogen/blender/pose.py, in the GLB's frame. Blender is +Z up with the character facing -Y;
// the GLB is +Y up facing +Z. (x, y, z) -> (x, z, -y) is a proper rotation, so axes and angles carry over as they
// are and the poses here turn every section exactly as the pose test does.
import * as THREE from 'three';

export const SECTIONS = [
  'Head', 'UpperTorso', 'LowerTorso',
  'LeftUpperArm', 'LeftLowerArm', 'LeftHand',
  'RightUpperArm', 'RightLowerArm', 'RightHand',
  'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot',
  'RightUpperLeg', 'RightLowerLeg', 'RightFoot',
];
// herogen/blender/__init__.py PARENT: used when sections.json does not name a section's parent.
export const R15_PARENT = {
  Head: 'UpperTorso', UpperTorso: 'LowerTorso', LowerTorso: null,
  LeftUpperArm: 'UpperTorso', LeftLowerArm: 'LeftUpperArm', LeftHand: 'LeftLowerArm',
  RightUpperArm: 'UpperTorso', RightLowerArm: 'RightUpperArm', RightHand: 'RightLowerArm',
  LeftUpperLeg: 'LowerTorso', LeftLowerLeg: 'LeftUpperLeg', LeftFoot: 'LeftLowerLeg',
  RightUpperLeg: 'LowerTorso', RightLowerLeg: 'RightUpperLeg', RightFoot: 'RightLowerLeg',
};
// The joint each section turns about (PIVOT_JOINT), the name the gap tables use.
export const PIVOT_JOINT = {
  Head: 'Neck', UpperTorso: 'Waist', LowerTorso: 'Root',
  LeftUpperArm: 'LeftShoulder', LeftLowerArm: 'LeftElbow', LeftHand: 'LeftWrist',
  RightUpperArm: 'RightShoulder', RightLowerArm: 'RightElbow', RightHand: 'RightWrist',
  LeftUpperLeg: 'LeftHip', LeftLowerLeg: 'LeftKnee', LeftFoot: 'LeftAnkle',
  RightUpperLeg: 'RightHip', RightLowerLeg: 'RightKnee', RightFoot: 'RightAnkle',
};

// The seven test poses, in pose.py's order, with the words and icons the page uses.
export const POSES = [
  { key: 'rest', label: 'Rest', icon: 'hg-pose-rest' },
  { key: 'walk_a', label: 'Walk', icon: 'hg-pose-walk' },
  { key: 'walk_b', label: 'Walk (other leg)', short: 'Walk B', icon: 'hg-pose-walk' },
  { key: 'arms_forward', label: 'Arms forward', icon: 'hg-pose-arms-forward' },
  { key: 'arms_up', label: 'Arms up', icon: 'hg-pose-arms-up' },
  { key: 'crouch', label: 'Crouch', icon: 'hg-pose-crouch' },
  { key: 'twist_turn', label: 'Twist', icon: 'hg-pose-twist' },
];
export const poseWord = (p) => ({ rest: 'rest', walk_a: 'walk', walk_b: 'walk (other leg)', arms_forward: 'arms forward',
  arms_up: 'arms up', crouch: 'crouch', twist_turn: 'twist' }[p] || p.replace(/_/g, ' '));

const V3 = (x, y, z) => new THREE.Vector3(x, y, z);
export const toGltf = (v) => V3(v[0], v[2], -v[1]);
const FWD = V3(0, 0, 1);
const BACK = V3(0, 0, -1);
const UP = V3(0, 1, 0);
const XAX = V3(1, 0, 0);

const rot = (axis, deg) => new THREE.Matrix4().makeRotationAxis(axis.clone().normalize(), THREE.MathUtils.degToRad(deg));
function swing(bone, target, deg) {
  const ax = new THREE.Vector3().crossVectors(bone, target);
  if (ax.length() < 1e-9) ax.crossVectors(bone, XAX);
  return rot(ax, deg);
}

// pose.bone_directions, from the joints and ends sections.json records (Blender frame). Falls back to the
// directions between the GLB's own section pivots when a joint is missing.
export function boneDirs(joints, ends, pivots) {
  const j = (n) => (joints && joints[n] ? toGltf(joints[n]) : null);
  const e = (n) => (ends && ends[n] ? toGltf(ends[n]) : null);
  const u = (a, b, fa, fb) => {
    const p = a || (pivots[fa] && pivots[fa].clone());
    const q = b || (pivots[fb] && pivots[fb].clone());
    return p && q ? q.clone().sub(p).normalize() : V3(0, -1, 0);
  };
  const d = {};
  for (const s of ['Left', 'Right']) {
    d[s + 'UpperArm'] = u(j(s + 'Shoulder'), j(s + 'Elbow'), s + 'UpperArm', s + 'LowerArm');
    d[s + 'LowerArm'] = u(j(s + 'Elbow'), j(s + 'Wrist'), s + 'LowerArm', s + 'Hand');
    d[s + 'UpperLeg'] = u(j(s + 'Hip'), j(s + 'Knee'), s + 'UpperLeg', s + 'LowerLeg');
    d[s + 'LowerLeg'] = u(j(s + 'Knee'), j(s + 'Ankle'), s + 'LowerLeg', s + 'Foot');
    d[s + 'Foot'] = j(s + 'Ankle') && e(s + 'Toe') ? e(s + 'Toe').sub(j(s + 'Ankle')).normalize() : FWD.clone();
  }
  return d;
}

// pose.pose_library: local rotations per section (in the parent's frame) for each test pose.
export function poseLibrary(d) {
  const outL = V3(1, 0, 0);
  const outR = V3(-1, 0, 0);
  const P = { rest: {} };
  for (const [name, s1, s2] of [['walk_a', 'Left', 'Right'], ['walk_b', 'Right', 'Left']]) {
    P[name] = {
      [s1 + 'UpperLeg']: swing(d[s1 + 'UpperLeg'], FWD, 30), [s1 + 'LowerLeg']: swing(d[s1 + 'LowerLeg'], BACK, 15),
      [s2 + 'UpperLeg']: swing(d[s2 + 'UpperLeg'], BACK, 25), [s2 + 'LowerLeg']: swing(d[s2 + 'LowerLeg'], BACK, 40),
      [s2 + 'Foot']: rot(XAX, -15),
      [s1 + 'UpperArm']: swing(d[s1 + 'UpperArm'], BACK, 25), [s2 + 'UpperArm']: swing(d[s2 + 'UpperArm'], FWD, 25),
      [s1 + 'LowerArm']: swing(d[s1 + 'LowerArm'], FWD, 20), [s2 + 'LowerArm']: swing(d[s2 + 'LowerArm'], FWD, 30),
    };
  }
  P.arms_forward = {
    LeftUpperArm: swing(d.LeftUpperArm, FWD, 80), RightUpperArm: swing(d.RightUpperArm, FWD, 80),
    LeftLowerArm: swing(d.LeftLowerArm, FWD, 20), RightLowerArm: swing(d.RightLowerArm, FWD, 20),
    LeftHand: rot(d.LeftLowerArm, 30), RightHand: rot(d.RightLowerArm, -30),
  };
  P.arms_up = {
    LeftUpperArm: swing(d.LeftUpperArm, outL.clone().add(UP), 75), RightUpperArm: swing(d.RightUpperArm, outR.clone().add(UP), 75),
    LeftLowerArm: swing(d.LeftLowerArm, UP, 30), RightLowerArm: swing(d.RightLowerArm, UP, 30),
  };
  P.crouch = {
    LeftUpperLeg: swing(d.LeftUpperLeg, FWD, 75), RightUpperLeg: swing(d.RightUpperLeg, FWD, 75),
    LeftLowerLeg: swing(d.LeftLowerLeg, BACK, 100), RightLowerLeg: swing(d.RightLowerLeg, BACK, 100),
    LeftFoot: rot(XAX, 25), RightFoot: rot(XAX, 25), UpperTorso: swing(UP, FWD, 25), Head: swing(UP, BACK, 20),
    LeftUpperArm: swing(d.LeftUpperArm, FWD, 40), RightUpperArm: swing(d.RightUpperArm, FWD, 40),
  };
  P.twist_turn = {
    UpperTorso: rot(UP, 30), Head: rot(UP, 45),
    LeftUpperArm: swing(d.LeftUpperArm, FWD, 30), RightUpperArm: swing(d.RightUpperArm, BACK, 30),
  };
  return P;
}

// pose.world_matrices: each section's transform relative to rest, turned about its pivot and carried by its parent.
export function worldMatrices(local, pivots, parent) {
  const out = {};
  const get = (n) => {
    if (out[n]) return out[n];
    const R = local[n] ? local[n].clone() : new THREE.Matrix4();
    const p = pivots[n] || V3(0, 0, 0);
    const rp = p.clone().applyMatrix4(R);
    R.setPosition(p.x - rp.x, p.y - rp.y, p.z - rp.z);
    out[n] = parent[n] && pivots[parent[n]] ? get(parent[n]).clone().multiply(R) : R;
    return out[n];
  };
  Object.keys(pivots).forEach(get);
  return out;
}
