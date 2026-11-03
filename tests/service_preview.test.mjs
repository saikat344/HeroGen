import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { SourceTextModule, SyntheticModule } from 'node:vm';

async function setup() {
  const loads=[], scenes=[], renderers=[], cameras=[], controls=[];
  class Vector3 { constructor(x=0,y=0,z=0){ this.x=x;this.y=y;this.z=z; } set(x,y,z){Object.assign(this,{x,y,z});return this;} copy(v){this.x=v.x;this.y=v.y;this.z=v.z;return this;} setScalar(){} toArray(){return [this.x,this.y,this.z];}
    fromBufferAttribute(a,i){return this.set(a.array[i*3],a.array[i*3+1],a.array[i*3+2]);} applyMatrix4(){return this;} }
  class Scene { constructor(){this.models=[];scenes.push(this);} add(n){this.models.push(n);n.parent=this;} remove(n){this.models=this.models.filter(v=>v!==n);} }
  class Renderer { constructor(){this.disposed=false;this.renders=0;this.lastModels=[];renderers.push(this);} dispose(){this.disposed=true;} setPixelRatio(){} setSize(){} setAnimationLoop(fn){this.loop=fn;} render(scene){this.renders++;this.lastModels=[...scene.models];} }
  class Camera { constructor(){this.position=new Vector3();cameras.push(this);} updateProjectionMatrix(){} }
  // An object may carry its own `box` ({size, centre}); everything else is 1 x 2 x 1 at the origin.
  class Box3 { constructor(){this.min={x:0,y:0,z:0};this.max={x:1,y:2,z:1};this.size=[1,2,1];this.centre=[0,0,0];} setFromObject(o){if(o?.box){this.size=o.box.size;this.centre=o.box.centre;}return this;}
    getSize(v){[v.x,v.y,v.z]=this.size;return v;} getCenter(v){[v.x,v.y,v.z]=this.centre;return v;} clone(){return this;} }
  // The figure of dots (preview.building): just enough of three.js to build, step and free it.
  const freed=[];
  class BufferAttribute { constructor(array,size){this.array=array;this.itemSize=size;this.count=array.length/size;} setXYZ(i,x,y,z){this.array.set([x,y,z],i*3);} }
  class BufferGeometry { constructor(){this.attributes={};} setAttribute(k,v){this.attributes[k]=v;} computeBoundingBox(){this.boundingBox=new Box3();} dispose(){freed.push('geometry');} }
  class ShaderMaterial { constructor(o){Object.assign(this,o);} dispose(){freed.push('material');} }
  class Points { constructor(geometry,material){this.geometry=geometry;this.material=material;} }
  class Group { constructor(){this.scale=new Vector3(1,1,1);this.position=new Vector3();this.matrixWorld={clone:()=>({invert(){return this;}})};this.children=[];}
    add(n){this.children.push(n);} updateMatrixWorld(){} removeFromParent(){this.parent?.remove(this);} }
  class Light { constructor(){this.position=new Vector3();this.scale=new Vector3();this.geometry={dispose(){}};this.material={color:{set(){}},dispose(){}};} }
  class Controls { constructor(){this.target=new Vector3();this.listeners={};controls.push(this);} addEventListener(name,callback){this.listeners[name]=callback;} update(){} dispose(){} }
  class Loader { load(url,done,progress,fail){loads.push({url,done,fail});} }
  class PMREMGenerator { fromScene(){return {texture:{dispose(){}}};} dispose(){} }
  const three={WebGLRenderer:Renderer,Scene,HemisphereLight:Light,DirectionalLight:Light,GridHelper:Light,PMREMGenerator,SRGBColorSpace:'srgb',ACESFilmicToneMapping:4,
    PerspectiveCamera:Camera,Vector3,Box3,BufferAttribute,BufferGeometry,ShaderMaterial,Points,Group,AdditiveBlending:2,Color:class {},MeshStandardMaterial:class { constructor(options){Object.assign(this,options);} dispose(){this.disposed=true;} }};
  const stub=(exports)=>new SyntheticModule(Object.keys(exports),function(){for(const [k,v] of Object.entries(exports))this.setExport(k,v);});
  const source=await readFile(new URL('../herogen/service/static/preview.js',import.meta.url),'utf8');
  globalThis.devicePixelRatio=1; globalThis.ResizeObserver=class {observe(){} disconnect(){}};
  globalThis.document={documentElement:{dataset:{theme:'light'}},addEventListener(){},removeEventListener(){}};
  const module=new SourceTextModule(source);
  await module.link((id)=>stub(id==='three'?three:id.includes('OrbitControls')?{OrbitControls:Controls}:id.includes('RoomEnvironment')?{RoomEnvironment:class {}}:{GLTFLoader:Loader}));
  await module.evaluate();
  const model=()=>({disposed:0,traverse(fn){fn({geometry:{dispose:()=>this.disposed++}});}});
  return {api:module.namespace,loads,scenes,renderers,cameras,controls,model,freed,canvas:{clientWidth:800,clientHeight:500}};
}
test('a late model response never replaces the selected hero',async()=>{
  const s=await setup(),old=s.model(),current=s.model();
  const a=s.api.show(s.canvas,'old.glb'); const b=s.api.show(s.canvas,'current.glb');
  s.loads[1].done({scene:current}); await b;
  s.loads[0].done({scene:old}); await a;
  assert.equal(s.scenes[0].models.includes(current),true);
  assert.equal(s.scenes[0].models.includes(old),false); assert.equal(old.disposed,1);
});
test('hiding the preview pauses rendering and discards an in-flight result',async()=>{
  const s=await setup(),model=s.model(),load=s.api.show(s.canvas,'hero.glb');
  s.api.hide(); assert.equal(s.renderers[0].loop,null);
  s.loads[0].done({scene:model}); await load;
  assert.equal(s.scenes[0].models.includes(model),false); assert.equal(model.disposed,1);
});
test('load failures reject explicitly and a loaded model resumes without downloading again',async()=>{
  const s=await setup(),bad=s.api.show(s.canvas,'bad.glb');
  s.loads[0].fail(new Error('offline')); await assert.rejects(bad,/offline/); assert.equal(s.renderers[0].loop,null);
  const good=s.api.show(s.canvas,'good.glb'); s.loads[1].done({scene:s.model()}); await good;
  s.api.hide(); await s.api.show(s.canvas,'good.glb');
  assert.equal(s.loads.length,2); assert.equal(typeof s.renderers[0].loop,'function');
});

test('independent viewers own their canvases and dispose without stopping another',async()=>{
  const s=await setup(),a=s.api.createPreview(),b=s.api.createPreview();
  const one=a.show(s.canvas,'one.glb'),two=b.show({...s.canvas},'two.glb');
  s.loads[0].done({scene:s.model()});s.loads[1].done({scene:s.model()});await Promise.all([one,two]);
  a.dispose(); assert.equal(s.renderers[0].disposed,true);assert.equal(typeof s.renderers[1].loop,'function');
  b.dispose();assert.equal(s.renderers[1].disposed,true);
});
test('clay and wire restore the original materials and preserve loaded camera',async()=>{
  const s=await setup(),v=s.api.createPreview(),original={dispose(){this.disposed=true;}};
  const mesh={isMesh:true,userData:{},material:original,geometry:{dispose(){}}};
  const model={traverse(fn){fn(mesh);}};const load=v.show(s.canvas,'one.glb');s.loads[0].done({scene:model});await load;
  v.shading('clay');const clay=mesh.material;assert.notEqual(clay,original);assert.equal(clay.wireframe,false);
  v.shading('wire');assert.equal(clay.disposed,true);assert.equal(mesh.material.wireframe,true);
  v.shading('textured');assert.equal(mesh.material,original);assert.equal(original.disposed,undefined);
  await v.show(s.canvas,'one.glb');assert.equal(s.loads.length,1);v.dispose();assert.equal(original.disposed,true);
});
test('failed model switch immediately flushes the cleared scene and recovers on retry',async()=>{
  const s=await setup(),v=s.api.createPreview(),first=s.model(),second=s.model();
  const loadFirst=v.show(s.canvas,'first.glb');
  s.loads[0].done({scene:first}); await loadFirst;
  assert.equal(s.renderers[0].lastModels.includes(first),true);
  const rendersBeforeSwitch=s.renderers[0].renders;
  const loadSecond=v.show(s.canvas,'second.glb');
  assert.equal(first.disposed,1);
  assert.equal(s.renderers[0].renders>rendersBeforeSwitch,true);
  assert.equal(s.renderers[0].lastModels.includes(first),false);
  s.loads[1].fail(new Error('asset 404'));
  await assert.rejects(loadSecond,/asset 404/);
  assert.equal(s.renderers[0].loop,null);
  assert.equal(s.renderers[0].lastModels.includes(first),false);
  const retrySecond=v.show(s.canvas,'second.glb');
  s.loads[2].done({scene:second}); await retrySecond;
  assert.equal(s.renderers[0].lastModels.includes(second),true);
  assert.equal(typeof s.renderers[0].loop,'function');
});
test('refinement updates preserve camera position, orbit target and zoom',async()=>{
  const s=await setup(),v=s.api.createPreview();
  const first=v.show(s.canvas,'source.glb');s.loads[0].done({scene:s.model()});await first;
  s.cameras[0].position.set(4,5,6);s.cameras[0].zoom=1.7;s.controls[0].target.set(1,2,3);
  const next=v.show(s.canvas,'version-1.glb',{preserveCamera:true});s.loads[1].done({scene:s.model()});await next;
  assert.deepEqual({...s.cameras[0].position},{x:4,y:5,z:6});assert.equal(s.cameras[0].zoom,1.7);
  assert.deepEqual({...s.controls[0].target},{x:1,y:2,z:3});
  const other=v.show(s.canvas,'another-hero.glb');s.loads[2].done({scene:s.model()});await other;
  assert.notDeepEqual({...s.cameras[0].position},{x:4,y:5,z:6});
});
test('interaction during a refinement download keeps the displayed model until Update',async()=>{
  const s=await setup(),v=s.api.createPreview(),first=s.model(),next=s.model();let untouched=true;
  const load=v.show(s.canvas,'source.glb');s.loads[0].done({scene:first});await load;
  const update=v.show(s.canvas,'version-1.glb',{preserveCamera:true,canReplace:()=>untouched});
  assert.equal(first.disposed,0);assert.equal(s.scenes[0].models.includes(first),true);
  untouched=false;s.loads[1].done({scene:next});assert.equal(await update,false);
  assert.equal(first.disposed,0);assert.equal(next.disposed,1);assert.equal(s.scenes[0].models.includes(first),true);
  const explicit=v.show(s.canvas,'version-1.glb',{preserveCamera:true});s.loads[2].done({scene:s.model()});
  assert.equal(await explicit,true);assert.equal(first.disposed,1);
});
test('an unsuccessful refinement keeps the existing model visible',async()=>{
  const s=await setup(),v=s.api.createPreview(),first=s.model();
  const load=v.show(s.canvas,'source.glb');s.loads[0].done({scene:first});await load;
  const update=v.show(s.canvas,'version-1.glb',{preserveCamera:true});s.loads[1].fail(new Error('offline'));
  await assert.rejects(update,/offline/);assert.equal(first.disposed,0);
  assert.equal(s.scenes[0].models.includes(first),true);assert.equal(typeof s.renderers[0].loop,'function');
});

test('comparison cameras mirror position, target and zoom in either direction',async()=>{
  const s=await setup(),a=s.api.createPreview({damping:false}),b=s.api.createPreview({damping:false});
  const one=a.show(s.canvas,'one.glb'),two=b.show({...s.canvas},'two.glb');
  s.loads[0].done({scene:s.model()});s.loads[1].done({scene:s.model()});await Promise.all([one,two]);
  assert.equal(s.controls[0].enableDamping,false);assert.equal(s.controls[1].enableDamping,false);
  a.onView(value=>b.setView(value));b.onView(value=>a.setView(value));
  a.setView({position:[4,5,6],target:[1,2,3],zoom:1.5});
  s.controls[0].listeners.change();
  assert.deepEqual(b.view(),a.view());
  b.setView({position:[6,5,4],target:[3,2,1],zoom:2});
  s.controls[1].listeners.change();
  assert.deepEqual(a.view(),b.view());
  a.dispose();b.dispose();
});

test('studio lighting changes illumination while preserving the loaded camera and model',async()=>{
  const s=await setup(),v=s.api.createPreview(),model=s.model();
  const load=v.show(s.canvas,'hero.glb');s.loads[0].done({scene:model});await load;
  const before=v.view(),sun=s.scenes[0].models[1];
  v.lighting(true);assert.equal(sun.intensity,2.4);
  assert.deepEqual(v.view(),before);assert.equal(s.scenes[0].models.includes(model),true);
  v.lighting(false);assert.equal(sun.intensity,1.2);
  v.dispose();
});

// The workspace stage: eased camera presets, an idle orbit after 4 s, and reduced motion.
function stageClock(reduced = false) {
  let t = 1000;
  Object.defineProperty(globalThis, 'performance', {value: {now: () => t}, configurable: true, writable: true});
  globalThis.matchMedia = query => ({matches: reduced && query.includes('reduced-motion')});
  const listeners = {};
  const canvas = {clientWidth: 800, clientHeight: 500, classList: {add() {}, remove() {}},
    addEventListener(type, fn) { listeners[type] = fn; }, removeEventListener() {}};
  return {canvas, listeners, advance(ms) { t += ms; }};
}
const azimuth = (s) => { const p = s.cameras[0].position, c = s.controls[0].target; return Math.atan2(p.x - c.x, p.z - c.z); };
async function stage(reduced = false) {
  const clock = stageClock(reduced), s = await setup(), presets = [];
  s.api.onPreset(value => presets.push(value));
  const load = s.api.show(clock.canvas, 'hero.glb'); s.loads[0].done({scene: s.model()}); await load;
  const frame = (ms) => { clock.advance(ms); s.renderers[0].loop(); };
  return {...s, clock, presets, frame};
}
test('the stage reveals the first model and settles on the front preset', async () => {
  const s = await stage();
  assert.ok(Math.abs(azimuth(s)) > 0.2, 'starts off-axis');
  s.frame(600); assert.equal(s.presets.at(-1), null);
  s.frame(700); assert.ok(Math.abs(azimuth(s)) < 1e-6); assert.equal(s.presets.at(-1), 'front');
  delete globalThis.matchMedia;
});
test('camera presets ease to 90 and 180 degrees and report the preset when they land', async () => {
  const s = await stage(); s.frame(1300);
  s.api.camera('back'); s.frame(100); assert.equal(s.presets.at(-1), null);
  s.frame(400); assert.ok(Math.abs(Math.abs(azimuth(s)) - Math.PI) < 1e-6); assert.equal(s.presets.at(-1), 'back');
  s.api.camera('side'); s.frame(500); assert.ok(Math.abs(azimuth(s) - Math.PI / 2) < 1e-6); assert.equal(s.presets.at(-1), 'side');
  delete globalThis.matchMedia;
});
test('the idle orbit starts after 4 s, stops on input and can be switched off', async () => {
  const s = await stage(); s.frame(1300);
  s.frame(3000); const still = azimuth(s); s.frame(500); assert.equal(azimuth(s), still);
  s.frame(600); s.frame(16); const a = azimuth(s); s.frame(16); assert.notEqual(azimuth(s), a);
  assert.equal(s.presets.at(-1), null);
  s.clock.listeners.pointerdown(); s.frame(16); const b = azimuth(s); s.frame(3000); assert.equal(azimuth(s), b);
  s.frame(1100); s.frame(16); const c = azimuth(s); s.frame(16); assert.notEqual(azimuth(s), c);
  s.api.orbit(false); s.frame(16); const d = azimuth(s); s.frame(9000); s.frame(16); assert.equal(azimuth(s), d);
  delete globalThis.matchMedia;
});
test('reduced motion skips the reveal, never orbits and snaps presets', async () => {
  const s = await stage(true);
  assert.ok(Math.abs(azimuth(s)) < 1e-6); assert.equal(s.presets.at(-1), 'front');
  s.frame(10000); s.frame(16); const a = azimuth(s); s.frame(16); assert.equal(azimuth(s), a);
  s.api.camera('back'); assert.ok(Math.abs(Math.abs(azimuth(s)) - Math.PI) < 1e-6); assert.equal(s.presets.at(-1), 'back');
  delete globalThis.matchMedia;
});

// While the 3D shape is built: the figure of dots, turning from the start, then gathering onto the first model.
const figure = (key = 'hero') => ({key, positions: new Float32Array([0, 0.5, 0, 0, -0.5, 0, 0.2, 0, 0.1, -0.2, 0, -0.1]),
  seeds: new Float32Array([0.1, 0.4, 0.7, 0.9])});
const dots = (s) => s.scenes[0].models.find(n => n.name === 'hg-shape-figure');
test('the figure of dots turns from the start and is shown once per figure', async () => {
  const clock = stageClock(), s = await setup(), frame = (ms) => { clock.advance(ms); s.renderers[0].loop(); };
  assert.equal(s.api.building(clock.canvas, figure()), true);
  const group = dots(s), material = group.children[0].material;
  frame(16); const a = azimuth(s); frame(16); assert.notEqual(azimuth(s), a, 'turns at once');
  frame(1000); assert.ok(material.uniforms.opacity.value > 0.8 && material.uniforms.time.value > 0, 'fades in and drifts');
  assert.equal(s.api.building(clock.canvas, figure()), true); assert.equal(dots(s), group, 'the same figure is kept');
  clock.listeners.pointerdown(); frame(16); const b = azimuth(s); frame(2000); assert.equal(azimuth(s), b, 'input stops the turn');
  s.api.hide(); assert.equal(dots(s), undefined); assert.deepEqual(s.freed, ['geometry', 'material'], 'leaving frees it');
  delete globalThis.matchMedia;
});
test('the first model takes over from the dots without moving the view, then the dots are freed', async () => {
  const clock = stageClock(), s = await setup(), frame = (ms) => { clock.advance(ms); s.renderers[0].loop(); };
  s.api.building(clock.canvas, figure()); frame(16); frame(700);
  const angle = azimuth(s), group = dots(s);
  const material = {opacity: 1, transparent: false};
  const position = {array: new Float32Array([0, 1, 0, 0, -1, 0]), count: 2};
  const model = {traverse(fn) { fn({isMesh: true, userData: {}, material, matrixWorld: {}, geometry: {attributes: {position}, dispose() {}}}); }};
  const load = s.api.show(clock.canvas, 'sam-raw.glb'); s.loads[0].done({scene: model}); assert.equal(await load, true);
  assert.ok(Math.abs(azimuth(s) - angle) < 1e-9, 'the view keeps its angle');
  assert.equal(material.opacity, 0); assert.equal(material.transparent, true);
  frame(600); assert.ok(group.children[0].material.uniforms.converge.value > 0, 'the dots gather');
  frame(600); frame(600);
  assert.deepEqual([material.opacity, material.transparent], [1, false]);
  assert.equal(dots(s), undefined); assert.deepEqual(s.freed, ['geometry', 'material']);
  assert.equal(s.scenes[0].models.includes(model), true);
  delete globalThis.matchMedia;
});
test('with reduced motion the dots hold still and the model fades in without them moving', async () => {
  const clock = stageClock(true), s = await setup(), frame = (ms) => { clock.advance(ms); s.renderers[0].loop(); };
  s.api.building(clock.canvas, figure());
  frame(16); const a = azimuth(s), material = dots(s).children[0].material; frame(5000);
  assert.equal(azimuth(s), a); assert.equal(material.uniforms.drift.value, 0); assert.equal(material.uniforms.time.value, 0);
  const load = s.api.show(clock.canvas, 'sam-raw.glb'); s.loads[0].done({scene: s.model()}); await load;
  frame(100); assert.equal(material.uniforms.converge.value, 0);
  frame(1000); assert.equal(dots(s), undefined);
  delete globalThis.matchMedia;
});

// The split stage while the texture refines: SAM 3D's raw shape and HeroGen's model, two panels under one camera.
function meshModel(){
  const original={dispose(){this.disposed=true;}},mesh={isMesh:true,userData:{},material:original,geometry:{dispose(){}}};
  return {mesh,original,visible:true,position:{set(...v){this.v=v;}},scale:{setScalar(k){this.k=k;}},updateMatrixWorld(){},traverse(fn){fn(mesh);}};
}
async function splitStage(reduced=false){
  const clock=stageClock(reduced),s=await setup(),hero=meshModel(),raw=meshModel(),passes=[];
  const load=s.api.show(clock.canvas,'version-1.glb');s.loads[0].done({scene:hero});await load;
  const r=s.renderers[0];
  r.setScissorTest=on=>passes.push(['scissor',on]);r.setViewport=(...v)=>passes.push(['viewport',...v]);r.setScissor=()=>{};
  const render=r.render.bind(r);r.render=scene=>{passes.push(['draw',hero.visible,raw.visible]);render(scene);};
  return {...s,clock,hero,raw,passes,frame(ms){clock.advance(ms);passes.length=0;r.loop();}};
}
test('the split stage draws SAM 3D and HeroGen side by side under one camera, then joins again',async()=>{
  const s=await splitStage();
  const shown=s.api.split(s.clock.canvas,{left:'source.glb',ratio:0.5,axis:'columns',waiting:false});
  assert.equal(s.loads[1].url,'source.glb');s.loads[1].done({scene:s.raw});assert.equal(await shown,true);
  assert.equal(s.scenes[0].models.includes(s.raw),true);
  s.frame(16);
  assert.deepEqual(s.passes.filter(p=>p[0]==='viewport').slice(1),[['viewport',0,0,400,500],['viewport',400,0,400,500]]);
  assert.deepEqual(s.passes.filter(p=>p[0]==='draw'),[['draw',false,true],['draw',true,false]],'each panel shows only its own model');
  assert.equal(s.cameras.length,1,'one camera for both panels');
  // Dragging the divider only moves the panels; the same raw shape is not fetched again.
  await s.api.split(s.clock.canvas,{left:'source.glb',ratio:0.95,axis:'columns'});s.frame(16);
  assert.deepEqual(s.passes.filter(p=>p[0]==='viewport')[1],['viewport',0,0,640,500],'the ratio stops at 80%');
  assert.equal(s.loads.length,2);
  // Stacked: SAM 3D on top (WebGL counts from the bottom).
  await s.api.split(s.clock.canvas,{left:'source.glb',ratio:0.5,axis:'rows'});s.frame(16);
  assert.deepEqual(s.passes.filter(p=>p[0]==='viewport').slice(1),[['viewport',0,250,800,250],['viewport',0,0,800,250]]);
  await s.api.split(s.clock.canvas,{left:'source.glb',ratio:0.5,axis:'rows',inset:100});s.frame(16);
  assert.deepEqual(s.passes.filter(p=>p[0]==='viewport').slice(1),[['viewport',0,300,800,200],['viewport',0,100,800,200]],'both stop above the dock');
  // A new HeroGen version keeps the camera.
  s.frame(1300);s.cameras[0].position.set(4,5,6);s.controls[0].target.set(1,2,3);s.clock.listeners.pointerdown();
  const next=s.api.show(s.clock.canvas,'version-2.glb',{preserveCamera:true});s.loads[2].done({scene:meshModel()});await next;
  assert.deepEqual({...s.cameras[0].position},{x:4,y:5,z:6});assert.deepEqual({...s.controls[0].target},{x:1,y:2,z:3});
  assert.equal(await s.api.animate('walk',{model:'body.glb',parts:[]}),false,'no animation while split');
  await s.api.split(s.clock.canvas,null);s.frame(16);
  assert.equal(s.scenes[0].models.includes(s.raw),false);assert.equal(s.raw.mesh.material,s.raw.original);
  assert.equal(s.passes.some(p=>p[0]==='viewport'),false,'one full view again');
  delete globalThis.matchMedia;
});
test('while the texture is repainted the HeroGen panel shows the shape in clay under a moving light',async()=>{
  const s=await splitStage();
  const shown=s.api.split(s.clock.canvas,{left:'source.glb',ratio:0.5,axis:'columns',waiting:true});s.loads[1].done({scene:s.raw});await shown;
  const swept=s.hero.mesh.material;
  assert.notEqual(swept,s.hero.original);assert.equal(typeof swept.onBeforeCompile,'function');assert.equal(swept.wireframe,false);
  assert.equal(s.raw.mesh.material,s.raw.original,'SAM 3D keeps its own texture');
  const shader={uniforms:{},vertexShader:'#include <common>\n#include <project_vertex>',fragmentShader:'#include <common>\n#include <emissivemap_fragment>'};
  swept.onBeforeCompile(shader);
  assert.match(shader.fragmentShader,/totalEmissiveRadiance \+=/);assert.ok('hgPhase' in shader.uniforms);
  s.frame(16);const a=shader.uniforms.hgPhase.value;s.frame(900);assert.notEqual(shader.uniforms.hgPhase.value,a,'the light moves');
  s.api.shading('wire');assert.equal(s.hero.mesh.material.wireframe,true);assert.equal(typeof s.hero.mesh.material.onBeforeCompile,'function');
  assert.equal(s.raw.mesh.material.wireframe,true,'the dock shading applies to both panels');
  s.api.shading('textured');
  await s.api.split(s.clock.canvas,{left:'source.glb',ratio:0.5,axis:'columns',waiting:false});
  assert.equal(s.hero.mesh.material,s.hero.original,'the refined texture shows once it lands');
  delete globalThis.matchMedia;
});
test('with reduced motion the repaint light stays parked',async()=>{
  const s=await splitStage(true);
  const shown=s.api.split(s.clock.canvas,{left:'source.glb',waiting:true});s.loads[1].done({scene:s.raw});await shown;
  const shader={uniforms:{},vertexShader:'',fragmentShader:''};s.hero.mesh.material.onBeforeCompile(shader);
  s.frame(16);s.frame(900);assert.equal(shader.uniforms.hgPhase.value,-10);
  delete globalThis.matchMedia;
});

test('a mesh in other units (the Roblox body in studs) takes the place of the last one under the same view',async()=>{
  const s=await setup(),v=s.api.createPreview(),refined=s.model();refined.box={size:[1,2,1],centre:[0,1,0]};
  const first=v.show(s.canvas,'refined.glb');s.loads[0].done({scene:refined});await first;
  s.cameras[0].position.set(0,1.5,4);s.controls[0].target.set(0,1,0);
  const body=s.model();body.box={size:[2.5,5.5,1.5],centre:[0,2.75,0]};
  const next=v.show(s.canvas,'high.glb',{preserveCamera:true});s.loads[1].done({scene:body});await next;
  assert.deepEqual({...s.controls[0].target},{x:0,y:2.75,z:0},'the target keeps its place on the hero');
  const p=s.cameras[0].position;assert.ok(Math.abs(p.y-4.125)<1e-9&&Math.abs(p.z-11)<1e-9,'the camera keeps its angle, scaled');
  const low=s.model();low.box={size:[2.5,5.5,1.5],centre:[0,2.75,0]};
  const third=v.show(s.canvas,'low.glb',{preserveCamera:true});s.loads[2].done({scene:low});await third;
  assert.ok(Math.abs(s.cameras[0].position.z-11)<1e-9,'high to low: the very same camera');
});
