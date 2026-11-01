import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { SourceTextModule, SyntheticModule } from 'node:vm';

async function setup() {
  const loads=[], scenes=[], renderers=[], cameras=[], controls=[];
  class Vector3 { constructor(){ this.x=0;this.y=0;this.z=0; } set(x,y,z){Object.assign(this,{x,y,z});} copy(v){Object.assign(this,v);} setScalar(){} toArray(){return [this.x,this.y,this.z];} }
  class Scene { constructor(){this.models=[];scenes.push(this);} add(n){this.models.push(n);} remove(n){this.models=this.models.filter(v=>v!==n);} }
  class Renderer { constructor(){this.disposed=false;this.renders=0;this.lastModels=[];renderers.push(this);} dispose(){this.disposed=true;} setPixelRatio(){} setSize(){} setAnimationLoop(fn){this.loop=fn;} render(scene){this.renders++;this.lastModels=[...scene.models];} }
  class Camera { constructor(){this.position=new Vector3();cameras.push(this);} updateProjectionMatrix(){} }
  class Box3 { constructor(){this.min={y:0};} setFromObject(){return this;} getSize(v){v.x=1;v.y=2;v.z=1;return v;} getCenter(v){return v;} }
  class Light { constructor(){this.position=new Vector3();this.scale=new Vector3();this.geometry={dispose(){}};this.material={color:{set(){}},dispose(){}};} }
  class Controls { constructor(){this.target=new Vector3();this.listeners={};controls.push(this);} addEventListener(name,callback){this.listeners[name]=callback;} update(){} dispose(){} }
  class Loader { load(url,done,progress,fail){loads.push({url,done,fail});} }
  class PMREMGenerator { fromScene(){return {texture:{dispose(){}}};} dispose(){} }
  const three={WebGLRenderer:Renderer,Scene,HemisphereLight:Light,DirectionalLight:Light,GridHelper:Light,PMREMGenerator,SRGBColorSpace:'srgb',ACESFilmicToneMapping:4,
    PerspectiveCamera:Camera,Vector3,Box3,MeshStandardMaterial:class { constructor(options){Object.assign(this,options);} dispose(){this.disposed=true;} }};
  const stub=(exports)=>new SyntheticModule(Object.keys(exports),function(){for(const [k,v] of Object.entries(exports))this.setExport(k,v);});
  const source=await readFile(new URL('../herogen/service/static/preview.js',import.meta.url),'utf8');
  globalThis.devicePixelRatio=1; globalThis.ResizeObserver=class {observe(){} disconnect(){}};
  globalThis.document={documentElement:{dataset:{theme:'light'}},addEventListener(){},removeEventListener(){}};
  const module=new SourceTextModule(source);
  await module.link((id)=>stub(id==='three'?three:id.includes('OrbitControls')?{OrbitControls:Controls}:id.includes('RoomEnvironment')?{RoomEnvironment:class {}}:{GLTFLoader:Loader}));
  await module.evaluate();
  const model=()=>({disposed:0,traverse(fn){fn({geometry:{dispose:()=>this.disposed++}});}});
  return {api:module.namespace,loads,scenes,renderers,cameras,controls,model,canvas:{clientWidth:800,clientHeight:500}};
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
