import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { SourceTextModule, SyntheticModule } from 'node:vm';

async function setup() {
  const loads=[], scenes=[], renderers=[];
  class Vector3 { constructor(){ this.x=0;this.y=0;this.z=0; } set(){} copy(){} }
  class Scene { constructor(){this.models=[];scenes.push(this);} add(n){this.models.push(n);} remove(n){this.models=this.models.filter(v=>v!==n);} }
  class Renderer { constructor(){renderers.push(this);} setPixelRatio(){} setSize(){} setAnimationLoop(fn){this.loop=fn;} render(){} }
  class Camera { constructor(){this.position=new Vector3();} updateProjectionMatrix(){} }
  class Box3 { setFromObject(){return this;} getSize(v){v.x=1;v.y=2;v.z=1;return v;} getCenter(v){return v;} }
  class Light { constructor(){this.position=new Vector3();} }
  class Controls { constructor(){this.target=new Vector3();} update(){} }
  class Loader { load(url,done,progress,fail){loads.push({url,done,fail});} }
  const three={WebGLRenderer:Renderer,Scene,HemisphereLight:Light,DirectionalLight:Light,GridHelper:Light,
    PerspectiveCamera:Camera,Vector3,Box3};
  const stub=(exports)=>new SyntheticModule(Object.keys(exports),function(){for(const [k,v] of Object.entries(exports))this.setExport(k,v);});
  const source=await readFile(new URL('../herogen/service/static/preview.js',import.meta.url),'utf8');
  globalThis.devicePixelRatio=1; globalThis.ResizeObserver=class {observe(){}};
  const module=new SourceTextModule(source);
  await module.link((id)=>stub(id==='three'?three:id.includes('OrbitControls')?{OrbitControls:Controls}:{GLTFLoader:Loader}));
  await module.evaluate();
  const model=()=>({disposed:0,traverse(fn){fn({geometry:{dispose:()=>this.disposed++}});}});
  return {api:module.namespace,loads,scenes,renderers,model,canvas:{clientWidth:800,clientHeight:500}};
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
  s.loads[0].fail(new Error('offline')); await assert.rejects(bad,/offline/);
  const good=s.api.show(s.canvas,'good.glb'); s.loads[1].done({scene:s.model()}); await good;
  s.api.hide(); await s.api.show(s.canvas,'good.glb');
  assert.equal(s.loads.length,2); assert.equal(typeof s.renderers[0].loop,'function');
});
