// Lagnav Level 1 viewer. Reads scene state from the server and applies it to the meshes;
// every change goes back to the server as a text command. The engine is the only writer.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
const css = getComputedStyle(document.documentElement);
scene.background = new THREE.Color(css.getPropertyValue('--canvas').trim() || '#e9e6e0');

// Viewer frame (manifest.frame): +X subject's left, +Y superior, +Z anterior.
// A camera on +Z looks at the subject from the front.
const camera = new THREE.PerspectiveCamera(35, 1, 1, 10000);
camera.position.set(0, 30, 600);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.1));
const key = new THREE.DirectionalLight(0xffffff, 1.4);
key.position.set(200, 300, 400);
scene.add(key);
const rim = new THREE.DirectionalLight(0xffffff, 0.5);
rim.position.set(-300, -100, -300);
scene.add(rim);

// three.js strips '.', ':', '/', '[' and ']' from glTF node names, so an entity id is
// matched to its mesh through the same rule rather than through the raw name.
const sanitise = (name) => name.replace(/\s/g, '_').replace(/[[\].:/]/g, '');
const meshes = new Map();       // entity id -> THREE.Mesh
let state = null;
let hovered = null;

function resize() {
  const { clientWidth: w, clientHeight: h } = canvas.parentElement;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

function frameScene() {
  const box = new THREE.Box3();
  meshes.forEach((m) => box.expandByObject(m));
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3()).length();
  const centre = box.getCenter(new THREE.Vector3());
  controls.target.copy(centre);
  // Distance at which the bounding sphere fits the narrower of the two fields of view.
  const radius = size / 2;
  const vertical = THREE.MathUtils.degToRad(camera.fov) / 2;
  const horizontal = Math.atan(Math.tan(vertical) * camera.aspect);
  const distance = (radius / Math.sin(Math.min(vertical, horizontal))) * 1.05;
  camera.position.copy(centre).add(new THREE.Vector3(0, radius * 0.1, distance));
  camera.near = size / 100;
  camera.far = size * 20;
  camera.updateProjectionMatrix();
}

function applyState() {
  if (!state) return;
  for (const e of state.entities) {
    const mesh = meshes.get(e.id);
    if (!mesh) continue;
    const m = mesh.material;
    m.color.set(e.colour);
    m.opacity = e.opacity;
    m.transparent = e.opacity < 0.999;
    m.depthWrite = !m.transparent;
    mesh.renderOrder = m.transparent ? 1 : 0;
    mesh.visible = e.visible;
  }
  $('lod').textContent =
    `Level of detail ${state.active_lod} · ${state.educational_level} · scene v${state.scene_version}`;
  renderTree();
}

function byId() {
  const map = new Map();
  state.entities.forEach((e) => map.set(e.id, e));
  return map;
}

function renderTree() {
  const tree = $('tree');
  tree.replaceChildren();
  const roots = state.entities.filter((e) => e.parent === 'thorax');
  const walk = (e, depth) => {
    const kids = state.entities.filter((c) => c.parent === e.id);
    const showable = e.renderable && !e.group;
    if (showable) tree.append(row(e, depth));
    else if (kids.length) {
      const li = document.createElement('li');
      li.className = 'group';
      li.style.paddingLeft = `${depth * 12}px`;
      li.textContent = e.name;
      tree.append(li);
    }
    kids.forEach((c) => walk(c, depth + 1));
  };
  roots.forEach((r) => walk(r, 0));
}

function row(e, depth) {
  const li = document.createElement('li');
  li.style.paddingLeft = `${depth * 12}px`;
  const dot = document.createElement('span');
  dot.className = 'dot';
  dot.style.background = e.visible ? e.colour : 'transparent';
  dot.title = e.visible ? 'visible' : 'hidden';
  const name = document.createElement('span');
  name.className = 'name';
  name.textContent = e.name;
  name.title = `Explain the ${e.name.toLowerCase()}`;
  name.addEventListener('click', () => run(`explain the ${e.name.toLowerCase()}`));
  li.append(dot, name);
  if (!e.has_geometry) {
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.textContent = 'no geometry yet';
    tag.title = 'In the anatomy model, but the source data has no shape for it';
    li.append(tag);
  }
  const toggle = document.createElement('button');
  toggle.className = 'toggle';
  toggle.textContent = e.visible ? 'Hide' : 'Show';
  toggle.addEventListener('click', () =>
    run(`${e.visible ? 'hide' : 'show'} the ${e.name.toLowerCase()}`));
  li.append(toggle);
  return li;
}

function log(text, kind) {
  const line = document.createElement('div');
  line.className = kind;
  line.textContent = text;
  const box = $('log');
  box.append(line);
  box.scrollTop = box.scrollHeight;
}

async function run(text) {
  log(`› ${text}`, 'cmd');
  const res = await fetch('/command', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  const body = await res.json();
  if (!res.ok) { log(body.error || 'Request failed', 'bad'); return; }
  log(body.message, body.ok ? 'ok' : 'bad');
  state = body.state;
  applyState();
}

$('form').addEventListener('submit', (ev) => {
  ev.preventDefault();
  const text = $('text').value.trim();
  if (text) { run(text); $('text').value = ''; }
});

const examples = [
  'Show only the lungs', 'Show everything', 'Make the left ventricle transparent',
  'Show more detail', 'Show less detail', 'Hide the trachea', 'Explain the right atrium',
];
for (const ex of examples) {
  const b = document.createElement('button');
  b.type = 'button';
  b.textContent = ex;
  b.addEventListener('click', () => run(ex));
  $('chips').append(b);
}
const reset = document.createElement('button');
reset.type = 'button';
reset.textContent = 'Reset';
reset.addEventListener('click', async () => {
  const res = await fetch('/reset', { method: 'POST' });
  const body = await res.json();
  log(body.message, 'ok');
  state = body.state;
  applyState();
});
$('chips').append(reset);

// Hover names a structure; a click explains it.
const ray = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const idOf = new Map();          // mesh uuid -> entity id
function pick(ev) {
  const r = canvas.getBoundingClientRect();
  pointer.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(pointer, camera);
  const hits = ray.intersectObjects([...meshes.values()].filter((m) => m.visible), false);
  return hits.length ? idOf.get(hits[0].object.uuid) : null;
}
canvas.addEventListener('pointermove', (ev) => {
  const id = pick(ev);
  if (id === hovered) return;
  if (hovered && meshes.get(hovered)) meshes.get(hovered).material.emissive.setHex(0x000000);
  hovered = id;
  if (hovered) meshes.get(hovered).material.emissive.setHex(0x333333);
  const e = hovered && state ? byId().get(hovered) : null;
  $('hud').textContent = e ? e.name : '';
});
canvas.addEventListener('click', (ev) => {
  const id = pick(ev);
  const e = id && state ? byId().get(id) : null;
  if (e) run(`explain the ${e.name.toLowerCase()}`);
});

async function start() {
  resize();
  $('status').textContent = 'Loading scene…';
  const [manifest, first] = await Promise.all([
    fetch('/manifest.json').then((r) => r.json()),
    fetch('/state').then((r) => r.json()),
  ]);
  state = first;
  $('banner').textContent =
    `${state.medical_status[0].toUpperCase()}${state.medical_status.slice(1)}.`;
  const c = state.credits;
  $('credits').innerHTML = '';
  $('credits').append(
    `Shapes derived from ${c.dataset} v${c.version} (subject ${c.subject}), ${c.license}. `,
    Object.assign(document.createElement('a'), { href: c.url, textContent: 'Source', target: '_blank', rel: 'noopener' }),
    ` · ${c.attribution}`,
  );

  const wanted = new Map(Object.keys(manifest.entities).map((id) => [sanitise(id), id]));
  if (wanted.size !== Object.keys(manifest.entities).length) {
    log('Two entity ids collide after glTF name sanitising; some meshes are ambiguous.', 'bad');
  }
  const gltf = await new GLTFLoader().loadAsync('/scene.glb');
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    const id = wanted.get(o.name) ?? wanted.get(o.parent?.name);
    if (!id) return;
    if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
    o.material = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.05 });
    meshes.set(id, o);
    idOf.set(o.uuid, id);
  });
  scene.add(gltf.scene);
  const missing = Object.keys(manifest.entities).filter((id) => !meshes.has(id));
  if (missing.length) log(`Meshes not found in scene.glb: ${missing.join(', ')}`, 'bad');
  frameScene();
  applyState();
  $('status').textContent = `${meshes.size} meshes`;
  log('Ready. Type a command, or click a structure.', 'ok');
}

function loop() {
  requestAnimationFrame(loop);
  controls.update();
  renderer.render(scene, camera);
}
start().catch((err) => { $('status').textContent = 'Failed to load'; log(String(err), 'bad'); });
loop();
