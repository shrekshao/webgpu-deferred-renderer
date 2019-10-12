import Renderer from './renderer/DeferredRenderer.js';
// import Renderer from './renderer/ForwardRenderer.js';
// import DeferredRenderer from './renderer/TestRT.js';




(async function () {
const gui = new dat.GUI();

const canvas = document.querySelector('canvas');

const renderer = new Renderer(canvas);

await renderer.init();

gui.add(renderer, 'debugViewOffset', 0.0, 5.0);

function frame() {
    renderer.frame();
    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
})();
