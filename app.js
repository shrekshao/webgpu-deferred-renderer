import Renderer from './renderer/DeferredRenderer.js';
// import Renderer from './renderer/ForwardRenderer.js';
// import DeferredRenderer from './renderer/TestRT.js';

(async function () {
const canvas = document.querySelector('canvas');

const renderer = new Renderer(canvas);

await renderer.init();

function frame() {
    renderer.frame();
    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
})();
