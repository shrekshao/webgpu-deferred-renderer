import DeferredRenderer from './renderer/DeferredRenderer.js';
// import DeferredRenderer from './renderer/TestRT.js';

(async function () {
const canvas = document.querySelector('canvas');

const renderer = new DeferredRenderer(canvas);

await renderer.init();

function frame() {
    renderer.frame();
    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
})();
