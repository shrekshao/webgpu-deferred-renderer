import Renderer from './renderer/DeferredRenderer.js';

(async function () {
    
    const stats = new Stats();
    stats.showPanel( 0 ); // 0: fps, 1: ms, 2: mb, 3+: custom
    document.body.appendChild( stats.dom );

    const gui = new dat.GUI();

    const canvas = document.querySelector('canvas');
    // canvas.width = window.innerWidth;
    // canvas.height = window.innerHeight;

    const renderer = new Renderer(canvas);

    await renderer.init();

    gui.add(renderer, 'debugViewOffset', 0.0, 5.0);
    gui.add(renderer, 'renderMode', renderer.renderModeLists).onChange(renderer.onChangeRenderMode.bind(renderer));


    function frame() {
        stats.begin();
        renderer.frame();
        stats.end();
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);

})();
