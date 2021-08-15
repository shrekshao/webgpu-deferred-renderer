# WebGPU Deferred Renderer

![](imgs/tiled-deferred-4096-lights.gif)

![](imgs/tile-light-count-heatmap.gif)

![](imgs/debug-view.gif)

### Run locally

```
/Applications/Google\ Chrome\ Canary.app/Contents/MacOS/Google\ Chrome\ Canary  --enable-unsafe-webgpu "http://localhost:8080"
```

### Status

This is now a basic working implementation of deferred renderer using WebGPU API, with a tile-based light culling implementation powered by compute shader.

I'm still doing cleaning, refactoring, fixing, etc. on the fly.

I have it working on Win10 Nvidia, Chrome Canary (92.0.4515.131 canary (64-bit))

