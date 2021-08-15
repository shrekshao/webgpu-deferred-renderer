import Geometry from './Geometry.js';
import Camera from './Camera.js';
import WriteGBufferMaterial from './WriteGBufferMaterial.js';
import Drawable from './Drawable.js';
import LightCulling from './LightCulling.js';

import {replaceArray} from './utils/stringReplaceArray.js'

const tmpVec3 = vec3.create();
const tmpVec4 = vec4.create();
const tmpMat4 = mat4.create();

const vertexShaderFullScreenQuad = `
struct VertexOutput {
    [[builtin(position)]] Position : vec4<f32>;
    [[location(0)]] fragUV: vec2<f32>;
};

[[stage(vertex)]]
fn main([[location(0)]] position : vec4<f32>,
        [[location(1)]] uv : vec2<f32>)
        -> VertexOutput {
    var output : VertexOutput;
    output.Position = position;
    output.fragUV = uv;
    return output;
}
`;

const fragmentShaderGBufferDebugView = `
[[group(0), binding(1)]] var gBufferPosition: texture_2d<f32>;
[[group(0), binding(2)]] var gBufferNormal: texture_2d<f32>;
[[group(0), binding(3)]] var gBufferAlbedo: texture_2d<f32>;

[[stage(fragment)]]
fn main([[builtin(position)]] coord : vec4<f32>,
        [[location(0)]] fragUV : vec2<f32>)
     -> [[location(0)]] vec4<f32> {
  var result : vec4<f32>;
  var c : vec2<f32> = fragUV;
  if (c.x < 0.33333) {
    result = textureLoad(
        gBufferPosition,
        vec2<i32>(floor(coord.xy)),
        0
    );
  } elseif (c.x < 0.66667) {
    result = textureLoad(
        gBufferNormal,
        vec2<i32>(floor(coord.xy)),
        0
    );
    result.x = (result.x + 1.0) * 0.5;
    result.y = (result.y + 1.0) * 0.5;
    result.z = (result.z + 1.0) * 0.5;
    result.a = 1.0;
  } else {
    result = textureLoad(
        gBufferAlbedo,
        vec2<i32>(floor(coord.xy)),
        0
    );
  }
  return result;
}
`;

const fragmentShaderDeferredShadingTiledLightDebug = `
struct TileLightIdData {
    count: atomic<u32>;
    lightId: array<u32, $NUM_TILE_LIGHT_SLOT>;
};
[[block]] struct Tiles {
    data: array<TileLightIdData, $NUM_TILES>;
};
[[group(0), binding(0)]] var<storage, read_write> tileLightId: Tiles;

[[stage(fragment)]]
fn main([[builtin(position)]] coord : vec4<f32>,
        [[location(0)]] fragUV : vec2<f32>)
     -> [[location(0)]] vec4<f32> {
    var result : vec4<f32>;
    let TILE_COUNT_X: u32 = $TILE_COUNT_Xu;
    let TILE_COUNT_Y: u32 = $TILE_COUNT_Yu;

    var tileScale = vec2<f32>(1.0 / f32(TILE_COUNT_X), 1.0 / f32(TILE_COUNT_Y));
    var flipUV = vec2<f32>(fragUV.x, 1.0 - fragUV.y);
    var tileCoord = vec2<u32>(floor( flipUV / tileScale ));
    var tileId: u32 = tileCoord.x + tileCoord.y * TILE_COUNT_X;

    var c: u32 = atomicLoad(&tileLightId.data[tileId].count);
    var t: f32 = f32(c) / ( f32($2) );
    result.r = 4.0 * t - 2.0;
    if (t < 0.5) {
        result.g = 4.0 * t;
    } else {
        result.g = 4.0 - 4.0 * t;
    }
    result.b = 2.0 - 4.0 * t;
    result.a = 1.0;

    return result;
}
`;

const fragmentShaderDeferredShadingTiled = `
[[group(0), binding(1)]] var gBufferPosition: texture_2d<f32>;
[[group(0), binding(2)]] var gBufferNormal: texture_2d<f32>;
[[group(0), binding(3)]] var gBufferAlbedo: texture_2d<f32>;

[[block]] struct CameraPositionUniform {
    position: vec4<f32>;
};
[[group(1), binding(0)]] var<uniform> camera: CameraPositionUniform;

struct LightData {
    position : vec4<f32>;
    color : vec3<f32>;
    radius : f32;
};
[[block]] struct LightsBuffer {
    lights: array<LightData>;
};
[[group(1), binding(1)]] var<storage, read_write> lightsBuffer: LightsBuffer;

struct TileLightIdData {
    count: atomic<u32>;
    lightId: array<u32, $NUM_TILE_LIGHT_SLOT>;
};
[[block]] struct Tiles {
    data: array<TileLightIdData, $NUM_TILES>;
};
[[group(1), binding(2)]] var<storage, read_write> tileLightId: Tiles;

[[stage(fragment)]]
fn main([[builtin(position)]] coord : vec4<f32>,
        [[location(0)]] fragUV : vec2<f32>)
     -> [[location(0)]] vec4<f32> {
  var result = vec3<f32>(0.0, 0.0, 0.0);
  var c = vec2<i32>(floor(coord.xy));

  var position = textureLoad(
    gBufferPosition,
    c,
    0
  ).xyz;

  if (position.z > 10000.0) {
    discard;
  }

  var normal = textureLoad(
    gBufferNormal,
    c,
    0
  ).xyz;

  var albedo = textureLoad(
    gBufferAlbedo,
    c,
    0
  ).rgb;

  var V = normalize(camera.position.xyz - position);

  let TILE_COUNT_X: u32 = $TILE_COUNT_Xu;
  let TILE_COUNT_Y: u32 = $TILE_COUNT_Yu;

  var tileScale = vec2<f32>(1.0 / f32(TILE_COUNT_X), 1.0 / f32(TILE_COUNT_Y));
  var flipUV = vec2<f32>(fragUV.x, 1.0 - fragUV.y);
  var tileCoord = vec2<u32>(floor( flipUV / tileScale ));

  var tileId: u32 = tileCoord.x + tileCoord.y * TILE_COUNT_X;

  var count: u32 = atomicLoad(&tileLightId.data[tileId].count);
  for (var i : u32 = 0u; i < $2u; i = i + 1u) {
    if (i >= count) {
        break;
    }
    var light = lightsBuffer.lights[ tileLightId.data[tileId].lightId[i] ];

    var L = light.position.xyz - position;
    var distance = length(L);
    if (distance  > light.radius) {
        continue;
    }
    L = normalize(L);

    var lambert = max(dot(L, normal), 0.0);
    var H = normalize(L + V);
    var specular = f32(lambert > 0.0) * pow(max(dot(H, normal), 0.0), 10.0);

    result = result +
            clamp(light.color * pow(1.0 - distance / light.radius, 2.0) *
            (
                albedo * lambert + 
                    vec3<f32>(1.0) * specular  // Assume white specular, modify if you add more specular info
            ), vec3<f32>(0.0), vec3<f32>(1.0));
  }

  return vec4<f32>(result, 1.0);
}
`;

async function createTextureFromImage(device, src, usage) {
    const img = document.createElement('img');
    img.src = src;
    await img.decode();
    const imageBitmap = await createImageBitmap(img, { imageOrientation: 'flipY' });

    const texture = device.createTexture({
      size: [imageBitmap.width, imageBitmap.height, 1],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture(
      { source: imageBitmap },
      { texture: texture },
      [imageBitmap.width, imageBitmap.height]
    );
    return texture;
}

const quadVertexSize = 4 * 6;
const quadUVOffset = 4 * 4;
const fullScreenQuadArray = new Float32Array([
    // float4 position, float2 uv
    -1, -1, 0.5, 1, 0, 0,
    1, -1, 0.5, 1, 1, 0,
    1, 1, 0.5, 1, 1, 1,
    -1, -1, 0.5, 1, 0, 0,
    1, 1, 0.5, 1, 1, 1,
    -1, 1, 0.5, 1, 0, 1,
]);

export default class DeferredRenderer {
    constructor(canvas) {
        this.canvas = canvas;

        this.drawableLists = [];

        this.camera = new Camera(canvas);

        // dat.gui controls
        this.renderFuncs = {
            'debugView': this.renderGBufferDebugView,
            'deferredTiledLightDebug': this.renderDeferredTiledLightDebug,
            'deferredTiledLightCulling': this.renderDeferredLightCulling,
        };
        this.renderModeLists = Object.keys(this.renderFuncs);

        const i = 0;
        // const i = 1;
        // const i = 2;
        // const i = 3;
        this.renderMode = this.renderModeLists[i];
        this.curRenderModeFunc = this.renderFuncs[this.renderMode];
    }

    onChangeRenderMode(v) {
        this.curRenderModeFunc = this.renderFuncs[v];
    }

    //------------------

    async init() {
        /* Context, Device, SwapChain */
        const adapter = await navigator.gpu.requestAdapter();
        const device = this.device = await adapter.requestDevice({});

        const canvas = this.canvas;
        const context = this.context = canvas.getContext('webgpu');

        context.configure({
            device,
            format: "bgra8unorm",
        });

        WriteGBufferMaterial.setup(device);

        const matrixSize = 4 * 16;  // 4x4 matrix
        const uniformBufferSize = 2 * matrixSize;

        const uniformBuffer = this.uniformBuffer = device.createBuffer({
            size: uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.lightCulling = new LightCulling(device);
        await this.lightCulling.init(canvas, this.camera);

        await this.setupScene(device);

        /* Render Pipeline */
        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [WriteGBufferMaterial.uniformsBindGroupLayout]
        });
        this.pipeline = device.createRenderPipeline({
            layout: pipelineLayout,

            vertex: {
                module: device.createShaderModule({
                    code: WriteGBufferMaterial.vertexShader,
                }),
                entryPoint: "main",
                buffers: this.drawableLists[0].geometry.vertexBuffers,
            },
            fragment: {
                module: device.createShaderModule({
                    code: WriteGBufferMaterial.fragmentShader,
                }),
                entryPoint: "main",
                targets: [
                    { format: 'rgba32float' },
                    { format: 'rgba32float' },
                    { format: 'bgra8unorm' },
                ],
            },
            primitive: {
                topology: 'triangle-list',
                frontFace: 'ccw',
                cullMode: 'none',
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: "depth24plus-stencil8",
            },
        });

        const depthTexture = this.depthTexture = device.createTexture({
            size: {
                width: canvas.width,
                height: canvas.height,
                depthOrArrayLayers: 1
            },
            mipLevelCount: 1,
            dimension: "2d",
            format: "depth24plus-stencil8",
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });


        // 10/11/2019 Unfortunately 
        // Currently Dawn does not support layered rendering.
        // https://cs.chromium.org/chromium/src/third_party/dawn/src/dawn_native/CommandEncoder.cpp?l=264
        // Might be a bit painful when you change number of gbuffer as you need to modify that in shader
        // Though you could have some helper function to inject strings that declares textures to shader
        // 08/14/2021 Not true anymore. But I'm too lazy to update this.
        // Look for https://github.com/austinEng/webgpu-samples/blob/main/src/sample/deferredRendering/main.ts
        // for array layer usage

        this.gbufferTextures = [
            device.createTexture({
                size: {
                    width: canvas.width,
                    height: canvas.height,
                    depthOrArrayLayers: 1
                },
                mipLevelCount: 1,
                sampleCount: 1,
                dimension: "2d",
                format: "rgba32float",
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
            }),
            device.createTexture({
                size: {
                    width: canvas.width,
                    height: canvas.height,
                    depthOrArrayLayers: 1
                },
                mipLevelCount: 1,
                sampleCount: 1,
                dimension: "2d",
                format: "rgba32float",
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
            }),
            device.createTexture({
                size: {
                    width: canvas.width,
                    height: canvas.height,
                    depthOrArrayLayers: 1
                },
                mipLevelCount: 1,
                sampleCount: 1,
                dimension: "2d",
                format: "bgra8unorm",
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
            }),
        ];

        this.renderPassDescriptor = {
            colorAttachments: [
                {
                    view: this.gbufferTextures[0].createView(),
                    loadValue: {
                        r: Number.MAX_VALUE,
                        g: Number.MAX_VALUE,
                        b: Number.MAX_VALUE,
                        a: 1.0,
                      },
                    storeOp: "store",
                },
                {
                    view: this.gbufferTextures[1].createView(),
                    loadValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                    storeOp: "store",
                },
                {
                    view: this.gbufferTextures[2].createView(),
                    loadValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                    storeOp: "store",
                },
            ],
            depthStencilAttachment: {
                view: depthTexture.createView(),

                depthLoadValue: 1.0,
                depthStoreOp: "store",
                stencilLoadValue: 0,
                stencilStoreOp: "store",
            }
        };

        this.setupQuadPipeline();
        
    }

    setupQuadPipeline() {

        const device = this.device;

        this.quadVerticesBuffer = device.createBuffer({
            size: fullScreenQuadArray.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(
            this.quadVerticesBuffer,
            0,
            fullScreenQuadArray.buffer,
            fullScreenQuadArray.byteOffset,
            fullScreenQuadArray.byteLength
        );

        this.quadUniformsBindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: {
                        type: "filtering",  //f32 is not filterable
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: {
                        sampleType: 'float',
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: {
                        sampleType: 'float',
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: {
                        sampleType: 'float',
                    }
                },
            ]
        });

        this.uniformBufferBindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: 'uniform',
                    }
                },
            ]
        });
        this.dynamicUniformBufferBindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: 'uniform',
                        hasDynamicOffsets: true,
                    }
                },
            ]
        });

        this.renderFullScreenPassDescriptor = {
            colorAttachments: [{
                view: undefined,
                loadValue: {r: 0.0, g: 0.0, b: 0.0, a: 1.0},
                storeOp: "store",
            }],
        };

        // Pass in camera position (eye) to compute specular (not important)
        this.cameraPositionUniformBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.setupGBufferDebugViewPipeline();
        this.setupDeferredTiledLightDebugPipeline();
        this.setupDeferredLightCullingPipeline();
    }

    setupGBufferDebugViewPipeline() {
        this.gbufferDebugViewPipeline = this.device.createRenderPipeline({
            vertex: {
                module: this.device.createShaderModule({
                    code: vertexShaderFullScreenQuad,
                }),
                entryPoint: "main",
                buffers: [{
                    arrayStride: quadVertexSize, //padding
                    stepMode: "vertex",
                    attributes: [{
                        // position
                        shaderLocation: 0,
                        offset: 0,
                        format: "float32x4"
                    },
                    {
                        // uv
                        shaderLocation: 1,
                        offset: quadUVOffset,
                        format: "float32x2"
                    }]
                }]
            },
            fragment: {
                module: this.device.createShaderModule({
                    code: fragmentShaderGBufferDebugView,
                }),
                entryPoint: "main",
                targets: [
                    { format: 'bgra8unorm' },
                ],
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'back',
            },
        });

        const debugViewUniformBufferSize = 16;
        const debugViewUniformBuffer = this.debugViewUniformBuffer = this.device.createBuffer({
            size: debugViewUniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.debugViewUniformBindGroup = this.device.createBindGroup({
            layout: this.uniformBufferBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: debugViewUniformBuffer,
                        offset: 0,
                    }
                }
            ]
        });

        this.quadUniformBindGroup = this.device.createBindGroup({
            layout: this.gbufferDebugViewPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 1,
                    resource: this.gbufferTextures[0].createView()
                },
                {
                    binding: 2,
                    resource: this.gbufferTextures[1].createView()
                },
                {
                    binding: 3,
                    resource: this.gbufferTextures[2].createView()
                },
            ]
        });
    }

    setupDeferredLightLoopPipeline() {
        this.deferredLightLoopPipeline = this.device.createRenderPipeline({
            vertex: {
                module: this.device.createShaderModule({
                    code: vertexShaderFullScreenQuad,
                }),
                entryPoint: "main",
            },
            fragment: {
                module: this.device.createShaderModule({
                    code: replaceArray(
                        fragmentShaderDeferredShadingLoopLights,
                        ["$1", "$2", "$3"],
                        [this.lightCulling.numLights, this.canvas.width.toFixed(1), this.canvas.height.toFixed(1)]
                    ),
                }),
                entryPoint: "main",
                targets: [
                    { format: 'bgra8unorm' },
                ],
            },
        });
    }

    setupDeferredTiledLightDebugPipeline() {
        this.deferredTiledLightDebugPipeline = this.device.createRenderPipeline({
            vertex: {
                module: this.device.createShaderModule({
                    code: vertexShaderFullScreenQuad,
                }),
                entryPoint: "main",
                buffers: [{
                    arrayStride: quadVertexSize, //padding
                    stepMode: "vertex",
                    attributes: [{
                        // position
                        shaderLocation: 0,
                        offset: 0,
                        format: "float32x4"
                    },
                    {
                        // uv
                        shaderLocation: 1,
                        offset: quadUVOffset,
                        format: "float32x2"
                    }]
                }]
            },
            fragment: {
                module: this.device.createShaderModule({
                    code: replaceArray(fragmentShaderDeferredShadingTiledLightDebug,
                        ['$NUM_TILE_LIGHT_SLOT', '$2', '$NUM_TILES', '$TILE_COUNT_Y', '$TILE_COUNT_X', '$TILE_SIZE'],
                        [this.lightCulling.tileLightSlot, this.lightCulling.tileLightSlot, this.lightCulling.numTiles, this.lightCulling.tileCount[1], this.lightCulling.tileCount[0], this.lightCulling.tileSize])
                }),
                entryPoint: "main",
                targets: [
                    { format: 'bgra8unorm' },
                ],
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'back',
            },
        });

        // For fragment usage
        this.tileLightIdBufferBindGroup = this.device.createBindGroup({
            layout: this.deferredTiledLightDebugPipeline.getBindGroupLayout(0),
            entries: [
              {
                binding: 0,
                resource: {
                  buffer: this.lightCulling.tileLightIdGPUBuffer,
                },
              },
            ],
          });
    }

    setupDeferredLightCullingPipeline() {
        this.deferredLightCullingPipeline = this.device.createRenderPipeline({
            vertex: {
                module: this.device.createShaderModule({
                    code: vertexShaderFullScreenQuad,
                }),
                entryPoint: "main",
                buffers: [{
                    arrayStride: quadVertexSize, //padding
                    stepMode: "vertex",
                    attributes: [{
                        // position
                        shaderLocation: 0,
                        offset: 0,
                        format: "float32x4"
                    },
                    {
                        // uv
                        shaderLocation: 1,
                        offset: quadUVOffset,
                        format: "float32x2"
                    }]
                }]
            },
            fragment: {
                module: this.device.createShaderModule({
                    code: replaceArray(fragmentShaderDeferredShadingTiled,
                        ['$NUM_TILE_LIGHT_SLOT', '$2', '$NUM_TILES', '$TILE_COUNT_Y', '$TILE_COUNT_X', '$TILE_SIZE'],
                        [this.lightCulling.tileLightSlot, this.lightCulling.tileLightSlot, this.lightCulling.numTiles, this.lightCulling.tileCount[1], this.lightCulling.tileCount[0], this.lightCulling.tileSize])
                }),
                entryPoint: "main",
                targets: [
                    { format: 'bgra8unorm' },
                ],
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'back',
            },
        });

        this.deferredFinalBindGroup = this.device.createBindGroup({
            layout: this.deferredLightCullingPipeline.getBindGroupLayout(1),
            entries: [
              {
                binding: 0,
                resource: {
                  buffer: this.cameraPositionUniformBuffer,
                },
              },
              {
                binding: 1,
                resource: {
                  buffer: this.lightCulling.lightDataGPUBuffer,
                },
              },
              {
                binding: 2,
                resource: {
                  buffer: this.lightCulling.tileLightIdGPUBuffer,
                },
              },
            ],
        });
    }

    loadModel(modelUrl, albedoUrl, normalUrl) {
        const device = this.device;
        const pModel = new Promise((resolve) => {
            OBJ.downloadMeshes({
                'obj': modelUrl
            }, resolve);
        });

        const pAlbedoMap = createTextureFromImage(device, albedoUrl, GPUTextureUsage.TEXTURE_BINDING);
        const pNormalMap = normalUrl ? createTextureFromImage(device, normalUrl, GPUTextureUsage.TEXTURE_BINDING) : null;

        // dModel, dAlbedoMap, dNormalMap
        return Promise.all([pModel, pAlbedoMap, pNormalMap]).then((values) => {
            const meshes = values[0];

            // build mesh, drawable list here
            const geometry = new Geometry(device);
            geometry.fromObjMesh(meshes['obj']);

            const albedoMap = values[1];

            const normalMap = values[2];

            const sampler = device.createSampler({
                addressModeU: "repeat",
                addressModeV: "repeat",
                magFilter: "linear",
                minFilter: "linear",
                mipmapFilter: "linear"
            });
            const material = new WriteGBufferMaterial(sampler, albedoMap, normalMap);

            const d = new Drawable(device, geometry, material, this.uniformBuffer);

            this.drawableLists.push(d);
        });
    }


    async setupScene(device) {
        await Promise.all([
            this.loadModel('models/sponza.obj', 'models/color.jpg', 'models/normal.png'),
            this.loadModel('models/di.obj', 'models/di.png', 'models/di-n.png'),
        ]);
    }

    frame() {
        this.device.queue.writeBuffer(
            this.uniformBuffer,
            64,
            this.camera.projectionMatrix.buffer,
            this.camera.projectionMatrix.byteOffset,
            this.camera.projectionMatrix.byteLength
          );

        const commandEncoder = this.device.createCommandEncoder();

        // draw geometry, write gbuffers

        const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);
        passEncoder.setPipeline(this.pipeline);

        for (let i = 0; i < this.drawableLists.length; i++) {
            const o = this.drawableLists[i];

            mat4.multiply(tmpMat4, this.camera.viewMatrix, o.transform.getModelMatrix());

            // TODO: There's problem here. should upload all at once
            this.device.queue.writeBuffer(
                this.uniformBuffer,
                0,
                tmpMat4.buffer,
                tmpMat4.byteOffset,
                tmpMat4.byteLength
              );

            o.draw(passEncoder);
        }

        passEncoder.endPass();

        // render full screen quad

        this.curRenderModeFunc(commandEncoder);

        this.device.queue.submit([commandEncoder.finish()]);
    }

    renderGBufferDebugView(commandEncoder) {
        const swapChainTexture = this.context.getCurrentTexture();

        this.renderFullScreenPassDescriptor.colorAttachments[0].view = swapChainTexture.createView();
        const quadPassEncoder = commandEncoder.beginRenderPass(this.renderFullScreenPassDescriptor);
        quadPassEncoder.setPipeline(this.gbufferDebugViewPipeline);
        quadPassEncoder.setVertexBuffer(0, this.quadVerticesBuffer);
        quadPassEncoder.setBindGroup(0, this.quadUniformBindGroup);
        quadPassEncoder.draw(6, 1, 0, 0);
        quadPassEncoder.endPass();
    }

    renderDeferredTiledLightDebug(commandEncoder) {
        this.lightCulling.update();

        const swapChainTexture = this.context.getCurrentTexture();

        this.renderFullScreenPassDescriptor.colorAttachments[0].view = swapChainTexture.createView();
        
        const quadPassEncoder = commandEncoder.beginRenderPass(this.renderFullScreenPassDescriptor);
        quadPassEncoder.setPipeline(this.deferredTiledLightDebugPipeline);
        quadPassEncoder.setVertexBuffer(0, this.quadVerticesBuffer);
        quadPassEncoder.setBindGroup(0, this.tileLightIdBufferBindGroup);

        quadPassEncoder.draw(6, 1, 0, 0);

        quadPassEncoder.endPass();
    }

    renderDeferredLightCulling(commandEncoder) {
        this.lightCulling.update();

        const swapChainTexture = this.context.getCurrentTexture();

        const eye = this.camera.getPosition();
        this.device.queue.writeBuffer(
            this.cameraPositionUniformBuffer,
            0,
            eye.buffer,
            eye.byteOffset,
            eye.byteLength,
        );

        this.renderFullScreenPassDescriptor.colorAttachments[0].view = swapChainTexture.createView();
        
        const quadPassEncoder = commandEncoder.beginRenderPass(this.renderFullScreenPassDescriptor);
        quadPassEncoder.setPipeline(this.deferredLightCullingPipeline);
        quadPassEncoder.setVertexBuffer(0, this.quadVerticesBuffer);
        quadPassEncoder.setBindGroup(0, this.quadUniformBindGroup);
        quadPassEncoder.setBindGroup(1, this.deferredFinalBindGroup);

        quadPassEncoder.draw(6, 1, 0, 0);

        quadPassEncoder.endPass();
    }
}