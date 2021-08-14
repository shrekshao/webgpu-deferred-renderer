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

// const vertexShaderFullScreenQuadGLSL = `#version 450
// layout(location = 0) in vec4 position;
// layout(location = 1) in vec2 uv;

// layout(location = 0) out vec2 fragUV;

// void main() {
//     gl_Position = position;
//     fragUV = uv;
// }
// `;

// const fragmentShaderGBufferDebugViewGLSL = `#version 450

// #define NUM_GBUFFERS 3

// #define A 3.0
// #define B 0.5

// layout(set = 0, binding = 0) uniform sampler quadSampler;

// // Layered texture array sampling is not supported for now
// // layout(set = 0, binding = 1) uniform texture2D gbufferTexture[NUM_GBUFFERS];
// layout(set = 0, binding = 1) uniform texture2D gbufferTexture0;
// layout(set = 0, binding = 2) uniform texture2D gbufferTexture1;
// layout(set = 0, binding = 3) uniform texture2D gbufferTexture2;

// layout(set = 1, binding = 0) uniform Uniforms {
//     float debugViewOffset;
//     vec3 padding;
// } uniforms;

// layout(location = 0) in vec2 fragUV;
// layout(location = 0) out vec4 outColor;

// void main() {
//     // fragUV.y > A * (fragUV.x - i / NUM_GBUFFERS) + B

//     float o = uniforms.debugViewOffset;

//     float r = mod( o + float(NUM_GBUFFERS) / A * (A * fragUV.x - fragUV.y + B), NUM_GBUFFERS);
//     if (r < 1)
//     {
//         outColor = texture(sampler2D(gbufferTexture0, quadSampler), fragUV);
//     }
//     else if (r < 2)
//     {
//         outColor = texture(sampler2D(gbufferTexture1, quadSampler), fragUV);
//     }
//     else if (r < 3)
//     {
//         outColor = texture(sampler2D(gbufferTexture2, quadSampler), fragUV);
//     }
// }
// `;

const fragmentShaderDeferredShadingLoopLights = `
[[group(0), binding(0)]] var mySampler: sampler;
[[group(0), binding(1)]] var gBufferPosition: texture_2d<f32>;
[[group(0), binding(2)]] var gBufferNormal: texture_2d<f32>;
[[group(0), binding(3)]] var gBufferAlbedo: texture_2d<f32>;

[[block]] struct CameraUniforms {
    position: vec4<f32>;
};
[[group(1), binding(0)]] var<uniform> camera : CameraUniforms;

struct LightData {
  position : vec4<f32>;
  color : vec3<f32>;
  radius : f32;
};
[[block]] struct LightsBuffer {
  lights: array<LightData>;
};
[[group(2), binding(0)]] var<storage, read> lightsBuffer: LightsBuffer;

// [[block]] struct Config {
//   numLights : u32;
// };
// [[group(1), binding(1)]] var<uniform> config: Config;

// [[block]] struct CanvasConstants {
//   size: vec2<f32>;
// };
// [[group(2), binding(0)]] var<uniform> canvas : CanvasConstants;

[[stage(fragment)]]
fn main([[builtin(position)]] coord : vec4<f32>)
     -> [[location(0)]] vec4<f32> {
  var result : vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
  var c : vec2<f32> = coord.xy / vec2<f32>($2, $3);

  var position : vec3<f32> = textureSample(
    gBufferPosition,
    mySampler,
    c
  ).xyz;

  if (position.z > 10000.0) {
    discard;
  }

  var normal : vec3<f32> = textureSample(
    gBufferNormal,
    mySampler,
    c
  ).xyz;

  var albedo : vec3<f32> = textureSample(
    gBufferAlbedo,
    mySampler,
    c
  ).rgb;

  var V: vec3<f32> = normalize(camera.position.xyz - position);

  for (var i : u32 = 0u; i < $1u; i = i + 1u) {
    var L : vec3<f32> = lightsBuffer.lights[i].position.xyz - position;
    var distance : f32 = length(L);
    if (distance > lightsBuffer.lights[i].radius) {
        continue;
    }

    L = L / distance;
    var lambert : f32 = max(dot(normal, L), 0.0);
    var H: vec3<f32> = normalize(L + V);
    var specular : f32 = sign(lambert) * pow(max(dot(H, normal), 0.0), 10.0);

    result = result + clamp(lightsBuffer.lights[i].color * (
      lambert * pow(1.0 - distance / lightsBuffer.lights[i].radius, 2.0) * albedo
      + specular * vec3<f32>(1.0, 1.0, 1.0)
    ), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
  }

//   // some manual ambient
//   result = result + vec3<f32>(0.2, 0.2, 0.2);

  return vec4<f32>(result, 1.0);
}
`;

// const fragmentShaderDeferredShadingLoopLightsGLSL = `#version 450
// #define NUM_GBUFFERS 3

// #define NUM_LIGHTS $1

// layout(set = 0, binding = 0) uniform sampler quadSampler;
// layout(set = 0, binding = 1) uniform texture2D gbufferTexture0;
// layout(set = 0, binding = 2) uniform texture2D gbufferTexture1;
// layout(set = 0, binding = 3) uniform texture2D gbufferTexture2;

// layout(set = 1, binding = 0) uniform CameraUniforms {
//     vec4 position;
// } camera;

// struct LightData
// {
//     vec4 position;
//     vec3 color;
//     float radius;
// };

// layout(std430, set = 2, binding = 0) buffer LightBuffer {
//     LightData data[];
// } lights;

// layout(location = 0) in vec2 fragUV;
// layout(location = 0) out vec4 outColor;

// void main() {

//     vec3 position = texture(sampler2D(gbufferTexture0, quadSampler), fragUV).xyz;
//     vec3 normal = texture(sampler2D(gbufferTexture1, quadSampler), fragUV).xyz;
//     vec3 albedo = texture(sampler2D(gbufferTexture2, quadSampler), fragUV).rgb;

//     vec3 V = normalize(camera.position.xyz - position);

//     vec3 finalColor = vec3(0);

//     for ( int i = 0; i < NUM_LIGHTS; i++) {
//         LightData light = lights.data[i];

//         float distance = distance(light.position.xyz, position);
//         if (distance  > light.radius) {
//             continue;
//         }

//         vec3 L = normalize(light.position.xyz - position);
//         float lambert = max(dot(L, normal), 0.0);
//         vec3 H = normalize(L + V);
//         float specular = float(lambert > 0.0) * pow(max(dot(H, normal), 0.0), 10.0);

//         finalColor += 
//             clamp(light.color * pow(1.0 - distance / light.radius, 2.0) *
//             (
//                 albedo * lambert + 
//                     vec3(1,1,1) * specular  // Assume white specular, modify if you add more specular info
//             ), vec3(0), vec3(1));

//     }

//     outColor = vec4(finalColor, 1);
// }
// `;

const fragmentShaderDeferredShadingTiledLightDebugGLSL = `#version 450

#define NUM_LIGHTS $1
#define NUM_TILES $2
#define TILE_COUNT_X $3
#define TILE_COUNT_Y $4
#define NUM_TILE_LIGHT_SLOT $5

struct TileLightIdData
{
    int count;
    int lightId[NUM_TILE_LIGHT_SLOT];
};

layout(std430, set = 0, binding = 0) buffer TileLightIdBuffer {
    TileLightIdData data[NUM_TILES];
} tileLightId;


layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 tileScale = vec2(1.0 / TILE_COUNT_X, 1.0 / TILE_COUNT_Y);
    ivec2 tileCoord = ivec2(floor( fragUV / tileScale ));
    int tileId = tileCoord.x + tileCoord.y * TILE_COUNT_X;

    float t = float(tileLightId.data[tileId].count) / ( float(NUM_TILE_LIGHT_SLOT) );
    vec3 color = vec3(4.0 * t - 2.0, t < 0.5 ? 4.0 * t: 4.0 - 4.0 * t , 2.0 - 4.0 * t);
    outColor = vec4(color, 1);
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
    var tileCoord = vec2<u32>(floor( fragUV / tileScale ));
    // var tileCoord = (vec2<u32>(floor(coord.xy)) / vec2<u32>(TILE_COUNT_X, TILE_COUNT_Y));
    var tileId: u32 = tileCoord.x + tileCoord.y * TILE_COUNT_X;

    var c: u32 = atomicLoad(&tileLightId.data[tileId].count);
    var t: f32 = f32(c) / ( f32($2) );
    t = t * 4.0;
    result.r = 4.0 * t - 2.0;
    if (t < 0.5) {
        result.g = 4.0 * t;
    } else {
        result.g = 4.0 - 4.0 * t;
    }
    result.b = 2.0 - 4.0 * t;
    result.a = 1.0;


    // result = vec4<f32>(f32(tileCoord.x) / 16.0, f32(tileCoord.y) / 16.0, 0.0, 1.0);
    // result = vec4<f32>(0.0, f32(tileId) / 256.0, 0.0, 1.0);


    return result;
}
`;

const fragmentShaderDeferredShadingTiledGLSL = `#version 450
#define NUM_GBUFFERS 3

#define NUM_LIGHTS $1
#define NUM_TILES $2
#define TILE_COUNT_X $3
#define TILE_COUNT_Y $4
#define NUM_TILE_LIGHT_SLOT $5

layout(set = 0, binding = 0) uniform sampler quadSampler;
layout(set = 0, binding = 1) uniform texture2D gbufferTexture0;
layout(set = 0, binding = 2) uniform texture2D gbufferTexture1;
layout(set = 0, binding = 3) uniform texture2D gbufferTexture2;

layout(set = 1, binding = 0) uniform CameraUniforms {
    vec4 position;
} camera;

struct LightData
{
    vec4 position;
    vec3 color;
    float radius;
};

layout(std430, set = 2, binding = 0) buffer LightBuffer {
    LightData data[];
} lights;

struct TileLightIdData
{
    int count;
    int lightId[NUM_TILE_LIGHT_SLOT];
};

layout(std430, set = 3, binding = 0) buffer TileLightIdBuffer {
    TileLightIdData data[NUM_TILES];
} tileLightId;

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

void main() {

    vec3 position = texture(sampler2D(gbufferTexture0, quadSampler), fragUV).xyz;
    vec3 normal = texture(sampler2D(gbufferTexture1, quadSampler), fragUV).xyz;
    vec3 albedo = texture(sampler2D(gbufferTexture2, quadSampler), fragUV).rgb;
    
    vec3 V = normalize(camera.position.xyz - position);

    vec2 tileScale = vec2(1.0 / TILE_COUNT_X, 1.0 / TILE_COUNT_Y);
    ivec2 tileCoord = ivec2(floor( fragUV / tileScale ));
    int tileId = tileCoord.x + tileCoord.y * TILE_COUNT_X;

    vec3 finalColor = vec3(0);

    for (int i = 0, len = tileLightId.data[tileId].count; i < len; i++) {
        LightData light = lights.data[ tileLightId.data[tileId].lightId[i] ];

        float distance = distance(light.position.xyz, position);
        if (distance  > light.radius) {
            continue;
        }

        vec3 L = normalize(light.position.xyz - position);
        float lambert = max(dot(L, normal), 0.0);
        vec3 H = normalize(L + V);
        float specular = float(lambert > 0.0) * pow(max(dot(H, normal), 0.0), 10.0);

        finalColor += 
            clamp(light.color * pow(1.0 - distance / light.radius, 2.0) *
            (
                albedo * lambert + 
                    vec3(1,1,1) * specular  // Assume white specular, modify if you add more specular info
            ), vec3(0), vec3(1));

    }

    outColor = vec4(finalColor, 1);
}
`;

const fragmentShaderDeferredShadingTiled = `
// [[group(0), binding(0)]] var mySampler: sampler;
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

// [[block]] struct Config {
//     numLights : u32;

//     numTiles : u32;
//     tileCountX : u32;
//     tileCountY : u32;
//     numTileLightSlot : u32;
//     tileSize : u32;
// };
// [[group(1), binding(3)]] var<uniform> config: Config;

// [[block]] struct CanvasConstants {
//   size: vec2<f32>;
// };
// [[group(4), binding(0)]] var<uniform> canvas : CanvasConstants;

[[stage(fragment)]]
fn main([[builtin(position)]] coord : vec4<f32>,
        [[location(0)]] fragUV : vec2<f32>)
     -> [[location(0)]] vec4<f32> {
  var result = vec3<f32>(0.0, 0.0, 0.0);
//   var c = coord.xy / canvas.size;
//   var c = fragUV;

  var position = textureLoad(
    gBufferPosition,
    vec2<i32>(floor(coord.xy)),
    0
  ).xyz;

  if (position.z > 10000.0) {
    discard;
  }

  var normal = textureLoad(
    gBufferNormal,
    vec2<i32>(floor(coord.xy)),
    0
  ).xyz;

  var albedo = textureLoad(
    gBufferAlbedo,
    vec2<i32>(floor(coord.xy)),
    0
  ).rgb;


  var V = normalize(camera.position.xyz - position);

  let TILE_COUNT_X: u32 = $TILE_COUNT_Xu;
  let TILE_COUNT_Y: u32 = $TILE_COUNT_Yu;

  var tileScale = vec2<f32>(1.0 / f32(TILE_COUNT_X), 1.0 / f32(TILE_COUNT_Y));
  var tileCoord = vec2<u32>(floor( fragUV / tileScale ));
  var tileId: u32 = tileCoord.x + tileCoord.y * TILE_COUNT_X;

  var c: u32 = atomicLoad(&tileLightId.data[tileId].count);
  for (var i : u32 = 0u; i < c; i = i + 1u) {
    var light: LightData = lightsBuffer.lights[ tileLightId.data[tileId].lightId[i] ];

    var distance = distance(light.position.xyz, position);
    if (distance  > light.radius) {
        continue;
    }

    var L = normalize(light.position.xyz - position);
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


//   for (var i : u32 = 0u; i < config.numLights; i = i + 1u) {
// //   for (var i : u32 = 0u; i < 1024u; i = i + 1u) {
//     var L = lightsBuffer.lights[i].position.xyz - position;
//     var distance = length(L);
//     if (distance > lightsBuffer.lights[i].radius) {
//         continue;
//     }
//     var lambert = max(dot(normal, normalize(L)), 0.0);
//     result = result + vec3<f32>(
//       lambert * pow(1.0 - distance / lightsBuffer.lights[i].radius, 2.0) * lightsBuffer.lights[i].color * albedo);
//   }

// //   // some manual ambient
// //   result = result + vec3<f32>(0.2, 0.2, 0.2);

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
    //   format: 'bgra8unorm',
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
        this.debugViewOffset = 0.5;
        this.renderFuncs = {
            'debugView': this.renderGBufferDebugView,
            // 'deferredLightLoop': this.renderDeferredLightLoop,
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
                // buffers: this.drawableLists[0].geometry.vertexInput.vertexBuffers,
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
            // arrayLayerCount: 1,
            mipLevelCount: 1,
            // sampleCount: 1,
            dimension: "2d",
            format: "depth24plus-stencil8",
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });


        // 10-11-2019 Unfortunately 
        // Currently Dawn does not support layered rendering.
        // https://cs.chromium.org/chromium/src/third_party/dawn/src/dawn_native/CommandEncoder.cpp?l=264
        // Might be a bit painful when you change number of gbuffer as you need to modify that in shader
        // Though you could have some helper function to inject strings that declares textures to shader

        this.gbufferTextures = [
            device.createTexture({
                size: {
                    width: canvas.width,
                    height: canvas.height,
                    depthOrArrayLayers: 1
                },
                // arrayLayerCount: 1,
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
                // arrayLayerCount: 1,
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
                // arrayLayerCount: 1,
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
                    loadValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
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
        // quadVerticesBuffer.setSubData(0, fullScreenQuadArray);
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
            // depthStencilAttachment: null
            // depthStencilAttachment: {
            //     view: undefined,

            //     depthLoadValue: 1.0,
            //     depthStoreOp: "store",
            //     stencilLoadValue: 0,
            //     stencilStoreOp: "store",
            // }
        };

        const sampler = this.sampler = device.createSampler({
            magFilter: "nearest",
            minFilter: "nearest"
        });

        const cameraPositionUniformBuffer = this.cameraPositionUniformBuffer = this.device.createBuffer({
            // size: 4 * 16,
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // this.cameraPositionUniformBindGroup = this.device.createBindGroup({
        //     layout: this.uniformBufferBindGroupLayout,
        //     // layout: this.pipeline.getBindGroupLayout(),
        //     entries: [
        //         {
        //             binding: 0,
        //             resource: {
        //                 buffer: cameraPositionUniformBuffer,
        //                 offset: 0,
        //                 size: 16
        //             }
        //         }
        //     ]
        // });

        this.setupGBufferDebugViewPipeline();
        // this.setupDeferredLightLoopPipeline();
        this.setupDeferredTiledLightDebugPipeline();
        this.setupDeferredLightCullingPipeline();
    }

    setupGBufferDebugViewPipeline() {
        // const quadPipeLineLayout = this.device.createPipelineLayout({
        //         bindGroupLayouts: [
        //             this.quadUniformsBindGroupLayout,
        //             this.uniformBufferBindGroupLayout
        //         ]
        //     });
        this.gbufferDebugViewPipeline = this.device.createRenderPipeline({
            // layout: quadPipeLineLayout,
            // layout: quadPipeLineLayout,

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
                // cullMode: 'none',
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
                        // size: debugViewUniformBufferSize
                    }
                }
            ]
        });

        this.quadUniformBindGroup = this.device.createBindGroup({
            // layout: this.quadUniformsBindGroupLayout,
            layout: this.gbufferDebugViewPipeline.getBindGroupLayout(0),
            entries: [
                // {
                //     binding: 0,
                //     resource: this.sampler,
                // },
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

        // const deferredLightLoopPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [
        //     this.quadUniformsBindGroupLayout,
        //     this.uniformBufferBindGroupLayout,
        //     this.lightCulling.storageBufferBindGroupLayout
        // ] });
        this.deferredLightLoopPipeline = this.device.createRenderPipeline({
            // layout: deferredLightLoopPipelineLayout,
            vertex: {
                module: this.device.createShaderModule({
                    code: vertexShaderFullScreenQuad,
                }),
                entryPoint: "main",
                // buffers: [{
                //     arrayStride: quadVertexSize, //padding
                //     stepMode: "vertex",
                //     attributes: [{
                //         // position
                //         shaderLocation: 0,
                //         offset: 0,
                //         format: "float32x4"
                //     },
                //     {
                //         // uv
                //         shaderLocation: 1,
                //         offset: quadUVOffset,
                //         format: "float32x2"
                //     }]
                // }]
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
        // const deferredTiledLightDebugPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [
        //     this.lightCulling.storageBufferBindGroupLayout
        // ] });
        this.deferredTiledLightDebugPipeline = this.device.createRenderPipeline({
            // layout: deferredTiledLightDebugPipelineLayout,

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
                    // code: this.glslang.compileGLSL(
                    //     replaceArray(fragmentShaderDeferredShadingTiledLightDebugGLSL,
                    //         ["$1", "$2", "$3", "$4", "$5"],
                    //         [this.lightCulling.numLights, this.lightCulling.numTiles, this.lightCulling.tileCount[0], this.lightCulling.tileCount[1], this.lightCulling.tileLightSlot]), "fragment"),
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
                // cullMode: 'none',
                cullMode: 'back',
            },

            // rasterizationState: {
            //     frontFace: 'ccw',
            //     cullMode: 'back'
            // },

            // colorStates: [{
            //     format: "bgra8unorm",
            //     alphaBlend: {},
            //     colorBlend: {}
            // }]
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
        // const deferredLightCullingPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [
        //     // this.quadUniformsBindGroupLayout,
        //     this.gbufferDebugViewPipeline.getBindGroupLayout(0),
        //     this.uniformBufferBindGroupLayout,
        //     this.lightCulling.storageBufferBindGroupLayout,
        //     this.lightCulling.storageBufferBindGroupLayout,
        // ] });
        this.deferredLightCullingPipeline = this.device.createRenderPipeline({
            // layout: deferredLightCullingPipelineLayout,

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
                    // code: replaceArray(fragmentShaderDeferredShadingTiled,
                    //         ["$1", "$2", "$3", "$4", "$5"],
                    //         [this.lightCulling.numLights, this.lightCulling.numTiles, this.lightCulling.tileCount[0], this.lightCulling.tileCount[1], this.lightCulling.tileLightSlot]),
                    // code: fragmentShaderDeferredShadingTiled,
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

            // rasterizationState: {
            //     frontFace: 'ccw',
            //     cullMode: 'back'
            // },

            // colorStates: [{
            //     format: "bgra8unorm",
            //     alphaBlend: {},
            //     colorBlend: {}
            // }]
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
            //   {
            //     binding: 3,
            //     resource: {
            //       buffer: this.lightCulling.configUniformBuffer,
            //     },
            //   },
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

        // this.uniformBuffer.setSubData(64, this.camera.projectionMatrix);
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
            // this.uniformBuffer.setSubData(0, tmpMat4);

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

        // this.debugViewUniformBuffer.setSubData(0, new Float32Array([this.debugViewOffset]));
        const a = new Float32Array([this.debugViewOffset]);
        this.device.queue.writeBuffer(
            this.debugViewUniformBuffer,
            0,
            a.buffer,
            a.byteOffset,
            a.byteLength
        );

        this.renderFullScreenPassDescriptor.colorAttachments[0].view = swapChainTexture.createView();
        const quadPassEncoder = commandEncoder.beginRenderPass(this.renderFullScreenPassDescriptor);
        quadPassEncoder.setPipeline(this.gbufferDebugViewPipeline);
        quadPassEncoder.setVertexBuffer(0, this.quadVerticesBuffer);
        quadPassEncoder.setBindGroup(0, this.quadUniformBindGroup);
        // quadPassEncoder.setBindGroup(1, this.debugViewUniformBindGroup);
        quadPassEncoder.draw(6, 1, 0, 0);
        quadPassEncoder.endPass();
    }

    // renderDeferredLightLoop(commandEncoder) {
    //     // this.lightCulling.update();     // TODO: only update light position (might have a separate compute shader that only do that)

    //     const swapChainTexture = this.context.getCurrentTexture();

    //     // this.cameraPositionUniformBuffer.setSubData(0, this.camera.getPosition());
    //     let v = this.camera.getPosition();
    //     this.device.queue.writeBuffer(
    //         this.cameraPositionUniformBuffer,
    //         0,
    //         v.buffer,
    //         v.byteOffset,
    //         v.byteLength
    //     );

    //     this.renderFullScreenPassDescriptor.colorAttachments[0].view = swapChainTexture.createView();
        
    //     const quadPassEncoder = commandEncoder.beginRenderPass(this.renderFullScreenPassDescriptor);
    //     quadPassEncoder.setPipeline(this.deferredLightLoopPipeline);
    //     quadPassEncoder.setVertexBuffer(0, this.quadVerticesBuffer);
    //     quadPassEncoder.setBindGroup(0, this.quadUniformBindGroup);
    //     quadPassEncoder.setBindGroup(1, this.cameraPositionUniformBindGroup);
    //     quadPassEncoder.setBindGroup(2, this.lightCulling.lightBufferBindGroup);

    //     quadPassEncoder.draw(6, 1, 0, 0);

    //     quadPassEncoder.endPass();
    // }

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
        // this.device.queue.submit([commandEncoder.finish()]);
    }

    renderDeferredLightCulling(commandEncoder) {
        this.lightCulling.update();

        const swapChainTexture = this.context.getCurrentTexture();

        // this.cameraPositionUniformBuffer.setSubData(0, this.camera.getPosition());

        this.renderFullScreenPassDescriptor.colorAttachments[0].view = swapChainTexture.createView();
        
        const quadPassEncoder = commandEncoder.beginRenderPass(this.renderFullScreenPassDescriptor);
        quadPassEncoder.setPipeline(this.deferredLightCullingPipeline);
        quadPassEncoder.setVertexBuffer(0, this.quadVerticesBuffer);
        quadPassEncoder.setBindGroup(0, this.quadUniformBindGroup);
        quadPassEncoder.setBindGroup(1, this.deferredFinalBindGroup);
        // quadPassEncoder.setBindGroup(1, this.cameraPositionUniformBindGroup);
        // quadPassEncoder.setBindGroup(2, this.lightCulling.lightBufferBindGroup);
        // quadPassEncoder.setBindGroup(3, this.lightCulling.tileLightIdBufferBindGroup);
        // quadPassEncoder.setBindGroup(4, this.lightCulling.configBindGroup);

        quadPassEncoder.draw(6, 1, 0, 0);

        quadPassEncoder.endPass();
        // this.device.queue.submit([commandEncoder.finish()]);
    }
}