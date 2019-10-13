import Geometry from './Geometry.js';
import Camera from './Camera.js';


const vertexShaderBlinnPhongGLSL = `#version 450
layout(set = 0, binding = 0) uniform Uniforms {
    mat4 modelViewMatrix;
    mat4 projectionMatrix;
    mat4 modelViewNormalMatrix;
} uniforms;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

layout(location = 0) out vec4 fragPosition;
layout(location = 1) out vec4 fragNormal;
layout(location = 2) out vec2 fragUV;
// Metallic, Roughness, Emissive, Motion, etc.

void main() {
    fragPosition = uniforms.modelViewMatrix * vec4(position, 1);
    fragNormal = normalize(uniforms.modelViewNormalMatrix * vec4(normal, 0));
    fragUV = uv;
    gl_Position = uniforms.projectionMatrix * fragPosition;
    // gl_Position = vec4(0.1 * position.xy, 0.5, 1);
    // gl_Position = vec4( clamp(position.xy, vec2(-0.5, -0.8), vec2(1, 1)), 0.5, 1);
}
`;
const vertexShaderGBufferGLSL = vertexShaderBlinnPhongGLSL;

const fragmentShaderGBufferGLSL = `#version 450
layout(set = 0, binding = 1) uniform sampler defaultSampler;
layout(set = 0, binding = 2) uniform texture2D albedoMap;
layout(set = 0, binding = 3) uniform texture2D normalMap;

layout(location = 0) in vec4 fragPosition;
layout(location = 1) in vec4 fragNormal;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outGBufferPosition;
layout(location = 1) out vec4 outGBufferNormal;
layout(location = 2) out vec4 outGBufferAlbedo;

vec3 applyNormalMap(vec3 geomnor, vec3 normap) {
    normap = normap * 2.0 - 1.0;
    vec3 up = normalize(vec3(0.001, 1, 0.001));
    vec3 surftan = normalize(cross(geomnor, up));
    vec3 surfbinor = cross(geomnor, surftan);
    return normap.y * surftan + normap.x * surfbinor + normap.z * geomnor;
}

void main() {
    outGBufferPosition = fragPosition;
    // outGBufferNormal = fragNormal;  // TODO: normal map
    outGBufferNormal = vec4(applyNormalMap(fragNormal.xyz, texture(sampler2D(normalMap, defaultSampler), fragUV).rgb), 1);
    outGBufferAlbedo = texture(sampler2D(albedoMap, defaultSampler), fragUV);
}
`;

const fragmentShaderBlinnPhongGLSL = `#version 450
layout(set = 0, binding = 1) uniform sampler defaultSampler;
layout(set = 0, binding = 2) uniform texture2D albedoMap;

layout(location = 0) in vec4 fragPosition;
layout(location = 1) in vec4 fragNormal;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outFragColor;

void main() {

    float intensity = dot( fragNormal.xyz, normalize( vec3(1,1,1) ) );
    outFragColor = vec4(intensity * texture(sampler2D(albedoMap, defaultSampler), fragUV).rgb, 1.0);

    // vec2 uv = vec2(fragUV.x, 1.0 - fragUV.y);
    // outFragColor = vec4(intensity * texture(sampler2D(albedoMap, defaultSampler), uv).rgb, 1.0);

    // outFragColor = vec4(fragUV, 0, 1);
    // outFragColor = vec4(clamp(fragNormal.xyz, vec3(0), vec3(1)), 1);
    // outFragColor = vec4(1, 0, 0, 1);
}
`;

const vertexShaderFullScreenQuadGLSL = `#version 450
layout(location = 0) in vec4 position;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec2 fragUV;

void main() {
    gl_Position = position;
    fragUV = uv;
}
`;

const fragmentShaderFullScreenQuadGLSL = `#version 450
layout(set = 0, binding = 0) uniform sampler quadSampler;
layout(set = 0, binding = 1) uniform texture2D quadTexture;

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(sampler2D(quadTexture, quadSampler), fragUV);
    // outColor = vec4(texture(sampler2D(quadTexture, quadSampler), fragUV).r, 0.0, 0.0, 1.0);
}
`;

const fragmentShaderGBufferDebugViewGLSL = `#version 450

#define NUM_GBUFFERS 3
#define POSITION_ID 0
#define NORMAL_ID 1
#define ALBEDOMAP_ID 2


#define A 3.0
#define B 0.5

layout(set = 0, binding = 0) uniform sampler quadSampler;
// layout(set = 0, binding = 1) uniform texture2D gbufferTexture[NUM_GBUFFERS];
layout(set = 0, binding = 1) uniform texture2D gbufferTexture0;
layout(set = 0, binding = 2) uniform texture2D gbufferTexture1;
layout(set = 0, binding = 3) uniform texture2D gbufferTexture2;

layout(set = 1, binding = 0) uniform Uniforms {
    float debugViewOffset;
    vec3 padding;
} uniforms;

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

void main() {

    // fragUV.y > A * (fragUV.x - i / NUM_GBUFFERS) + B

    float o = uniforms.debugViewOffset;
    // float o = 0.0;

    float r = mod( o + float(NUM_GBUFFERS) / A * (A * fragUV.x - fragUV.y + B), NUM_GBUFFERS);
    if (r < 1)
    {
        outColor = texture(sampler2D(gbufferTexture0, quadSampler), fragUV);
    }
    else if (r < 2)
    {
        outColor = texture(sampler2D(gbufferTexture1, quadSampler), fragUV);
    }
    else if (r < 3)
    {
        outColor = texture(sampler2D(gbufferTexture2, quadSampler), fragUV);
    }

    // int i = 0;
    // if (fragUV.y > A * (fragUV.x - 1.0 / NUM_GBUFFERS) + B ) {
    //     i = 0;
    // } else if (fragUV.y > A * (fragUV.x - 2.0 / NUM_GBUFFERS) + B) {
    //     i = 1;
    // } else if (fragUV.y > A * (fragUV.x - 2.0 / NUM_GBUFFERS) + B) {
    //     i = 2;
    // }
    // outColor = texture(sampler2D(gbufferTexture[i], quadSampler), fragUV);

    // vec4 position = texture(sampler2D(gbufferTexture[POSITION_ID], quadSampler), fragUV);
    // vec4 normal = texture(sampler2D(gbufferTexture[NORMAL_ID], quadSampler), fragUV);

    // outColor = texture(sampler2D(quadTexture, quadSampler), fragUV);
    // outColor = vec4(texture(sampler2D(quadTexture, quadSampler), fragUV).r, 0.0, 0.0, 1.0);
}
`;


async function createTextureFromImage(device, src, usage) {
    // Current hacky Texture impl for WebGPU
    // Upload texture image data to gpu by uploading data array
    // retrieved from a 2d canvas

    const img = document.createElement('img');
    img.src = src;
    await img.decode();

    const imageCanvas = document.createElement('canvas');
    imageCanvas.width = img.width;
    imageCanvas.height = img.height;

    const imageCanvasContext = imageCanvas.getContext('2d');
    imageCanvasContext.translate(0, img.height);
    imageCanvasContext.scale(1, -1);
    imageCanvasContext.drawImage(img, 0, 0, img.width, img.height);
    const imageData = imageCanvasContext.getImageData(0, 0, img.width, img.height);
    let data = null;

    const rowPitch = Math.ceil(img.width * 4 / 256) * 256;
    if (rowPitch == img.width * 4) {
        data = imageData.data;
    } else {
        data = new Uint8Array(rowPitch * img.height);
        for (let y = 0; y < canvas.height; ++y) {
            for (let x = 0; x < canvas.width; ++x) {
                let i = x * 4 + y * rowPitch;
                data[i] = imageData.data[i];
                data[i + 1] = imageData.data[i + 1];
                data[i + 2] = imageData.data[i + 2];
                data[i + 3] = imageData.data[i + 3];
            }
        }
    }

    const texture = device.createTexture({
        size: {
            width: img.width,
            height: img.height,
            depth: 1,
        },
        arrayLayerCount: 1,
        mipLevelCount: 1,
        sampleCount: 1,
        dimension: "2d",
        format: "rgba8unorm",
        usage: GPUTextureUsage.COPY_DST | usage,
    });

    const textureDataBuffer = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    textureDataBuffer.setSubData(0, data);

    const commandEncoder = device.createCommandEncoder({});
    commandEncoder.copyBufferToTexture({
        buffer: textureDataBuffer,
        rowPitch: rowPitch,
        arrayLayer: 0,
        mipLevel: 0,
        imageHeight: 0,
    }, {
            texture: texture,
            mipLevel: 0,
            arrayLayer: 0,
            origin: { x: 0, y: 0, z: 0 }
        }, {
            width: img.width,
            height: img.height,
            depth: 1,
        });

    device.getQueue().submit([commandEncoder.finish()]);

    // this.texture = texture;
    return texture;
}



const quadVertexSize = 4 * 6;   // padding?
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


// // let T = mat4.create();
// // T[5] = 0;
// // T[6] = 1;
// // T[9] = 1;
// // T[10] = 0;
// const T = mat4.create();
// function R2L(M) {
//     // mat4.copy(T, M);
//     // M[1] = T[2];
//     // M[2] = T[1];

//     // M[4] = T[8];
//     // M[5] = T[10];
//     // M[6] = T[9];

//     // M[8] = T[4];
//     // M[9] = T[6];
//     // M[10] = T[5];

//     // M[13] = T[14];
//     // M[14] = T[13];

//     mat4.set(M, 
//         M[0], M[2], M[1], M[3],
//         M[8], M[10], M[9], M[11],
//         M[4], M[6], M[5], M[7],
//         M[12], M[14], M[13], M[15],
//     );
// }

let modelMatrix1 = mat4.create();
mat4.translate(modelMatrix1, modelMatrix1, vec3.fromValues(0, 0, 0));
let modelViewMatrix1 = mat4.create();

let tmpMat41 = mat4.create();
let tmpMat42 = mat4.create();


export default class DeferredRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        // this.drawables = [];

        this.drawableLists = [];    // {renderpass: GPURenderPass, drawables: []}

        // dat.gui controls
        this.debugViewOffset = 0.5;

        this.camera = new Camera(canvas);
    }

    // draw() {

    //     // Render pass 1: MRT to G-Buffers
    //     for (let i = 0; i < this.drawables.length; i++) {
    //         this.drawables[i].draw();
    //     }
    // }

    //------------------

    async init() {
        /* Context, Device, SwapChain */
        const adapter = await navigator.gpu.requestAdapter();
        // const device = await adapter.requestDevice({});
        const device = this.device = await adapter.requestDevice({});

        const glslangModule = await import('https://unpkg.com/@webgpu/glslang@0.0.7/web/glslang.js');
        const glslang = await glslangModule.default();

        const canvas = this.canvas;
        const context = canvas.getContext('gpupresent');

        this.swapChain = context.configureSwapChain({
            device,
            format: "bgra8unorm",
        });

        await this.setupScene();

        const quadVerticesBuffer = this.quadVerticesBuffer = device.createBuffer({
            size: fullScreenQuadArray.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        quadVerticesBuffer.setSubData(0, fullScreenQuadArray);

        const uniformsBindGroupLayout = device.createBindGroupLayout({
            bindings: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    type: "uniform-buffer"
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    type: "sampler"
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    type: "sampled-texture",
                    textureComponentType: "float"
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.FRAGMENT,
                    type: "sampled-texture",
                    textureComponentType: "float"
                },
            ]
        });
        
        const quadUniformsBindGroupLayout = this.quadUniformsBindGroupLayout = device.createBindGroupLayout({
            bindings: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    type: "sampler"
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    type: "sampled-texture",
                    textureComponentType: "float"
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    type: "sampled-texture",
                    textureComponentType: "float"
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.FRAGMENT,
                    type: "sampled-texture",
                    textureComponentType: "float"
                },
            ]
        });

        const debugViewBindGroupLayout = this.debugViewsBindGroupLayout = device.createBindGroupLayout({
            bindings: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    type: "uniform-buffer"
                },
            ]
        });


        /* Render Pipeline */
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [uniformsBindGroupLayout] });
        const pipeline = this.pipeline = device.createRenderPipeline({
            layout: pipelineLayout,

            vertexStage: {
                module: device.createShaderModule({
                    code: glslang.compileGLSL(vertexShaderGBufferGLSL, "vertex"),
                }),
                entryPoint: "main"
            },
            fragmentStage: {
                module: device.createShaderModule({
                    // code: glslang.compileGLSL(fragmentShaderBlinnPhongGLSL, "fragment"),
                    code: glslang.compileGLSL(fragmentShaderGBufferGLSL, "fragment"),
                }),
                entryPoint: "main"
            },

            primitiveTopology: "triangle-list",
            depthStencilState: {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: "depth24plus-stencil8",
                stencilFront: {},
                stencilBack: {},
            },
            vertexInput: this.sponza.vertexInput,

            rasterizationState: {
                frontFace: 'cw',
                cullMode: 'back',
            },

            colorStates: [
                {
                    format: "bgra8unorm",
                    alphaBlend: {},
                    colorBlend: {},
                },
                {
                    format: "bgra8unorm",
                    alphaBlend: {},
                    colorBlend: {},
                },
                {
                    format: "bgra8unorm",
                    alphaBlend: {},
                    colorBlend: {},
                },
            ],
        });

        const quadPipeLineLayout = device.createPipelineLayout({ bindGroupLayouts: [quadUniformsBindGroupLayout, debugViewBindGroupLayout] });
        const quadPipeline = this.quadPipeline = device.createRenderPipeline({
            layout: quadPipeLineLayout,

            vertexStage: {
                module: device.createShaderModule({
                    code: glslang.compileGLSL(vertexShaderFullScreenQuadGLSL, "vertex"),
                }),
                entryPoint: "main"
            },
            fragmentStage: {
                module: device.createShaderModule({
                    // code: glslang.compileGLSL(fragmentShaderFullScreenQuadGLSL, "fragment"),
                    code: glslang.compileGLSL(fragmentShaderGBufferDebugViewGLSL, "fragment"),
                }),
                entryPoint: "main"
            },

            primitiveTopology: "triangle-list",
            
            vertexInput: {
                indexFormat: "uint32",
                vertexBuffers: [{
                    stride: quadVertexSize, //padding
                    stepMode: "vertex",
                    attributeSet: [{
                        // position
                        shaderLocation: 0,
                        offset: 0,
                        format: "float4"
                    },
                    {
                        // uv
                        shaderLocation: 1,
                        offset: quadUVOffset,
                        format: "float2"
                    }]
                }]
            },

            rasterizationState: {
                frontFace: 'ccw',
                cullMode: 'back'
            },

            colorStates: [{
                format: "bgra8unorm",
                alphaBlend: {},
                colorBlend: {}
            }]
        });

        const depthTexture = this.depthTexture = device.createTexture({
            size: {
                width: canvas.width,
                height: canvas.height,
                depth: 1
            },
            arrayLayerCount: 1,
            mipLevelCount: 1,
            sampleCount: 1,
            dimension: "2d",
            format: "depth24plus-stencil8",
            usage: GPUTextureUsage.OUTPUT_ATTACHMENT
            // usage: GPUTextureUsage.OUTPUT_ATTACHMENT | GPUTextureUsage.COPY_DST | GPUTextureUsage.SAMPLED
        });


        // 10-11-2019 Unfortunately 
        // Currently Dawn does not support layered rendering.
        // https://cs.chromium.org/chromium/src/third_party/dawn/src/dawn_native/CommandEncoder.cpp?l=264
        // Sample from depth is not supported


        // this.gbufferTextures = device.createTexture({
        //     size: {
        //         width: canvas.width,
        //         height: canvas.height,
        //         depth: 1
        //     },
        //     arrayLayerCount: 3,     // Gbuffer count
        //     mipLevelCount: 1,
        //     sampleCount: 1,
        //     dimension: "2d",
        //     format: "bgra8unorm",
        //     usage: GPUTextureUsage.OUTPUT_ATTACHMENT | GPUTextureUsage.SAMPLED
        // });

        // // Loop
        // const colorAttachment0 = this.gbufferTextures.createView({
        //     dimension: "2d-array",
        //     baseArrayLayer: 0,
        //     arrayLayerCount: 1
        // });
        // const colorAttachment1 = this.gbufferTextures.createView({
        //     dimension: "2d-array",
        //     baseArrayLayer: 1,
        //     arrayLayerCount: 1
        // });
        // const colorAttachment2 = this.gbufferTextures.createView({
        //     dimension: "2d-array",
        //     baseArrayLayer: 2,
        //     arrayLayerCount: 1
        // });

        this.gbufferTextures = [
            device.createTexture({
                size: {
                    width: canvas.width,
                    height: canvas.height,
                    depth: 1
                },
                arrayLayerCount: 1,
                mipLevelCount: 1,
                sampleCount: 1,
                dimension: "2d",
                format: "bgra8unorm",
                usage: GPUTextureUsage.OUTPUT_ATTACHMENT | GPUTextureUsage.SAMPLED
            }),
            device.createTexture({
                size: {
                    width: canvas.width,
                    height: canvas.height,
                    depth: 1
                },
                arrayLayerCount: 1,
                mipLevelCount: 1,
                sampleCount: 1,
                dimension: "2d",
                format: "bgra8unorm",
                usage: GPUTextureUsage.OUTPUT_ATTACHMENT | GPUTextureUsage.SAMPLED
            }),
            device.createTexture({
                size: {
                    width: canvas.width,
                    height: canvas.height,
                    depth: 1
                },
                arrayLayerCount: 1,
                mipLevelCount: 1,
                sampleCount: 1,
                dimension: "2d",
                format: "bgra8unorm",
                usage: GPUTextureUsage.OUTPUT_ATTACHMENT | GPUTextureUsage.SAMPLED
            }),
        ];


        // const renderPassDescriptor = {
        this.renderPassDescriptor = {
            colorAttachments: [
                {
                    // attachment: colorAttachment0,
                    attachment: this.gbufferTextures[0].createView(),
                    loadValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                    storeOp: "store",
                },
                {
                    attachment: this.gbufferTextures[1].createView(),
                    loadValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                    storeOp: "store",
                },
                {
                    attachment: this.gbufferTextures[2].createView(),
                    loadValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                    storeOp: "store",
                },
            ],
            depthStencilAttachment: {
                attachment: depthTexture.createView(),

                depthLoadValue: 1.0,
                depthStoreOp: "store",
                stencilLoadValue: 0,
                stencilStoreOp: "store",
            }
        };

        this.renderFullScreenPassDescriptor = {
            colorAttachments: [{
                loadValue: {r: 0.0, g: 0.0, b: 0.0, a: 1.0},
                storeOp: "store",
            }],
        };

        const matrixSize = 4 * 16;  // 4x4 matrix
        // const offset = 256; // uniformBindGroup offset must be 256-byte aligned
        // const uniformBufferSize = offset + matrixSize;
        const uniformBufferSize = 3 * matrixSize;

        const uniformBuffer = this.uniformBuffer = device.createBuffer({
            size: uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const samplerRepeat = this.sampler = device.createSampler({
            addressModeU: "repeat",
            addressModeV: "repeat",
            magFilter: "linear",
            minFilter: "linear",
            mipmapFilter: "linear"
        });

        const sampler = this.sampler = device.createSampler({
            magFilter: "linear",
            minFilter: "linear"
        });

        this.uniformBindGroupWriteGBuffer = device.createBindGroup({
            layout: uniformsBindGroupLayout,
            bindings: [
                {
                    binding: 0,
                    resource: {
                        buffer: uniformBuffer,
                        offset: 0,
                        // size: matrixSize
                        size: uniformBufferSize
                    }
                },
                {
                    binding: 1,
                    resource: samplerRepeat,
                },
                {
                    binding: 2,
                    resource: this.albedoMap.createView(),
                },
                {
                    binding: 3,
                    resource: this.normalMap.createView(),
                },
            ],
        });

        

        // // const uniformBindGroup2 = device.createBindGroup({
        // this.uniformBindGroup2 = device.createBindGroup({
        //     layout: uniformsBindGroupLayout,
        //     bindings: [{
        //         binding: 0,
        //         resource: {
        //             buffer: uniformBuffer,
        //             offset: offset,
        //             size: matrixSize
        //         }
        //     }]
        // });

        

        this.quadUniformBindGroup = this.device.createBindGroup({
            layout: this.quadUniformsBindGroupLayout,
            bindings: [
                {
                    binding: 0,
                    resource: sampler,
                },
                // {
                //     binding: 1,
                //     // resource: this.rttTexture.createView()
                //     resource: this.gbufferTextures.createView()
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

        // const debugViewUniformBufferSize = 4;
        const debugViewUniformBufferSize = 16;
        const debugViewUniformBuffer = this.debugViewUniformBuffer = this.device.createBuffer({
            size: debugViewUniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.debugViewUniformBindGroup = this.device.createBindGroup({
            layout: this.debugViewsBindGroupLayout,
            bindings: [
                {
                    binding: 0,
                    resource: {
                        buffer: debugViewUniformBuffer,
                        offset: 0,
                        size: debugViewUniformBufferSize
                    }
                }
            ]
        });
    }

    async setupScene() {

        // return new Promise((resolve) => {
        //     OBJ.downloadMeshes({
        //         'sponza': 'models/sponza.obj'
        //         // 'sponza': 'models/di.obj'
        //     }, resolve);
        // }).then((meshes) => {
        //     this.meshes = meshes;

        //     // build mesh, drawable list here
        //     const geometry = this.sponza = new Geometry(this.device);
        //     geometry.fromObjMesh(meshes['sponza']);

        // });

        const modelUrl = 'models/sponza.obj';
        const albedoUrl = 'models/color.jpg';
        const normalUrl = 'models/normal.png';

        // const modelUrl = 'models/di.obj';
        // const albedoUrl = 'models/di.png';
        // const normalUrl = null;

        const pModel = new Promise((resolve) => {
            OBJ.downloadMeshes({
                'obj': modelUrl
            }, resolve);
        });

        const pAlbedoMap = createTextureFromImage(this.device, albedoUrl, GPUTextureUsage.SAMPLED);

        const pNormalMap = normalUrl ? createTextureFromImage(this.device, normalUrl, GPUTextureUsage.SAMPLED) : null;

        await Promise.all([pModel, pAlbedoMap, pNormalMap]).then((values) => {
            this.meshes = values[0];

            // build mesh, drawable list here
            const geometry = this.sponza = new Geometry(this.device);
            geometry.fromObjMesh(this.meshes['obj']);

            const albedoMap = this.albedoMap = values[1];
            // console.log(albedoMap);

            const normalMap = this.normalMap = values[2];
            // console.log('normalmap: ', normalMap);
        });
    }

    updateTransformationMatrix() {
        // let now = Date.now() / 1000;

        // mat4.copy(tmpMat41, modelMatrix1);
        // mat4.rotate(tmpMat41, modelMatrix1, now, vec3.fromValues(0, 1, 0));


        mat4.multiply(modelViewMatrix1, this.camera.viewMatrix, modelMatrix1);
    
        mat4.invert(tmpMat41, modelViewMatrix1);
        mat4.transpose(tmpMat41, tmpMat41); //normal matrix


        // R2L(modelViewMatrix1);
        // R2L(tmpMat41);


        // mat4.multiply(modelViewMatrix1, T, modelViewMatrix1);
        // mat4.multiply(tmpMat41, T, tmpMat41);


        // mat4.multiply(modelViewProjectionMatrix1, projectionMatrix, modelViewProjectionMatrix1);
        // mat4.multiply(modelViewProjectionMatrix2, viewMatrix, tmpMat42);
        // mat4.multiply(modelViewProjectionMatrix2, projectionMatrix, modelViewProjectionMatrix2);
    }


    frame() {
        // updateTransformationMatrix();
        this.updateTransformationMatrix();

        // this.renderPassDescriptor.colorAttachments[0].attachment = swapChainTexture.createView();
        // this.renderPassDescriptor.colorAttachments[0].attachment = this.rttTexture.createView();

        const commandEncoder = this.device.createCommandEncoder({});

        this.uniformBuffer.setSubData(0, modelViewMatrix1);
        this.uniformBuffer.setSubData(64, this.camera.projectionMatrix);
        this.uniformBuffer.setSubData(128, tmpMat41);
        // this.uniformBuffer.setSubData(offset, modelViewProjectionMatrix2);
        // this.uniformBuffer.setSubData(256, modelViewProjectionMatrix2);
        const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setVertexBuffers(
            0,
            [this.sponza.verticesBuffer, this.sponza.normalsBuffer, this.sponza.uvsBuffer],
            [0, 0, 0]
        );
        // passEncoder.setVertexBuffer(0, this.sponza.verticesBuffer);
        // passEncoder.setVertexBuffer(1, this.sponza.normalsBuffer);
        // passEncoder.setVertexBuffer(2, this.sponza.uvsBuffer);
        passEncoder.setIndexBuffer(this.sponza.indicesBuffer);

        passEncoder.setBindGroup(0, this.uniformBindGroupWriteGBuffer);

        // console.log(this.sponza.indices.length);
        passEncoder.drawIndexed(this.sponza.indices.length, 1, 0, 0, 0);
        // passEncoder.draw(36, 1, 0, 0);

        // passEncoder.setBindGroup(0, this.uniformBindGroup2);
        // passEncoder.draw(36, 1, 0, 0);

        passEncoder.endPass();

        // // Copy depth texture
        // commandEncoder.copyTextureToTexture({
        //     texture: this.depthTexture,
        //     mipLevel: 0,
        //     arrayLayer: 0,
        //     origin: { x: 0, y: 0, z: 0 }
        // }, {
        //     texture: fullScreenQuadTexture,
        //     mipLevel: 0,
        //     arrayLayer: 0,
        //     origin: { x: 0, y: 0, z: 0 }
        // }, {
        //     width: this.canvas.width,
        //     height: this.canvas.height,
        //     depth: 1
        // });
        // this.device.getQueue().submit([commandEncoder.finish()]);


        // // render full screen quad

        const swapChainTexture = this.swapChain.getCurrentTexture();

        this.debugViewUniformBuffer.setSubData(0, new Float32Array([this.debugViewOffset]));
        // this.debugViewUniformBuffer.setSubData(0, new Float32Array([this.debugViewOffset, 0, 0, 0]));
        // this.debugViewUniformBuffer.setSubData(0, new Float32Array([0.5, 0, 0, 0]));

        this.renderFullScreenPassDescriptor.colorAttachments[0].attachment = swapChainTexture.createView();
        const quadPassEncoder = commandEncoder.beginRenderPass(this.renderFullScreenPassDescriptor);
        quadPassEncoder.setPipeline(this.quadPipeline);
        quadPassEncoder.setVertexBuffers(0, [this.quadVerticesBuffer], [0]);
        // quadPassEncoder.setVertexBuffer(0, this.quadVerticesBuffer);
        quadPassEncoder.setBindGroup(0, this.quadUniformBindGroup);
        quadPassEncoder.setBindGroup(1, this.debugViewUniformBindGroup);
        quadPassEncoder.draw(6, 1, 0, 0);
        quadPassEncoder.endPass();

        this.device.getQueue().submit([commandEncoder.finish()]);
    }
}