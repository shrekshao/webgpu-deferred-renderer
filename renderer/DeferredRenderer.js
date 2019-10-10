// import {vec3, vec4, mat4} from '../third_party/gl-matrix-min.js';
// import glmatrix from '../third_party/gl-matrix-min.js';

import Geometry from './Geometry.js';


const vertexShaderGLSL = `#version 450
layout(set = 0, binding = 0) uniform Uniforms {
mat4 modelViewProjectionMatrix;
} uniforms;

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 color;

layout(location = 0) out vec4 fragColor;

void main() {
gl_Position = uniforms.modelViewProjectionMatrix * position;
fragColor = color;
}
`;

const fragmentShaderGLSL = `#version 450
layout(location = 0) in vec4 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
outColor = fragColor;
}
`;


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
    fragNormal = uniforms.modelViewNormalMatrix * vec4(normal, 0);
    fragUV = uv;
    gl_Position = uniforms.projectionMatrix * fragPosition;
    // gl_Position = vec4(0.1 * position.xy, 0.5, 1);
    // gl_Position = vec4( clamp(position.xy, vec2(-0.5, -0.8), vec2(1, 1)), 0.5, 1);
}
`;

const fragmentShaderGBufferGLSL = `#version 450
layout(location = 0) out vec4 fragPosition;
layout(location = 1) out vec4 fragNormal;
layout(location = 2) out vec2 fragUV;

layout(location = 0) out vec4 outGBufferPosition;
layout(location = 1) out vec4 outGBufferNormal;
layout(location = 2) out vec4 outGBufferAlbedo;

void main() {
    outGBufferPosition = fragPosition;
    outGBufferNormal = fragNormal;
    outGBufferAlbedo = vec4(fragUV, 0, 1);  // temp
}
`;

const fragmentShaderBlinnPhongGLSL = `#version 450
layout(location = 0) in vec4 fragPosition;
layout(location = 1) in vec4 fragNormal;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outFragColor;

// TODO: texture

void main() {
    outFragColor = vec4(fragNormal.xyz, 1);  // temp
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


// const fragmentShaderFullScreenQuadGLSL = `#version 450

// layout(location = 0) in vec2 fragUV;
// layout(location = 0) out vec4 outColor;

// void main() {
//     outColor = vec4(fragUV, 0.0, 1.0);
// }
// `;

const vertexSize = 4 * 8; // Byte size of one cube vertex.
const colorOffset = 4 * 4; // Byte offset of cube vertex color attribute.
const cubeVerticesArray = new Float32Array([
    // float4 position, float4 color
    1, -1, 1, 1, 1, 0, 1, 1,
    -1, -1, 1, 1, 0, 0, 1, 1,
    -1, -1, -1, 1, 0, 0, 0, 1,
    1, -1, -1, 1, 1, 0, 0, 1,
    1, -1, 1, 1, 1, 0, 1, 1,
    -1, -1, -1, 1, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, -1, 1, 1, 1, 0, 1, 1,
    1, -1, -1, 1, 1, 0, 0, 1,
    1, 1, -1, 1, 1, 1, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, -1, -1, 1, 1, 0, 0, 1,
    -1, 1, 1, 1, 0, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, -1, 1, 1, 1, 0, 1,
    -1, 1, -1, 1, 0, 1, 0, 1,
    -1, 1, 1, 1, 0, 1, 1, 1,
    1, 1, -1, 1, 1, 1, 0, 1,
    -1, -1, 1, 1, 0, 0, 1, 1,
    -1, 1, 1, 1, 0, 1, 1, 1,
    -1, 1, -1, 1, 0, 1, 0, 1,
    -1, -1, -1, 1, 0, 0, 0, 1,
    -1, -1, 1, 1, 0, 0, 1, 1,
    -1, 1, -1, 1, 0, 1, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    -1, 1, 1, 1, 0, 1, 1, 1,
    -1, -1, 1, 1, 0, 0, 1, 1,
    -1, -1, 1, 1, 0, 0, 1, 1,
    1, -1, 1, 1, 1, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, -1, -1, 1, 1, 0, 0, 1,
    -1, -1, -1, 1, 0, 0, 0, 1,
    -1, 1, -1, 1, 0, 1, 0, 1,
    1, 1, -1, 1, 1, 1, 0, 1,
    1, -1, -1, 1, 1, 0, 0, 1,
    -1, 1, -1, 1, 0, 1, 0, 1,
]);

// const uvCubeVerticesArray = new Float32Array([
//     // float3 position, float3 normal, float2 uv
//     1, -1, 1, 0, -1, 0, 1, 1,
//     -1, -1, -1, 0, -1, 0, 0, 0,
//     -1, -1, 1, 0, -1, 0, 0, 1,
//     1, -1, -1, 0, -1, 0, 1, 0,
//     -1, -1, -1, 0, -1, 0, 0, 0,
//     1, -1, 1, 0, -1, 0, 1, 1,

//     1, 1, 1, 1, 0, 0, 1, 1,
//     1, -1, -1, 1, 0, 0, 0, 0,
//     1, -1, 1, 1, 0, 0, 0, 1,
//     1, 1, -1, 1, 0, 0, 1, 0,
//     1, -1, -1, 1, 0, 0, 0, 0,
//     1, 1, 1, 1, 0, 0, 1, 1,

//     -1, 1, 1, 0, 1, 0, 0, 1,
//     1, 1, -1, 0, 1, 0, 1, 0,
//     1, 1, 1, 0, 1, 0, 1, 1,
//     -1, 1, -1, 0, 1, 0, 0, 0,
//     1, 1, -1, 0, 1, 0, 1, 0,
//     -1, 1, 1, 0, 1, 0, 0, 1,

//     -1, -1, 1, -1, 0, 0, 0, 1,
//     -1, 1, -1, -1, 0, 0, 1, 0,
//     -1, 1, 1, -1, 0, 0, 1, 1,
//     -1, -1, -1, -1, 0, 0, 0, 0,
//     -1, 1, -1, -1, 0, 0, 1, 0,
//     -1, -1, 1, -1, 0, 0, 0, 1,

//     1, 1, 1, 0, 0, 1, 1, 1,
//     -1, -1, 1, 0, 0, 1, 0, 0,
//     -1, 1, 1, 0, 0, 1, 0, 1,
//     -1, -1, 1, 0, 0, 1, 0, 0,
//     1, 1, 1, 0, 0, 1, 1, 1,
//     1, -1, 1, 0, 0, 1, 1, 0,

//     1, -1, -1, 0, 0, -1, 1, 0,
//     -1, 1, -1, 0, 0, -1, 0, 1,
//     -1, -1, -1, 0, 0, -1, 0, 0,
//     1, 1, -1, 0, 0, -1, 1, 1,
//     -1, 1, -1, 0, 0, -1, 0, 1,
//     1, -1, -1, 0, 0, -1, 1, 0,
// ]);

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

let modelMatrix1 = mat4.create();
// mat4.scale(modelMatrix1, modelMatrix1, vec3.fromValues(0.1, 0.1, 0.1));
// mat4.translate(modelMatrix1, modelMatrix1, vec3.fromValues(-2, 0, 0));
mat4.translate(modelMatrix1, modelMatrix1, vec3.fromValues(0, 0, 0));
// let modelMatrix2 = mat4.create();
// mat4.translate(modelMatrix2, modelMatrix2, vec3.fromValues(2, 0, 0));
let modelViewMatrix1 = mat4.create();
// let modelViewProjectionMatrix2 = mat4.create();
let viewMatrix = mat4.create();
// mat4.translate(viewMatrix, viewMatrix, vec3.fromValues(0, 0, -7));
mat4.translate(viewMatrix, viewMatrix, vec3.fromValues(0, 0, -50));

let projectionMatrix = mat4.create();

let tmpMat41 = mat4.create();
let tmpMat42 = mat4.create();


export default class DeferredRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.drawables = [];
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

        // const canvas = document.querySelector('canvas');
        const canvas = this.canvas;

        const aspect = Math.abs(canvas.width / canvas.height);
        // let projectionMatrix = mat4.create();
        mat4.perspective(projectionMatrix, (2 * Math.PI) / 5, aspect, 1, 1000.0);

        const context = canvas.getContext('gpupresent');

        // const swapChain = context.configureSwapChain({
        this.swapChain = context.configureSwapChain({
            device,
            format: "bgra8unorm",
        });

        await this.setupScene();
        // console.log(this.meshes);

        // const verticesBuffer = this.verticesBuffer = device.createBuffer({
        //     size: cubeVerticesArray.byteLength,
        //     usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        // });
        // verticesBuffer.setSubData(0, cubeVerticesArray);

        const quadVerticesBuffer = this.quadVerticesBuffer = device.createBuffer({
            size: fullScreenQuadArray.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        quadVerticesBuffer.setSubData(0, fullScreenQuadArray);

        const uniformsBindGroupLayout = device.createBindGroupLayout({
            bindings: [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                type: "uniform-buffer"
            }]
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
            ]
        });


        /* Render Pipeline */
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [uniformsBindGroupLayout] });
        const pipeline = this.pipeline = device.createRenderPipeline({
            layout: pipelineLayout,

            vertexStage: {
                module: device.createShaderModule({
                    code: glslang.compileGLSL(vertexShaderBlinnPhongGLSL, "vertex"),
                }),
                entryPoint: "main"
            },
            fragmentStage: {
                module: device.createShaderModule({
                    code: glslang.compileGLSL(fragmentShaderBlinnPhongGLSL, "fragment"),
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
                frontFace: 'ccw',
                cullMode: 'back',
            },

            colorStates: [{
                format: "bgra8unorm",
                alphaBlend: {},
                colorBlend: {},
            }],
        });

        const quadPipeLineLayout = device.createPipelineLayout({ bindGroupLayouts: [quadUniformsBindGroupLayout] });
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
                    code: glslang.compileGLSL(fragmentShaderFullScreenQuadGLSL, "fragment"),
                }),
                entryPoint: "main"
            },

            primitiveTopology: "triangle-list",
            // depthStencilState: {
            //     depthWriteEnabled: true,
            //     depthCompare: "less",
            //     format: "depth24plus-stencil8",
            //     // format: "depth24plus",
            // },
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

        const rttTexture = this.rttTexture = device.createTexture({
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
        });

        // const renderPassDescriptor = {
        this.renderPassDescriptor = {
            colorAttachments: [{
                // attachment is acquired in render loop.
                loadValue: { r: 0.5, g: 0.5, b: 0.5, a: 1.0 },
                storeOp: "store",
            }],
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
                loadValue: {r: 0.0, g: 0.5, b: 0.0, a: 1.0},
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

        // const uniformBindGroup1 = device.createBindGroup({
        this.uniformBindGroup1 = device.createBindGroup({
            layout: uniformsBindGroupLayout,
            bindings: [{
                binding: 0,
                resource: {
                    buffer: uniformBuffer,
                    offset: 0,
                    // size: matrixSize
                    size: uniformBufferSize
                }
            }],
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

        const sampler = this.sampler = device.createSampler({
            magFilter: "nearest",
            minFilter: "nearest"
        });

        const quadUniformBindGroup = this.quadUniformBindGroup = this.device.createBindGroup({
            layout: this.quadUniformsBindGroupLayout,
            bindings: [
                {
                    binding: 0,
                    resource: this.sampler,
                },
                {
                    binding: 1,
                    resource: this.rttTexture.createView()
                }
            ]
        });
    }

    async setupScene() {
        // OBJ.downloadMeshes({
        //     'sponza': 'models/sponza.obj',
        // }, (meshes) => {
        //     console.log(meshes);
        // });

        return new Promise((resolve) => {
            OBJ.downloadMeshes({
                'sponza': 'models/sponza.obj'
            }, resolve);
        }).then((meshes) => {
            this.meshes = meshes;

            // build mesh, drawable list here
            const geometry = this.sponza = new Geometry(this.device);
            geometry.fromObjMesh(meshes['sponza']);

        });

        // const g = this.sponza = new Geometry(this.device);
        // g.fromData(
        //     new Float32Array([
        //             // float position
        //             1, -1, 1,
        //             -1, -1, 1,
        //             -1, -1, -1,
        //             1, -1, -1,
        //             1, -1, 1,
        //             -1, -1, -1,

        //             1, 1, 1,
        //             1, -1, 1,
        //             1, -1, -1,
        //             1, 1, -1,
        //             1, 1, 1,
        //             1, -1, -1,

        //             -1, 1, 1,
        //             1, 1, 1,
        //             1, 1, -1,
        //             -1, 1, -1,
        //             -1, 1, 1,
        //             1, 1, -1,

        //             -1, -1, 1,
        //             -1, 1, 1,
        //             -1, 1, -1,
        //             -1, -1, -1,
        //             -1, -1, 1,
        //             -1, 1, -1,

        //             1, 1, 1,
        //             -1, 1, 1,
        //             -1, -1, 1,
        //             -1, -1, 1,
        //             1, -1, 1,
        //             1, 1, 1,

        //             1, -1, -1,
        //             -1, -1, -1,
        //             -1, 1, -1,
        //             1, 1, -1,
        //             1, -1, -1,
        //             -1, 1, -1,
        //     ]),
        //     new Float32Array([
        //             // float normal
        //             0, -1, 0,
        //             0, -1, 0,
        //             0, -1, 0,
        //             0, -1, 0,
        //             0, -1, 0,
        //             0, -1, 0,

        //             1, 0, 0,
        //             1, 0, 0,
        //             1, 0, 0,
        //             1, 0, 0,
        //             1, 0, 0,
        //             1, 0, 0,

        //             0, 1, 0,
        //             0, 1, 0,
        //             0, 1, 0,
        //             0, 1, 0,
        //             0, 1, 0,
        //             0, 1, 0,

        //             -1, 0, 0,
        //             -1, 0, 0,
        //             -1, 0, 0,
        //             -1, 0, 0,
        //             -1, 0, 0,
        //             -1, 0, 0,
        //             -1, 0, 0,

        //             0, 0, 1,
        //             0, 0, 1,
        //             0, 0, 1,
        //             0, 0, 1,
        //             0, 0, 1,
        //             0, 0, 1,

        //             0, 0, -1,
        //             0, 0, -1,
        //             0, 0, -1,
        //             0, 0, -1,
        //             0, 0, -1,
        //             0, 0, -1,
        //     ]),
        //     new Float32Array([
        //             // float2 uv,
        //             1, 1,
        //             0, 1,
        //             0, 0,
        //             1, 0,
        //             1, 1,
        //             0, 0,

        //             1, 1,
        //             0, 1,
        //             0, 0,
        //             1, 0,
        //             1, 1,
        //             0, 0,

        //             1, 1,
        //             0, 1,
        //             0, 0,
        //             1, 0,
        //             1, 1,
        //             0, 0,

        //             1, 1,
        //             0, 1,
        //             0, 0,
        //             1, 0,
        //             1, 1,
        //             0, 0,

        //             1, 1,
        //             0, 1,
        //             0, 0,
        //             1, 0,
        //             1, 1,
        //             0, 0,

        //             1, 1,
        //             0, 1,
        //             0, 0,
        //             1, 0,
        //             1, 1,
        //             0, 0,
        //     ]),
        //     new Uint32Array([
        //             0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
        //             21,22,23,24,25,26,27,28,29,30,
        //             31,32,33,34,35
        //     ]),

        // );
    }

    updateTransformationMatrix() {
        let now = Date.now() / 1000;

        mat4.rotate(tmpMat41, modelMatrix1, now, vec3.fromValues(0, 1, 0));
        // mat4.rotate(tmpMat41, modelMatrix1, 1, vec3.fromValues(Math.sin(now), Math.cos(now), 0));
        // mat4.rotate(tmpMat42, modelMatrix2, 1, vec3.fromValues(Math.cos(now), Math.sin(now), 0));

        mat4.multiply(modelViewMatrix1, viewMatrix, tmpMat41);
        mat4.invert(tmpMat41, modelViewMatrix1);
        mat4.transpose(tmpMat41, tmpMat41);
        // mat4.multiply(modelViewProjectionMatrix1, projectionMatrix, modelViewProjectionMatrix1);
        // mat4.multiply(modelViewProjectionMatrix2, viewMatrix, tmpMat42);
        // mat4.multiply(modelViewProjectionMatrix2, projectionMatrix, modelViewProjectionMatrix2);
    }


    frame() {
        // updateTransformationMatrix();
        this.updateTransformationMatrix();

        // this.renderPassDescriptor.colorAttachments[0].attachment = swapChainTexture.createView();
        this.renderPassDescriptor.colorAttachments[0].attachment = this.rttTexture.createView();
        // this.renderPassDescriptor.depthStencilAttachment = this.fullScreenQuadTexture.createView();
        // this.renderPassDescriptor.depthStencilAttachment = this.depthTexture.createView();

        const commandEncoder = this.device.createCommandEncoder({});

        this.uniformBuffer.setSubData(0, modelViewMatrix1);
        this.uniformBuffer.setSubData(64, projectionMatrix);
        this.uniformBuffer.setSubData(128, tmpMat41);
        // this.uniformBuffer.setSubData(offset, modelViewProjectionMatrix2);
        // this.uniformBuffer.setSubData(256, modelViewProjectionMatrix2);
        const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);
        passEncoder.setPipeline(this.pipeline);
        // passEncoder.setVertexBuffers(0, [this.verticesBuffer], [0]);
        passEncoder.setVertexBuffers(
            0,
            [this.sponza.verticesBuffer, this.sponza.normalsBuffer, this.sponza.uvsBuffer],
            [0, 0, 0]
        );
        passEncoder.setIndexBuffer(this.sponza.indicesBuffer);

        passEncoder.setBindGroup(0, this.uniformBindGroup1);

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

        

        this.renderFullScreenPassDescriptor.colorAttachments[0].attachment = swapChainTexture.createView();
        const quadPassEncoder = commandEncoder.beginRenderPass(this.renderFullScreenPassDescriptor);
        quadPassEncoder.setPipeline(this.quadPipeline);
        quadPassEncoder.setVertexBuffers(0, [this.quadVerticesBuffer], [0]);
        quadPassEncoder.setBindGroup(0, this.quadUniformBindGroup);
        quadPassEncoder.draw(6, 1, 0, 0);
        quadPassEncoder.endPass();

        this.device.getQueue().submit([commandEncoder.finish()]);
    }
}