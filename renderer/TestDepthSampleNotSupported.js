// import {vec3, vec4, mat4} from '../third_party/gl-matrix-min.js';
// import glmatrix from '../third_party/gl-matrix-min.js';

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
`

const fragmentShaderGLSL = `#version 450
layout(location = 0) in vec4 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
outColor = fragColor;
}
`

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
    // outColor = texture(sampler2D(quadTexture, quadSampler), fragUV);
    outColor = vec4(texture(sampler2D(quadTexture, quadSampler), fragUV).r, 0.0, 0.0, 1.0);
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
    -1, -1, -1, 1, 0, 0, 0, 1,
    -1, -1, 1, 1, 0, 0, 1, 1,
    1, -1, -1, 1, 1, 0, 0, 1,
    -1, -1, -1, 1, 0, 0, 0, 1,
    1, -1, 1, 1, 1, 0, 1, 1,

    1, 1, 1, 1, 1, 1, 1, 1,
    1, -1, -1, 1, 1, 0, 0, 1,
    1, -1, 1, 1, 1, 0, 1, 1,
    1, 1, -1, 1, 1, 1, 0, 1,
    1, -1, -1, 1, 1, 0, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 1,

    -1, 1, 1, 1, 0, 1, 1, 1,
    1, 1, -1, 1, 1, 1, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    -1, 1, -1, 1, 0, 1, 0, 1,
    1, 1, -1, 1, 1, 1, 0, 1,
    -1, 1, 1, 1, 0, 1, 1, 1,

    -1, -1, 1, 1, 0, 0, 1, 1,
    -1, 1, -1, 1, 0, 1, 0, 1,
    -1, 1, 1, 1, 0, 1, 1, 1,
    -1, -1, -1, 1, 0, 0, 0, 1,
    -1, 1, -1, 1, 0, 1, 0, 1,
    -1, -1, 1, 1, 0, 0, 1, 1,

    1, 1, 1, 1, 1, 1, 1, 1,
    -1, -1, 1, 1, 0, 0, 1, 1,
    -1, 1, 1, 1, 0, 1, 1, 1,
    -1, -1, 1, 1, 0, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, -1, 1, 1, 1, 0, 1, 1,

    1, -1, -1, 1, 1, 0, 0, 1,
    -1, 1, -1, 1, 0, 1, 0, 1,
    -1, -1, -1, 1, 0, 0, 0, 1,
    1, 1, -1, 1, 1, 1, 0, 1,
    -1, 1, -1, 1, 0, 1, 0, 1,
    1, -1, -1, 1, 1, 0, 0, 1,
]);

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
mat4.translate(modelMatrix1, modelMatrix1, vec3.fromValues(-2, 0, 0));
let modelMatrix2 = mat4.create();
mat4.translate(modelMatrix2, modelMatrix2, vec3.fromValues(2, 0, 0));
let modelViewProjectionMatrix1 = mat4.create();
let modelViewProjectionMatrix2 = mat4.create();
let viewMatrix = mat4.create();
mat4.translate(viewMatrix, viewMatrix, vec3.fromValues(0, 0, -7));

let projectionMatrix = mat4.create();

let tmpMat41 = mat4.create();
let tmpMat42 = mat4.create();




export default class DeferredRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.drawables = [];
    }

    draw() {

        // Render pass 1 MRT to G-Buffers
        for (let i = 0; i < this.drawables.length; i++) {
            this.drawables[i].draw();
        }
    }

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
        mat4.perspective(projectionMatrix, (2 * Math.PI) / 5, -aspect, 1, 100.0);

        const context = canvas.getContext('gpupresent');

        // const swapChain = context.configureSwapChain({
        this.swapChain = context.configureSwapChain({
            device,
            format: "bgra8unorm"
        });

        const verticesBuffer = this.verticesBuffer = device.createBuffer({
            size: cubeVerticesArray.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        verticesBuffer.setSubData(0, cubeVerticesArray);

        const quadVerticesBuffer = this.quadVerticesBuffer = device.createBuffer({
            size: fullScreenQuadArray.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        quadVerticesBuffer.setSubData(0, fullScreenQuadArray);

        const uniformsBindGroupLayout = device.createBindGroupLayout({
            bindings: [{
                binding: 0,
                // visibility: 1,
                visibility: GPUShaderStage.VERTEX,
                type: "uniform-buffer"
            }]
        });
        
        const quadUniformsBindGroupLayout = device.createBindGroupLayout({
            bindings: [
                {
                    binding: 0,
                    // visibility: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    type: "sampler"
                },
                {
                    binding: 1,
                    // visibility: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    type: "sampled-texture",
                    // textureComponentType: "float",
                    textureComponentType: "uint",
                    // textureComponentType: "sint",
                },
            ]
        });


        /* Render Pipeline */
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [uniformsBindGroupLayout] });
        const pipeline = this.pipeline = device.createRenderPipeline({
            layout: pipelineLayout,

            vertexStage: {
                module: device.createShaderModule({
                    code: glslang.compileGLSL(vertexShaderGLSL, "vertex"),
                }),
                entryPoint: "main"
            },
            fragmentStage: {
                module: device.createShaderModule({
                    code: glslang.compileGLSL(fragmentShaderGLSL, "fragment"),
                }),
                entryPoint: "main"
            },

            primitiveTopology: "triangle-list",
            depthStencilState: {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: "depth24plus-stencil8",
                // format: "depth24plus",
                // format: "depth32float",
                stencilFront: {},
                stencilBack: {},
            },
            vertexInput: {
                indexFormat: "uint32",
                vertexBuffers: [{
                    stride: vertexSize,
                    stepMode: "vertex",
                    attributeSet: [{
                        // position
                        shaderLocation: 0,
                        offset: 0,
                        format: "float4"
                    }, {
                        // color
                        shaderLocation: 1,
                        offset: colorOffset,
                        format: "float4"
                    }]
                }],
            },

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
            // format: "depth24plus",
            // format: "depth32float",
            // usage: GPUTextureUsage.OUTPUT_ATTACHMENT
            usage: GPUTextureUsage.OUTPUT_ATTACHMENT | GPUTextureUsage.COPY_DST | GPUTextureUsage.SAMPLED
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
        const offset = 256; // uniformBindGroup offset must be 256-byte aligned
        const uniformBufferSize = offset + matrixSize;

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
                    size: matrixSize
                }
            }],
        });

        // const uniformBindGroup2 = device.createBindGroup({
        this.uniformBindGroup2 = device.createBindGroup({
            layout: uniformsBindGroupLayout,
            bindings: [{
                binding: 0,
                resource: {
                    buffer: uniformBuffer,
                    offset: offset,
                    size: matrixSize
                }
            }]
        });

        const sampler = device.createSampler({
            magFilter: "nearest",
            minFilter: "nearest"
        });

        this.quadUniformBindGroup = device.createBindGroup({
            layout: quadUniformsBindGroupLayout,
            bindings: [
                {
                    binding: 0,
                    resource: sampler,
                },
                {
                    binding: 1,
                    resource: depthTexture.createView({
                        // format: 'depth32float'
                        // format: 'depth24plus'
                        format: 'depth24plus-stencil8'
                    })
                    // resource: depthSampledTexture
                }
            ]
        });
    }

    updateTransformationMatrix() {

        let now = Date.now() / 1000;

        mat4.rotate(tmpMat41, modelMatrix1, 1, vec3.fromValues(Math.sin(now), Math.cos(now), 0));
        mat4.rotate(tmpMat42, modelMatrix2, 1, vec3.fromValues(Math.cos(now), Math.sin(now), 0));

        mat4.multiply(modelViewProjectionMatrix1, viewMatrix, tmpMat41);
        mat4.multiply(modelViewProjectionMatrix1, projectionMatrix, modelViewProjectionMatrix1);
        mat4.multiply(modelViewProjectionMatrix2, viewMatrix, tmpMat42);
        mat4.multiply(modelViewProjectionMatrix2, projectionMatrix, modelViewProjectionMatrix2);
    }


    frame() {
        // updateTransformationMatrix();
        this.updateTransformationMatrix();

        const swapChainTexture = this.swapChain.getCurrentTexture();
        this.renderPassDescriptor.colorAttachments[0].attachment = swapChainTexture.createView();
        // this.renderPassDescriptor.depthStencilAttachment = this.fullScreenQuadTexture.createView();
        // this.renderPassDescriptor.depthStencilAttachment = this.depthTexture.createView();

        const commandEncoder = this.device.createCommandEncoder({});

        this.uniformBuffer.setSubData(0, modelViewProjectionMatrix1);
        // this.uniformBuffer.setSubData(offset, modelViewProjectionMatrix2);
        this.uniformBuffer.setSubData(256, modelViewProjectionMatrix2);
        const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setVertexBuffers(0, [this.verticesBuffer], [0]);

        passEncoder.setBindGroup(0, this.uniformBindGroup1);
        passEncoder.draw(36, 1, 0, 0);

        passEncoder.setBindGroup(0, this.uniformBindGroup2);
        passEncoder.draw(36, 1, 0, 0);

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

        // this.renderFullScreenPassDescriptor.colorAttachments[0].attachment = swapChainTexture.createView();
        // const quadPassEncoder = commandEncoder.beginRenderPass(this.renderFullScreenPassDescriptor);
        // quadPassEncoder.setPipeline(this.quadPipeline);
        // quadPassEncoder.setVertexBuffers(0, [this.quadVerticesBuffer], [0]);
        // quadPassEncoder.setBindGroup(0, this.quadUniformBindGroup);
        // quadPassEncoder.draw(3, 1, 0, 0);
        // quadPassEncoder.endPass();

        this.device.getQueue().submit([commandEncoder.finish()]);
    }
}