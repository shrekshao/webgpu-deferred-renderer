
if (!navigator.gpu) {
    document.getElementById('not-supported').style.display = 'block';
    alert('WebGPU not supported! Please visit webgpu.io to see the current implementation status.');
}

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

async function init() {
    /* Context, Device, SwapChain */
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice({});

    const glslangModule = await import('https://unpkg.com/@webgpu/glslang@0.0.7/web/glslang.js');
    const glslang = await glslangModule.default();

    const canvas = document.querySelector('canvas');

    const aspect = Math.abs(canvas.width / canvas.height);
    let projectionMatrix = mat4.create();
    mat4.perspective(projectionMatrix, (2 * Math.PI) / 5, -aspect, 1, 100.0);

    const context = canvas.getContext('gpupresent');

    const swapChain = context.configureSwapChain({
        device,
        format: "bgra8unorm"
    });

    const verticesBuffer = device.createBuffer({
        size: cubeVerticesArray.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    verticesBuffer.setSubData(0, cubeVerticesArray);

    const uniformsBindGroupLayout = device.createBindGroupLayout({
        entries: [{
            binding: 0,
            visibility: 1,
            type: "uniform-buffer"
        }]
    });

    /* Render Pipeline */
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [uniformsBindGroupLayout] });
    const pipeline = device.createRenderPipeline({
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

    const depthTexture = device.createTexture({
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
    });

    const renderPassDescriptor = {
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

    const matrixSize = 4 * 16;  // 4x4 matrix
    const offset = 256; // uniformBindGroup offset must be 256-byte aligned
    const uniformBufferSize = offset + matrixSize;

    const uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const uniformBindGroup1 = device.createBindGroup({
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

    const uniformBindGroup2 = device.createBindGroup({
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

    let modelMatrix1 = mat4.create();
    mat4.translate(modelMatrix1, modelMatrix1, vec3.fromValues(-2, 0, 0));
    let modelMatrix2 = mat4.create();
    mat4.translate(modelMatrix2, modelMatrix2, vec3.fromValues(2, 0, 0));
    let modelViewProjectionMatrix1 = mat4.create();
    let modelViewProjectionMatrix2 = mat4.create();
    let viewMatrix = mat4.create();
    mat4.translate(viewMatrix, viewMatrix, vec3.fromValues(0, 0, -7));

    let tmpMat41 = mat4.create();
    let tmpMat42 = mat4.create();

    function updateTransformationMatrix() {

        let now = Date.now() / 1000;

        mat4.rotate(tmpMat41, modelMatrix1, 1, vec3.fromValues(Math.sin(now), Math.cos(now), 0));
        mat4.rotate(tmpMat42, modelMatrix2, 1, vec3.fromValues(Math.cos(now), Math.sin(now), 0));

        mat4.multiply(modelViewProjectionMatrix1, viewMatrix, tmpMat41);
        mat4.multiply(modelViewProjectionMatrix1, projectionMatrix, modelViewProjectionMatrix1);
        mat4.multiply(modelViewProjectionMatrix2, viewMatrix, tmpMat42);
        mat4.multiply(modelViewProjectionMatrix2, projectionMatrix, modelViewProjectionMatrix2);
    }

    function frame() {
        updateTransformationMatrix();

        renderPassDescriptor.colorAttachments[0].attachment = swapChain.getCurrentTexture().createView();

        const commandEncoder = device.createCommandEncoder({});

        uniformBuffer.setSubData(0, modelViewProjectionMatrix1);
        uniformBuffer.setSubData(offset, modelViewProjectionMatrix2);
        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(pipeline);
        passEncoder.setVertexBuffers(0, [verticesBuffer], [0]);

        passEncoder.setBindGroup(0, uniformBindGroup1);
        passEncoder.draw(36, 1, 0, 0);

        passEncoder.setBindGroup(0, uniformBindGroup2);
        passEncoder.draw(36, 1, 0, 0);

        passEncoder.endPass();

        device.getQueue().submit([commandEncoder.finish()]);

        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

init();