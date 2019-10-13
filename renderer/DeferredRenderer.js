import Geometry from './Geometry.js';
import Camera from './Camera.js';
import BlinnPhongDeferredMaterial from './BlinnPhongDeferredMaterial.js';
import Drawable from './Drawable.js';


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
// #define POSITION_ID 0
// #define NORMAL_ID 1
// #define ALBEDOMAP_ID 2

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

let modelViewMatrix = mat4.create();
let tmpMat4 = mat4.create();


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
        const glslang = this.glslang = await glslangModule.default();

        const canvas = this.canvas;
        const context = canvas.getContext('gpupresent');

        this.swapChain = context.configureSwapChain({
            device,
            format: "bgra8unorm",
        });


        BlinnPhongDeferredMaterial.setup(device);


        const matrixSize = 4 * 16;  // 4x4 matrix
        // const offset = 256; // uniformBindGroup offset must be 256-byte aligned
        // const uniformBufferSize = offset + matrixSize;
        const uniformBufferSize = 3 * matrixSize;

        const uniformBuffer = this.uniformBuffer = device.createBuffer({
            size: uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        await this.setupScene(device);

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

        /* Render Pipeline */
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [uniformsBindGroupLayout] });
        const pipeline = this.pipeline = device.createRenderPipeline({
            layout: pipelineLayout,

            vertexStage: {
                module: device.createShaderModule({
                    // code: glslang.compileGLSL(vertexShaderGBufferGLSL, "vertex"),
                    code: glslang.compileGLSL(BlinnPhongDeferredMaterial.vertexShaderGLSL, "vertex"),
                }),
                entryPoint: "main"
            },
            fragmentStage: {
                module: device.createShaderModule({
                    // code: glslang.compileGLSL(fragmentShaderBlinnPhongGLSL, "fragment"),
                    // code: glslang.compileGLSL(fragmentShaderGBufferGLSL, "fragment"),
                    code: glslang.compileGLSL(BlinnPhongDeferredMaterial.fragmentShaderGLSL, "fragment"),
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
            // vertexInput: this.sponza.vertexInput,
            vertexInput: this.drawableLists[0].geometry.vertexInput,

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
        // Sample from depth is not supported either

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



        

        const samplerRepeat = this.sampler = device.createSampler({
            addressModeU: "repeat",
            addressModeV: "repeat",
            magFilter: "linear",
            minFilter: "linear",
            mipmapFilter: "linear"
        });

        // this.uniformBindGroupWriteGBuffer = device.createBindGroup({
        //     layout: uniformsBindGroupLayout,
        //     bindings: [
        //         {
        //             binding: 0,
        //             resource: {
        //                 buffer: uniformBuffer,
        //                 offset: 0,
        //                 // size: matrixSize
        //                 size: uniformBufferSize
        //             }
        //         },
        //         {
        //             binding: 1,
        //             resource: samplerRepeat,
        //         },
        //         {
        //             binding: 2,
        //             resource: this.albedoMap.createView(),
        //         },
        //         {
        //             binding: 3,
        //             resource: this.normalMap.createView(),
        //         },
        //     ],
        // });



        this.setupQuadPipeline();
        
    }

    setupQuadPipeline() {

        const device = this.device;
        const glslang = this.glslang;

        const quadVerticesBuffer = this.quadVerticesBuffer = device.createBuffer({
            size: fullScreenQuadArray.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        quadVerticesBuffer.setSubData(0, fullScreenQuadArray);

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

        this.renderFullScreenPassDescriptor = {
            colorAttachments: [{
                loadValue: {r: 0.0, g: 0.0, b: 0.0, a: 1.0},
                storeOp: "store",
            }],
        };

        const sampler = this.sampler = device.createSampler({
            magFilter: "linear",
            minFilter: "linear"
        });

        this.quadUniformBindGroup = this.device.createBindGroup({
            layout: this.quadUniformsBindGroupLayout,
            bindings: [
                {
                    binding: 0,
                    resource: sampler,
                },
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


    async loadModel(modelUrl, albedoUrl, normalUrl) {
        const device = this.device;
        const pModel = new Promise((resolve) => {
            OBJ.downloadMeshes({
                'obj': modelUrl
            }, resolve);
        });

        const pAlbedoMap = createTextureFromImage(device, albedoUrl, GPUTextureUsage.SAMPLED);
        const pNormalMap = normalUrl ? createTextureFromImage(device, normalUrl, GPUTextureUsage.SAMPLED) : null;

        //, dModel, dAlbedoMap, dNormalMap
        return Promise.all([pModel, pAlbedoMap, pNormalMap]).then((values) => {
            const meshes = values[0];

            // build mesh, drawable list here
            // const geometry = this.sponza = new Geometry(device);
            const geometry = new Geometry(device);
            geometry.fromObjMesh(meshes['obj']);

            const albedoMap = values[1];
            // console.log(albedoMap);

            const normalMap = values[2];
            // console.log('normalmap: ', normalMap);

            const sampler = device.createSampler({
                addressModeU: "repeat",
                addressModeV: "repeat",
                magFilter: "linear",
                minFilter: "linear",
                mipmapFilter: "linear"
            });
            const material = new BlinnPhongDeferredMaterial(sampler, albedoMap, normalMap);

            const d = new Drawable(device, geometry, material, this.uniformBuffer);

            this.drawableLists.push(d);
        });
    }


    async setupScene(device) {

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

        // const modelUrl = 'models/sponza.obj';
        // const albedoUrl = 'models/color.jpg';
        // const normalUrl = 'models/normal.png';

        await Promise.all([
            this.loadModel('models/sponza.obj', 'models/color.jpg', 'models/normal.png'),
            this.loadModel('models/di.obj', 'models/di.png', 'models/di-n.png'),
        ]);
    }

    // updateTransformationMatrix() {
    //     // let now = Date.now() / 1000;

    //     this.uniformBuffer.setSubData(64, this.camera.projectionMatrix);

    //     for (let i = 0; i < this.drawableLists.length; i++) {
    //         const o = this.drawableLists[i];
    //         // o.updateTransfrom(this.uniformBuffer);

    //         // if (o.transform.needUpdate()) {
    //         //     this.uniformBuffer.setSubData(0, o.transform.modelMatrix);
    //         // }

    //         this.uniformBuffer.setSubData(0, o.transform.modelMatrix);

    //         mat4.multiply(tmpMat4, this.camera.viewMatrix, o.transform.modelMatrix);
    //         mat4.invert(tmpMat4, tmpMat4);
    //         mat4.transpose(tmpMat4, tmpMat4);
    //         this.uniformBuffer.setSubData(128, tmpMat4);
    //     }

    //     // mat4.multiply(modelViewMatrix1, this.camera.viewMatrix, modelMatrix1);
    //     // mat4.invert(tmpMat41, modelViewMatrix1);
    //     // mat4.transpose(tmpMat41, tmpMat41); //normal matrix
    // }


    frame() {
        // updateTransformationMatrix();
        // this.updateTransformationMatrix();

        const commandEncoder = this.device.createCommandEncoder({});

        // this.uniformBuffer.setSubData(0, modelViewMatrix1);

        

        // this.uniformBuffer.setSubData(offset, modelViewProjectionMatrix2);
        // this.uniformBuffer.setSubData(256, modelViewProjectionMatrix2);
        const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);
        passEncoder.setPipeline(this.pipeline);


        // passEncoder.setBindGroup(0, this.uniformBindGroupWriteGBuffer);

        this.uniformBuffer.setSubData(64, this.camera.projectionMatrix);

        for (let i = 0; i < this.drawableLists.length; i++) {
            const o = this.drawableLists[i];

            

            mat4.multiply(tmpMat4, this.camera.viewMatrix, o.transform.modelMatrix);
            this.uniformBuffer.setSubData(0, tmpMat4);
            mat4.invert(tmpMat4, tmpMat4);
            mat4.transpose(tmpMat4, tmpMat4);
            this.uniformBuffer.setSubData(128, tmpMat4);
            this.uniformBuffer.setSubData(128, tmpMat4);

            o.draw(passEncoder);
        }

        passEncoder.endPass();


        // render full screen quad

        const swapChainTexture = this.swapChain.getCurrentTexture();

        this.debugViewUniformBuffer.setSubData(0, new Float32Array([this.debugViewOffset]));
        // this.debugViewUniformBuffer.setSubData(0, new Float32Array([this.debugViewOffset, 0, 0, 0]));
        // this.debugViewUniformBuffer.setSubData(0, new Float32Array([0.5, 0, 0, 0]));

        this.renderFullScreenPassDescriptor.colorAttachments[0].attachment = swapChainTexture.createView();
        const quadPassEncoder = commandEncoder.beginRenderPass(this.renderFullScreenPassDescriptor);
        quadPassEncoder.setPipeline(this.quadPipeline);
        // quadPassEncoder.setVertexBuffers(0, [this.quadVerticesBuffer], [0]);
        quadPassEncoder.setVertexBuffer(0, this.quadVerticesBuffer);
        quadPassEncoder.setBindGroup(0, this.quadUniformBindGroup);
        quadPassEncoder.setBindGroup(1, this.debugViewUniformBindGroup);
        quadPassEncoder.draw(6, 1, 0, 0);
        quadPassEncoder.endPass();

        this.device.getQueue().submit([commandEncoder.finish()]);
    }
}