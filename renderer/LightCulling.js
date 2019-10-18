const tmpVec3 = vec3.create();
const tmpVec4 = vec4.create();

const computeShaderCode = `#version 450

struct LightData
{
    vec4 position;
    vec3 color;
    float radius;
};

layout(std430, set = 0, binding = 0) buffer ResultMatrix {
    LightData data[];
} lightArrays;

layout(std140, set = 1, binding = 0) uniform Uniforms {
    vec3 extentMin;
    vec3 extentMax;
} uniforms;

void main() {
    // Light position animating
    // Feel free to customize
    float y = lightArrays.data[gl_GlobalInvocationID.x].position.y;
    y = y < uniforms.extentMin.y ? uniforms.extentMax.y : y - 0.1;
    lightArrays.data[gl_GlobalInvocationID.x].position.y  = y;

    // TODO: Light culling
}
`;

export default class LightCulling {

    constructor(device, glslang) {
        this.device = device;
        this.glslang = glslang;
    }

    async init() {
        this.tileSize = 16;

        this.numLights = 500;
        this.extentMin = vec3.fromValues(-14, -1, -6);
        this.extentMax = vec3.fromValues(14, 20, 6);
        const extent = vec3.create();
        vec3.sub(extent, this.extentMax, this.extentMin);

        this.lightDataStride = 8;
        const bufferSizeInByte = 4 * this.lightDataStride * this.numLights;

        this.lightBufferSize = bufferSizeInByte;

        const [lightDataBuffer, arrayBuffer] = await this.device.createBufferMappedAsync({
            size: bufferSizeInByte,
            usage: GPUBufferUsage.STORAGE,
        });

        const lightData = new Float32Array(arrayBuffer);
        // const lightData = new Float32Array(this.lightDataStride * this.numLights);

        let offset = 0;
        for (let i = 0; i < this.numLights; i++) {
            offset = this.lightDataStride * i;
            //position
            for (let i = 0; i < 3; i++) {
                tmpVec4[i] = Math.random() * extent[i] + this.extentMin[i];
            }
            tmpVec4[3] = 1;
            this.setV4(lightData, offset, tmpVec4);
            // color
            tmpVec4[0] = Math.random() * 2;
            tmpVec4[1] = Math.random() * 2;
            tmpVec4[2] = Math.random() * 2;
            // radius
            tmpVec4[3] = 4.0;
            this.setV4(lightData, offset + 4, tmpVec4);
        }

        // lightData[0] = 0;
        // lightData[1] = 2;
        // lightData[2] = 0;
        // lightData[3] = 1;

        // lightData[8] = -5;
        // lightData[9] = 4;
        // lightData[10] = -2;
        // lightData[11] = 1;

        // new Float32Array(arrayBuffer).set(lightData);

        // Unmap buffer so that it can be used later for copy.
        lightDataBuffer.unmap();
        this.lightDataBuffer = lightDataBuffer;

        this.lightBufferBindGroupLayout = this.device.createBindGroupLayout({
            bindings: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                    type: "storage-buffer"
                },
            ]
        });

        this.lightBufferBindGroup = this.device.createBindGroup({
            layout: this.lightBufferBindGroupLayout,
            bindings: [
              {
                binding: 0,
                resource: {
                  buffer: lightDataBuffer
                }
              },
            ]
        });

        // Readonly is not supported yet
        // this.lightBufferReadOnlyBindGroupLayout = this.device.createBindGroupLayout({
        //     bindings: [
        //         {
        //             binding: 0,
        //             visibility: GPUShaderStage.COMPUTE,
        //             type: "readonly-storage-buffer"
        //         },
        //     ]
        // });

        // this.lightBufferReadOnlyBindgGroup = this.device.createBindGroup({
        //     layout: this.lightBufferReadOnlyBindGroupLayout,
        //     bindings: [
        //       {
        //         binding: 0,
        //         resource: {
        //           buffer: lightDataBuffer
        //         }
        //       },
        //     ]
        // });

        const extentUniformBufferSize = 4 * 4 * 2;
        const extentUniformBuffer = this.device.createBuffer({
            size: extentUniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        extentUniformBuffer.setSubData(0, this.extentMin);
        extentUniformBuffer.setSubData(16, this.extentMax);

        const uniformBufferBindGroupLayout = this.device.createBindGroupLayout({
            bindings: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    type: "uniform-buffer"
                },
            ]
        });

        this.extentUniformBindGroup = this.device.createBindGroup({
            layout: uniformBufferBindGroupLayout,
            bindings: [
              {
                binding: 0,
                resource: {
                  buffer: extentUniformBuffer
                }
              },
            ]
        });

        this.lightCullingComputePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({
              bindGroupLayouts: [this.lightBufferBindGroupLayout, uniformBufferBindGroupLayout]
            }),
            computeStage: {
              module: this.device.createShaderModule({
                code: this.glslang.compileGLSL(computeShaderCode, "compute")
              }),
              entryPoint: "main"
            }
        });
    }

    setV4(array, offset, v4) {
        array[offset] = v4[0];
        array[offset + 1] = v4[1];
        array[offset + 2] = v4[2];
        array[offset + 3] = v4[3];
    }

    getV4(array, offset, v4) {
        v4[0] = array[offset];
        v4[1] = array[offset + 1];
        v4[2] = array[offset + 2];
        v4[3] = array[offset + 3];
    }

    update() {
        const commandEncoder = this.device.createCommandEncoder();

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.lightCullingComputePipeline);
        passEncoder.setBindGroup(0, this.lightBufferBindGroup);
        passEncoder.setBindGroup(1, this.extentUniformBindGroup);
        passEncoder.dispatch(this.numLights);
        passEncoder.endPass();

        this.device.getQueue().submit([commandEncoder.finish()]);
    }
}