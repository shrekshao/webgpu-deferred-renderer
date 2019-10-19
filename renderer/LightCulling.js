import {replaceArray} from './utils/stringReplaceArray.js'

const tmpVec3 = vec3.create();
const tmpVec4 = vec4.create();
const tmpMat4 = mat4.create();


// const clearTileLightIdComputeShaderCode = `#version 450
// #define NUM_TILES $2

// struct TileLightIdData
// {
//     int count;
//     int lightId[NUM_LIGHTS];
// };

// layout(std430, set = 2, binding = 0) buffer TileLightIdBuffer {
//     TileLightIdData data[NUM_TILES];
// } tileLightId;

// void main()
// {
//     if (gl_GlobalInvocationID.x >= NUM_LIGHTS) return;
// }
// `;

const computeShaderCode = `#version 450

#define NUM_LIGHTS $1
#define NUM_TILES $2
#define TILE_COUNT_X $3
#define TILE_COUNT_Y $4
#define NUM_TILE_LIGHT_SLOT $5

struct LightData
{
    vec4 position;
    vec3 color;
    float radius;
};

layout(std430, set = 0, binding = 0) buffer LightDataBuffer {
    LightData data[];
} lightArrays;


layout(std140, set = 1, binding = 0) uniform Uniforms {
    vec3 extentMin;
    vec3 extentMax;

    // camera
    mat4 viewProjectionMatrix;

    // // Tile info
    // ivec4 screen;   //width, height

} uniforms;


struct TileLightIdData
{
    int count;
    int lightId[NUM_TILE_LIGHT_SLOT];
};

layout(std430, set = 2, binding = 0) buffer TileLightIdBuffer {
    TileLightIdData data[NUM_TILES];
} tileLightId;


// // Tile plane info
// // could be precomputed by another compute pass
// struct TileInfo
// {
//     // planes
// };

// layout(std430, set = 3, binding = 0) buffer Tiles {
//     TileInfo info[TILE_COUNT_X][TILE_COUNT_Y];
// } tiles;

void main() {

    if (gl_GlobalInvocationID.x >= NUM_LIGHTS) return;

    vec3 position = lightArrays.data[gl_GlobalInvocationID.x].position.xyz;
    // Light position animating
    // Feel free to customize
    position.y = position.y < uniforms.extentMin.y ? uniforms.extentMax.y : position.y - 0.1;
    lightArrays.data[gl_GlobalInvocationID.x].position.y  = position.y;

    // Light culling

    // TODO: view frustum culling for tile based, or aabb for cluster, etc.

    // now just do uniform grid assign to complete pipeline
    // make a heatmap

    vec4 p = uniforms.viewProjectionMatrix * vec4(position, 1);
    p /= p.w;   // in NDC

    if (p.x > 1 || p.x < -1 || p.y > 1 || p.y < -1 || p.z < -1 || p.z > 1) return;

    vec2 tileScale = vec2(2.0 / TILE_COUNT_X, 2.0 / TILE_COUNT_Y);

    ivec2 tileCoord = ivec2(floor( (p.xy - vec2(-1, -1)) / tileScale ));
    int tileId = tileCoord.x + tileCoord.y * TILE_COUNT_X;

    if (tileId < 0 || tileId >= NUM_TILES) return;

    int offset = atomicAdd(tileLightId.data[tileId].count, 1);

    if (offset >= NUM_TILE_LIGHT_SLOT) return;

    tileLightId.data[tileId].lightId[offset] = int(gl_GlobalInvocationID.x);
}
`;

function getCount(t, f) {
    return Math.floor( (t + f - 1) / f);
}

// function replaceArray(str, find, replace) {
//     for (var i = 0; i < find.length; i++) {
//         str = str.replace(find[i], replace[i]);
//     }
//     return str;
// };

export default class LightCulling {

    constructor(device, glslang) {
        this.device = device;
        this.glslang = glslang;
    }

    async init(canvas, camera) {
        this.camera = camera;

        this.numLights = 500;
        // this.numLights = 2;
        this.extentMin = vec3.fromValues(-14, -1, -6);
        this.extentMax = vec3.fromValues(14, 20, 6);
        const extent = vec3.create();
        vec3.sub(extent, this.extentMax, this.extentMin);

        this.lightDataStride = 8;
        const bufferSizeInByte = 4 * this.lightDataStride * this.numLights;

        this.lightBufferSize = bufferSizeInByte;

        const [lightDataGPUBuffer, arrayBuffer] = await this.device.createBufferMappedAsync({
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

        // // lightData[8] = 5;
        // // lightData[9] = 4;
        // // lightData[10] = -2;
        // // lightData[11] = 1;

        // lightData[8] = -5;
        // lightData[9] = 4;
        // lightData[10] = -2;
        // lightData[11] = 1;

        // new Float32Array(arrayBuffer).set(lightData);

        // Unmap buffer so that it can be used later for copy.
        lightDataGPUBuffer.unmap();
        this.lightDataGPUBuffer = lightDataGPUBuffer;

        this.storageBufferBindGroupLayout = this.device.createBindGroupLayout({
            bindings: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                    type: "storage-buffer"
                },
            ]
        });

        this.lightBufferBindGroup = this.device.createBindGroup({
            layout: this.storageBufferBindGroupLayout,
            bindings: [
              {
                binding: 0,
                resource: {
                  buffer: lightDataGPUBuffer
                }
              },
            ]
        });

        const uniformBufferSize = 4 * 4 * 2 + 4 * 16;
        const uniformBuffer = this.uniformBuffer = this.device.createBuffer({
            size: uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        uniformBuffer.setSubData(0, this.extentMin);
        uniformBuffer.setSubData(16, this.extentMax);
        

        const uniformBufferBindGroupLayout = this.device.createBindGroupLayout({
            bindings: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    type: "uniform-buffer"
                },
            ]
        });

        this.uniformBindGroup = this.device.createBindGroup({
            layout: uniformBufferBindGroupLayout,
            bindings: [
              {
                binding: 0,
                resource: {
                  buffer: uniformBuffer
                }
              },
            ]
        });


        // Tile buffer init

        this.tileSize = 16;
        this.tileCount = Int32Array.from([ getCount(canvas.width, this.tileSize), getCount(canvas.width, this.tileSize) ]);
        this.numTiles = this.tileCount[0] * this.tileCount[1];

        // numLights slot + 1 (for count)
        this.tileLightSlot = Math.ceil(this.numLights * 0.1);    // safe factor
        this.tileLightIdBufferSize = (this.tileLightSlot + 1) * this.numTiles;
        this.tileLightIDBufferSizeInByte = this.tileLightIdBufferSize * 4;
        // Easy approach
        // each tile has numLights (max) entry for light id
        // use atmoicAdd to trace offset for each tile (avoid implementing parallel reduction)
        const [tileLightIdGPUBuffer, tileLightIdArrayBuffer] = await this.device.createBufferMappedAsync({
            size: this.tileLightIDBufferSizeInByte,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // const tileLightIdData = new Float32Array(tileLightIdArrayBuffer);
        const tileLightIdData = this.tileLightIdData = new Int32Array(tileLightIdArrayBuffer);
        tileLightIdData.fill(0);

        tileLightIdGPUBuffer.unmap();
        this.tileLightIdGPUBuffer = tileLightIdGPUBuffer;

        this.tileLightIdBufferBindGroup = this.device.createBindGroup({
            layout: this.storageBufferBindGroupLayout,
            bindings: [
              {
                binding: 0,
                resource: {
                  buffer: tileLightIdGPUBuffer
                }
              },
            ]
        });


        // console.log(computeShaderCode.replace("$1", this.numLights));
        // console.log(
        //     replaceArray(computeShaderCode, ["$1", "$2"], [this.numLights, this.tileCount[0] * this.tileCount[1]])
        // );

        this.lightCullingComputePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({
              bindGroupLayouts: [this.storageBufferBindGroupLayout, uniformBufferBindGroupLayout, this.storageBufferBindGroupLayout]
            }),
            computeStage: {
              module: this.device.createShaderModule({
                code: this.glslang.compileGLSL(
                    replaceArray(computeShaderCode, ["$1", "$2", "$3", "$4", "$5"], [this.numLights, this.numTiles, this.tileCount[0], this.tileCount[1], this.tileLightSlot]),
                    "compute")
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
        mat4.multiply(tmpMat4, this.camera.projectionMatrix, this.camera.viewMatrix);
        this.uniformBuffer.setSubData(32, tmpMat4);

        this.tileLightIdGPUBuffer.setSubData(0, this.tileLightIdData);  // temp clear
        // this.tileLightIdGPUBuffer.setSubData(0, this.tileLightIdClearData);  // temp clear

        const commandEncoder = this.device.createCommandEncoder();

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.lightCullingComputePipeline);
        passEncoder.setBindGroup(0, this.lightBufferBindGroup);
        passEncoder.setBindGroup(1, this.uniformBindGroup);
        passEncoder.setBindGroup(2, this.tileLightIdBufferBindGroup);
        passEncoder.dispatch(this.numLights);
        passEncoder.endPass();

        this.device.getQueue().submit([commandEncoder.finish()]);
    }
}