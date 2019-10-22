import {replaceArray} from './utils/stringReplaceArray.js'

const tmpVec3 = vec3.create();
const tmpVec4 = vec4.create();
const tmpMat4 = mat4.create();

const computeShaderCode = `#version 450

#define NUM_LIGHTS $1
#define NUM_TILES $2
#define TILE_COUNT_X $3
#define TILE_COUNT_Y $4
#define NUM_TILE_LIGHT_SLOT $5

#define TILE_SIZE $6

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
    mat4 viewMatrix;
    mat4 projectionMatrix;

    // Tile info
    vec4 fullScreenSize;    // width, height

} uniforms;


struct TileLightIdData
{
    int count;
    int lightId[NUM_TILE_LIGHT_SLOT];
};

layout(std430, set = 2, binding = 0) buffer TileLightIdBuffer {
    TileLightIdData data[NUM_TILES];
} tileLightId;


// TODO: tile/cluster frustum plane could be generated using another compute pass
// for better performance. The new storage buffer could look like this
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
    // Implementation here is Tiled without per tile min-max depth
    // You could also implement cluster culling
    // Feel free to add more compute passes if necessary

    // some math reference: http://www.txutxi.com/?p=444

    mat4 M = uniforms.projectionMatrix;

    float viewNear = - M[3][2] / ( -1.0 + M[2][2]);
    float viewFar = - M[3][2] / (1.0 + M[2][2]);

    vec4 lightPos = uniforms.viewMatrix * vec4(position, 1);
    lightPos /= lightPos.w;

    float lightRadius = lightArrays.data[gl_GlobalInvocationID.x].radius;

    vec4 boxMin = lightPos - vec4( vec3(lightRadius), 0.0);
    vec4 boxMax = lightPos + vec4( vec3(lightRadius), 0.0);

    // vec4 frustumPlanes[4];
    vec4 frustumPlanes[6];

    frustumPlanes[4] = vec4(0.0, 0.0, -1.0, viewNear);    // near
    frustumPlanes[5] = vec4(0.0, 0.0, 1.0, -viewFar);    // far

    for (int y = 0; y < TILE_COUNT_Y; y++) {
        for (int x = 0; x < TILE_COUNT_X; x++) {

            ivec2 tilePixel0Idx = ivec2(x * TILE_SIZE, y * TILE_SIZE);

            // tile position in NDC space
            vec2 floorCoord = 2.0 * vec2(tilePixel0Idx) / uniforms.fullScreenSize.xy - vec2(1.0);  // -1, 1
            vec2 ceilCoord = 2.0 * vec2(tilePixel0Idx + ivec2(TILE_SIZE)) / uniforms.fullScreenSize.xy - vec2(1.0);  // -1, 1

            vec2 viewFloorCoord = vec2( (- viewNear * floorCoord.x - M[2][0] * viewNear) / M[0][0] , (- viewNear * floorCoord.y - M[2][1] * viewNear) / M[1][1] );
            vec2 viewCeilCoord = vec2( (- viewNear * ceilCoord.x - M[2][0] * viewNear) / M[0][0] , (- viewNear * ceilCoord.y - M[2][1] * viewNear) / M[1][1] );

            frustumPlanes[0] = vec4(1.0, 0.0, - viewFloorCoord.x / viewNear, 0.0);       // left
            frustumPlanes[1] = vec4(-1.0, 0.0, viewCeilCoord.x / viewNear, 0.0);   // right
            frustumPlanes[2] = vec4(0.0, 1.0, - viewFloorCoord.y / viewNear, 0.0);       // bottom
            frustumPlanes[3] = vec4(0.0, -1.0, viewCeilCoord.y / viewNear, 0.0);   // top

            float dp = 0.0;     //dot product

            for (int i = 0; i < 6; i++)
            {
                dp += min(0.0, dot(
                    vec4( 
                        frustumPlanes[i].x > 0.0 ? boxMax.x : boxMin.x, 
                        frustumPlanes[i].y > 0.0 ? boxMax.y : boxMin.y, 
                        frustumPlanes[i].z > 0.0 ? boxMax.z : boxMin.z, 
                        1.0), 
                    frustumPlanes[i]));
            }

            if (dp >= 0.0) {
                // overlapping
                int tileId = x + y * TILE_COUNT_X;

                if (tileId < 0 || tileId >= NUM_TILES) continue;
            
                int offset = atomicAdd(tileLightId.data[tileId].count, 1);
            
                if (offset >= NUM_TILE_LIGHT_SLOT) continue;
            
                tileLightId.data[tileId].lightId[offset] = int(gl_GlobalInvocationID.x);
            }
        }
    }
}
`;

function getCount(t, f) {
    return Math.floor( (t + f - 1) / f);
}

export default class LightCulling {

    constructor(device, glslang) {
        this.device = device;
        this.glslang = glslang;
    }

    async init(canvas, camera) {
        this.camera = camera;

        // this.numLights = 4096;
        this.numLights = 2048;
        // this.numLights = 500;
        // this.numLights = 15;
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
            // tmpVec4[3] = 4.0;
            tmpVec4[3] = 2.0;
            this.setV4(lightData, offset + 4, tmpVec4);
        }

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

        const uniformBufferSize = 4 * 4 * 2 + 4 * 16 * 2 + 4 * 4;
        const uniformBuffer = this.uniformBuffer = this.device.createBuffer({
            size: uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        uniformBuffer.setSubData(0, this.extentMin);
        uniformBuffer.setSubData(16, this.extentMax);
        vec4.set(tmpVec4, canvas.width, canvas.height, 0, 0);
        uniformBuffer.setSubData(160, tmpVec4);
        

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
        const safeFactor = 0.1;
        // This is a magic number.
        this.tileLightSlot = Math.max( 127, Math.ceil(this.numLights * safeFactor / 128) * 128 - 1 );
        this.tileLightIdBufferSize = (this.tileLightSlot + 1) * this.numTiles;
        this.tileLightIDBufferSizeInByte = this.tileLightIdBufferSize * 4;

        console.log('Lights: ' + this.numLights);
        console.log('Tile light slots: ' + this.tileLightSlot);
        console.log('tiles ' + this.numTiles);
        // Easy approach
        // use atmoicAdd to trace offset for each tile's light id array
        // (avoid implementing parallel reduction)
        const [tileLightIdGPUBuffer, tileLightIdArrayBuffer] = await this.device.createBufferMappedAsync({
            size: this.tileLightIDBufferSizeInByte,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

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

        this.lightCullingComputePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({
              bindGroupLayouts: [this.storageBufferBindGroupLayout, uniformBufferBindGroupLayout, this.storageBufferBindGroupLayout]
            }),
            computeStage: {
              module: this.device.createShaderModule({
                code: this.glslang.compileGLSL(
                    replaceArray(computeShaderCode, ["$1", "$2", "$3", "$4", "$5", "$6"], [this.numLights, this.numTiles, this.tileCount[0], this.tileCount[1], this.tileLightSlot, this.tileSize]),
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
        this.uniformBuffer.setSubData(32, this.camera.viewMatrix);
        // projectionMatrix generated by gl-matrix is in right-handed coordinates.
        // we flipped it on y to make it work with the webgpu left handed coordinate system.
        // we reset it here to make sure the light culling math are all done in right handed system.
        mat4.scale(tmpMat4, this.camera.projectionMatrix, vec3.fromValues(1, -1, 1));
        this.uniformBuffer.setSubData(96, tmpMat4);

        // clear
        this.tileLightIdGPUBuffer.setSubData(0, this.tileLightIdData);

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