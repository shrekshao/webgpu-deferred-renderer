import {replaceArray} from './utils/stringReplaceArray.js'

const tmpVec3 = vec3.create();
const tmpVec4 = vec4.create();
const tmpMat4 = mat4.create();

const computeShader = `
struct LightData {
    position : vec4<f32>;
    color : vec3<f32>;
    radius : f32;
};
[[block]] struct LightsBuffer {
    lights: array<LightData>;
};
[[group(0), binding(0)]] var<storage, read_write> lightsBuffer: LightsBuffer;

struct TileLightIdData {
    count: atomic<u32>;
    lightId: array<u32, $NUM_TILE_LIGHT_SLOT>;
};
[[block]] struct Tiles {
    data: array<TileLightIdData, $NUM_TILES>;
};
[[group(1), binding(0)]] var<storage, read_write> tileLightId: Tiles;
  
[[block]] struct Config {
    numLights : u32;

    numTiles : u32;
    tileCountX : u32;
    tileCountY : u32;
    numTileLightSlot : u32;
    tileSize : u32;
};
[[group(2), binding(0)]] var<uniform> config: Config;

[[block]] struct Uniforms {
    min : vec4<f32>;
    max : vec4<f32>;

    // camera
    viewMatrix : mat4x4<f32>;
    projectionMatrix : mat4x4<f32>;

    // Tile info
    fullScreenSize : vec4<f32>;    // width, height
};
[[group(3), binding(0)]] var<uniform> uniforms: Uniforms;

[[stage(compute), workgroup_size(64, 1, 1)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
    var index = GlobalInvocationID.x;
    if (index >= config.numLights) {
        return;
    }

    // Light position updating
    lightsBuffer.lights[index].position.y = lightsBuffer.lights[index].position.y - 0.1 + 0.001 * (f32(index) - 64.0 * floor(f32(index) / 64.0));
  
    if (lightsBuffer.lights[index].position.y < uniforms.min.y) {
        lightsBuffer.lights[index].position.y = uniforms.max.y;
    }

    // Light culling
    // Implementation here is Tiled without per tile min-max depth
    // You could also implement cluster culling
    // Feel free to add more compute passes if necessary

    // some math reference: http://www.txutxi.com/?p=444
    var M: mat4x4<f32> = uniforms.projectionMatrix;

    var viewNear: f32 = - M[3][2] / ( -1.0 + M[2][2]);
    var viewFar: f32 = - M[3][2] / (1.0 + M[2][2]);

    var lightPos: vec4<f32> = uniforms.viewMatrix * lightsBuffer.lights[index].position;
    lightPos = lightPos / lightPos.w;

    var lightRadius: f32 = lightsBuffer.lights[index].radius;

    var boxMin: vec4<f32> = lightPos - vec4<f32>(vec3<f32>(lightRadius), 0.0);
    var boxMax: vec4<f32> = lightPos + vec4<f32>(vec3<f32>(lightRadius), 0.0);

    var frustumPlanes: array<vec4<f32>, 6>;
    frustumPlanes[4] = vec4<f32>(0.0, 0.0, -1.0, viewNear);    // near
    frustumPlanes[5] = vec4<f32>(0.0, 0.0, 1.0, -viewFar);    // far

    let TILE_SIZE: i32 = $TILE_SIZE;
    let TILE_COUNT_X: i32 = $TILE_COUNT_X;
    let TILE_COUNT_Y: i32 = $TILE_COUNT_Y;
    for (var y : i32 = 0; y < TILE_COUNT_Y; y = y + 1) {
        for (var x : i32 = 0; x < TILE_COUNT_X; x = x + 1) {
            var tilePixel0Idx : vec2<i32> = vec2<i32>(x * TILE_SIZE, y * TILE_SIZE);

            // tile position in NDC space
            var floorCoord: vec2<f32> = 2.0 * vec2<f32>(tilePixel0Idx) / uniforms.fullScreenSize.xy - vec2<f32>(1.0);  // -1, 1
            var ceilCoord: vec2<f32> = 2.0 * vec2<f32>(tilePixel0Idx + vec2<i32>(TILE_SIZE)) / uniforms.fullScreenSize.xy - vec2<f32>(1.0);  // -1, 1

            var viewFloorCoord: vec2<f32> = vec2<f32>( (- viewNear * floorCoord.x - M[2][0] * viewNear) / M[0][0] , (- viewNear * floorCoord.y - M[2][1] * viewNear) / M[1][1] );
            var viewCeilCoord: vec2<f32> = vec2<f32>( (- viewNear * ceilCoord.x - M[2][0] * viewNear) / M[0][0] , (- viewNear * ceilCoord.y - M[2][1] * viewNear) / M[1][1] );

            frustumPlanes[0] = vec4<f32>(1.0, 0.0, - viewFloorCoord.x / viewNear, 0.0);       // left
            frustumPlanes[1] = vec4<f32>(-1.0, 0.0, viewCeilCoord.x / viewNear, 0.0);   // right
            frustumPlanes[2] = vec4<f32>(0.0, 1.0, - viewFloorCoord.y / viewNear, 0.0);       // bottom
            frustumPlanes[3] = vec4<f32>(0.0, -1.0, viewCeilCoord.y / viewNear, 0.0);   // top

            var dp: f32 = 0.0;  // dot product

            for (var i: u32 = 0u; i < 6u; i = i + 1u)
            {
                var p: vec4<f32>;
                if (frustumPlanes[i].x > 0.0) {
                    p.x = boxMax.x;
                } else {
                    p.x = boxMin.x;
                }
                if (frustumPlanes[i].y > 0.0) {
                    p.y = boxMax.y;
                } else {
                    p.y = boxMin.y;
                }
                if (frustumPlanes[i].z > 0.0) {
                    p.z = boxMax.z;
                } else {
                    p.z = boxMin.z;
                }
                p.w = 1.0;
                dp = dp + min(0.0, dot(p, frustumPlanes[i]));
            }

            if (dp >= 0.0) {
                // light is overlapping with the tile
                var tileId: u32 = u32(x + y * TILE_COUNT_X);
                if (tileId < 0u || tileId >= config.numTiles) {
                    continue;
                }
                var offset: u32 = atomicAdd(&(tileLightId.data[tileId].count), 1u);
                if (offset >= config.numTileLightSlot) {
                    continue;
                }
                tileLightId.data[tileId].lightId[offset] = GlobalInvocationID.x;
            }
        }
    }
}
  
`;

function getCount(t, f) {
    return Math.floor( (t + f - 1) / f);
}

export default class LightCulling {

    constructor(device) {
        this.device = device;
    }

    async init(canvas, camera) {
        this.camera = camera;

        // this.numLights = 4096;
        this.numLights = 2048;
        // this.numLights = 512;
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

        const lightDataGPUBuffer = this.device.createBuffer({
            size: bufferSizeInByte,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
        });

        const lightData = new Float32Array(lightDataGPUBuffer.getMappedRange());

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
            tmpVec4[3] = 2.0;
            this.setV4(lightData, offset + 4, tmpVec4);
        }

        // Unmap buffer so that it can be used later for copy.
        lightDataGPUBuffer.unmap();
        this.lightDataGPUBuffer = lightDataGPUBuffer;

        this.storageBufferBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: "storage"
                    }
                },
            ]
        });

        const uniformBufferSize = 4 * 4 * 2 + 4 * 16 * 2 + 4 * 4;
        const uniformBuffer = this.uniformBuffer = this.device.createBuffer({
            size: uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(
            uniformBuffer,
            0,
            this.extentMin.buffer,
            this.extentMin.byteOffset,
            this.extentMin.byteLength,
        );
        this.device.queue.writeBuffer(
            uniformBuffer,
            16,
            this.extentMax.buffer,
            this.extentMax.byteOffset,
            this.extentMax.byteLength,
        );
        vec4.set(tmpVec4, canvas.width, canvas.height, 0, 0);
        this.device.queue.writeBuffer(
            uniformBuffer,
            160,
            tmpVec4.buffer,
            tmpVec4.byteOffset,
            tmpVec4.byteLength,
        );

        // Tile buffer init

        this.tileSize = 16;
        this.tileCount = Int32Array.from([ getCount(canvas.width, this.tileSize), getCount(canvas.width, this.tileSize) ]);
        this.numTiles = this.tileCount[0] * this.tileCount[1];

        // numLights slot + 1 (for count)
        const safeFactor = 0.1;
        // This is a magic number.
        this.tileLightSlot = Math.max( 127, Math.ceil(this.numLights * safeFactor / 128) * 128 - 1 );
        this.tileLightIdBufferSize = (this.tileLightSlot + 1) * this.numTiles;
        this.tileLightIDBufferSizeInByte = this.tileLightIdBufferSize * Uint32Array.BYTES_PER_ELEMENT;

        console.log('Lights: ' + this.numLights);
        console.log('Tile light slots: ' + this.tileLightSlot);
        console.log('tiles ' + this.numTiles);
        // Easy approach
        // use atmoicAdd to trace offset for each tile's light id array
        // (avoid implementing parallel reduction)

        const tileLightIdGPUBuffer = this.device.createBuffer({
            size: this.tileLightIDBufferSizeInByte,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.tileLightIdGPUBuffer = tileLightIdGPUBuffer;

        this.tileLightIdZeroData = new Uint32Array(this.tileLightIdBufferSize);
        this.tileLightIdZeroData.fill(0);

        this.lightCullingComputePipeline = this.device.createComputePipeline({
            compute: {
              module: this.device.createShaderModule({
                code: replaceArray(computeShader,
                    ['$NUM_TILE_LIGHT_SLOT', '$NUM_TILES', '$TILE_COUNT_Y', '$TILE_COUNT_X', '$TILE_SIZE'],
                    [this.tileLightSlot, this.numTiles, this.tileCount[1], this.tileCount[0], this.tileSize]
                )
              }),
              entryPoint: "main"
            }
        });

        const configUniformBuffer = (() => {
            const buffer = this.device.createBuffer({
                size: Uint32Array.BYTES_PER_ELEMENT * 6,
                mappedAtCreation: true,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            const v = new Uint32Array(buffer.getMappedRange())
            v[0] = this.numLights;
            v[1] = this.numTiles;
            v[2] = this.tileCount[0];
            v[3] = this.tileCount[1];
            v[4] = this.tileLightSlot;
            v[5] = this.tileSize;
            buffer.unmap();
            return buffer;
        })();
        this.configUniformBuffer = configUniformBuffer;

        this.lightBufferBindGroup = this.device.createBindGroup({
            layout: this.lightCullingComputePipeline.getBindGroupLayout(0),
            entries: [
              {
                binding: 0,
                resource: {
                  buffer: lightDataGPUBuffer
                }
              },
            ]
        });
        this.tileLightIdBufferBindGroup = this.device.createBindGroup({
            layout: this.lightCullingComputePipeline.getBindGroupLayout(1),
            entries: [
              {
                binding: 0,
                resource: {
                  buffer: tileLightIdGPUBuffer,
                }
              },
            ]
        });

        this.configBindGroup = this.device.createBindGroup({
            layout: this.lightCullingComputePipeline.getBindGroupLayout(2),
            entries: [
              {
                binding: 0,
                resource: {
                  buffer: configUniformBuffer
                }
              },
            ]
        });

        this.uniformBindGroup = this.device.createBindGroup({
            layout: this.lightCullingComputePipeline.getBindGroupLayout(3),
            entries: [
              {
                binding: 0,
                resource: {
                  buffer: uniformBuffer
                }
              },
            ]
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
        this.device.queue.writeBuffer(
            this.uniformBuffer,
            32,
            this.camera.viewMatrix.buffer,
            this.camera.viewMatrix.byteOffset,
            this.camera.viewMatrix.byteLength
          );
        // my shader math used to work with flipped projection Matrix. I forgot why.
        // just don't touch...
        mat4.scale(tmpMat4, this.camera.projectionMatrix, vec3.fromValues(1, -1, 1));
        this.device.queue.writeBuffer(
            this.uniformBuffer,
            96,
            tmpMat4.buffer,
            tmpMat4.byteOffset,
            tmpMat4.byteLength
          );

        // clear to 0
        this.device.queue.writeBuffer(
            this.tileLightIdGPUBuffer,
            0,
            this.tileLightIdZeroData.buffer,
            this.tileLightIdZeroData.byteOffset,
            this.tileLightIdZeroData.byteLength
          );

        const commandEncoder = this.device.createCommandEncoder();

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.lightCullingComputePipeline);
        passEncoder.setBindGroup(0, this.lightBufferBindGroup);
        passEncoder.setBindGroup(1, this.tileLightIdBufferBindGroup);
        passEncoder.setBindGroup(2, this.configBindGroup);
        passEncoder.setBindGroup(3, this.uniformBindGroup);
        passEncoder.dispatch(this.numLights);
        passEncoder.endPass();

        this.device.queue.submit([commandEncoder.finish()]);
    }
}