const vertexShaderBlinnPhong = `
[[block]] struct Uniforms {
    modelViewMatrix : mat4x4<f32>;
    projectionMatrix : mat4x4<f32>;
};
[[group(0), binding(0)]] var<uniform> uniforms : Uniforms;

struct VertexOutput {
    [[builtin(position)]] Position : vec4<f32>;
    [[location(0)]] fragPosition: vec3<f32>;
    [[location(1)]] fragNormal: vec3<f32>;
    [[location(2)]] fragUV: vec2<f32>;
};

[[stage(vertex)]]
fn main([[location(0)]] position : vec3<f32>,
        [[location(1)]] normal : vec3<f32>,
        [[location(2)]] uv : vec2<f32>) -> VertexOutput {
var output : VertexOutput;
output.fragPosition = position;
output.Position = uniforms.projectionMatrix * uniforms.modelViewMatrix * vec4<f32>(position, 1.0);
output.fragNormal = normalize(normal);
output.fragUV = uv;
return output;
}
`;

const fragmentShaderGBuffer = `
struct GBufferOutput {
    [[location(0)]] position : vec4<f32>;
    [[location(1)]] normal : vec4<f32>;

    // Textures: diffuse color, specular color, smoothness, emissive etc. could go here
    [[location(2)]] albedo : vec4<f32>;
};

fn applyNormalMap(geomnor: vec3<f32>, normap: vec3<f32>) -> vec3<f32> {
    var nm: vec3<f32> = 2.0 * normap - vec3<f32>(1.0, 1.0, 1.0);
    var up: vec3<f32> = normalize(vec3<f32>(0.001, 1.0, 0.001));
    var surftan: vec3<f32> = normalize(cross(geomnor, up));
    var surfbinor: vec3<f32> = cross(geomnor, surftan);
    return nm.y * surftan + nm.x * surfbinor + nm.z * geomnor;
}

[[group(0), binding(1)]] var mySampler: sampler;
[[group(0), binding(2)]] var albedoMap: texture_2d<f32>;
[[group(0), binding(3)]] var normalMap: texture_2d<f32>;

[[stage(fragment)]]
fn main([[location(0)]] fragPosition: vec3<f32>,
        [[location(1)]] fragNormal: vec3<f32>,
        [[location(2)]] fragUV : vec2<f32>) -> GBufferOutput {
    var output : GBufferOutput;
    output.position = vec4<f32>(fragPosition, 1.0);
    output.normal = vec4<f32>(applyNormalMap(fragNormal, textureSample(normalMap, mySampler, fragUV).xyz), 1.0);
    output.albedo = textureSample(albedoMap, mySampler, fragUV);
    return output;
}
`;

export default class WriteGBufferMaterial {
    static setup(device) {
        this.uniformsBindGroupLayout = device.createBindGroupLayout({
            entries: [
                // concat with drawable uniform bindgroup
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: {
                        type: 'uniform',
                    },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: {
                        type: 'filtering',
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

        this.vertexShader = vertexShaderBlinnPhong;
        this.fragmentShader = fragmentShaderGBuffer;
    }

    constructor(sampler, albedoMap, normalMap) {

        this.uniformsBindGroupLayout = WriteGBufferMaterial.uniformsBindGroupLayout;

        // to concat
        this.bindings = [
            {
                binding: 1,
                resource: sampler,
            },
            {
                binding: 2,
                resource: albedoMap.createView(),
            },
            {
                binding: 3,
                resource: normalMap.createView(),
            },
        ];
    }
}