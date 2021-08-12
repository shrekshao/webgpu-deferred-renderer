// const vertexShaderBlinnPhongGLSL = `#version 450
// layout(set = 0, binding = 0) uniform Uniforms {
//     mat4 modelViewMatrix;
//     mat4 projectionMatrix;
// } uniforms;

// layout(location = 0) in vec3 position;
// layout(location = 1) in vec3 normal;
// layout(location = 2) in vec2 uv;

// layout(location = 0) out vec4 fragPosition;
// layout(location = 1) out vec4 fragNormal;
// layout(location = 2) out vec2 fragUV;
// // Metallic, Roughness, Emissive, Motion, etc.

// void main() {
//     fragPosition = vec4(position, 1);
//     fragNormal = normalize(vec4(normal, 0));
//     fragUV = uv;
//     gl_Position = uniforms.projectionMatrix * uniforms.modelViewMatrix * fragPosition;
// }
// `;

// const fragmentShaderGBufferGLSL = `#version 450
// layout(set = 0, binding = 1) uniform sampler defaultSampler;
// layout(set = 0, binding = 2) uniform texture2D albedoMap;
// layout(set = 0, binding = 3) uniform texture2D normalMap;

// layout(location = 0) in vec4 fragPosition;
// layout(location = 1) in vec4 fragNormal;
// layout(location = 2) in vec2 fragUV;

// layout(location = 0) out vec4 outGBufferPosition;
// layout(location = 1) out vec4 outGBufferNormal;
// layout(location = 2) out vec4 outGBufferAlbedo;

// vec3 applyNormalMap(vec3 geomnor, vec3 normap) {
//     normap = normap * 2.0 - 1.0;
//     vec3 up = normalize(vec3(0.001, 1, 0.001));
//     vec3 surftan = normalize(cross(geomnor, up));
//     vec3 surfbinor = cross(geomnor, surftan);
//     return normap.y * surftan + normap.x * surfbinor + normap.z * geomnor;
// }

// void main() {
//     outGBufferPosition = fragPosition;
//     outGBufferNormal = vec4(applyNormalMap(fragNormal.xyz, texture(sampler2D(normalMap, defaultSampler), fragUV).rgb), 1);
//     outGBufferAlbedo = texture(sampler2D(albedoMap, defaultSampler), fragUV);
// }
// `;

const vertexShaderBlinnPhong = `
[[block]] struct Uniforms {
    modelViewMatrix : mat4x4<f32>;
    projectionMatrix : mat4x4<f32>;
    // modelMatrix : mat4x4<f32>;
    // normalModelMatrix : mat4x4<f32>;
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
// output.Position = uniforms.viewProjectionMatrix * uniforms.modelMatrix * vec4<f32>(output.fragPosition, 1.0);
// output.fragNormal = normalize((uniforms.normalModelMatrix * vec4<f32>(normal, 1.0)).xyz);
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
    // output.albedo = textureSample(albedoMap, mySampler, vec2<f32>(fragUV.x, 1.0-fragUV.y));
    // output.albedo = vec4<f32>(fragUV.x, fragUV.y, 0.0, 1.0);
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
                    // type: "uniform-buffer"
                    buffer: {
                        type: 'uniform',
                        // minBindingSize: 4,
                      },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    // type: "sampler"
                    sampler: {
                        type: 'filtering',
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    // type: "sampled-texture",
                    // textureComponentType: "float"
                    texture: {
                        sampleType: 'float',
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.FRAGMENT,
                    // type: "sampled-texture",
                    // textureComponentType: "float"
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