const vertexShaderBlinnPhongGLSL = `#version 450
layout(set = 0, binding = 0) uniform Uniforms {
    mat4 modelViewMatrix;
    mat4 projectionMatrix;
} uniforms;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

layout(location = 0) out vec4 fragPosition;
layout(location = 1) out vec4 fragNormal;
layout(location = 2) out vec2 fragUV;
// Metallic, Roughness, Emissive, Motion, etc.

void main() {
    fragPosition = vec4(position, 1);
    fragNormal = normalize(vec4(normal, 0));
    fragUV = uv;
    gl_Position = uniforms.projectionMatrix * uniforms.modelViewMatrix * fragPosition;
}
`;

const fragmentShaderGBufferGLSL = `#version 450
layout(set = 0, binding = 1) uniform sampler defaultSampler;
layout(set = 0, binding = 2) uniform texture2D albedoMap;
layout(set = 0, binding = 3) uniform texture2D normalMap;

layout(location = 0) in vec4 fragPosition;
layout(location = 1) in vec4 fragNormal;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outGBufferPosition;
layout(location = 1) out vec4 outGBufferNormal;
layout(location = 2) out vec4 outGBufferAlbedo;

vec3 applyNormalMap(vec3 geomnor, vec3 normap) {
    normap = normap * 2.0 - 1.0;
    vec3 up = normalize(vec3(0.001, 1, 0.001));
    vec3 surftan = normalize(cross(geomnor, up));
    vec3 surfbinor = cross(geomnor, surftan);
    return normap.y * surftan + normap.x * surfbinor + normap.z * geomnor;
}

void main() {
    outGBufferPosition = fragPosition;
    outGBufferNormal = vec4(applyNormalMap(fragNormal.xyz, texture(sampler2D(normalMap, defaultSampler), fragUV).rgb), 1);
    outGBufferAlbedo = texture(sampler2D(albedoMap, defaultSampler), fragUV);
}
`;

export default class WriteGBufferMaterial {
    static setup(device) {
        this.uniformsBindGroupLayout = device.createBindGroupLayout({
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

        this.vertexShaderGLSL = vertexShaderBlinnPhongGLSL;
        this.fragmentShaderGLSL = fragmentShaderGBufferGLSL;
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