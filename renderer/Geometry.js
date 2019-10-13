export default class Geometry {
    constructor(device) {
        this.device = device;
    }

    fromObjMesh(mesh) {
        this.fromData(
            new Float32Array(mesh.vertices),
            new Float32Array(mesh.vertexNormals),
            new Float32Array(mesh.textures),
            new Uint32Array(mesh.indices)
        );
    }

    fromData(vertices, normals, uvs, indices) {
        this.vertices = vertices;
        this.normals = normals;
        this.uvs = uvs;
        this.indices = indices;

        this.vertexInput = {
            indexFormat: "uint32",
            vertexBuffers: [
                {
                    stride: 4 * 3,
                    stepMode: "vertex",
                    attributeSet: [{
                        // position
                        shaderLocation: 0,
                        offset: 0,
                        format: "float3"
                    }]
                },
                {
                    stride: 4 * 3,
                    stepMode: "vertex",
                    attributeSet: [{
                        // normal
                        shaderLocation: 1,
                        offset: 0,
                        format: "float3"
                    }]
                },
                {
                    stride: 4 * 2,
                    stepMode: "vertex",
                    attributeSet: [{
                        // uv
                        shaderLocation: 2,
                        offset: 0,
                        format: "float2"
                    }]
                }
            ],
        };

        // Complex scene management shall update when in drawing list
        this.verticesBuffer = this.device.createBuffer({
            size: this.vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.verticesBuffer.setSubData(0, this.vertices);

        this.normalsBuffer = this.device.createBuffer({
            size: this.normals.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.normalsBuffer.setSubData(0, this.normals);

        this.uvsBuffer = this.device.createBuffer({
            size: this.uvs.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.uvsBuffer.setSubData(0, this.uvs);

        this.indicesBuffer = this.device.createBuffer({
            size: this.indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
        });
        this.indicesBuffer.setSubData(0, this.indices);
    }

    draw(passEncoder, instanceCount) {
        passEncoder.setVertexBuffer(0, this.verticesBuffer);
        passEncoder.setVertexBuffer(1, this.normalsBuffer);
        passEncoder.setVertexBuffer(2, this.uvsBuffer);
        passEncoder.setIndexBuffer(this.indicesBuffer);

        // indexed
        passEncoder.drawIndexed(this.indices.length, instanceCount, 0, 0, 0);
    }

    
}