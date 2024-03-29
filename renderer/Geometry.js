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

        this.vertexBuffers = [
            {
                arrayStride: Float32Array.BYTES_PER_ELEMENT * 3,
                attributes: [{
                    // position
                    shaderLocation: 0,
                    offset: 0,
                    format: "float32x3"
                }]
            },
            {
                arrayStride: Float32Array.BYTES_PER_ELEMENT * 3,
                attributes: [{
                    // normal
                    shaderLocation: 1,
                    offset: 0,
                    format: "float32x3"
                }]
            },
            {
                arrayStride: Float32Array.BYTES_PER_ELEMENT * 2,
                attributes: [{
                    // uv
                    shaderLocation: 2,
                    offset: 0,
                    format: "float32x2"
                }]
            }
          ];

        // Complex scene management shall update when in drawing list
        this.verticesBuffer = this.device.createBuffer({
            size: this.vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(
            this.verticesBuffer,
            0,
            this.vertices.buffer,
            this.vertices.byteOffset,
            this.vertices.byteLength
        );

        this.normalsBuffer = this.device.createBuffer({
            size: this.normals.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(
            this.normalsBuffer,
            0,
            this.normals.buffer,
            this.normals.byteOffset,
            this.normals.byteLength
        );

        this.uvsBuffer = this.device.createBuffer({
            size: this.uvs.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(
            this.uvsBuffer,
            0,
            this.uvs.buffer,
            this.uvs.byteOffset,
            this.uvs.byteLength
        );

        this.indicesBuffer = this.device.createBuffer({
            size: this.indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(
            this.indicesBuffer,
            0,
            this.indices.buffer,
            this.indices.byteOffset,
            this.indices.byteLength
        );
    }

    draw(passEncoder, instanceCount) {
        passEncoder.setVertexBuffer(0, this.verticesBuffer);
        passEncoder.setVertexBuffer(1, this.normalsBuffer);
        passEncoder.setVertexBuffer(2, this.uvsBuffer);
        passEncoder.setIndexBuffer(this.indicesBuffer, "uint32");

        // indexed
        passEncoder.drawIndexed(this.indices.length, instanceCount, 0, 0, 0);
    }

    
}