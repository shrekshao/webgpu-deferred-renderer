// import Geometry from "./Geometry.js";
import Transform from "./Transform.js";

export default class Drawable {
    // aggregate of assets data
    // geometry, textures, etc.

    constructor(device, geometry, material, uniformBuffer) {
        this.geometry = geometry;
        this.material = material;

        this.transform = new Transform();

        this.uniformBindGroup = device.createBindGroup({
            layout: this.material.uniformsBindGroupLayout,
            bindings: [{
                binding: 0,
                resource: {
                    buffer: uniformBuffer,
                    offset: 0,
                    // size: this.transform.modelMatrix.byteSize    // size of a mat4
                    size: 2 * 4 * 16    // this needs shader string stiching to fully make sense
                }
            }].concat(this.material.bindings)
        });
    }

    // updateTransform(uniformBuffer) {
    //     if (this.transform.needUpdate()) {
    //         uniformBuffer.setSubData(0, this.transform.modelMatrix);
    //     }
    // }

    draw(passEncoder) {
        // TODO: this is not enough (need dynamic bind group layout and prestore all data in one pass)
        passEncoder.setBindGroup(0, this.uniformBindGroup);

        this.geometry.draw(passEncoder, 1);
    }
}