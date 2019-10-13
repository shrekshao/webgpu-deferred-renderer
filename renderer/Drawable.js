// import Geometry from "./Geometry.js";
import Transform from "./Transform.js";

export default class Drawable {
    // aggregate of assets data
    // geometry, textures, etc.

    // DrawableInstanced
    // 

    // // user needs to make sure this fits current renderPipeline, bindGroup, etc.
    

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
                    size: 3 * 4 * 16    // this needs shader string stiching to fully make sense
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
        passEncoder.setBindGroup(0, this.uniformBindGroup);
        this.geometry.draw(passEncoder, 1);
    }
}