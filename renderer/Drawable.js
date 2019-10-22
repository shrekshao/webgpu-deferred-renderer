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
                    size: 2 * 4 * 16    // this needs shader string stiching to fully make sense
                }
            }].concat(this.material.bindings)
        });
    }

    draw(passEncoder) {
        // TODO: this is not enough (might need bind group layout with dynamic offset and prestore all data in one pass)
        // Current implementation won't handle transform of multiple object properly
        passEncoder.setBindGroup(0, this.uniformBindGroup);

        this.geometry.draw(passEncoder, 1);
    }
}