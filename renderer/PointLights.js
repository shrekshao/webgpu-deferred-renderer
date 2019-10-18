const tmpVec3 = vec3.create();
const tmpVec4 = vec4.create();

export default class PointLights {
    // May use compute shader to simulate (particle) movement
    constructor() {
        this.numLights = 100;
        this.extentMin = vec3.fromValues(-14, -1, -6);
        this.extentMax = vec3.fromValues(14, 20, 6);
        const extent = vec3.create();
        vec3.sub(extent, this.extentMax, this.extentMin);

        // layout:
        // vec4 position;
        // vec3 color; float radius;
        this.lightDataStride = 64;  // offset 256 / 4 byte
        this.data = new Float32Array(this.numLights * this.lightDataStride);

        // init
        let offset = 0;
        for (let i = 0; i < this.numLights; i++) {
            offset = this.lightDataStride * i;
            //position
            for (let i = 0; i < 3; i++) {
                tmpVec4[i] = Math.random() * extent[i] + this.extentMin[i];
            }
            tmpVec4[3] = 1;
            this.setV4(this.data, offset, tmpVec4);
            // color
            tmpVec4[0] = Math.random() * 2;
            tmpVec4[1] = Math.random() * 2;
            tmpVec4[2] = Math.random() * 2;
            // radius
            tmpVec4[3] = 4.0;
            this.setV4(this.data, offset + 4, tmpVec4);
        }
    }

    setV3(array, offset, v3) {
        array[offset] = v3[0];
        array[offset + 1] = v3[1];
        array[offset + 2] = v3[2];
    }

    getV3(array, offset, v3) {
        v3[0] = array[offset];
        v3[1] = array[offset + 1];
        v3[2] = array[offset + 2];
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

    getPosition(i, v4) {
        this.getV4(this.data, this.lightDataStride * i, v4);
    }

    getColorAndRadius(i, v4) {
        this.getV4(this.data, this.lightDataStride * i + 4, v4);
    }

    update() {
        let o = 0;
        for (let i = 0; i < this.numLights; i++) {
            o = this.lightDataStride * i;
            this.getV3(this.data, o, tmpVec3);
            tmpVec3[1] += -0.1;
            if (tmpVec3[1] < this.extentMin[1]) {
                tmpVec3[1] = this.extentMax[1];
            }
            this.setV3(this.data, o, tmpVec3);
        }
    }
}