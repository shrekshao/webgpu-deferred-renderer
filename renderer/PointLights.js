const tmpVec3 = vec3.create();

export default class PointLights {
    // May use compute shader to simulate (particle) movement
    constructor() {
        this.numLights = 50;
        this.extentMin = vec3.fromValues(-14, -1, -6);
        this.extentMax = vec3.fromValues(14, 15, 6);
        // this.extentMin = vec3.fromValues(-1, -1, -1);
        // this.extentMax = vec3.fromValues(1, 1, 1);
        // this.extentMin = vec3.fromValues(-14, -1, -6);
        // this.extentMax = vec3.fromValues(14, 20, 6);
        this.positions = new Float32Array(this.numLights * 3);
        this.colors = new Float32Array(this.numLights * 3);
        this.radius = new Float32Array(this.numLights);

        // this.velocity = new Float32Array(this.numLights);    // downwards

        // init

        const extent = vec3.create();
        vec3.sub(extent, this.extentMax, this.extentMin);

        

        // Math.seedrandom(0);


        let offset = 0;
        for (let i = 0; i < this.numLights; i++) {
            offset = 3 * i;
            //position
            for (let i = 0; i < 3; i++) {
                tmpVec3[i] = Math.random() * extent[i] + this.extentMin[i];
            }
            this.setV3(this.positions, offset, tmpVec3);
            // color
            tmpVec3[0] = Math.random() * 2;
            tmpVec3[1] = Math.random() * 2;
            tmpVec3[2] = Math.random() * 2;
            this.setV3(this.colors, offset, tmpVec3);
            this.radius[i] = 4.0;
        }


        // // test
        // this.positions[0] = 0;
        // this.positions[1] = 10;
        // this.positions[2] = 0;

        // this.positions[3] = 0;
        // this.positions[4] = 4;
        // this.positions[5] = 4;
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

    update() {
        let o = 0;
        for (let i = 0; i < this.numLights; i++) {
            o = 3 * i;
            this.getV3(this.positions, o, tmpVec3);
            tmpVec3[1] += -0.1;
            if (tmpVec3[1] < this.extentMin[1]) {
                tmpVec3[1] = this.extentMax[1];
            }
            this.setV3(this.positions, o, tmpVec3);
        }
    }
}