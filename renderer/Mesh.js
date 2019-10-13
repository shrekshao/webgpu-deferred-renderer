import { Transform } from "./Transform";

export default class Mesh {
    constructor(geometry, material) {
        this.geometry = geometry;
        this.material = material;

        this.transform = new Transform();
    }

    
}