export default class Transform {
    constructor() {
        this.scale = vec3.fromValues(1, 1, 1);
        this.rotation = quat.create();
        this.translation = vec3.fromValues(0, 0, 0);

        this.modelMatrix = mat4.create();

        this.dirty = true;

        this.needUpdate();

        this.dirty = true;
    }

    setScale(s) {
        vec3.copy(this.scale, s);
        this.dirty = true;
    }

    setRotation(q) {
        quat.copy(this.rotation, q);
        this.dirty = true;
    }

    setTranslation(t) {
        vec3.copy(this.translation, t);
        this.dirty = true;
    }

    needUpdate() {
        if (this.dirty) {
            mat4.fromRotationTranslationScale(this.modelMatrix, this.rotation, this.translation, this.scale);
            this.dirty = false;
            return true;
        }
        return false;
    }

    getModelMatrix() {
        this.needUpdate();
        return this.modelMatrix;
    }

}