export default class Camera {

    constructor (canvas) {

        let eulerX = 0;
        let eulerY = -Math.PI / 2;

        let mouseDown = false;
        let mouseButtonId = 0;
        let lastMouseY = 0;
        let lastMouseX = 0;

        let zero = vec3.fromValues(0, 0, 0);
        let forward = vec3.fromValues(0, 0, 1);
        let tmpVec3 = vec3.create();

        // this.eye = vec3.fromValues(0, 0.5, 0);
        // this.eye = vec3.fromValues(1.5, 0.5, 0.5);
        this.eye = vec3.fromValues(4.5, 0.5, 0.5);
        this.view = vec3.fromValues(-1, 0, 0);
        this.up = vec3.fromValues(0, 1, 0);

        this.center = vec3.create();
        vec3.add(this.center, this.eye, this.view);


        this.viewMatrix = mat4.create();
        mat4.lookAt(this.viewMatrix, this.eye, this.center, this.up);
        this.projectionMatrix = mat4.create();

        canvas.oncontextmenu = function (e) {
            e.preventDefault();
        };
        const aspect = Math.abs(canvas.width / canvas.height);
        
        mat4.perspective(this.projectionMatrix, (2 * Math.PI) / 5, aspect, 0.1, 100.0);

        // workaround for left handness
        // together with 'cw' being front face
        mat4.scale(this.projectionMatrix, this.projectionMatrix, vec3.fromValues(1, -1, 1));

        canvas.onmousedown = function(event) {
            mouseDown = true;
            mouseButtonId = event.which;
            lastMouseY = event.clientY;
            lastMouseX = event.clientX;
        };
        window.onmouseup = function(event) {
            mouseDown = false;
        };
        window.onmousemove = function(event) {
            if(!mouseDown) {
                return;
            }
            let newY = event.clientY;
            let newX = event.clientX;
            
            let deltaY = newY - lastMouseY;
            let deltaX = newX - lastMouseX;
    
            switch(mouseButtonId) {
                case 1:
                // left: rotation
                eulerX += deltaY * 0.01;
                eulerY += -deltaX * 0.01;
                vec3.rotateX(this.view, forward, zero, eulerX);
                vec3.rotateY(this.view, this.view, zero, eulerY);
                break;
                case 3:
                // right: panning
                vec3.scale(tmpVec3, this.up, -deltaY * 0.01);
                vec3.add(this.eye, this.eye, tmpVec3);
                vec3.cross(tmpVec3, this.view, this.up);
                vec3.scale(tmpVec3, tmpVec3, deltaX * 0.01);
                vec3.add(this.eye, this.eye, tmpVec3);
                break;
            }
            
            lastMouseY = newY;
            lastMouseX = newX;
            
            vec3.add(this.center, this.eye, this.view);
            mat4.lookAt(this.viewMatrix, this.eye, this.center, this.up);


        }.bind(this);
        window.onwheel = function(event) {
            // panning
            // vec3.scale(tmpVec3, this.view, delta * 0.01);
            vec3.scale(tmpVec3, this.view, event.deltaY * 0.01);
            vec3.add(this.eye, this.eye, tmpVec3);

            vec3.add(this.center, this.eye, this.view);
            mat4.lookAt(this.viewMatrix, this.eye, this.center, this.up);
        }.bind(this);
    }

    getPosition() {
        return this.eye;
    }
}