#include "window.hpp"
#include <iostream>

CGLWindow::CGLWindow(CGLCamera *camera, int width, int height){
    this->camera = camera;
    this->width = width;
    this->height = height;
}

int CGLWindow::init(){
    this->display = XOpenDisplay(getenv("DISPLAY"));
	if (display == NULL) {
		printf("Couldn't open display.\n");
		return -1;
	}

    int screen = DefaultScreen(display);
    this->rootWindow = DefaultRootWindow(display);
    this->window = XCreateSimpleWindow(display, rootWindow, 0, 0, width, height, 1, BlackPixel(display, screen), WhitePixel(display, screen));

    XMapWindow(display, window);
    XStoreName(display, window, "CPP Graphics");

    return 0;
}

void CGLWindow::show(){
    XSelectInput(display, window, KeyPressMask | KeyReleaseMask | ExposureMask);
 
	XEvent ev;
    this->gc = XCreateGC(display, rootWindow, 0, NULL);
    float xSpeed = 0;
    float ySpeed = 0;
    float zSpeed = 0;
    float yawSpeed = 0;
    float pitchSpeed = 0;
    XAutoRepeatOff(display);

	while (True) {

        while(XPending(display)) { //Repeats until all events are computed
            XEvent KeyEvent;
            XNextEvent(display,&KeyEvent); //Gets exactly one event
            if(KeyEvent.type==KeyPress) {
                auto key =KeyEvent.xkey.keycode; //Gets the key code, NOT HIS CHAR EQUIVALENT
                // std::cout << key << " pressed \n"; //Displays the key code

                //W
                if (key == 25)  {
                    zSpeed = 1;
                }
                //S
                if(key == 39){
                    zSpeed = -1;
                }
                //A
                if(key == 38){
                    xSpeed = -1;
                }
                //D
                if(key == 40){
                    xSpeed = 1;
                }
                //Space
                if (key == 65)  {
                    ySpeed = 1;
                }
                //Control
                if(key == 37){
                    ySpeed = -1;
                }

                //Left
                if(key == 113){
                    yawSpeed = -0.05;
                }
                //Right
                if(key == 114){
                    yawSpeed = 0.05;
                }
                //Up
                if(key == 111){
                    pitchSpeed = -0.05;
                }
                //Down
                if(key == 116){
                    pitchSpeed = 0.05;
                }
                
                /* Code handling a Keypress event */

            } if(KeyEvent.type==KeyRelease) {
                auto key = KeyEvent.xkey.keycode;
                // std::cout << key << " released \n"; //Displays the key code

                //W
                if (key == 25)  {
                    zSpeed = 0;
                }
                //S
                if(key == 39){
                    zSpeed = 0;
                }
                //A
                if(key == 38){
                    xSpeed = 0;
                }
                //D
                if(key == 40){
                    xSpeed = 0;
                }
                //Space
                if (key == 65)  {
                    ySpeed = 0;
                }
                //Control
                if(key == 37){
                    ySpeed = 0;
                }

                //Left
                if(key == 113){
                    yawSpeed = 0;
                }
                //Right
                if(key == 114){
                    yawSpeed = 0;
                }
                //Up
                if(key == 111){
                    pitchSpeed = 0;
                }
                //Down
                if(key == 116){
                    pitchSpeed = 0;
                }
                
                /* Code handling a KeyRelease event */
            } if (KeyEvent.type == Expose) {
                drawPixels(this->camera->render());
            }
        }
		// int a = XNextEvent(display, &ev);
		// if (ev.type == KeyPress) {
        //     auto key = XLookupKeysym(&ev.xkey, 0);
        //     std::cout << "Key pressed: " << key << std::endl;

        //     //W
        //     if (key == 119)  {
        //         this->camera->origin = this->camera->origin + Vector3(0 * cos(this->camera->rotation.x) + 1 * sin(this->camera->rotation.x), 0, 0 * sin(this->camera->rotation.x) + 1 * cos(this->camera->rotation.x));
        //     }
        //     //S
        //     if(key == 115){
        //         this->camera->origin = this->camera->origin + Vector3(0 * cos(this->camera->rotation.x) - 1 * sin(this->camera->rotation.x), 0, 0 * sin(this->camera->rotation.x) + -1 * cos(this->camera->rotation.x));    
        //     }
        //     //A
        //     if(key == 97){
        //         this->camera->origin = this->camera->origin + Vector3(-1 * cos(this->camera->rotation.x) + 0 * sin(this->camera->rotation.x), 0, 1 * sin(this->camera->rotation.x) + 0 * cos(this->camera->rotation.x));
        //     }
        //     //D
        //     if(key == 100){
        //         this->camera->origin = this->camera->origin + Vector3(1 * cos(this->camera->rotation.x) + 0 * sin(this->camera->rotation.x), 0, -1 * sin(this->camera->rotation.x) + 0 * cos(this->camera->rotation.x));
        //     }
        //     //Space
        //     if (key == 32)  {
        //         this->camera->origin = this->camera->origin + Vector3(0, 1, 0);
        //     }
        //     //Control
        //     if(key == 65507){
        //         this->camera->origin = this->camera->origin + Vector3(0, -1, 0);
        //     }

        //     //Left
        //     if(key == 65361){
        //         this->camera->rotation = this->camera->rotation + Vector3(-.1, 0, 0);
        //     }
        //     if(key == 65363){
        //         this->camera->rotation = this->camera->rotation + Vector3(.1, 0, 0);
        //     }
        //     if(key == 65362){

        //     }
        //     if(key == 65364){

        //     }
            
        //     drawPixels(this->camera->render());
        // }
		// if (ev.type == Expose) {

        if(xSpeed != 0 || ySpeed != 0 || zSpeed != 0 || yawSpeed != 0 || pitchSpeed != 0){
            this->camera->rotation = this->camera->rotation + Vector3(pitchSpeed, yawSpeed, 0);

            float dx = xSpeed;
            float dy = ySpeed;
            float dz = zSpeed;
                
            float alpha = this->camera->rotation.x;
            float beta = this->camera->rotation.y;
            float gamma = this->camera->rotation.z;

            // this->camera->origin = this->camera->origin + Vector3(
            //     dx * (cos(beta)*cos(gamma)) + dy * (sin(alpha)*sin(beta)*cos(gamma) - cos(alpha)*sin(gamma)) + dz * (cos(alpha)*sin(beta)*cos(gamma) + sin(alpha)*sin(gamma)),
            //     dx * (cos(beta)*sin(gamma)) + dy * (sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma)) + dz * (cos(alpha)*sin(beta)*sin(gamma) - sin(alpha)*cos(gamma)), 
            //     -dx * (sin(beta)) + dy * (sin(alpha)*sin(beta)) + dz * (cos(alpha)*cos(beta))
            // );

            Vector3 delta = Vector3(
			    dx,
			    dy * cos(alpha) - dz * sin(alpha),
			    dy * sin(alpha) + dz * cos(alpha)
		    );

		    delta = Vector3(
			    delta.x * cos(beta) + delta.z * sin(beta),
			    delta.y,
			    -delta.x * sin(beta) + delta.z * cos(beta)
		    );

            this->camera->origin = this->camera->origin + delta; 
		
            //  = this->camera->origin + Vector3(xSpeed * cos(this->camera->rotation.y) + zSpeed * sin(this->camera->rotation.y), ySpeed, -xSpeed * sin(this->camera->rotation.y) + zSpeed * cos(this->camera->rotation.y));
            
            drawPixels(this->camera->render());
        }

        // }

    }
}

void CGLWindow::close(){
    XFreeGC(display, gc);
	XDestroyWindow(display, window);
	XCloseDisplay(display);
}

void CGLWindow::drawPixel(int x, int y, int color) {
	XSetForeground(display, gc, color);
	XDrawPoint(display, window, gc, x, y);
}

void CGLWindow::drawPixels(int*** pixels){ 
    int** pixels1 = *pixels;
    // std::cout << sizeof(pixels1) / sizeof(int) << std::endl;
    
    for(int x = 0; x < camera->getWidth(); x++){
        for(int y = 0; y < camera->getHeight(); y++){
            drawPixel(x, y, (*pixels)[x][y]);
        }
    }
}