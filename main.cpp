#include <stdio.h>
#include <stdlib.h>
#include "X11/Xlib.h"
#include <chrono>
#include <thread>
#include <iostream>
#include "test_renderer.hpp"

int main() {
    int width = 800, height = 800;
    TestRenderer renderer = TestRenderer(width, height);
    CGLWindow window = CGLWindow(&renderer, width, height);
    window.init();
    window.show();
    window.close();

	return 0;
}