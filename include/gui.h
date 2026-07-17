#pragma once 

#define Matrix RMatrix
#include "../include/raylib.h"
#undef Matrix // raylib also has a Matrix struct

#include "../include/config.h"
#include "../include/image.h"
#include "../include/matrix.h"

#define WINDOW_WIDTH 1000
#define WINDOW_HEIGHT 500

#define CANVAS_SIZE 28
#define DISPLAY_SCALE 16

void display(Network *network);