#include "../include/gui.h"

const float BRUSH_RADIUS = 1.0f;

int mouse_in_canvas(int cx, int cy, int dw, int dh, Vector2 mouse_position) {
    int x = mouse_position.x, y = mouse_position.y;
    return (x >= cx && y >= cy && x < cx + dw && y < cy + dh);
}

void display(Network *network) {
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "FFNN-SGD"); 
    RenderTexture2D canvas = LoadRenderTexture(28, 28);

    BeginTextureMode(canvas);
    ClearBackground(BLACK);
    EndTextureMode();

    int display_width = CANVAS_SIZE * DISPLAY_SCALE;
    int display_height = CANVAS_SIZE * DISPLAY_SCALE;

    int cx = (WINDOW_WIDTH - display_width) / 16;
    int cy = (WINDOW_HEIGHT - display_height) / 2;

    char results_buffer[2048];
    Font jb_mono = LoadFontEx("res/jbmono.ttf", 128, NULL, 0);

    GenTextureMipmaps(&jb_mono.texture);

    SetTextureFilter(jb_mono.texture, TEXTURE_FILTER_BILINEAR);



    while (!WindowShouldClose()) {
        Vector2 mouse_position = GetMousePosition();
        
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            if (mouse_in_canvas(cx, cy, display_width, display_height, mouse_position)) {
                float tx = (mouse_position.x - cx) / DISPLAY_SCALE;
                float ty = (mouse_position.y - cy) / DISPLAY_SCALE;

                BeginTextureMode(canvas);
                DrawCircleV((Vector2){tx, ty}, BRUSH_RADIUS, WHITE);
                EndTextureMode();
            }
        }

        if (IsKeyPressed(KEY_C)) {
            BeginTextureMode(canvas);
            ClearBackground(BLACK);
            EndTextureMode();

            results_buffer[0] = '\0';
        }

        if (IsKeyPressed(KEY_ENTER)) {
            char *output_filename = "test.png";
            Image img = LoadImageFromTexture(canvas.texture);
            ImageFlipVertical(&img);
            ExportImage(img, output_filename);
            UnloadImage(img);

            Matrix *image_matrix = load_image(output_filename);
            char *results = test_image(image_matrix, network);     
            
            snprintf(results_buffer, sizeof(results_buffer), "%s", results);
            
            free(results);
        }

        BeginDrawing();
        ClearBackground(BLACK);

        DrawTextEx(jb_mono, "Left Click: Draw\nEnter: Get Prediction\nC: Clear", (Vector2){500, 30}, 24, 1.0f, RAYWHITE);
        DrawRectangleLines(cx - 1, cy - 1, display_width + 2, display_height + 2, WHITE);
        Rectangle srcRec = { 0.0f, 0.0f, (float)canvas.texture.width, -(float)canvas.texture.height };
        Rectangle destRec = { (float)cx, (float)cy, (float)display_width, (float)display_height };
        DrawTexturePro(canvas.texture, srcRec, destRec, (Vector2){0, 0}, 0.0f, WHITE);

        if (results_buffer[0] != '\0') {
            DrawTextEx(jb_mono, results_buffer, (Vector2){500, 120}, 24, 1.0f, RAYWHITE);
        }

        EndDrawing();
    }

    UnloadRenderTexture(canvas);
    CloseWindow();
}  