#include "../include/progress_bar.h"
#include <windows.h>

// custom progress bar for the memes, kinda dogshit but it works 

void progress(int current, int end) {
    printf("\e[?25l");

    float percentage = (float)current / end;
    int progress_count = (int)(percentage * BAR_WIDTH);
    int remaining_count = BAR_WIDTH - progress_count;
    char progress_buffer[BAR_WIDTH + 1], remaining_buffer[BAR_WIDTH + 1];

    memset(progress_buffer, 219, progress_count);
    progress_buffer[progress_count] = '\0';
    
    memset(remaining_buffer, 219, remaining_count); 
    remaining_buffer[remaining_count] = '\0';

    if (progress_count > 0) printf("\e[0;92m%s\e[0m", progress_buffer);
    if (remaining_count > 0) printf("\e[0;37m%s\e[0m", remaining_buffer);

    printf(" %3d%%\e[K", (int)(percentage * 100));
    fflush(stdout);

    for (int i = 0; i < BAR_WIDTH + 5; i++) printf("\b");
}