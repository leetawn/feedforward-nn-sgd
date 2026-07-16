BUILD_DIR = build
TARGET = ffnn-sgd.exe

.PHONY: all clean run setup

all: setup
	@cmake --build $(BUILD_DIR)

setup:
	@cmake -E make_directory $(BUILD_DIR)
	@cmake -B $(BUILD_DIR) -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release

run: all
	@./$(BUILD_DIR)/$(TARGET)

clean:
	@echo "Cleaning build directory..."
	@cmake -E rm -rf $(BUILD_DIR)