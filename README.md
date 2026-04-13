# CUDA Path Tracer

A CUDA-based path tracer with an OpenGL/GLFW preview window and an ImGui control panel. The renderer supports progressive accumulation, environment lighting, emissive area lights, and multiple importance sampling (MIS).

## Features

- CUDA path tracing with progressive accumulation
- OpenGL display through GLFW + GLEW
- ImGui overlay for runtime controls
- Scene loading through Assimp
- Environment light toggle
- Emissive triangle area lights
- MIS for direct lighting
- PNG screenshot export
- Headless batch rendering to a PNG file

## Build

This project uses CMake.

```bash
cmake -S . -B build
cmake --build build --config Release --target pathtracer
```

On Windows, the executable is typically located at:

```bash
build/Release/pathtracer.exe
```

## Usage

### Interactive mode

Run the renderer with a scene file to open the GUI window:

```bash
build/Release/pathtracer.exe assets/CBbunny.dae
```

### Headless mode

If `-f` is provided, the renderer runs without the GUI, renders a still image, saves it as PNG, and exits:

```bash
build/Release/pathtracer.exe assets/CBbunny.dae -f output.png
```

## Command-Line Options

- `scene_file`  
  Positional argument. Path to the scene file to load.

- `-m N`  
  Set the maximum number of path bounces. Default: `8`.

- `-s N`  
  Set the number of samples per pixel for headless rendering. Default: `1`.

- `-f output.png`  
  Save a single rendered image to the given PNG file and skip the GUI.

- `-r width height`  
  Set the render resolution. Default: `1280 720`.

### Examples

Render interactively at 1920x1080 with 6 max bounces:

```bash
build/Release/pathtracer.exe assets/CBbunny.dae -r 1920 1080 -m 6
```

Render a single image headlessly with 64 samples per pixel:

```bash
build/Release/pathtracer.exe assets/CBbunny.dae -s 64 -f output.png
```

Render a smaller image headlessly for faster iteration:

```bash
build/Release/pathtracer.exe assets/CBbunny.dae -r 320 240 -s 8 -m 5 -f preview.png
```

## GUI Controls

The ImGui overlay shows runtime statistics and controls.

- FPS: current frame rate
- Samples: accumulated sample count
- Resolution: current render size
- Environment Light: enable or disable the environment light
- Invert Mouse Y: invert mouse look vertical movement
- Max Bounces: adjust the maximum path length from 1 to 16

### Keyboard and Mouse

- `W`, `A`, `S`, `D`: move the camera
- `Space`: move up
- `Left Shift`: move down
- Right mouse button: hold to look around
- `Esc`: close the window
- `F12`: save the current frame as a PNG in `screenshots/`

## Output

- GUI screenshots are saved as `screenshots/frame_XXXXXX.png`
- Headless renders are saved to the path passed to `-f`
- Headless runs also print the total render time to the console before saving

## Notes

- Environment lighting is disabled by default.
- If no scene file is provided, the app starts without loading a scene.
- Supported scene formats depend on Assimp.
