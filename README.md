# Forma — STL Viewer

A browser-based STL viewer for quickly opening and inspecting 3D models. Files are parsed locally in the browser and are never uploaded to a server.

## Features

- Open binary and ASCII `.stl` files, or drag and drop a file into the viewport
- Rename the project and download the original STL data using that name as the `.stl` filename
- Interactive 3D preview with orbit, zoom, and reset/fit camera controls
- Rotate tool with 45° X/Y/Z increments, axis-arrow dragging for constrained rotation, and free rotation by dragging the model
- A 220 × 220 mm reference build plate with a ground grid
- Toggle the ground grid and wireframe view
- Inspect model dimensions, triangle count, approximate surface area, and volume
- Responsive workspace for desktop and mobile

## Run locally

```sh
npm install
npm run dev
```

Vite prints a local development URL when ready. To create a production build:

```sh
npm run build
```
