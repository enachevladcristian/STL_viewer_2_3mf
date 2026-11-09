# Forma — STL Viewer

A browser-based STL viewer for quickly opening and inspecting 3D models. Files are parsed locally in the browser and are never uploaded to a server.

## Features

- Open binary and ASCII `.stl` files, or drag and drop a file into the viewport
- Parse, center, and calculate mesh statistics in background workers to keep the UI responsive on dense STL files
- Run the layer-support check in a background worker; large geometry is not duplicated for viewport rendering
- Rename the project and download the original STL data using that name as the `.stl` filename
- Interactive 3D preview with orbit, zoom, and reset/fit camera controls
- Rotate tool with 45° X/Y/Z increments, axis-arrow dragging for constrained rotation, and free rotation by dragging the model
- Move the model along visible X/Y/Z handles or enter millimeter offsets; horizontal movement stays within the build plate
- Scale with visible axis handles or percentage inputs, with uniform scaling enabled by default and independent X/Y/Z scaling available
- Move, rotate, and scale are viewport transforms; downloads preserve the original STL data
- Preliminary FDM support check that marks unsupported surface triangles red and previews green columns from the bed to them, using a 0.2 mm layer and a 45° support reach; steep downward facets are conservatively marked for supports
- A 220 × 220 mm reference build plate with a ground grid
- Toggle the ground grid
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
