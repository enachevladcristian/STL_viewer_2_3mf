# Forma — STL Viewer

A browser-based STL viewer for quickly opening and inspecting 3D models. Files are parsed locally in the browser and are never uploaded to a server.

## Features

- Open binary and ASCII `.stl` files and `.3mf` projects, or drag and drop a file into the viewport
- Read common embedded slicer settings from 3MF metadata/configuration files alongside the imported mesh
- 3MF parsing and settings extraction run locally in a background worker; exporting an imported 3MF produces an STL mesh
- Parse, center, and calculate mesh statistics in background workers to keep the UI responsive on dense STL files
- Run the layer-support check in a background worker; meshes above 100,000 triangles are welded and simplified to an analysis-only proxy while the full-resolution model remains unchanged
- Rename the project and download the original STL data using that name as the `.stl` filename
- Interactive 3D preview with orbit, zoom, and reset/fit camera controls
- Toggle a measured bounding box with colored X/Y/Z dimension callouts around the model
- Rotate tool with 45° X/Y/Z increments, axis-arrow dragging for constrained rotation, and free rotation by dragging the model
- Move the model along visible X/Y/Z handles or enter millimeter offsets; horizontal movement stays within the build plate
- Scale with visible axis handles or percentage inputs, with uniform scaling enabled by default and independent X/Y/Z scaling available
- Move, rotate, and scale are viewport transforms; downloads preserve the original STL data
- Preliminary FDM check marks unsupported surface triangles red and previews only the voxelized box-minus-model columns that rise from the bed to the model; the box is padded by 8% (at least 2 mm), and the green preview is not print-ready support material
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
