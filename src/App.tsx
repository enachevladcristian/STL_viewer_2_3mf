import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Box, Check, ChevronDown, ClipboardCheck, CloudUpload, FileBox, FileUp, Grid2X2, Info, Move,
  Maximize, MousePointer2, Pencil, Rotate3D, RotateCcw, Ruler, Scan,
  ShieldCheck, Sparkles, Triangle, Upload, X,
} from 'lucide-react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { OverhangCheck } from './printability'

type ModelInfo = {
  name: string
  size: number
  format: 'STL' | '3MF'
  parameters: { key: string; value: string; source: string }[]
  triangles: number
  dimensions: [number, number, number]
  volume: number
  area: number
  geometry: THREE.BufferGeometry
}

type MeshIntegrityIssue = 'openEdges' | 'nonManifold' | 'invertedNormals' | 'degenerateFaces'
type MeshIntegrityWorkerResponse = {
  stage?: string
  error?: string
  counts?: { openEdges: number; nonManifoldEdges: number; invertedTriangles: number; degenerateTriangles: number }
  triangleCounts?: { openEdges: number; nonManifold: number; invertedNormals: number; degenerateFaces: number }
  highlights?: Record<MeshIntegrityIssue, ArrayBuffer>
  sampledHighlights?: boolean
}
type MeshIntegrityResult = {
  counts: NonNullable<MeshIntegrityWorkerResponse['counts']>
  triangleCounts: NonNullable<MeshIntegrityWorkerResponse['triangleCounts']>
  highlights: Record<MeshIntegrityIssue, THREE.BufferGeometry>
  sampledHighlights: boolean
}

const INTEGRITY_ISSUE_COLORS: Record<MeshIntegrityIssue, string> = {
  openEdges: '#ff9a3d',
  nonManifold: '#e84855',
  invertedNormals: '#a45be0',
  degenerateFaces: '#f2bd3d',
}

type OrientationSuggestion = {
  rotation: Rotation
  unsupportedArea: number
  unsupportedPercent: number
  bedContactArea: number
  minimumContactArea: number
  dimensions: [number, number, number]
  fits: boolean
  overflow: number
}
type OrientationWorkerResponse = { stage?: string; error?: string; suggestions?: OrientationSuggestion[] }
type LayerPreviewData = {
  layerNumber: number
  layerHeight: number
  currentPositions: ArrayBuffer
  islandPositions: ArrayBuffer
  islandCount: number
  triangleCount: number
  originalTriangleCount: number
  simplified: boolean
}
type LayerPreviewWorkerResponse = LayerPreviewData | { stage: string } | { error: string }
type LayerScanWarning = { layerNumber: number; layerHeight: number; islandCount: number }
type LayerScanData = { scanWarnings: LayerScanWarning[]; scannedLayers: number; triangleCount: number; originalTriangleCount: number; simplified: boolean }
type LayerScanWorkerResponse = LayerScanData | { stage: string } | { error: string }

type CheckWorkerResponse = {
  stage?: string
  originalTriangleCount?: number
  simplified?: boolean
  simplificationError?: number
  triangleCount?: number
  flaggedTriangles?: number
  flaggedArea?: number
  thinWallTriangles?: number
  undersizedFeatureTriangles?: number
  minimumWallThickness?: number
  minimumFeatureWidth?: number
  bedContactArea?: number
  minimumContactArea?: number
  floating?: boolean
  limitDegrees?: number
  highlightPositions?: ArrayBuffer
  highlightNormals?: ArrayBuffer
  supportPositions?: ArrayBuffer
  supportNormals?: ArrayBuffer
  error?: string
}

type STLWorkerResponse = {
  positions?: ArrayBuffer
  normals?: ArrayBuffer
  triangles?: number
  dimensions?: [number, number, number]
  bounds?: [number, number, number, number, number, number]
  volume?: number
  area?: number
  error?: string
}

type ThreeMFWorkerResponse = STLWorkerResponse & {
  parameters?: { key: string; value: string; source: string }[]
  unit?: string
}

type Rotation = [number, number, number]
type AxisValues = [number, number, number]
type AxisTransformMode = 'move' | 'scale'
type ViewPreset = 'X' | 'Y' | 'Z' | 'home'
type FilamentMaterial = 'PLA' | 'PETG'
type ProfileSetting = { label: string; value: string }

const BUILD_VOLUME = { width: 330, depth: 240, height: 300 }

const PROFILE_DEFAULTS: Record<FilamentMaterial, ProfileSetting[]> = {
  PLA: [
    { label: 'Profile', value: 'Speed' },
    { label: 'Layer Height', value: '0.2 mm' },
    { label: 'Infill', value: '20 %' },
    { label: 'Material temperature', value: '215 C' },
    { label: 'Bed Temperature', value: '70 C' },
    { label: 'Build plate adhesion', value: 'Brim' },
    { label: 'Brim Width', value: '4 mm' },
  ],
  PETG: [
    { label: 'Profile', value: 'Speed' },
    { label: 'Layer Height', value: '0.2 mm' },
    { label: 'Infill', value: '20%' },
    { label: 'Material', value: '255 C' },
    { label: 'Build plate', value: '85 C' },
  ],
}

function getFlatSupportNormal(geometry: THREE.BufferGeometry, normal: THREE.Vector3, point: THREE.Vector3, minimumArea: number) {
  const position = geometry.getAttribute('position')
  const index = geometry.getIndex()
  const triangleCount = Math.floor((index?.count ?? position.count) / 3)
  const tolerance = Math.max(geometry.boundingBox?.getSize(new THREE.Vector3()).length() ?? 0, 1) * 0.00002
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3()
  const ab = new THREE.Vector3(), ac = new THREE.Vector3(), faceNormal = new THREE.Vector3()
  let coplanarArea = 0

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const offset = triangle * 3
    const ia = index ? index.getX(offset) : offset
    const ib = index ? index.getX(offset + 1) : offset + 1
    const ic = index ? index.getX(offset + 2) : offset + 2
    a.fromBufferAttribute(position, ia)
    b.fromBufferAttribute(position, ib)
    c.fromBufferAttribute(position, ic)
    ab.subVectors(b, a)
    ac.subVectors(c, a)
    faceNormal.crossVectors(ab, ac)
    const twiceArea = faceNormal.length()
    if (twiceArea < 1e-12) continue
    faceNormal.multiplyScalar(1 / twiceArea)
    if (Math.abs(faceNormal.dot(normal)) < 0.99995) continue
    if (Math.abs(normal.dot(a.sub(point))) > tolerance
      || Math.abs(normal.dot(b.sub(point))) > tolerance
      || Math.abs(normal.dot(c.sub(point))) > tolerance) continue
    coplanarArea += twiceArea * 0.5
  }
  if (coplanarArea < minimumArea) return null

  // A bed-contact face must be a support plane: the rest of the mesh must lie
  // entirely on one side. This also corrects inward/reversed STL facet normals.
  let minDistance = Infinity
  let maxDistance = -Infinity
  for (let vertex = 0; vertex < position.count; vertex += 1) {
    a.fromBufferAttribute(position, vertex)
    const distance = normal.dot(a.sub(point))
    minDistance = Math.min(minDistance, distance)
    maxDistance = Math.max(maxDistance, distance)
  }
  if (minDistance < -tolerance && maxDistance > tolerance) return null
  if (minDistance >= -tolerance && maxDistance > tolerance) return normal.clone().negate()
  return normal.clone()
}

function constrainToBuildPlate(geometry: THREE.BufferGeometry, rotation: Rotation, scale: AxisValues, position: AxisValues): AxisValues {
  const bounds = geometry.boundingBox?.clone()
  if (!bounds) return position
  const center = bounds.getCenter(new THREE.Vector3())
  bounds.min.sub(center)
  bounds.max.sub(center)
  const orientation = new THREE.Quaternion().setFromEuler(new THREE.Euler(
    THREE.MathUtils.degToRad(rotation[0]),
    THREE.MathUtils.degToRad(rotation[1]),
    THREE.MathUtils.degToRad(rotation[2]),
  ))
  bounds.applyMatrix4(new THREE.Matrix4().compose(new THREE.Vector3(), orientation, new THREE.Vector3(...scale)))
  const minX = -BUILD_VOLUME.width / 2 - bounds.min.x
  const maxX = BUILD_VOLUME.width / 2 - bounds.max.x
  const minZ = -BUILD_VOLUME.depth / 2 - bounds.min.z
  const maxZ = BUILD_VOLUME.depth / 2 - bounds.max.z
  const x = minX <= maxX ? THREE.MathUtils.clamp(position[0], minX, maxX) : 0
  const z = minZ <= maxZ ? THREE.MathUtils.clamp(position[2], minZ, maxZ) : 0
  const height = bounds.max.y - bounds.min.y
  const y = THREE.MathUtils.clamp(position[1], 0, Math.max(0, BUILD_VOLUME.height - height))
  return [x, y, z]
}
const formatSize = (bytes: number) => bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / (1024 * 1024)).toFixed(2)} MB`
const formatMeasure = (n: number) => `${n.toFixed(2)} mm`

function getStats(geometry: THREE.BufferGeometry, scaleFactors: AxisValues = [1, 1, 1]) {
  const position = geometry.getAttribute('position')
  const scale = new THREE.Vector3(...scaleFactors)
  let volume = 0
  let area = 0
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3()
  const edge1 = new THREE.Vector3(), edge2 = new THREE.Vector3(), normal = new THREE.Vector3()
  for (let i = 0; i < position.count; i += 3) {
    a.fromBufferAttribute(position, i).multiply(scale)
    b.fromBufferAttribute(position, i + 1).multiply(scale)
    c.fromBufferAttribute(position, i + 2).multiply(scale)
    volume += a.dot(b.clone().cross(c)) / 6
    edge1.subVectors(b, a)
    edge2.subVectors(c, a)
    area += normal.crossVectors(edge1, edge2).length() / 2
  }
  return { volume: Math.abs(volume), area }
}

const MAX_CUBE_GRID_CELLS = 12_000

type CubeGridLayout = { min: THREE.Vector3; counts: [number, number, number]; total: number }

function getCubeGridLayout(bounds: THREE.Box3, cubeSize: number): CubeGridLayout | null {
  if (!Number.isFinite(cubeSize) || cubeSize <= 0 || bounds.isEmpty()) return null
  const size = bounds.getSize(new THREE.Vector3())
  const counts: [number, number, number] = [
    Math.max(1, Math.ceil((size.x - 1e-8) / cubeSize)),
    Math.max(1, Math.ceil((size.y - 1e-8) / cubeSize)),
    Math.max(1, Math.ceil((size.z - 1e-8) / cubeSize)),
  ]
  const expandedSize = new THREE.Vector3(counts[0] * cubeSize, counts[1] * cubeSize, counts[2] * cubeSize)
  const min = bounds.getCenter(new THREE.Vector3()).sub(expandedSize.multiplyScalar(0.5))
  return { min, counts, total: counts[0] * counts[1] * counts[2] }
}

function makeCubeGridGeometry(layout: CubeGridLayout, cubeSize: number) {
  const [nx, ny, nz] = layout.counts
  const { min } = layout
  const points: number[] = []
  const pushSegment = (x0: number, y0: number, z0: number, x1: number, y1: number, z1: number) => {
    points.push(x0, y0, z0, x1, y1, z1)
  }

  for (let iy = 0; iy <= ny; iy += 1) for (let iz = 0; iz <= nz; iz += 1) for (let ix = 0; ix < nx; ix += 1) {
    const y = min.y + iy * cubeSize, z = min.z + iz * cubeSize, x = min.x + ix * cubeSize
    pushSegment(x, y, z, x + cubeSize, y, z)
  }
  for (let ix = 0; ix <= nx; ix += 1) for (let iz = 0; iz <= nz; iz += 1) for (let iy = 0; iy < ny; iy += 1) {
    const x = min.x + ix * cubeSize, z = min.z + iz * cubeSize, y = min.y + iy * cubeSize
    pushSegment(x, y, z, x, y + cubeSize, z)
  }
  for (let ix = 0; ix <= nx; ix += 1) for (let iy = 0; iy <= ny; iy += 1) for (let iz = 0; iz < nz; iz += 1) {
    const x = min.x + ix * cubeSize, y = min.y + iy * cubeSize, z = min.z + iz * cubeSize
    pushSegment(x, y, z, x, y, z + cubeSize)
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3))
  return geometry
}

function makeBuildPlateGrid(width: number, depth: number, spacing = 10, majorSpacing = 50) {
  const minorPoints: number[] = []
  const majorPoints: number[] = []
  const addLine = (list: number[], x1: number, z1: number, x2: number, z2: number) => list.push(x1, 0, z1, x2, 0, z2)
  const isMajor = (value: number) => Math.abs(value / majorSpacing - Math.round(value / majorSpacing)) < 1e-6
  for (let x = -width / 2; x <= width / 2 + 1e-6; x += spacing) addLine(isMajor(x) ? majorPoints : minorPoints, x, -depth / 2, x, depth / 2)
  for (let z = -depth / 2; z <= depth / 2 + 1e-6; z += spacing) addLine(isMajor(z) ? majorPoints : minorPoints, -width / 2, z, width / 2, z)
  const makeLines = (points: number[], color: number) => {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3))
    return new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9, depthWrite: false }))
  }
  const group = new THREE.Group()
  group.name = 'rectangular-build-plate-grid'
  group.add(makeLines(minorPoints, 0xcbd4e0), makeLines(majorPoints, 0x9baabe))
  return group
}

type AdaptiveGridData = { positions: Float32Array; cellCount: number; limited: boolean }

type TriangleFeature = { triangle: THREE.Triangle; bounds: THREE.Box3 }

function makeAdaptiveGridData(geometry: THREE.BufferGeometry, bounds: THREE.Box3, transform: THREE.Matrix4, minCellSize: number, maxCellSize: number): AdaptiveGridData {
  const position = geometry.getAttribute('position')
  const index = geometry.getIndex()
  const triangleCount = Math.floor((index?.count ?? position.count) / 3)
  const sampleStep = Math.max(1, Math.ceil(triangleCount / 50_000))
  const triangles: TriangleFeature[] = []
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3()
  for (let triangleIndex = 0; triangleIndex < triangleCount; triangleIndex += sampleStep) {
    const ia = index ? index.getX(triangleIndex * 3) : triangleIndex * 3
    const ib = index ? index.getX(triangleIndex * 3 + 1) : triangleIndex * 3 + 1
    const ic = index ? index.getX(triangleIndex * 3 + 2) : triangleIndex * 3 + 2
    a.fromBufferAttribute(position, ia).applyMatrix4(transform)
    b.fromBufferAttribute(position, ib).applyMatrix4(transform)
    c.fromBufferAttribute(position, ic).applyMatrix4(transform)
    const triangle = new THREE.Triangle(a.clone(), b.clone(), c.clone())
    triangles.push({ triangle, bounds: new THREE.Box3().setFromPoints([triangle.a, triangle.b, triangle.c]) })
  }

  const boxes: THREE.Box3[] = []
  const fitBoxToMesh = (cell: THREE.Box3, features: TriangleFeature[]) => {
    const fitted = new THREE.Box3().makeEmpty()
    for (const feature of features) {
      const overlap = feature.bounds.clone().intersect(cell)
      if (!overlap.isEmpty()) fitted.union(overlap)
    }
    if (fitted.isEmpty()) return null
    const cellSize = cell.getSize(new THREE.Vector3())
    const center = fitted.getCenter(new THREE.Vector3())
    const min = fitted.min.clone(), max = fitted.max.clone()
    for (const axis of ['x', 'y', 'z'] as const) {
      const targetSize = Math.min(minCellSize, cellSize[axis])
      if (max[axis] - min[axis] < targetSize) {
        min[axis] = Math.max(cell.min[axis], Math.min(center[axis] - targetSize / 2, cell.max[axis] - targetSize))
        max[axis] = min[axis] + targetSize
      }
    }
    return new THREE.Box3(min, max)
  }
  const maxNodes = MAX_CUBE_GRID_CELLS * 2 - 1
  const detailThreshold = 12
  const maxTriangleTests = 3_000_000
  let visitedNodes = 0
  let triangleTests = 0
  let limited = false
  const subdivide = (cell: THREE.Box3, candidates: TriangleFeature[]) => {
    visitedNodes += 1
    if (visitedNodes >= maxNodes || boxes.length >= MAX_CUBE_GRID_CELLS - 8) {
      limited = true
      boxes.push(cell)
      return
    }
    const relevant: TriangleFeature[] = []
    for (const feature of candidates) {
      triangleTests += 1
      if (triangleTests > maxTriangleTests) {
        limited = true
        boxes.push(cell)
        return
      }
      if (feature.bounds.intersectsBox(cell) && cell.intersectsTriangle(feature.triangle)) relevant.push(feature)
    }
    if (relevant.length === 0) return
    const size = cell.getSize(new THREE.Vector3())
    const longestEdge = Math.max(size.x, size.y, size.z)
    if (longestEdge <= minCellSize || (longestEdge <= maxCellSize && relevant.length <= detailThreshold)) {
      const fitted = fitBoxToMesh(cell, relevant)
      if (fitted) boxes.push(fitted)
      return
    }

    const center = cell.getCenter(new THREE.Vector3())
    const min = cell.min, max = cell.max
    const startingBoxCount = boxes.length
    for (let octant = 0; octant < 8; octant += 1) {
      if (boxes.length >= MAX_CUBE_GRID_CELLS - 8) {
        limited = true
        boxes.splice(startingBoxCount)
        boxes.push(cell)
        return
      }
      const child = new THREE.Box3(
        new THREE.Vector3(octant & 1 ? center.x : min.x, octant & 2 ? center.y : min.y, octant & 4 ? center.z : min.z),
        new THREE.Vector3(octant & 1 ? max.x : center.x, octant & 2 ? max.y : center.y, octant & 4 ? max.z : center.z),
      )
      const childCandidates = relevant.filter(feature => feature.bounds.intersectsBox(child))
      if (childCandidates.length > 0) subdivide(child, childCandidates)
      if (boxes.length >= MAX_CUBE_GRID_CELLS - 8) {
        limited = true
        boxes.splice(startingBoxCount)
        boxes.push(cell)
        return
      }
    }
  }
  const boundsSize = bounds.getSize(new THREE.Vector3())
  const rootEdge = Math.max(boundsSize.x, boundsSize.y, boundsSize.z)
  const rootCenter = bounds.getCenter(new THREE.Vector3())
  const rootBounds = new THREE.Box3(
    rootCenter.clone().addScalar(-rootEdge / 2),
    rootCenter.clone().addScalar(rootEdge / 2),
  )
  subdivide(rootBounds, triangles.filter(feature => feature.bounds.intersectsBox(rootBounds)))

  const positions: number[] = []
  const corners = Array.from({ length: 8 }, (_, index) => index)
  const edgePairs = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
  for (const box of boxes) {
    const points = corners.map(index => new THREE.Vector3(
      index & 1 ? box.max.x : box.min.x,
      index & 2 ? box.max.y : box.min.y,
      index & 4 ? box.max.z : box.min.z,
    ))
    for (const [start, end] of edgePairs) positions.push(...points[start].toArray(), ...points[end].toArray())
  }
  return { positions: new Float32Array(positions), cellCount: boxes.length, limited }
}

function makeAxisLabels(length: number) {
  const group = new THREE.Group()
  group.name = 'model-axis-arrows'
  group.visible = false
  const axes = [
    { name: 'X', direction: new THREE.Vector3(1, 0, 0), color: '#d66f70', hex: 0xd66f70 },
    { name: 'Y', direction: new THREE.Vector3(0, 1, 0), color: '#59a27a', hex: 0x59a27a },
    { name: 'Z', direction: new THREE.Vector3(0, 0, 1), color: '#6f92c4', hex: 0x6f92c4 },
  ]
  for (const [axisIndex, axis] of axes.entries()) {
    const arrow = new THREE.ArrowHelper(axis.direction, new THREE.Vector3(0, 0, 0), length, axis.hex, length * 0.16, length * 0.075)
    arrow.line.userData.rotationAxis = axisIndex
    arrow.cone.userData.rotationAxis = axisIndex
    const setAlwaysVisible = (material: THREE.Material | THREE.Material[]) => {
      for (const item of Array.isArray(material) ? material : [material]) {
        item.depthTest = false
        item.depthWrite = false
      }
    }
    setAlwaysVisible(arrow.line.material)
    setAlwaysVisible(arrow.cone.material)
    arrow.line.renderOrder = 20
    arrow.cone.renderOrder = 20
    group.add(arrow)

    const hitArea = new THREE.Mesh(
      new THREE.CylinderGeometry(length * 0.065, length * 0.065, length * 0.78, 10, 1, true),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, colorWrite: false, depthWrite: false, depthTest: false, side: THREE.DoubleSide }),
    )
    hitArea.position.copy(axis.direction).multiplyScalar(length * 0.47)
    hitArea.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis.direction)
    hitArea.userData.rotationAxis = axisIndex
    group.add(hitArea)

    const canvas = document.createElement('canvas')
    canvas.width = 96
    canvas.height = 96
    const context = canvas.getContext('2d')
    if (context) {
      context.beginPath()
      context.arc(48, 48, 43, 0, Math.PI * 2)
      context.fillStyle = axis.color
      context.fill()
      context.lineWidth = 5
      context.strokeStyle = '#ffffff'
      context.stroke()
      context.font = '700 58px Poppins, sans-serif'
      context.textAlign = 'center'
      context.textBaseline = 'middle'
      context.fillStyle = '#ffffff'
      context.fillText(axis.name, 48, 50)
    }
    const texture = new THREE.CanvasTexture(canvas)
    texture.colorSpace = THREE.SRGBColorSpace
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, depthWrite: false }))
    sprite.userData.rotationAxis = axisIndex
    const labelSize = Math.max(length * 0.19, 1.8)
    sprite.scale.set(labelSize, labelSize, 1)
    sprite.position.copy(axis.direction).multiplyScalar(length * 1.24)
    sprite.renderOrder = 21
    group.add(sprite)
  }
  return group
}

function makeDimensionsOverlay(geometry: THREE.BufferGeometry, scale: AxisValues) {
  const bounds = geometry.boundingBox!
  const center = bounds.getCenter(new THREE.Vector3())
  const size = bounds.getSize(new THREE.Vector3())
  const half = size.clone().multiplyScalar(0.5)
  const group = new THREE.Group()
  group.name = 'model-dimensions'

  const box = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(size.x, size.y, size.z)),
    new THREE.LineBasicMaterial({ color: '#63748b', transparent: true, opacity: 0.72, depthTest: false }),
  )
  box.position.copy(center)
  box.renderOrder = 25
  group.add(box)

  const maximum = Math.max(size.x, size.y, size.z, 1)
  const offset = maximum * 0.09
  const tick = maximum * 0.025
  const labelHeight = maximum * 0.085
  const axes = [
    { name: 'X', color: 0xd66f70, colorCss: '#d66f70', from: new THREE.Vector3(-half.x, -half.y - offset, half.z + offset), to: new THREE.Vector3(half.x, -half.y - offset, half.z + offset), tickA: new THREE.Vector3(0, tick, 0), value: size.x * scale[0] },
    { name: 'Y', color: 0x59a27a, colorCss: '#59a27a', from: new THREE.Vector3(half.x + offset, -half.y, half.z + offset), to: new THREE.Vector3(half.x + offset, half.y, half.z + offset), tickA: new THREE.Vector3(tick, 0, 0), value: size.y * scale[1] },
    { name: 'Z', color: 0x6f92c4, colorCss: '#6f92c4', from: new THREE.Vector3(half.x + offset, -half.y - offset, -half.z), to: new THREE.Vector3(half.x + offset, -half.y - offset, half.z), tickA: new THREE.Vector3(0, tick, 0), value: size.z * scale[2] },
  ]
  for (const axis of axes) {
    const linePoints = [axis.from, axis.to, axis.from.clone().sub(axis.tickA), axis.from.clone().add(axis.tickA), axis.to.clone().sub(axis.tickA), axis.to.clone().add(axis.tickA)]
    const measureLine = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(linePoints), new THREE.LineBasicMaterial({ color: axis.color, depthTest: false }))
    measureLine.renderOrder = 26
    group.add(measureLine)

    const canvas = document.createElement('canvas')
    canvas.width = 360
    canvas.height = 88
    const context = canvas.getContext('2d')
    if (context) {
      context.fillStyle = '#ffffffed'
      context.fillRect(4, 4, 352, 80)
      context.strokeStyle = axis.colorCss
      context.lineWidth = 5
      context.strokeRect(4, 4, 352, 80)
      context.font = '600 40px Poppins, sans-serif'
      context.textAlign = 'center'
      context.textBaseline = 'middle'
      context.fillStyle = axis.colorCss
      context.fillText(`${axis.name}  ${axis.value.toFixed(1)} mm`, 180, 45)
    }
    const texture = new THREE.CanvasTexture(canvas)
    texture.colorSpace = THREE.SRGBColorSpace
    const label = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, depthWrite: false }))
    label.scale.set(labelHeight * 3.8, labelHeight, 1)
    label.position.copy(axis.from).add(axis.to).multiplyScalar(0.5)
    if (axis.name === 'X') label.position.z += labelHeight * 0.65
    if (axis.name === 'Y') label.position.x += labelHeight * 2.0
    if (axis.name === 'Z') label.position.y += labelHeight * 0.75
    label.renderOrder = 27
    group.add(label)
  }
  return group
}

function disposeObjectTree(root: THREE.Object3D) {
  root.traverse(object => {
    const renderable = object as THREE.Mesh
    renderable.geometry?.dispose?.()
    if (renderable.material) {
      const materials = Array.isArray(renderable.material) ? renderable.material : [renderable.material]
      materials.forEach(material => {
        const texture = (material as THREE.Material & { map?: THREE.Texture | null }).map
        texture?.dispose()
        material.dispose()
      })
    }
  })
}

function Viewport({
  model, showGrid, showAxes, showDimensions, rotation, position, scale, showCubeGrid, adaptiveCubeGrid, cubeGridSize, cubeGridBounds, adaptiveGridData, rotateMode, placeOnBedMode, axisTransformMode, analysisOverlay, integrityOverlay, integrityIssue, supportOverlay, layerPreviewOverlay, onRotationChange, onPlaceSurface, onFreeRotateEnd, onAxisTransformStart, onAxisTransform, onAxisTransformEnd, onError, resetRef, viewPresetRef,
}: {
  model: ModelInfo | null
  showGrid: boolean
  showAxes: boolean
  showDimensions: boolean
  rotation: Rotation
  position: AxisValues
  scale: AxisValues
  showCubeGrid: boolean
  adaptiveCubeGrid: boolean
  cubeGridSize: number
  cubeGridBounds: THREE.Box3 | null
  adaptiveGridData: AdaptiveGridData | null
  analysisOverlay: THREE.BufferGeometry | null
  integrityOverlay: THREE.BufferGeometry | null
  integrityIssue: MeshIntegrityIssue | null
  supportOverlay: THREE.BufferGeometry | null
  layerPreviewOverlay: LayerPreviewData | null
  rotateMode: boolean
  placeOnBedMode: boolean
  axisTransformMode: AxisTransformMode | null
  onRotationChange: (value: Rotation) => void
  onPlaceSurface: (worldNormal: THREE.Vector3) => void
  onFreeRotateEnd: () => void
  onAxisTransformStart: (mode: AxisTransformMode) => void
  onAxisTransform: (mode: AxisTransformMode, axis: number, delta: number) => void
  onAxisTransformEnd: (mode: AxisTransformMode) => void
  onError: (message: string) => void
  resetRef: React.MutableRefObject<(() => void) | null>
  viewPresetRef: React.MutableRefObject<((view: ViewPreset) => void) | null>
}) {
  const hostRef = useRef<HTMLDivElement>(null)
  const rotationRef = useRef<Rotation>(rotation)
  const axisTransformModeRef = useRef<AxisTransformMode | null>(axisTransformMode)
  axisTransformModeRef.current = axisTransformMode
  const engineRef = useRef<{
    renderer: THREE.WebGLRenderer
    scene: THREE.Scene
    camera: THREE.PerspectiveCamera
    controls: OrbitControls
    mesh?: THREE.Mesh
    axisGroup?: THREE.Group
    cubeGrid?: THREE.LineSegments
    adaptiveGrid?: THREE.LineSegments
    grid?: THREE.Group
  } | null>(null)

  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    } catch {
      onError('Your browser could not start WebGL. Try enabling hardware acceleration.')
      return
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(host.clientWidth, host.clientHeight)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.15
    host.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#edf1f6')
    const camera = new THREE.PerspectiveCamera(38, host.clientWidth / host.clientHeight, 0.01, 10000)
    camera.position.set(5.3, 4.2, 6.5)
    scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa8bc, 2.0))
    const key = new THREE.DirectionalLight(0xffffff, 3.2)
    key.position.set(5, 8, 6)
    scene.add(key)
    const fill = new THREE.DirectionalLight(0xc9d9ff, 1.4)
    fill.position.set(-6, 2, -4)
    scene.add(fill)

    const bed = new THREE.Mesh(
      new THREE.BoxGeometry(BUILD_VOLUME.width, 0.18, BUILD_VOLUME.depth),
      new THREE.MeshStandardMaterial({ color: '#dfe5ed', roughness: 0.92, metalness: 0.02 }),
    )
    bed.position.y = -0.16
    scene.add(bed)
    const grid = makeBuildPlateGrid(BUILD_VOLUME.width, BUILD_VOLUME.depth)
    grid.position.y = -0.065
    scene.add(grid)
    const halfWidth = BUILD_VOLUME.width / 2
    const halfDepth = BUILD_VOLUME.depth / 2
    const bedOutline = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-halfWidth, -0.06, -halfDepth), new THREE.Vector3(halfWidth, -0.06, -halfDepth),
        new THREE.Vector3(halfWidth, -0.06, halfDepth), new THREE.Vector3(-halfWidth, -0.06, halfDepth),
      ]),
      new THREE.LineBasicMaterial({ color: '#91a1b6' }),
    )
    scene.add(bedOutline)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.07
    controls.target.set(0, 0, 0)
    controls.minDistance = 0.1
    controls.maxDistance = 10000
    controls.update()
    const resize = () => {
      camera.aspect = host.clientWidth / host.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(host.clientWidth, host.clientHeight)
    }
    const observer = new ResizeObserver(resize)
    observer.observe(host)
    let frame = 0
    const render = () => {
      frame = requestAnimationFrame(render)
      controls.update()
      const currentEngine = engineRef.current
      if (currentEngine?.axisGroup && currentEngine.mesh) {
        currentEngine.axisGroup.position.copy(currentEngine.mesh.position)
        if (axisTransformModeRef.current === 'move') currentEngine.axisGroup.quaternion.identity()
        else currentEngine.axisGroup.quaternion.copy(currentEngine.mesh.quaternion)
      }
      renderer.render(scene, camera)
    }
    render()
    engineRef.current = { renderer, scene, camera, controls, grid }
    return () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
      controls.dispose()
      renderer.dispose()
      renderer.domElement.remove()
      engineRef.current = null
    }
  }, [onError])

  useEffect(() => {
    const engine = engineRef.current
    if (!engine) return
    if (engine.mesh) {
      engine.scene.remove(engine.mesh)
      disposeObjectTree(engine.mesh)
      engine.mesh = undefined
    }
    if (engine.axisGroup) {
      engine.scene.remove(engine.axisGroup)
      disposeObjectTree(engine.axisGroup)
      engine.axisGroup = undefined
    }
    if (!model) return

    const geometry = model.geometry
    const material = new THREE.MeshStandardMaterial({ color: '#c4d4e8', roughness: 0.34, metalness: 0.12, flatShading: true })
    const mesh = new THREE.Mesh(geometry, material)
    const axisGroup = makeAxisLabels(Math.max(...model.dimensions) * 0.58)
    axisGroup.name = 'model-axis-arrows'
    engine.scene.add(mesh, axisGroup)
    engine.mesh = mesh
    engine.axisGroup = axisGroup

    const modelBounds = geometry.boundingBox!.clone()
    mesh.position.y = -modelBounds.min.y
    const extent = Math.max(...model.dimensions, BUILD_VOLUME.width, BUILD_VOLUME.depth)
    engine.camera.near = Math.max(extent / 1000, 0.001)
    engine.camera.far = extent * 100
    engine.camera.updateProjectionMatrix()
    engine.controls.minDistance = extent * 0.08
    engine.controls.maxDistance = extent * 20
    const setView = (view: ViewPreset) => {
      const focus = engine.mesh?.getWorldPosition(new THREE.Vector3()) ?? new THREE.Vector3(0, model.dimensions[1] / 2, 0)
      const distance = extent * 2.4
      engine.controls.target.copy(focus)
      if (view === 'home') {
        engine.camera.up.set(0, 1, 0)
        engine.camera.position.set(focus.x + extent * 1.35, focus.y + extent * 1.05, focus.z + extent * 1.65)
      } else if (view === 'X') {
        engine.camera.up.set(0, 1, 0)
        engine.camera.position.set(focus.x + distance, focus.y, focus.z)
      } else if (view === 'Y') {
        engine.camera.up.set(0, 0, -1)
        engine.camera.position.set(focus.x, focus.y + distance, focus.z)
      } else {
        engine.camera.up.set(0, 1, 0)
        engine.camera.position.set(focus.x, focus.y, focus.z + distance)
      }
      engine.camera.lookAt(focus)
      engine.camera.updateMatrixWorld(true)
      engine.controls.update()
    }
    resetRef.current = () => setView('home')
    viewPresetRef.current = setView
    setView('home')
  }, [model, resetRef])

  useEffect(() => {
    rotationRef.current = rotation
    const mesh = engineRef.current?.mesh
    if (!mesh || !model) return
    mesh.rotation.set(THREE.MathUtils.degToRad(rotation[0]), THREE.MathUtils.degToRad(rotation[1]), THREE.MathUtils.degToRad(rotation[2]))
    mesh.scale.set(scale[0], scale[1], scale[2])
    const bounds = mesh.geometry.boundingBox!.clone()
    const orientation = new THREE.Matrix4().compose(new THREE.Vector3(), mesh.quaternion, mesh.scale)
    bounds.applyMatrix4(orientation)
    mesh.position.set(position[0], -bounds.min.y + position[1], position[2])
    mesh.updateMatrixWorld(true)
  }, [model, position, rotation, scale])

  useEffect(() => {
    const engine = engineRef.current
    if (!engine || !model || !cubeGridBounds) return
    const layout = getCubeGridLayout(cubeGridBounds, cubeGridSize)
    if (!layout) return
    const grids: { object: THREE.LineSegments; key: 'cubeGrid' | 'adaptiveGrid' }[] = []
    const makeLines = (geometry: THREE.BufferGeometry, name: string) => {
      const object = new THREE.LineSegments(
        geometry,
        new THREE.LineBasicMaterial({ color: '#526981', transparent: true, opacity: 0.78, depthTest: true, depthWrite: false }),
      )
      object.name = name
      object.renderOrder = 3
      engine.scene.add(object)
      return object
    }
    if (showCubeGrid && layout.total <= MAX_CUBE_GRID_CELLS) {
      const object = makeLines(makeCubeGridGeometry(layout, cubeGridSize), 'bounding-box-cube-grid')
      engine.cubeGrid = object
      grids.push({ object, key: 'cubeGrid' })
    }
    if (adaptiveCubeGrid && adaptiveGridData) {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(adaptiveGridData.positions, 3))
      const object = makeLines(geometry, 'adaptive-feature-grid')
      engine.adaptiveGrid = object
      grids.push({ object, key: 'adaptiveGrid' })
    }
    return () => {
      for (const { object, key } of grids) {
        engine.scene.remove(object)
        disposeObjectTree(object)
        if (engine[key] === object) engine[key] = undefined
      }
    }
  }, [model, showCubeGrid, adaptiveCubeGrid, cubeGridSize, cubeGridBounds, adaptiveGridData])

  useEffect(() => {
    const axes = engineRef.current?.axisGroup
    if (axes) axes.visible = showAxes
  }, [model, showAxes])

  useEffect(() => {
    const mesh = engineRef.current?.mesh
    if (!mesh) return
    const previous = mesh.getObjectByName('model-dimensions')
    if (previous) {
      mesh.remove(previous)
      disposeObjectTree(previous)
    }
    if (showDimensions && model) mesh.add(makeDimensionsOverlay(model.geometry, scale))
  }, [model, scale, showDimensions])

  useEffect(() => {
    const mesh = engineRef.current?.mesh
    if (!mesh) return
    const previous = mesh.getObjectByName('printability-highlights')
    if (previous) {
      mesh.remove(previous)
      disposeObjectTree(previous)
    }
    if (analysisOverlay && analysisOverlay.getAttribute('position')?.count) {
      const overlay = new THREE.Mesh(
        analysisOverlay,
        new THREE.MeshBasicMaterial({
          color: '#f05d5a', side: THREE.DoubleSide, transparent: true, opacity: 0.82,
          depthTest: true, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1,
        }),
      )
      overlay.name = 'printability-highlights'
      overlay.renderOrder = 2
      mesh.add(overlay)
    }
  }, [analysisOverlay, model])

  useEffect(() => {
    const mesh = engineRef.current?.mesh
    if (!mesh) return
    const previous = mesh.getObjectByName('mesh-integrity-highlights')
    if (previous) {
      mesh.remove(previous)
      disposeObjectTree(previous)
    }
    if (!integrityOverlay || !integrityOverlay.getAttribute('position')?.count || !integrityIssue) return
    const overlay = new THREE.Mesh(
      integrityOverlay.clone(),
      new THREE.MeshBasicMaterial({
        color: INTEGRITY_ISSUE_COLORS[integrityIssue], side: THREE.DoubleSide, transparent: true, opacity: 0.9,
        depthTest: true, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
      }),
    )
    overlay.name = 'mesh-integrity-highlights'
    overlay.renderOrder = 4
    mesh.add(overlay)
  }, [integrityOverlay, integrityIssue, model])

  useEffect(() => {
    const scene = engineRef.current?.scene
    if (!scene) return
    const previous = scene.getObjectByName('printability-support-columns')
    if (previous) {
      scene.remove(previous)
      disposeObjectTree(previous)
    }
    if (supportOverlay?.getAttribute('position')?.count) {
      const supportMesh = new THREE.Mesh(
        supportOverlay,
        new THREE.MeshBasicMaterial({ color: '#50ad91', side: THREE.DoubleSide, transparent: true, opacity: 0.16, depthTest: true, depthWrite: false }),
      )
      supportMesh.name = 'printability-support-columns'
      supportMesh.renderOrder = 1
      scene.add(supportMesh)
    }
  }, [model, supportOverlay])

  useEffect(() => {
    const scene = engineRef.current?.scene
    if (!scene || !layerPreviewOverlay) return
    const lines: THREE.LineSegments[] = []
    const addLines = (name: string, positions: ArrayBuffer, color: string) => {
      const vertices = new Float32Array(positions)
      if (!vertices.length) return
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3))
      const line = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.96 }))
      line.name = name
      line.renderOrder = 8
      scene.add(line)
      lines.push(line)
    }
    addLines('layer-cross-section', layerPreviewOverlay.currentPositions, '#3979b5')
    addLines('layer-island-warnings', layerPreviewOverlay.islandPositions, '#e6504f')
    return () => {
      for (const line of lines) { scene.remove(line); disposeObjectTree(line) }
    }
  }, [model, layerPreviewOverlay])

  useEffect(() => {
    const engine = engineRef.current
    if (!engine) return
    engine.controls.enabled = !rotateMode && !placeOnBedMode
    const canvas = engine.renderer.domElement
    if ((!showAxes && !placeOnBedMode) || !model) return () => { engine.controls.enabled = true }

    const raycaster = new THREE.Raycaster()
    raycaster.params.Line.threshold = Math.max(Math.max(...model.dimensions) * 0.035, 0.5)
    const pointer = new THREE.Vector2()
    let drag: { pointerId: number; x: number; y: number; axis: number | null; action: 'rotate' | 'free' | AxisTransformMode; axisScreenX: number; axisScreenY: number; pixelsPerUnit: number } | null = null
    const onPointerDown = (event: PointerEvent) => {
      const mesh = engine.mesh
      if (!mesh) return
      const rect = canvas.getBoundingClientRect()
      pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1)
      raycaster.setFromCamera(pointer, engine.camera)
      if (placeOnBedMode) {
        event.preventDefault()
        event.stopImmediatePropagation()
        const hit = raycaster.intersectObject(mesh, false)[0]
        if (!hit?.face) return
        const localPoint = mesh.worldToLocal(hit.point.clone())
        const minimumArea = Math.max(Math.max(...model.dimensions) ** 2 * 0.0001, 0.25)
        const supportNormal = getFlatSupportNormal(mesh.geometry, hit.face.normal, localPoint, minimumArea)
        if (!supportNormal) {
          onError('Choose a broad, flat outside face that can support the model on the bed.')
          return
        }
        const worldNormal = supportNormal.applyMatrix3(new THREE.Matrix3().getNormalMatrix(mesh.matrixWorld)).normalize()
        onPlaceSurface(worldNormal)
        return
      }
      const axisGroup = engine.axisGroup
      const axisHit = axisGroup
        ? raycaster.intersectObject(axisGroup, true).find(hit => Number.isInteger(hit.object.userData.rotationAxis))
        : undefined
      const axis = axisHit ? Number(axisHit.object.userData.rotationAxis) : null
      const hitModel = rotateMode && raycaster.intersectObject(mesh, false).length > 0
      const transformMode = axisTransformModeRef.current
      if (axis !== null && transformMode && axisGroup) {
        event.preventDefault()
        event.stopImmediatePropagation()
        canvas.setPointerCapture(event.pointerId)
        const axisVector = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)][axis].applyQuaternion(axisGroup.quaternion)
        const sampleDistance = Math.max(...model.dimensions) * 0.5
        const origin = axisGroup.position.clone()
        const projectedOrigin = origin.clone().project(engine.camera)
        const projectedTip = origin.clone().addScaledVector(axisVector, sampleDistance).project(engine.camera)
        const axisScreenX = (projectedTip.x - projectedOrigin.x) * rect.width * 0.5
        const axisScreenY = -(projectedTip.y - projectedOrigin.y) * rect.height * 0.5
        const screenLength = Math.hypot(axisScreenX, axisScreenY)
        drag = {
          pointerId: event.pointerId, x: event.clientX, y: event.clientY, axis, action: transformMode,
          axisScreenX: screenLength > 1e-4 ? axisScreenX / screenLength : 1,
          axisScreenY: screenLength > 1e-4 ? axisScreenY / screenLength : 0,
          pixelsPerUnit: screenLength > 1e-4 ? screenLength / sampleDistance : 1,
        }
        onAxisTransformStart(transformMode)
        return
      }
      if (axis === null && !hitModel) return
      event.preventDefault()
      event.stopImmediatePropagation()
      canvas.setPointerCapture(event.pointerId)
      drag = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, axis, action: axis === null ? 'free' : 'rotate', axisScreenX: 1, axisScreenY: 0, pixelsPerUnit: 1 }
    }
    const onPointerMove = (event: PointerEvent) => {
      if (!drag || event.pointerId !== drag.pointerId) return
      event.stopImmediatePropagation()
      const dx = event.clientX - drag.x
      const dy = event.clientY - drag.y
      drag.x = event.clientX
      drag.y = event.clientY
      if ((drag.action === 'move' || drag.action === 'scale') && drag.axis !== null) {
        const projectedDelta = dx * drag.axisScreenX + dy * drag.axisScreenY
        const delta = drag.action === 'move' ? projectedDelta / drag.pixelsPerUnit : projectedDelta * 0.008
        onAxisTransform(drag.action, drag.axis, delta)
        return
      }
      let next: Rotation
      if (drag.action === 'rotate' && drag.axis !== null) {
        next = [...rotationRef.current] as Rotation
        const delta = Math.abs(dx) >= Math.abs(dy) ? dx : -dy
        next[drag.axis] += delta * 0.65
      } else {
        next = [rotationRef.current[0] + dy * 0.65, rotationRef.current[1] + dx * 0.65, rotationRef.current[2]]
      }
      rotationRef.current = next
      onRotationChange(next)
    }
    const onPointerUp = (event: PointerEvent) => {
      if (!drag || event.pointerId !== drag.pointerId) return
      event.stopImmediatePropagation()
      const finishedAction = drag.action
      drag = null
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId)
      if (finishedAction === 'free') onFreeRotateEnd()
      if (finishedAction === 'move' || finishedAction === 'scale') onAxisTransformEnd(finishedAction)
    }
    canvas.addEventListener('pointerdown', onPointerDown, true)
    canvas.addEventListener('pointermove', onPointerMove)
    canvas.addEventListener('pointerup', onPointerUp)
    canvas.addEventListener('pointercancel', onPointerUp)
    return () => {
      engine.controls.enabled = true
      canvas.removeEventListener('pointerdown', onPointerDown, true)
      canvas.removeEventListener('pointermove', onPointerMove)
      canvas.removeEventListener('pointerup', onPointerUp)
      canvas.removeEventListener('pointercancel', onPointerUp)
    }
  }, [model, onAxisTransform, onAxisTransformEnd, onAxisTransformStart, onFreeRotateEnd, onPlaceSurface, onError, onRotationChange, placeOnBedMode, rotateMode, showAxes])

  useEffect(() => {
    const engine = engineRef.current
    if (engine?.grid) engine.grid.visible = showGrid
  }, [showGrid])

  return <div className={`viewport${showAxes ? ' rotate-axes-active' : ''}${rotateMode ? ' rotate-mode' : ''}${placeOnBedMode ? ' place-on-bed-mode' : ''}`} ref={hostRef} />
}

function FilamentSpoolIcon({ size = 14 }: { size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M4 4h12v2c0 1.1-2.7 2-6 2S4 7.1 4 6V4Z"/>
    <path d="M4 6v8c0 1.1 2.7 2 6 2s6-.9 6-2V6"/>
    <path d="M4 10c0 1.1 2.7 2 6 2s6-.9 6-2"/>
    <path d="M7 4V3h6v1"/>
  </svg>
}

function App() {
  const [model, setModel] = useState<ModelInfo | null>(null)
  const [inspectorTab, setInspectorTab] = useState<'model' | 'starred' | 'filament'>('model')
  const [suggestedMaterial, setSuggestedMaterial] = useState<FilamentMaterial>('PLA')
  const [nozzleDiameter, setNozzleDiameter] = useState(0.4)
  const [profileSettings, setProfileSettings] = useState<Record<FilamentMaterial, ProfileSetting[]>>(() => ({
    PLA: PROFILE_DEFAULTS.PLA.map(setting => ({ ...setting })),
    PETG: PROFILE_DEFAULTS.PETG.map(setting => ({ ...setting })),
  }))
  const [projectName, setProjectName] = useState('Untitled project')
  const [rotation, setRotation] = useState<Rotation>([0, 0, 0])
  const rotationStateRef = useRef<Rotation>(rotation)
  rotationStateRef.current = rotation
  const [position, setPosition] = useState<AxisValues>([0, 0, 0])
  const [scale, setScale] = useState<AxisValues>([1, 1, 1])
  const scaleStateRef = useRef<AxisValues>(scale)
  scaleStateRef.current = scale
  const [uniformScale, setUniformScale] = useState(true)
  const uniformScaleRef = useRef(uniformScale)
  uniformScaleRef.current = uniformScale
  const [axisDragMode, setAxisDragMode] = useState<AxisTransformMode | null>(null)
  const modelStateRef = useRef<ModelInfo | null>(model)
  modelStateRef.current = model
  const transformedStats = useMemo(() => {
    if (!model) return null
    return scale[0] === 1 && scale[1] === 1 && scale[2] === 1
      ? { area: model.area, volume: model.volume }
      : getStats(model.geometry, scale)
  }, [model, scale])
  const [rotateMenuOpen, setRotateMenuOpen] = useState(false)
  const [movePanelOpen, setMovePanelOpen] = useState(false)
  const [scalePanelOpen, setScalePanelOpen] = useState(false)
  const [rotateMode, setRotateMode] = useState(false)
  const [placeOnBedMode, setPlaceOnBedMode] = useState(false)
  const [checkPanelOpen, setCheckPanelOpen] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<OverhangCheck | null>(null)
  const [integrityResult, setIntegrityResult] = useState<MeshIntegrityResult | null>(null)
  const [selectedIntegrityIssue, setSelectedIntegrityIssue] = useState<MeshIntegrityIssue | null>(null)
  const [isCheckingIntegrity, setIsCheckingIntegrity] = useState(false)
  const [integrityStatus, setIntegrityStatus] = useState('')
  useEffect(() => {
    if (!analysisResult && inspectorTab === 'starred') setInspectorTab('model')
  }, [analysisResult, inspectorTab])
  const [showGrid, setShowGrid] = useState(true)
  const [showDimensions, setShowDimensions] = useState(false)
  const [advancedToolsEnabled, setAdvancedToolsEnabled] = useState(false)
  const [showCubeGrid, setShowCubeGrid] = useState(false)
  const [adaptiveCubeGrid, setAdaptiveCubeGrid] = useState(false)
  const [cubeGridSize, setCubeGridSize] = useState(30)
  const [adaptiveMinSize, setAdaptiveMinSize] = useState(5)
  const cubeGridTransform = useMemo(() => {
    if (!model) return null
    const quaternion = new THREE.Quaternion().setFromEuler(new THREE.Euler(
      THREE.MathUtils.degToRad(rotation[0]),
      THREE.MathUtils.degToRad(rotation[1]),
      THREE.MathUtils.degToRad(rotation[2]),
    ))
    const scaleVector = new THREE.Vector3(...scale)
    const rotatedBounds = model.geometry.boundingBox!.clone()
      .applyMatrix4(new THREE.Matrix4().compose(new THREE.Vector3(), quaternion, scaleVector))
    return new THREE.Matrix4().compose(
      new THREE.Vector3(position[0], -rotatedBounds.min.y + position[1], position[2]),
      quaternion,
      scaleVector,
    )
  }, [model, position, rotation, scale])
  const cubeGridBounds = useMemo(() => {
    if (!model || !cubeGridTransform) return null
    return model.geometry.boundingBox!.clone().applyMatrix4(cubeGridTransform)
  }, [model, cubeGridTransform])
  const layerHeight = useMemo(() => {
    const profileValue = profileSettings[suggestedMaterial].find(setting => setting.label === 'Layer Height')?.value
    const parsed = Number.parseFloat(profileValue ?? '0.2')
    return Number.isFinite(parsed) ? THREE.MathUtils.clamp(parsed, 0.05, 2) : 0.2
  }, [profileSettings, suggestedMaterial])
  const layerCount = useMemo(() => cubeGridBounds ? Math.max(1, Math.ceil(cubeGridBounds.max.y / layerHeight)) : 1, [cubeGridBounds, layerHeight])
  useEffect(() => setLayerNumber(current => Math.max(1, Math.min(current, layerCount))), [layerCount])
  const buildVolumeFit = useMemo(() => {
    if (!cubeGridBounds) return null
    const dimensions = cubeGridBounds.getSize(new THREE.Vector3())
    const halfWidth = BUILD_VOLUME.width / 2, halfDepth = BUILD_VOLUME.depth / 2
    const overflow = {
      X: Math.max(0, -halfWidth - cubeGridBounds.min.x, cubeGridBounds.max.x - halfWidth),
      Y: Math.max(0, -cubeGridBounds.min.y, cubeGridBounds.max.y - BUILD_VOLUME.height),
      Z: Math.max(0, -halfDepth - cubeGridBounds.min.z, cubeGridBounds.max.z - halfDepth),
    }
    return { dimensions, overflow, fits: Object.values(overflow).every(value => value <= 0.05) }
  }, [cubeGridBounds])
  const cubeGridLayout = useMemo(
    () => cubeGridBounds ? getCubeGridLayout(cubeGridBounds, cubeGridSize) : null,
    [cubeGridBounds, cubeGridSize],
  )
  const adaptiveMaxSize = useMemo(() => {
    if (!cubeGridBounds) return adaptiveMinSize
    const dimensions = cubeGridBounds.getSize(new THREE.Vector3())
    return Math.max(adaptiveMinSize, Math.max(dimensions.x, dimensions.y, dimensions.z) / 2)
  }, [cubeGridBounds, adaptiveMinSize])
  const adaptiveGridData = useMemo(() => {
    if (!model || !advancedToolsEnabled || !adaptiveCubeGrid || !cubeGridBounds || !cubeGridTransform) return null
    return makeAdaptiveGridData(model.geometry, cubeGridBounds, cubeGridTransform, adaptiveMinSize, adaptiveMaxSize)
  }, [model, advancedToolsEnabled, adaptiveCubeGrid, cubeGridBounds, cubeGridTransform, adaptiveMinSize, adaptiveMaxSize])
  const [isDragging, setIsDragging] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isChecking, setIsChecking] = useState(false)
  const [checkStatus, setCheckStatus] = useState('')
  const [isFindingOrientations, setIsFindingOrientations] = useState(false)
  const [orientationStatus, setOrientationStatus] = useState('')
  const [orientationSuggestions, setOrientationSuggestions] = useState<OrientationSuggestion[] | null>(null)
  const [layerPreviewEnabled, setLayerPreviewEnabled] = useState(false)
  const [layerNumber, setLayerNumber] = useState(1)
  const [layerPreviewData, setLayerPreviewData] = useState<LayerPreviewData | null>(null)
  const [isSlicingLayer, setIsSlicingLayer] = useState(false)
  const [layerPreviewStatus, setLayerPreviewStatus] = useState('')
  const [isScanningLayers, setIsScanningLayers] = useState(false)
  const [layerScanStatus, setLayerScanStatus] = useState('')
  const [layerScanData, setLayerScanData] = useState<LayerScanData | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const projectNameInputRef = useRef<HTMLInputElement>(null)
  const cameraControls = useRef<(() => void) | null>(null)
  const viewPresetControls = useRef<((view: ViewPreset) => void) | null>(null)
  const loadWorkerRef = useRef<Worker | null>(null)
  const checkWorkerRef = useRef<Worker | null>(null)
  const orientationWorkerRef = useRef<Worker | null>(null)
  const layerPreviewWorkerRef = useRef<Worker | null>(null)
  const layerScanWorkerRef = useRef<Worker | null>(null)
  const integrityWorkerRef = useRef<Worker | null>(null)
  const cancelIntegrityCheck = useCallback(() => {
    integrityWorkerRef.current?.terminate()
    integrityWorkerRef.current = null
    setIsCheckingIntegrity(false)
    setIntegrityStatus('')
  }, [])
  const cancelCheck = useCallback(() => {
    checkWorkerRef.current?.terminate()
    checkWorkerRef.current = null
    setIsChecking(false)
    setCheckStatus('')
  }, [])
  const cancelOrientationCheck = useCallback(() => {
    orientationWorkerRef.current?.terminate()
    orientationWorkerRef.current = null
    setIsFindingOrientations(false)
    setOrientationStatus('')
  }, [])
  const cancelLayerScan = useCallback(() => {
    layerScanWorkerRef.current?.terminate()
    layerScanWorkerRef.current = null
    setIsScanningLayers(false)
    setLayerScanStatus('')
  }, [])

  useEffect(() => () => {
    loadWorkerRef.current?.terminate()
    checkWorkerRef.current?.terminate()
    orientationWorkerRef.current?.terminate()
    layerPreviewWorkerRef.current?.terminate()
    layerScanWorkerRef.current?.terminate()
    integrityWorkerRef.current?.terminate()
  }, [])

  useEffect(() => {
    cancelOrientationCheck()
    setOrientationSuggestions(null)
  }, [model, rotation, position, scale, nozzleDiameter, cancelOrientationCheck])

  useEffect(() => {
    cancelLayerScan()
    setLayerScanData(null)
  }, [model, rotation, position, scale, layerHeight, cancelLayerScan])

  useEffect(() => {
    if (!layerPreviewEnabled || !model || isScanningLayers) {
      layerPreviewWorkerRef.current?.terminate()
      layerPreviewWorkerRef.current = null
      setLayerPreviewData(null)
      setIsSlicingLayer(false)
      setLayerPreviewStatus('')
      return
    }
    let worker: Worker | null = null
    let canceled = false
    setLayerPreviewData(null)
    setIsSlicingLayer(true)
    setLayerPreviewStatus('Preparing layer slice…')
    const timer = window.setTimeout(() => {
      if (canceled) return
      try {
        worker = new Worker(new URL('./layer-preview-worker.ts', import.meta.url), { type: 'module' })
        layerPreviewWorkerRef.current = worker
        const positionAttribute = model.geometry.getAttribute('position')
        const positions = (positionAttribute.array as Float32Array).slice()
        const bounds = model.geometry.boundingBox!
        const boundsTuple: [number, number, number, number, number, number] = [bounds.min.x, bounds.min.y, bounds.min.z, bounds.max.x, bounds.max.y, bounds.max.z]
        worker.onmessage = event => {
          if (canceled || layerPreviewWorkerRef.current !== worker) return
          const result = event.data as LayerPreviewWorkerResponse
          if ('stage' in result) { setLayerPreviewStatus(result.stage); return }
          layerPreviewWorkerRef.current = null
          worker?.terminate()
          if ('error' in result) {
            setLayerPreviewData(null)
            setLayerPreviewStatus(result.error)
            setIsSlicingLayer(false)
            return
          }
          setLayerPreviewData(result)
          setLayerPreviewStatus('')
          setIsSlicingLayer(false)
        }
        worker.onerror = () => {
          if (layerPreviewWorkerRef.current !== worker) return
          layerPreviewWorkerRef.current = null
          worker?.terminate()
          setLayerPreviewData(null)
          setLayerPreviewStatus('Layer preview failed. Try a smaller model.')
          setIsSlicingLayer(false)
        }
        worker.postMessage({
          positions: positions.buffer,
          bounds: boundsTuple,
          rotation,
          scale,
          translation: position,
          layerHeight,
          layerNumber: Math.min(layerNumber, layerCount),
        }, [positions.buffer])
      } catch {
        if (worker) worker.terminate()
        if (layerPreviewWorkerRef.current === worker) layerPreviewWorkerRef.current = null
        setLayerPreviewData(null)
        setLayerPreviewStatus('Not enough memory to calculate this layer.')
        setIsSlicingLayer(false)
      }
    }, 140)
    return () => {
      canceled = true
      window.clearTimeout(timer)
      if (worker) worker.terminate()
      if (layerPreviewWorkerRef.current === worker) layerPreviewWorkerRef.current = null
    }
  }, [layerPreviewEnabled, isScanningLayers, layerNumber, layerCount, layerHeight, model, rotation, position, scale])

  useEffect(() => {
    if (!model) return
    setPosition(current => {
      const constrained = constrainToBuildPlate(model.geometry, rotation, scale, current)
      return current[0] === constrained[0] && current[2] === constrained[2] ? current : constrained
    })
  }, [model, rotation, scale])

  const loadFile = useCallback((file?: File) => {
    if (!file) return
    cancelCheck()
    cancelOrientationCheck()
    setOrientationSuggestions(null)
    cancelLayerScan()
    setLayerScanData(null)
    cancelIntegrityCheck()
    setIntegrityResult(null)
    setSelectedIntegrityIssue(null)
    loadWorkerRef.current?.terminate()
    loadWorkerRef.current = null
    setIsProcessing(false)
    setError('')
    setNotice('')
    const extension = file.name.toLowerCase().split('.').pop()
    if (extension !== 'stl' && extension !== '3mf') {
      setError('That file type is not supported yet. Choose an .stl or .3mf file.')
      return
    }
    if (!file.size) {
      setError('This file appears to be empty. Please choose another model file.')
      return
    }
    if (extension === '3mf') {
      let worker: Worker
      try {
        worker = new Worker(new URL('./3mf-worker.ts', import.meta.url), { type: 'module' })
      } catch {
        setError('This browser could not start the 3MF processing worker.')
        return
      }
      loadWorkerRef.current = worker
      setIsProcessing(true)
      worker.onmessage = event => {
        if (loadWorkerRef.current !== worker) return
        loadWorkerRef.current = null
        worker.terminate()
        setIsProcessing(false)
        const result = event.data as ThreeMFWorkerResponse
        if (result.error || !result.positions || !result.normals || !result.dimensions || !result.bounds) {
          setError(result.error || 'This 3MF could not be parsed. Check that it is a valid 3MF archive.')
          return
        }
        const geometry = new THREE.BufferGeometry()
        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(result.positions), 3))
        geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(result.normals), 3))
        const [minX, minY, minZ, maxX, maxY, maxZ] = result.bounds
        geometry.boundingBox = new THREE.Box3(new THREE.Vector3(minX, minY, minZ), new THREE.Vector3(maxX, maxY, maxZ))
        geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Math.hypot(...result.dimensions.map(value => value * 0.5)))
        setModel({
          name: file.name, size: file.size, format: '3MF', parameters: result.parameters ?? [],
          triangles: result.triangles ?? Math.floor(geometry.getAttribute('position').count / 3),
          dimensions: result.dimensions, volume: result.volume ?? 0, area: result.area ?? 0, geometry,
        })
        setProjectName(file.name.replace(/\.3mf$/i, '') || 'Untitled project')
        setRotation([0, 0, 0]); setPosition([0, 0, 0]); setScale([1, 1, 1]); setUniformScale(true)
        setMovePanelOpen(false); setScalePanelOpen(false); setAxisDragMode(null); setRotateMode(false); setPlaceOnBedMode(false)
        setAnalysisResult(null); setCheckPanelOpen(false)
        const count = result.parameters?.length ?? 0
        setNotice(count ? `3MF model loaded · ${count} print parameters found` : '3MF model loaded · no slicer settings found')
        window.setTimeout(() => setNotice(''), 3500)
      }
      worker.onerror = () => {
        if (loadWorkerRef.current !== worker) return
        loadWorkerRef.current = null; worker.terminate(); setIsProcessing(false)
        setError('The 3MF processing worker stopped unexpectedly. Try exporting the 3MF again or use a smaller file.')
      }
      file.arrayBuffer().then(buffer => {
        if (loadWorkerRef.current === worker) worker.postMessage({ buffer }, [buffer])
      }).catch(() => {
        if (loadWorkerRef.current !== worker) return
        loadWorkerRef.current = null; worker.terminate(); setIsProcessing(false)
        setError('We could not read this 3MF file. Please try another file.')
      })
      return
    }
    if (!file.size) {
      setError('This file appears to be empty. Please choose another STL.')
      return
    }

    let worker: Worker
    try {
      worker = new Worker(new URL('./stl-worker.ts', import.meta.url), { type: 'module' })
    } catch {
      setError('This browser could not start the STL processing worker.')
      return
    }
    loadWorkerRef.current = worker
    setIsProcessing(true)
    worker.onmessage = event => {
      if (loadWorkerRef.current !== worker) return
      loadWorkerRef.current = null
      worker.terminate()
      setIsProcessing(false)
      const result = event.data as STLWorkerResponse
      if (result.error || !result.positions || !result.normals || !result.dimensions || !result.bounds) {
        setError(result.error || 'This STL could not be parsed. It may be damaged or contain no valid triangles.')
        return
      }
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(result.positions), 3))
      geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(result.normals), 3))
      const [minX, minY, minZ, maxX, maxY, maxZ] = result.bounds
      geometry.boundingBox = new THREE.Box3(new THREE.Vector3(minX, minY, minZ), new THREE.Vector3(maxX, maxY, maxZ))
      geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Math.hypot(...result.dimensions.map(value => value * 0.5)))
      setModel({
        name: file.name,
        size: file.size,
        format: 'STL',
        parameters: [],
        triangles: result.triangles ?? Math.floor(geometry.getAttribute('position').count / 3),
        dimensions: result.dimensions,
        volume: result.volume ?? 0,
        area: result.area ?? 0,
        geometry,
      })
      setProjectName(file.name.replace(/\.stl$/i, '') || 'Untitled project')
      setRotation([0, 0, 0])
      setPosition([0, 0, 0])
      setScale([1, 1, 1])
      setUniformScale(true)
      setMovePanelOpen(false)
      setScalePanelOpen(false)
      setAxisDragMode(null)
      setRotateMode(false)
      setPlaceOnBedMode(false)
      setAnalysisResult(null)
      setCheckPanelOpen(false)
      setNotice('Model loaded successfully')
      window.setTimeout(() => setNotice(''), 2800)
    }
    worker.onerror = () => {
      if (loadWorkerRef.current !== worker) return
      loadWorkerRef.current = null
      worker.terminate()
      setIsProcessing(false)
      setError('The STL processing worker stopped unexpectedly. Try exporting the STL again or use a smaller file.')
    }
    file.arrayBuffer().then(buffer => {
      if (loadWorkerRef.current !== worker) return
      worker.postMessage({ buffer }, [buffer])
    }).catch(() => {
      if (loadWorkerRef.current !== worker) return
      loadWorkerRef.current = null
      worker.terminate()
      setIsProcessing(false)
      setError('We could not read this file. Please try another STL.')
    })
  }, [cancelCheck, cancelIntegrityCheck, cancelOrientationCheck, cancelLayerScan])

  const chooseFile = () => inputRef.current?.click()
  const resetView = () => cameraControls.current?.()
  const toggleRotatePanel = () => {
    setPlaceOnBedMode(false)
    setMovePanelOpen(false)
    setScalePanelOpen(false)
    if (rotateMenuOpen) {
      setRotateMenuOpen(false)
      setRotateMode(false)
    } else {
      setCheckPanelOpen(false)
      setRotateMenuOpen(true)
    }
  }
  const closeRotatePanel = () => {
    setRotateMenuOpen(false)
    setRotateMode(false)
    setPlaceOnBedMode(false)
    setMovePanelOpen(false)
    setScalePanelOpen(false)
  }
  const toggleCheckPanel = () => {
    cancelCheck()
    setPlaceOnBedMode(false)
    setMovePanelOpen(false)
    setScalePanelOpen(false)
    setRotateMenuOpen(false)
    setRotateMode(false)
    if (checkPanelOpen) {
      setCheckPanelOpen(false)
    } else {
      setRotateMenuOpen(false)
      setRotateMode(false)
      setCheckPanelOpen(true)
    }
  }
  const toggleMovePanel = () => {
    const open = !movePanelOpen
    setMovePanelOpen(open)
    setScalePanelOpen(false)
    setRotateMenuOpen(false)
    setRotateMode(false)
    setPlaceOnBedMode(false)
    setCheckPanelOpen(false)
  }
  const toggleScalePanel = () => {
    const open = !scalePanelOpen
    setScalePanelOpen(open)
    setMovePanelOpen(false)
    setRotateMenuOpen(false)
    setRotateMode(false)
    setPlaceOnBedMode(false)
    setCheckPanelOpen(false)
  }
  const runMeshIntegrityCheck = () => {
    if (!model) return
    cancelIntegrityCheck()
    setIntegrityResult(null)
    setSelectedIntegrityIssue(null)
    setError('')
    let worker: Worker
    try {
      worker = new Worker(new URL('./mesh-integrity-worker.ts', import.meta.url), { type: 'module' })
    } catch {
      setError('This browser could not start the mesh integrity worker.')
      return
    }
    integrityWorkerRef.current = worker
    setIsCheckingIntegrity(true)
    setIntegrityStatus('Preparing mesh integrity check…')
    let positions: Float32Array
    let indices: Uint32Array | undefined
    let bounds: [number, number, number, number, number, number]
    try {
      const positionAttribute = model.geometry.getAttribute('position')
      positions = new Float32Array(positionAttribute.array as ArrayLike<number>)
      const indexAttribute = model.geometry.getIndex()
      if (indexAttribute) indices = Uint32Array.from(indexAttribute.array)
      const modelBounds = model.geometry.boundingBox!
      bounds = [modelBounds.min.x, modelBounds.min.y, modelBounds.min.z, modelBounds.max.x, modelBounds.max.y, modelBounds.max.z]
    } catch {
      integrityWorkerRef.current = null
      worker.terminate()
      setIsCheckingIntegrity(false)
      setIntegrityStatus('')
      setError('Not enough memory to prepare this model for the mesh integrity check.')
      return
    }
    const request: { positions: ArrayBuffer; indices?: ArrayBuffer; bounds: typeof bounds } = {
      positions: positions.buffer,
      ...(indices ? { indices: indices.buffer } : {}),
      bounds,
    }
    const transfers: Transferable[] = [request.positions]
    if (request.indices) transfers.push(request.indices)
    worker.onmessage = event => {
      if (integrityWorkerRef.current !== worker) return
      const result = event.data as MeshIntegrityWorkerResponse
      if (result.stage) {
        setIntegrityStatus(result.stage)
        return
      }
      integrityWorkerRef.current = null
      worker.terminate()
      setIsCheckingIntegrity(false)
      setIntegrityStatus('')
      if (result.error || !result.counts || !result.triangleCounts || !result.highlights) {
        setError(result.error || 'The mesh integrity check could not process this model.')
        return
      }
      const highlights = {} as Record<MeshIntegrityIssue, THREE.BufferGeometry>
      for (const issue of ['openEdges', 'nonManifold', 'invertedNormals', 'degenerateFaces'] as const) {
        const geometry = new THREE.BufferGeometry()
        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(result.highlights[issue]), 3))
        if (geometry.getAttribute('position').count) geometry.computeVertexNormals()
        highlights[issue] = geometry
      }
      setIntegrityResult({ counts: result.counts, triangleCounts: result.triangleCounts, highlights, sampledHighlights: result.sampledHighlights ?? false })
    }
    worker.onerror = () => {
      if (integrityWorkerRef.current !== worker) return
      integrityWorkerRef.current = null
      worker.terminate()
      setIsCheckingIntegrity(false)
      setIntegrityStatus('')
      setError('The mesh integrity worker stopped unexpectedly.')
    }
    worker.postMessage(request, transfers)
  }

  const runPrintabilityCheck = () => {
    if (!model) return
    cancelCheck()
    setError('')
    let worker: Worker
    try {
      worker = new Worker(new URL('./check-worker.ts', import.meta.url), { type: 'module' })
    } catch {
      setError('This browser could not start the printability worker.')
      return
    }
    checkWorkerRef.current = worker
    setIsChecking(true)
    setCheckStatus('Preparing analysis…')
    let positions: Float32Array
    let boundsTuple: [number, number, number, number, number, number]
    try {
      const positionAttribute = model.geometry.getAttribute('position')
      positions = (positionAttribute.array as Float32Array).slice()
      const bounds = model.geometry.boundingBox!
      boundsTuple = [bounds.min.x, bounds.min.y, bounds.min.z, bounds.max.x, bounds.max.y, bounds.max.z]
    } catch {
      checkWorkerRef.current = null
      worker.terminate()
      setIsChecking(false)
      setCheckStatus('')
      setError('Not enough memory to prepare this model for the printability check.')
      return
    }
    worker.onmessage = event => {
      if (checkWorkerRef.current !== worker) return
      const result = event.data as CheckWorkerResponse
      if (result.stage) {
        setCheckStatus(result.stage)
        return
      }
      checkWorkerRef.current = null
      worker.terminate()
      setIsChecking(false)
      setCheckStatus('')
      if (result.error || !result.highlightPositions || !result.highlightNormals || !result.supportPositions || !result.supportNormals) {
        setError(result.error || 'The printability check could not process this model.')
        return
      }
      const highlights = new THREE.BufferGeometry()
      highlights.setAttribute('position', new THREE.BufferAttribute(new Float32Array(result.highlightPositions), 3))
      highlights.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(result.highlightNormals), 3))
      const supports = new THREE.BufferGeometry()
      supports.setAttribute('position', new THREE.BufferAttribute(new Float32Array(result.supportPositions), 3))
      supports.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(result.supportNormals), 3))
      setAnalysisResult({
        triangleCount: result.triangleCount ?? model.triangles,
        originalTriangleCount: result.originalTriangleCount ?? model.triangles,
        simplified: result.simplified ?? false,
        simplificationError: result.simplificationError,
        flaggedTriangles: result.flaggedTriangles ?? 0,
        flaggedArea: result.flaggedArea ?? 0,
        thinWallTriangles: result.thinWallTriangles ?? 0,
        undersizedFeatureTriangles: result.undersizedFeatureTriangles ?? 0,
        minimumWallThickness: result.minimumWallThickness ?? nozzleDiameter * 2,
        minimumFeatureWidth: result.minimumFeatureWidth ?? nozzleDiameter,
        bedContactArea: result.bedContactArea ?? 0,
        minimumContactArea: result.minimumContactArea ?? (nozzleDiameter * 10) ** 2,
        floating: result.floating ?? false,
        limitDegrees: result.limitDegrees ?? 45,
        highlights,
        supports,
      })
    }
    worker.onerror = () => {
      if (checkWorkerRef.current !== worker) return
      checkWorkerRef.current = null
      worker.terminate()
      setIsChecking(false)
      setCheckStatus('')
      setError('The printability worker stopped unexpectedly. Try checking a smaller or repaired STL.')
    }
    try {
      worker.postMessage({ positions: positions.buffer, bounds: boundsTuple, rotation, scale, translation: position, nozzleDiameter }, [positions.buffer])
    } catch {
      checkWorkerRef.current = null
      worker.terminate()
      setIsChecking(false)
      setCheckStatus('')
      setError('Not enough memory to run the printability check on this model.')
    }
  }
  const findOrientationSuggestions = () => {
    if (!model) return
    cancelOrientationCheck()
    setError('')
    setOrientationSuggestions(null)
    let worker: Worker
    try {
      worker = new Worker(new URL('./orientation-worker.ts', import.meta.url), { type: 'module' })
    } catch {
      setError('This browser could not start the orientation assistant.')
      return
    }
    orientationWorkerRef.current = worker
    setIsFindingOrientations(true)
    setOrientationStatus('Preparing geometry…')
    try {
      const positionAttribute = model.geometry.getAttribute('position')
      const positions = (positionAttribute.array as Float32Array).slice()
      const bounds = model.geometry.boundingBox!
      const boundsTuple: [number, number, number, number, number, number] = [bounds.min.x, bounds.min.y, bounds.min.z, bounds.max.x, bounds.max.y, bounds.max.z]
      worker.onmessage = event => {
        if (orientationWorkerRef.current !== worker) return
        const result = event.data as OrientationWorkerResponse
        if (result.stage) { setOrientationStatus(result.stage); return }
        orientationWorkerRef.current = null
        worker.terminate()
        setIsFindingOrientations(false)
        setOrientationStatus('')
        if (result.error || !result.suggestions) {
          setError(result.error || 'The orientation assistant could not find suggestions.')
          return
        }
        setOrientationSuggestions(result.suggestions)
      }
      worker.onerror = () => {
        if (orientationWorkerRef.current !== worker) return
        orientationWorkerRef.current = null
        worker.terminate()
        setIsFindingOrientations(false)
        setOrientationStatus('')
        setError('The orientation assistant stopped unexpectedly. Try a smaller model.')
      }
      worker.postMessage({
        positions: positions.buffer,
        bounds: boundsTuple,
        rotation,
        scale,
        translation: position,
        nozzleDiameter,
        buildVolume: BUILD_VOLUME,
      }, [positions.buffer])
    } catch {
      orientationWorkerRef.current = null
      worker.terminate()
      setIsFindingOrientations(false)
      setOrientationStatus('')
      setError('Not enough memory to prepare orientation suggestions for this model.')
    }
  }

  const runLayerStackScan = () => {
    if (!model || isScanningLayers) return
    cancelLayerScan()
    setLayerScanData(null)
    setError('')
    let worker: Worker
    try {
      worker = new Worker(new URL('./layer-preview-worker.ts', import.meta.url), { type: 'module' })
    } catch {
      setLayerScanStatus('This browser could not start the layer scan worker.')
      return
    }
    layerScanWorkerRef.current = worker
    setIsScanningLayers(true)
    setLayerScanStatus('Preparing whole-model scan…')
    try {
      const positionAttribute = model.geometry.getAttribute('position')
      const positions = (positionAttribute.array as Float32Array).slice()
      const bounds = model.geometry.boundingBox!
      const boundsTuple: [number, number, number, number, number, number] = [bounds.min.x, bounds.min.y, bounds.min.z, bounds.max.x, bounds.max.y, bounds.max.z]
      worker.onmessage = event => {
        if (layerScanWorkerRef.current !== worker) return
        const result = event.data as LayerScanWorkerResponse
        if ('stage' in result) { setLayerScanStatus(result.stage); return }
        layerScanWorkerRef.current = null
        worker.terminate()
        setIsScanningLayers(false)
        setLayerScanStatus('')
        if ('error' in result) {
          setLayerScanStatus(result.error)
          return
        }
        setLayerScanData(result)
      }
      worker.onerror = () => {
        if (layerScanWorkerRef.current !== worker) return
        layerScanWorkerRef.current = null
        worker.terminate()
        setIsScanningLayers(false)
        setLayerScanStatus('The layer scan stopped unexpectedly. Try a smaller model.')
      }
      worker.postMessage({
        positions: positions.buffer,
        bounds: boundsTuple,
        rotation,
        scale,
        translation: position,
        layerHeight,
        layerNumber: 1,
        scanAll: true,
        maxLayers: layerCount,
      }, [positions.buffer])
    } catch {
      layerScanWorkerRef.current = null
      worker.terminate()
      setIsScanningLayers(false)
      setLayerScanStatus('Not enough memory to prepare a full layer scan.')
    }
  }

  const onRotationChange = useCallback((next: Rotation) => {
    cancelCheck()
    setRotation(next)
    setAnalysisResult(null)
  }, [cancelCheck])
  const onFreeRotateEnd = useCallback(() => {
    setRotateMode(false)
    setRotateMenuOpen(true)
  }, [])
  const onAxisTransformStart = useCallback((mode: AxisTransformMode) => {
    setAxisDragMode(mode)
    if (mode === 'move') setMovePanelOpen(false)
    else setScalePanelOpen(false)
  }, [])
  const onAxisTransformEnd = useCallback((mode: AxisTransformMode) => {
    setAxisDragMode(null)
    if (mode === 'move') setMovePanelOpen(true)
    else setScalePanelOpen(true)
  }, [])
  const onAxisTransform = useCallback((mode: AxisTransformMode, axis: number, delta: number) => {
    cancelCheck()
    if (mode === 'move') {
      setPosition(current => {
        const next = [...current] as AxisValues
        next[axis] += delta
        const currentModel = modelStateRef.current
        return currentModel ? constrainToBuildPlate(currentModel.geometry, rotationStateRef.current, scaleStateRef.current, next) : next
      })
      setAnalysisResult(null)
      return
    }
    const multiplier = Math.exp(delta)
    setScale(current => {
      if (uniformScaleRef.current) {
        const uniform = Math.max(0.01, Math.min(5, current[axis] * multiplier))
        return [uniform, uniform, uniform]
      }
      const next = [...current] as AxisValues
      next[axis] = Math.max(0.01, Math.min(5, next[axis] * multiplier))
      return next
    })
    setAnalysisResult(null)
  }, [cancelCheck])
  const onPlaceSurface = useCallback((surfaceNormal: THREE.Vector3) => {
    cancelCheck()
    const currentRotation = rotationStateRef.current
    const currentQuaternion = new THREE.Quaternion().setFromEuler(new THREE.Euler(
      THREE.MathUtils.degToRad(currentRotation[0]),
      THREE.MathUtils.degToRad(currentRotation[1]),
      THREE.MathUtils.degToRad(currentRotation[2]),
    ))
    const alignToBed = new THREE.Quaternion().setFromUnitVectors(surfaceNormal, new THREE.Vector3(0, -1, 0))
    currentQuaternion.premultiply(alignToBed)
    const alignedRotation = new THREE.Euler().setFromQuaternion(currentQuaternion, 'XYZ')
    setRotation([
      THREE.MathUtils.radToDeg(alignedRotation.x),
      THREE.MathUtils.radToDeg(alignedRotation.y),
      THREE.MathUtils.radToDeg(alignedRotation.z),
    ])
    setPlaceOnBedMode(false)
    setRotateMenuOpen(true)
    setAnalysisResult(null)
    setError('')
    setNotice('Model placed on the selected flat surface')
    window.setTimeout(() => setNotice(''), 2800)
  }, [cancelCheck])
  const adjustRotation = (axis: number, amount: number) => {
    cancelCheck()
    setRotation(current => current.map((value, index) => index === axis ? value + amount : value) as Rotation)
    setAnalysisResult(null)
  }
  const resetModelPosition = () => {
    cancelCheck()
    setRotation([0, 0, 0])
    setPosition([0, 0, 0])
    setPlaceOnBedMode(false)
    setAnalysisResult(null)
  }
  const resetModelScale = () => {
    cancelCheck()
    setScale([1, 1, 1])
    setAnalysisResult(null)
  }
  const togglePlaceOnBedMode = () => {
    if (!placeOnBedMode) setRotateMenuOpen(false)
    setPlaceOnBedMode(value => !value)
    setRotateMode(false)
    setError('')
  }
  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'o') {
        event.preventDefault()
        inputRef.current?.click()
      }
    }
    window.addEventListener('keydown', handleShortcut)
    return () => window.removeEventListener('keydown', handleShortcut)
  }, [])

  const onDrop = (event: React.DragEvent) => {
    event.preventDefault()
    setIsDragging(false)
    loadFile(event.dataTransfer.files[0])
  }
  const profileHasChanges = profileSettings[suggestedMaterial].some((setting, index) => setting.value !== PROFILE_DEFAULTS[suggestedMaterial][index]?.value)
  const updateProfileSetting = (index: number, value: string) => setProfileSettings(current => ({
    ...current,
    [suggestedMaterial]: current[suggestedMaterial].map((setting, settingIndex) => settingIndex === index ? { ...setting, value } : setting),
  }))
  const resetProfileDefaults = () => setProfileSettings(current => ({
    ...current,
    [suggestedMaterial]: PROFILE_DEFAULTS[suggestedMaterial].map(setting => ({ ...setting })),
  }))
  const integrityOverlay = selectedIntegrityIssue && integrityResult ? integrityResult.highlights[selectedIntegrityIssue] : null
  const hasMeshIntegrityIssues = Boolean(integrityResult && (
    integrityResult.counts.openEdges || integrityResult.counts.nonManifoldEdges || integrityResult.counts.invertedTriangles || integrityResult.counts.degenerateTriangles
  ))
  const hasPrintabilityWarnings = Boolean(analysisResult && (
    analysisResult.flaggedTriangles || analysisResult.thinWallTriangles || analysisResult.undersizedFeatureTriangles ||
    analysisResult.floating || analysisResult.bedContactArea < analysisResult.minimumContactArea
  ))
  const printabilityVerdict = buildVolumeFit && (!buildVolumeFit.fits || analysisResult?.floating)
    ? 'Not printable'
    : analysisResult && integrityResult && !hasMeshIntegrityIssues && !hasPrintabilityWarnings
      ? 'Ready'
      : 'Review'
  const fitOverflowText = buildVolumeFit
    ? Object.entries(buildVolumeFit.overflow).filter(([, amount]) => amount > 0.05).map(([axis, amount]) => `${axis} +${amount.toFixed(1)} mm`).join(', ')
    : ''
  const layerWarningGroups = useMemo(() => {
    const groups: { startLayer: number; endLayer: number; maxIslands: number }[] = []
    for (const warning of layerScanData?.scanWarnings ?? []) {
      const last = groups[groups.length - 1]
      if (last && warning.layerNumber <= last.endLayer + 1) {
        last.endLayer = warning.layerNumber
        last.maxIslands = Math.max(last.maxIslands, warning.islandCount)
      } else groups.push({ startLayer: warning.layerNumber, endLayer: warning.layerNumber, maxIslands: warning.islandCount })
    }
    return groups
  }, [layerScanData])

  return <div className="app-shell" onDragOver={event => { event.preventDefault(); setIsDragging(true) }} onDragLeave={event => { if (event.target === event.currentTarget) setIsDragging(false) }} onDrop={onDrop}>
    <main className="workspace">
      <section className="canvas-panel">
        <div className="canvas-header">
          <div className="file-crumb">
            <span className="crumb-icon"><FileBox size={16}/></span>
            <input ref={projectNameInputRef} className="project-name-input" aria-label="Project name" title="Edit project name" value={projectName} onChange={event => setProjectName(event.target.value)} onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur() }}/>
            <button className="rename-project-button" type="button" title="Rename project" aria-label="Rename project" onClick={() => projectNameInputRef.current?.focus()}><Pencil size={13}/></button>
            {model && <span className="file-ext">.{model.format.toLowerCase()}</span>}
            {model && <span className="crumb-divider"/>}
            {model && <span className="local-badge"><ShieldCheck size={12}/> Local file</span>}
          </div>
          <div className="canvas-actions">
            <button className={`canvas-tool ${showGrid ? 'selected' : ''}`} title="Toggle grid" onClick={() => setShowGrid(value => !value)}><Grid2X2 size={16}/></button>
            <span className="tool-separator"/>
            <button className="canvas-tool" title="Fit model to view" onClick={resetView}><Maximize size={16}/></button>
          </div>

        </div>

        <div className={`canvas-body${(rotateMenuOpen || movePanelOpen || scalePanelOpen || checkPanelOpen) && model ? ' with-rotate-panel' : ''}`}>
          <Viewport model={model} showGrid={showGrid} showAxes={rotateMenuOpen || rotateMode || movePanelOpen || scalePanelOpen || axisDragMode !== null} showDimensions={showDimensions} rotation={rotation} position={position} scale={scale} showCubeGrid={advancedToolsEnabled && showCubeGrid} adaptiveCubeGrid={advancedToolsEnabled && adaptiveCubeGrid} cubeGridSize={cubeGridSize} cubeGridBounds={cubeGridBounds} adaptiveGridData={adaptiveGridData} rotateMode={rotateMode} placeOnBedMode={placeOnBedMode} axisTransformMode={axisDragMode ?? (movePanelOpen ? 'move' : scalePanelOpen ? 'scale' : null)} analysisOverlay={analysisResult?.highlights ?? null} integrityOverlay={integrityOverlay} integrityIssue={selectedIntegrityIssue} supportOverlay={analysisResult?.supports ?? null} layerPreviewOverlay={layerPreviewEnabled ? layerPreviewData : null} onRotationChange={onRotationChange} onPlaceSurface={onPlaceSurface} onFreeRotateEnd={onFreeRotateEnd} onAxisTransformStart={onAxisTransformStart} onAxisTransform={onAxisTransform} onAxisTransformEnd={onAxisTransformEnd} onError={setError} resetRef={cameraControls} viewPresetRef={viewPresetControls}/>
          {model && <div className="left-tool-rail">
            <button className={`left-tool-button ${rotateMenuOpen || rotateMode || placeOnBedMode ? 'selected' : ''}`} title="Open rotate tools" aria-label="Open rotate tools" aria-expanded={rotateMenuOpen} onClick={toggleRotatePanel}><Rotate3D size={18}/><span>Rotate</span></button>
            <button className={`left-tool-button ${movePanelOpen || axisDragMode === 'move' ? 'selected' : ''}`} title="Move model" aria-label="Move model" aria-expanded={movePanelOpen} onClick={toggleMovePanel}><Move size={18}/><span>Move</span></button>
            <button className={`left-tool-button ${scalePanelOpen || axisDragMode === 'scale' ? 'selected' : ''}`} title="Scale model" aria-label="Scale model" aria-expanded={scalePanelOpen} onClick={toggleScalePanel}><Maximize size={18}/><span>Scale</span></button>
            <button className={`left-tool-button ${checkPanelOpen || analysisResult ? 'selected' : ''}`} title="Check printability" aria-label="Check printability" aria-pressed={checkPanelOpen || Boolean(analysisResult)} onClick={toggleCheckPanel}><ClipboardCheck size={18}/><span>Check</span></button>
            <button className={`left-tool-button ${showDimensions ? 'selected' : ''}`} title="Show measured bounding box" aria-label="Show measured bounding box" aria-pressed={showDimensions} onClick={() => setShowDimensions(value => !value)} disabled={!model}><Ruler size={18}/></button>
          </div>}
          {rotateMenuOpen && model && <div className="rotate-panel" role="dialog" aria-label="Rotate model">
            <div className="rotate-menu-header"><div><Rotate3D size={16}/><strong>Rotate model</strong></div><button title="Close rotate menu" aria-label="Close rotate menu" onClick={closeRotatePanel}><X size={15}/></button></div>
            <button className="free-rotate-toggle" aria-pressed={rotateMode} onClick={() => { setPlaceOnBedMode(false); setRotateMode(true); setRotateMenuOpen(false) }}><span><strong>Free rotate</strong><small>{rotateMode ? 'Drag the model in the viewport' : 'Drag directly on the model'}</small></span><span className={`toggle ${rotateMode ? 'on' : ''}`}><i/></span></button>
            <div className="rotate-menu-divider"/>
            <div className="rotate-menu-label">ROTATE BY 45°</div>
            {(['X', 'Y', 'Z'] as const).map((axis, index) => <div className="rotate-axis-row" key={axis}><span className={`rotate-axis-tag axis-${index === 0 ? 'red' : index === 1 ? 'green' : 'blue'}`}>{axis}</span><span className="rotate-angle">{Math.round(rotation[index])}°</span><button aria-label={`Rotate ${axis} negative 45 degrees`} onClick={() => adjustRotation(index, -45)}>−45°</button><button aria-label={`Rotate ${axis} positive 45 degrees`} onClick={() => adjustRotation(index, 45)}>+45°</button></div>)}
            <button className="reset-rotation" title="Return the model to its initial orientation and centered bed position" onClick={resetModelPosition}><RotateCcw size={14}/> Reset position</button>
            <button className={`run-analysis-button place-surface-button${placeOnBedMode ? ' active' : ''}`} aria-pressed={placeOnBedMode} onClick={togglePlaceOnBedMode}><Triangle size={14}/>{placeOnBedMode ? 'Click a flat face on the model' : 'Place on bed from a flat face'}</button>
          </div>}
          {movePanelOpen && model && <div className="rotate-panel transform-panel" role="dialog" aria-label="Move model">
            <div className="rotate-menu-header"><div><Move size={16}/><strong>Move model</strong></div><button title="Close move panel" aria-label="Close move panel" onClick={() => setMovePanelOpen(false)}><X size={15}/></button></div>
            <p className="analysis-intro">Move the model along the world axes. Values are in millimeters.</p>
            <div className="rotate-menu-label">POSITION · MM</div>
            {(['X', 'Y', 'Z'] as const).map((axis, index) => <label className="axis-transform-row" key={axis}><span className={`rotate-axis-tag axis-${index === 0 ? 'red' : index === 1 ? 'green' : 'blue'}`}>{axis}</span><input type="number" step="1" value={position[index]} aria-label={`Move on ${axis} axis in millimeters`} onChange={event => { const value = Number(event.target.value); cancelCheck(); setPosition(current => current.map((item, i) => i === index ? value : item) as AxisValues); setAnalysisResult(null) }}/><span className="axis-transform-unit">mm</span></label>)}
            <button className="reset-rotation" onClick={() => { cancelCheck(); setPosition([0, 0, 0]); setAnalysisResult(null) }}><RotateCcw size={14}/> Reset movement</button>
          </div>}
          {scalePanelOpen && model && <div className="rotate-panel transform-panel" role="dialog" aria-label="Scale model">
            <div className="rotate-menu-header"><div><Maximize size={16}/><strong>Scale model</strong></div><button title="Close scale panel" aria-label="Close scale panel" onClick={() => setScalePanelOpen(false)}><X size={15}/></button></div>
            <p className="analysis-intro">Drag a colored axis to scale, or enter percentages below.</p>
            <label className="uniform-scale-toggle"><input type="checkbox" checked={uniformScale} onChange={event => { const checked = event.target.checked; cancelCheck(); setAnalysisResult(null); setUniformScale(checked); if (checked) setScale(current => [current[0], current[0], current[0]]) }}/><span>Uniform scaling</span></label>
            <div className="rotate-menu-label">SCALE · PERCENT</div>
            {(['X', 'Y', 'Z'] as const).map((axis, index) => <label className="axis-transform-row" key={axis}><span className={`rotate-axis-tag axis-${index === 0 ? 'red' : index === 1 ? 'green' : 'blue'}`}>{axis}</span><input type="number" min="1" max="500" step="1" value={Math.round(scale[index] * 100)} aria-label={`Scale ${axis} axis in percent`} onChange={event => { const percent = Math.max(1, Math.min(500, Number(event.target.value) || 1)); cancelCheck(); setScale(current => uniformScale ? [percent / 100, percent / 100, percent / 100] : current.map((item, i) => i === index ? percent / 100 : item) as AxisValues); setAnalysisResult(null) }}/><span className="axis-transform-unit">%</span></label>)}
            <button className="reset-rotation" onClick={resetModelScale}><RotateCcw size={14}/> Reset scale</button>
          </div>}
          {checkPanelOpen && model && <div className="analysis-panel" role="dialog" aria-label="Printability check">
            <div className="rotate-menu-header"><div><ClipboardCheck size={16}/><strong>Printability check</strong></div><button title="Close printability panel" aria-label="Close printability panel" onClick={() => setCheckPanelOpen(false)}><X size={15}/></button></div>
            <div className="analysis-profile"><span className="analysis-profile-dot"/>Ultimaker S5 <span>·</span> {nozzleDiameter.toFixed(2).replace(/0$/, '')} mm nozzle</div>
            <section className={`unified-printability-summary ${printabilityVerdict.toLowerCase().replace(' ', '-')}`} aria-label="Unified printability summary">
              <div className="unified-summary-heading"><span>Printability summary</span><strong>{printabilityVerdict}</strong></div>
              {printabilityVerdict === 'Review' && (!analysisResult || !integrityResult) && <p>Run both checks to complete the verdict.</p>}
              <div className="unified-summary-row"><span>Build volume</span><strong>{buildVolumeFit?.fits ? 'Fits' : `Exceeds ${fitOverflowText || 'build limits'}`}</strong></div>
              <div className="unified-summary-row"><span>Bed contact</span><strong>{!analysisResult ? 'Check pending' : analysisResult.floating ? `Floating · ${Math.max(0, cubeGridBounds?.min.y ?? 0).toFixed(2)} mm gap` : analysisResult.bedContactArea < analysisResult.minimumContactArea ? `${analysisResult.bedContactArea.toFixed(1)} mm² · small (<${analysisResult.minimumContactArea.toFixed(1)})` : `${analysisResult.bedContactArea.toFixed(1)} mm²`}</strong></div>
              <div className="unified-summary-row"><span>Overhangs</span><strong>{analysisResult ? analysisResult.flaggedTriangles ? `${analysisResult.flaggedTriangles.toLocaleString()} faces need review` : 'No unsupported faces' : 'Check pending'}</strong></div>
              <div className="unified-summary-row"><span>Thin features</span><strong>{!analysisResult ? 'Check pending' : analysisResult.thinWallTriangles || analysisResult.undersizedFeatureTriangles ? `${analysisResult.thinWallTriangles.toLocaleString()} thin walls · ${analysisResult.undersizedFeatureTriangles.toLocaleString()} fine details` : 'No small features detected'}</strong></div>
              <div className="unified-summary-row"><span>Mesh integrity</span><strong>{integrityResult ? hasMeshIntegrityIssues ? 'Issues found' : 'Pass' : 'Check pending'}</strong></div>
            </section>
            <section className="layer-preview-panel" aria-label="Layer-by-layer preview">
              <label className="layer-preview-toggle"><input type="checkbox" checked={layerPreviewEnabled} onChange={event => { setLayerPreviewEnabled(event.target.checked); setLayerPreviewData(null); setLayerNumber(current => Math.min(current, layerCount)) }}/><span><strong>Layer-by-layer preview</strong><small>Inspect {layerHeight.toFixed(2)} mm layers and unsupported regions.</small></span></label>
              {layerPreviewEnabled && <>
                <div className="layer-preview-range-label"><span>Layer {layerNumber} of {layerCount}</span><strong>Z {(layerPreviewData?.layerHeight ?? Math.max(layerHeight / 2, layerNumber * layerHeight - layerHeight / 2)).toFixed(2)} mm</strong></div>
                <input className="layer-preview-slider" type="range" min={1} max={layerCount} step={1} value={Math.min(layerNumber, layerCount)} aria-label="Preview layer number" onChange={event => setLayerNumber(Number(event.currentTarget.value))}/>
                <div className="layer-preview-status" aria-live="polite">{isSlicingLayer ? layerPreviewStatus || 'Calculating layer…' : layerPreviewData ? layerNumber === 1 ? (cubeGridBounds && cubeGridBounds.min.y > 0.05 ? 'No first-layer section at the bed; the model is floating above it.' : 'First layer rests directly on the bed.') : layerPreviewData.islandCount ? `${layerPreviewData.islandCount} contour${layerPreviewData.islandCount === 1 ? '' : 's'} have no overlap with the previous layer.` : 'No disconnected unsupported contours detected at this layer.' : layerPreviewStatus}</div>
                {layerPreviewData && <div className="layer-preview-legend"><span><i/>Cross-section</span><span><i/>New / unsupported</span></div>}
                {layerPreviewData?.simplified && <div className="analysis-approx-note">Layer analysis simplified from {layerPreviewData.originalTriangleCount.toLocaleString()} to {layerPreviewData.triangleCount.toLocaleString()} triangles; contours and island warnings are approximate.</div>}
                <div className="layer-scan-actions"><button className="run-analysis-button layer-scan-button" onClick={runLayerStackScan} disabled={isScanningLayers || isChecking || isCheckingIntegrity || isFindingOrientations}><Scan size={14}/>{isScanningLayers ? 'Scanning layer stack…' : `Scan all ${layerCount.toLocaleString()} layers`}</button>{isScanningLayers && <button type="button" className="layer-scan-cancel" onClick={cancelLayerScan}>Cancel</button>}</div>
                {layerScanData && <div className="layer-scan-results">
                  <strong>{layerScanData.scanWarnings.length ? `${layerScanData.scanWarnings.length} flagged layer${layerScanData.scanWarnings.length === 1 ? '' : 's'} in ${layerScanData.scannedLayers.toLocaleString()} scanned` : `No unsupported islands found in ${layerScanData.scannedLayers.toLocaleString()} scanned layers`}</strong>
                  {layerWarningGroups.length > 0 && <>
                    <div className="layer-scan-timeline" role="group" aria-label="Layer warning timeline"><i/>{layerWarningGroups.map(group => <button type="button" key={`${group.startLayer}-${group.endLayer}`} style={{ left: `${(group.startLayer - 1) / Math.max(1, layerScanData.scannedLayers - 1) * 100}%` }} title={`Jump to layer ${group.startLayer}`} aria-label={`View layer ${group.startLayer} warning`} onClick={() => setLayerNumber(group.startLayer)}/>)}</div>
                    <div className="layer-scan-warning-list">{layerWarningGroups.map(group => <button type="button" key={`${group.startLayer}-${group.endLayer}`} onClick={() => setLayerNumber(group.startLayer)}><span>{group.startLayer === group.endLayer ? `Layer ${group.startLayer}` : `Layers ${group.startLayer}–${group.endLayer}`}</span><strong>{group.maxIslands} region{group.maxIslands === 1 ? '' : 's'}</strong><small>View</small></button>)}</div>
                  </>}
                  {layerScanData.simplified && <div className="analysis-approx-note">Full scan used a simplified mesh ({layerScanData.triangleCount.toLocaleString()} of {layerScanData.originalTriangleCount.toLocaleString()} triangles); warnings are approximate.</div>}
                </div>}
                {(isScanningLayers || layerScanStatus) && <div className="layer-scan-progress" role="status">{layerScanStatus}</div>}
              </>}
            </section>
            <section className="orientation-assistant" aria-label="Orientation assistant">
              <div className="orientation-assistant-heading"><div><Sparkles size={14}/><strong>Orientation assistant</strong></div><span>Preview-only</span></div>
              <p>Compare a few bed-facing rotations to reduce unsupported surface area while keeping the model inside the build volume.</p>
              <button className="run-analysis-button orientation-run-button" onClick={findOrientationSuggestions} disabled={isFindingOrientations || isChecking || isCheckingIntegrity}>
                <Rotate3D size={14}/>{isFindingOrientations ? (orientationStatus || 'Finding orientations…') : 'Find better orientations'}
              </button>
              {orientationSuggestions && <div className="orientation-suggestions">
                {orientationSuggestions.map((suggestion, index) => <article className="orientation-suggestion-card" key={`${suggestion.rotation.join('-')}-${index}`}>
                  <div className="orientation-suggestion-title"><strong>{index === 0 ? 'Best balance' : `Alternative ${index + 1}`}</strong><span className={suggestion.fits ? 'fits' : 'exceeds'}>{suggestion.fits ? 'Fits build volume' : `Exceeds by ${suggestion.overflow.toFixed(1)} mm`}</span></div>
                  <div className="orientation-suggestion-metrics"><span>Est. unsupported area <b>{suggestion.unsupportedArea.toFixed(1)} mm² ({suggestion.unsupportedPercent.toFixed(1)}%)</b></span><span>Bed contact <b>{suggestion.bedContactArea.toFixed(1)} mm²{suggestion.bedContactArea < suggestion.minimumContactArea ? ' · small' : ''}</b></span><span>Oriented size <b>{suggestion.dimensions.map(value => `${value.toFixed(1)} mm`).join(' × ')}</b></span></div>
                  <button type="button" className="orientation-apply-button" onClick={() => { cancelCheck(); setOrientationSuggestions(null); setRotation(suggestion.rotation); setPosition(current => [current[0], 0, current[2]]); setAnalysisResult(null); setRotateMode(false); setPlaceOnBedMode(false); setRotateMenuOpen(false) }}>Apply orientation</button>
                </article>)}
                <p className="orientation-disclaimer">These are fast geometry estimates, not slicer support paths. Run the printability and mesh checks again after applying.</p>
              </div>}
            </section>
            <section className="mesh-integrity-block" aria-label="Mesh integrity checks">
              <div className="mesh-integrity-title">Mesh integrity</div>
              {!integrityResult && <p className="mesh-integrity-intro">Check for open edges, non-manifold areas, inverted face winding, and degenerate faces.</p>}
              <button className="run-analysis-button secondary integrity-run-button" onClick={runMeshIntegrityCheck} disabled={isCheckingIntegrity || isChecking}>
                <ClipboardCheck size={14}/>{isCheckingIntegrity ? (integrityStatus || 'Checking mesh…') : integrityResult ? 'Run integrity check again' : 'Run mesh integrity check'}
              </button>
              {integrityResult && <>
                <div className={`integrity-summary ${integrityResult.counts.openEdges + integrityResult.counts.nonManifoldEdges + integrityResult.counts.invertedTriangles + integrityResult.counts.degenerateTriangles ? 'warning' : 'pass'}`}>
                  {integrityResult.counts.openEdges + integrityResult.counts.nonManifoldEdges + integrityResult.counts.invertedTriangles + integrityResult.counts.degenerateTriangles ? 'Potential mesh issues found' : 'No mesh integrity issues detected'}
                </div>
                <div className="integrity-issue-list">
                  <button type="button" className={selectedIntegrityIssue === 'openEdges' ? 'selected' : ''} disabled={!integrityResult.counts.openEdges} onClick={() => setSelectedIntegrityIssue(current => current === 'openEdges' ? null : 'openEdges')}><span><i style={{ backgroundColor: INTEGRITY_ISSUE_COLORS.openEdges }}/>Open edges</span><strong>{integrityResult.counts.openEdges.toLocaleString()}</strong><small>{integrityResult.triangleCounts.openEdges.toLocaleString()} adjacent faces</small></button>
                  <button type="button" className={selectedIntegrityIssue === 'nonManifold' ? 'selected' : ''} disabled={!integrityResult.counts.nonManifoldEdges} onClick={() => setSelectedIntegrityIssue(current => current === 'nonManifold' ? null : 'nonManifold')}><span><i style={{ backgroundColor: INTEGRITY_ISSUE_COLORS.nonManifold }}/>Non-manifold edges</span><strong>{integrityResult.counts.nonManifoldEdges.toLocaleString()}</strong><small>{integrityResult.triangleCounts.nonManifold.toLocaleString()} affected faces</small></button>
                  <button type="button" className={selectedIntegrityIssue === 'invertedNormals' ? 'selected' : ''} disabled={!integrityResult.counts.invertedTriangles} onClick={() => setSelectedIntegrityIssue(current => current === 'invertedNormals' ? null : 'invertedNormals')}><span><i style={{ backgroundColor: INTEGRITY_ISSUE_COLORS.invertedNormals }}/>Inverted normals / winding</span><strong>{integrityResult.counts.invertedTriangles.toLocaleString()}</strong><small>affected faces</small></button>
                  <button type="button" className={selectedIntegrityIssue === 'degenerateFaces' ? 'selected' : ''} disabled={!integrityResult.counts.degenerateTriangles} onClick={() => setSelectedIntegrityIssue(current => current === 'degenerateFaces' ? null : 'degenerateFaces')}><span><i style={{ backgroundColor: INTEGRITY_ISSUE_COLORS.degenerateFaces }}/>Degenerate faces</span><strong>{integrityResult.counts.degenerateTriangles.toLocaleString()}</strong><small>affected faces</small></button>
                </div>
                {selectedIntegrityIssue && <button className="clear-analysis-button" onClick={() => setSelectedIntegrityIssue(null)}><X size={13}/> Clear mesh highlight</button>}
                {integrityResult.sampledHighlights && <div className="analysis-approx-note">Highlights are sampled for display; issue counts use the full mesh.</div>}
              </>}
            </section>
            {!analysisResult ? <>
              <p className="analysis-intro">Checks surfaces above the bed against the previous 0.2 mm layer. Green shows only the box-minus-model volume that rises from the bed to the model.</p>
              <button className="run-analysis-button" onClick={runPrintabilityCheck} disabled={isChecking || isCheckingIntegrity}><ClipboardCheck size={15}/>{isChecking ? (checkStatus || 'Checking geometry…') : 'Run check'}</button>
            </> : <>
              <div className={`analysis-summary ${analysisResult.flaggedTriangles ? 'warning' : 'pass'}`}>
                <strong>{analysisResult.flaggedTriangles ? 'Potential supports needed' : 'No unsupported surfaces found'}</strong>
                <span>{analysisResult.flaggedTriangles.toLocaleString()} of {analysisResult.triangleCount.toLocaleString()} analyzed triangles are flagged for support</span>
                {analysisResult.simplified && <span className="analysis-approx-note">Quick approximate analysis: simplified from {analysisResult.originalTriangleCount?.toLocaleString()} triangles. The displayed model and export are unchanged.</span>}
              </div>
              {(analysisResult.thinWallTriangles > 0 || analysisResult.undersizedFeatureTriangles > 0) && <div className="thin-wall-warning"><strong>Thin features detected</strong><span>{analysisResult.thinWallTriangles.toLocaleString()} wall faces below {analysisResult.minimumWallThickness.toFixed(2)} mm; {analysisResult.undersizedFeatureTriangles.toLocaleString()} detail faces below the {analysisResult.minimumFeatureWidth.toFixed(2)} mm nozzle width.</span></div>}
              {analysisResult.flaggedTriangles > 0 && <div className="analysis-legend"><i/> Red: unsupported</div>}
              <div className="analysis-legend"><i className="support-legend-swatch"/> Green: bed-to-model difference (voxelized)</div>
              <p className="analysis-intro">This envelope visualizes the volume difference; it is not print-ready support material.</p>
              <div className="analysis-result-row"><span>Unsupported surface area</span><strong>{analysisResult.flaggedArea.toFixed(1)} mm²</strong></div>
              <button className="run-analysis-button secondary" onClick={runPrintabilityCheck} disabled={isChecking || isCheckingIntegrity}><ClipboardCheck size={14}/>{isChecking ? (checkStatus || 'Checking geometry…') : 'Run again'}</button>
            </>}
            <div className="analysis-divider"/>
            <div className="analysis-manual"><strong>Other FDM rules</strong><p>Walls below twice the selected nozzle width and opposing details below one nozzle width are flagged as estimates. Bridges (10 mm), holes (Ø2 mm), clearances (0.5 mm), and pins (3 mm) still need manual review.</p></div>
            {analysisResult && <button className="clear-analysis-button" title="Remove check highlights and preview" onClick={() => { cancelCheck(); setAnalysisResult(null) }}><X size={14}/> Clear highlights</button>}
          </div>}
          {isDragging && <div className="drag-overlay"><div className="drag-card"><CloudUpload size={30}/><strong>Drop an STL or 3MF to open</strong><span>Release to start viewing</span></div></div>}
          {!model && !isDragging && <div className="welcome-card">
            <div className="welcome-art"><img className="printer-illustration" src="/images/printer-illustration.png" alt="Detailed line drawing of a 3D printer"/></div>
            <div className="welcome-copy"><span className="eyebrow"><Sparkles size={12}/> YOUR 3D WORKSPACE</span><h1>Bring your model<br/>into <em>focus.</em></h1><p>Open an STL or 3MF to explore the mesh, inspect dimensions, and read print settings embedded in a 3MF.</p><button className="primary-cta" onClick={chooseFile}><FileUp size={17}/> Choose a model file <span>↗</span></button><div className="drop-hint"><span className="hint-line"/>or drag and drop it here<span className="hint-line"/></div></div>
          </div>}
          {model && <div className="canvas-hint"><MousePointer2 size={13}/><span>{placeOnBedMode ? 'Click a broad, flat face' : axisDragMode === 'move' ? 'Drag a colored axis to move' : axisDragMode === 'scale' ? 'Drag a colored axis to scale' : rotateMode ? 'Drag the model to rotate' : rotateMenuOpen ? 'Drag a colored axis to rotate' : movePanelOpen || scalePanelOpen ? 'Drag a colored axis or edit values' : 'Drag to orbit'}</span><i/><span>{placeOnBedMode ? 'Selected face will rest on the bed' : axisDragMode ? 'Menu returns when you release' : rotateMode ? 'Turn off Free rotate in menu' : rotateMenuOpen ? 'Or use 45° controls' : movePanelOpen || scalePanelOpen ? 'Select an axis handle' : 'Scroll to zoom'}</span></div>}
          <div className="view-cube" role="group" aria-label="Choose a 3D view"><button className="axis-y" title="View from +Y" aria-label="Top view from positive Y" disabled={!model} onClick={() => viewPresetControls.current?.('Y')}>Y</button><button className="axis-x" title="View from +X" aria-label="Side view from positive X" disabled={!model} onClick={() => viewPresetControls.current?.('X')}>X</button><button className="axis-z" title="View from +Z" aria-label="Front view from positive Z" disabled={!model} onClick={() => viewPresetControls.current?.('Z')}>Z</button><button className="cube-center" title="Reset to isometric view" aria-label="Reset to isometric view" disabled={!model} onClick={resetView}><Box size={14}/></button></div>
          {model && <div className="scale-indicator"><span className="scale-line"/><span>{Math.max(...model.dimensions).toFixed(0)} mm</span></div>}
          {error && <div className="toast error-toast"><Info size={16}/><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss"><X size={15}/></button></div>}
          {notice && <div className="toast success-toast"><Check size={16}/><span>{notice}</span><button onClick={() => setNotice('')} aria-label="Dismiss"><X size={15}/></button></div>}
        </div>
        <div className="canvas-footer"><div className="footer-left"><span className="render-indicator"/><span>{model ? 'Perspective' : '3D viewport'}</span><span className="footer-divider"/><span className="units-label">mm</span></div><div className="footer-right"><span>{model ? `${model.triangles.toLocaleString()} triangles` : 'WebGL ready'}</span><span className="footer-divider"/><span>100%</span></div></div>
      </section>

      <aside className="inspector">
        <div className="inspector-heading"><div><span className="panel-kicker">WORKSPACE</span><h2>Inspector</h2></div></div>
        <div className="inspector-tabs" role="tablist" aria-label="Inspector sections">
          <button type="button" role="tab" aria-label="Model" title="Model" aria-selected={inspectorTab === 'model'} className={inspectorTab === 'model' ? 'active' : ''} onClick={() => setInspectorTab('model')}><Box size={14} aria-hidden="true"/></button>
          <button type="button" role="tab" aria-label="Filament" title="Filament" aria-selected={inspectorTab === 'filament'} className={inspectorTab === 'filament' ? 'active' : ''} onClick={() => setInspectorTab('filament')}><FilamentSpoolIcon/></button>
          {analysisResult && <button type="button" role="tab" aria-selected={inspectorTab === 'starred'} className={inspectorTab === 'starred' ? 'active' : ''} onClick={() => setInspectorTab('starred')}>Suggestions</button>}
        </div>
        <div className="inspector-content" role="tabpanel">
          {inspectorTab === 'filament' ? <div className="filament-tab-content">
            <div className="filament-tab-title"><FilamentSpoolIcon size={17}/><strong>Printing code suggestions</strong></div>
            <label className="printer-select-field"><span>3D printer</span><select aria-label="3D printer" defaultValue="ultimaker-s5"><option value="ultimaker-s5">Ultimaker S5</option></select></label>
            <label className="printer-select-field"><span>Material</span><select aria-label="Material" value={suggestedMaterial} onChange={event => setSuggestedMaterial(event.currentTarget.value as FilamentMaterial)}><option value="PLA">PLA</option><option value="PETG">PETG</option></select></label>
            <label className="printer-select-field"><span>Nozzle diameter</span><select aria-label="Nozzle diameter" value={nozzleDiameter} onChange={event => { cancelCheck(); setNozzleDiameter(Number(event.currentTarget.value)); setAnalysisResult(null) }}><option value={0.25}>0.25 mm</option><option value={0.4}>0.4 mm</option><option value={0.8}>0.8 mm</option><option value={1}>1.0 mm</option></select></label>
            <div className="profile-actions"><span>Profile defaults</span><button type="button" onClick={resetProfileDefaults} disabled={!profileHasChanges}>Reset to profile defaults</button></div>
            <table className="material-settings-table" aria-label={`${suggestedMaterial} settings`}>
              <caption>{suggestedMaterial} settings</caption>
              <tbody>{profileSettings[suggestedMaterial].map((setting, index) => {
                const changed = setting.value !== PROFILE_DEFAULTS[suggestedMaterial][index]?.value
                return <tr key={setting.label}>
                  <th scope="row">{setting.label}</th>
                  <td><div className="profile-value-cell"><input type="text" value={setting.value} aria-label={`${suggestedMaterial} ${setting.label}`} onChange={event => updateProfileSetting(index, event.currentTarget.value)}/>{changed && <span className="profile-modified">Modified</span>}</div></td>
                </tr>
              })}</tbody>
            </table>
          </div> : inspectorTab === 'model' || !analysisResult ? <>
          <div className="section-label">MODEL</div>
          {model ? <>
            <div className="model-card"><div className="model-card-copy"><strong title={model.name}>{model.name}</strong><span>{formatSize(model.size)}</span></div><button className="more-button" title="Remove model" aria-label="Remove model" onClick={() => { cancelCheck(); cancelIntegrityCheck(); setModel(null); setProjectName('Untitled project'); setRotation([0, 0, 0]); setPosition([0, 0, 0]); setScale([1, 1, 1]); setAxisDragMode(null); setRotateMode(false); setPlaceOnBedMode(false); setMovePanelOpen(false); setScalePanelOpen(false); setCheckPanelOpen(false); setAnalysisResult(null); setIntegrityResult(null); setSelectedIntegrityIssue(null); setError('') }}><X size={15}/></button></div>
            <div className="section-label section-gap">DIMENSIONS <span>mm</span></div>
            <table className="dimensions-table" aria-label="Model dimensions in millimeters"><tbody>
              <tr><th scope="row"><span className="dim-axis axis-red">X</span><span>:</span></th><td>{formatMeasure(model.dimensions[0] * scale[0])}</td></tr>
              <tr><th scope="row"><span className="dim-axis axis-green">Y</span><span>:</span></th><td>{formatMeasure(model.dimensions[1] * scale[1])}</td></tr>
              <tr><th scope="row"><span className="dim-axis axis-blue">Z</span><span>:</span></th><td>{formatMeasure(model.dimensions[2] * scale[2])}</td></tr>
            </tbody></table>
            <details className="mesh-statistics-dropdown">
              <summary className="section-label section-gap"><span>MESH STATISTICS</span><ChevronDown size={14}/></summary>
              <div className="stat-list"><div><span><Triangle size={14}/> Triangles</span><strong>{model.triangles.toLocaleString()}</strong></div><div><span><Scan size={14}/> Surface area</span><strong>{(transformedStats?.area ?? model.area).toFixed(2)} mm²</strong></div><div><span><Box size={14}/> Volume</span><strong>{(transformedStats?.volume ?? model.volume).toFixed(2)} mm³</strong></div></div>
            </details>
            {model.format === '3MF' && <>
              <div className="section-label section-gap">3MF PRINT PARAMETERS <span>{model.parameters.length}</span></div>
              {model.parameters.length ? <div className="parameter-list">{model.parameters.map((parameter, index) => <div className="parameter-row" key={`${parameter.key}-${index}`} title={`Found in ${parameter.source}`}><span>{parameter.key}</span><strong>{parameter.value}</strong></div>)}</div> : <div className="parameter-empty">This 3MF contains model data, but no readable slicer settings were found. Settings can vary by slicer.</div>}
            </>}
            <div className="health-card"><div className="health-icon"><Check size={14}/></div><div><strong>Mesh loaded</strong><span>Ready to inspect in 3D</span></div><span className="health-dot"/></div>
            <div className="cube-grid-settings">
              <label className="advanced-section-toggle">
                <span className="cube-grid-heading">ADVANCED</span>
                <span className={`small-toggle${advancedToolsEnabled ? ' is-on' : ''}`}>
                  <input type="checkbox" role="switch" aria-label="Enable advanced grid options" checked={advancedToolsEnabled} onChange={event => setAdvancedToolsEnabled(event.currentTarget.checked)}/>
                  <span className="small-toggle-thumb"/>
                </span>
              </label>
              {advancedToolsEnabled && <div className="advanced-grid-options">
                <label className="cube-grid-toggle"><input type="checkbox" checked={showCubeGrid} onChange={event => setShowCubeGrid(event.currentTarget.checked)}/><span>Bounding-box cubes</span></label>
                <label className="cube-grid-toggle adaptive-grid-toggle"><input type="checkbox" checked={adaptiveCubeGrid} onChange={event => setAdaptiveCubeGrid(event.currentTarget.checked)}/><span>Adaptive feature boxes</span></label>
                {(showCubeGrid || adaptiveCubeGrid) && <>
                  {showCubeGrid && <label className="cube-grid-size-row"><span>Cube edge length</span><input type="number" min="0.1" max="500" step="0.1" value={cubeGridSize} onChange={event => { const value = Number(event.currentTarget.value); if (Number.isFinite(value) && value >= 0.1 && value <= 500) setCubeGridSize(value) }}/><span>mm</span></label>}
                  {adaptiveCubeGrid && <>
                    <label className="cube-grid-size-row"><span>Minimum box edge</span><input type="number" min="5" max="500" step="0.1" value={adaptiveMinSize} onChange={event => { const value = Number(event.currentTarget.value); if (Number.isFinite(value) && value >= 5 && value <= 500) setAdaptiveMinSize(value) }}/><span>mm</span></label>
                    <div className="cube-grid-count">Adaptive range: {adaptiveMinSize.toFixed(1)}–{adaptiveMaxSize.toFixed(1)} mm (max = half the longest side, or minimum for small models)</div>
                  </>}
                  {showCubeGrid && cubeGridLayout && <div className="cube-grid-count">Uniform grid: {cubeGridLayout.counts.join(' × ')} · {cubeGridLayout.total.toLocaleString()} cubes</div>}
                  {showCubeGrid && cubeGridLayout && cubeGridLayout.total > MAX_CUBE_GRID_CELLS && <div className="cube-grid-warning" role="status">Uniform grid is limited to {MAX_CUBE_GRID_CELLS.toLocaleString()} cubes. Increase the cube size.</div>}
                  {adaptiveCubeGrid && adaptiveGridData && <div className="cube-grid-count">Adaptive grid: {adaptiveGridData.cellCount.toLocaleString()} boxes</div>}
                  {adaptiveCubeGrid && adaptiveGridData?.limited && <div className="cube-grid-warning" role="status">Detail limit reached; some boxes may exceed the selected size range. Raise the minimum edge to reduce refinement.</div>}
                  {adaptiveCubeGrid && <div className="cube-grid-hint">Larger boxes in simpler areas; dense mesh regions refine down to the selected minimum edge.</div>}
                  {showCubeGrid && !adaptiveCubeGrid && <div className="cube-grid-hint">Uniform boxes expand evenly to complete equal-sized cubes.</div>}
                </>}
              </div>}
            </div>
          </> : <>
            <div className="empty-model empty-upload-only"><button className="empty-upload-button" onClick={chooseFile} disabled={isProcessing}><Upload size={15}/> {isProcessing ? 'Processing…' : 'Open file'}</button><span className="empty-format-hint">Supports .STL and .3MF</span></div>
          </>}
          </> : <div className="suggestions-panel"><strong>Recomandari pentru Ultimaker S5</strong><select aria-label="Filament material" value={suggestedMaterial} onChange={event => setSuggestedMaterial(event.currentTarget.value as 'PLA' | 'PETG')}><option value="PLA">PLA</option><option value="PETG">PETG</option></select>{suggestedMaterial === 'PLA' && <table className="suggestions-table" aria-label="PLA printing settings"><tbody><tr><th scope="row">Printing temperature</th><td>215 C</td></tr><tr><th scope="row">Bed temperature</th><td>70 C</td></tr><tr><th scope="row">Brim</th><td>Yes</td></tr><tr><th scope="row">Brim width</th><td>3 - 7 mm</td></tr><tr><th scope="row">Brim only on outside</th><td>Yes</td></tr></tbody></table>}<div className="support-suggestion"><strong>Support:</strong><span>{analysisResult && analysisResult.flaggedTriangles > 0 ? 'Check printer support' : 'Analizati modelul pentru a vedea daca este nevoie de suporti'}</span></div></div>}
        </div>
      </aside>
    </main>
    <input ref={inputRef} type="file" accept=".stl,.3mf,model/stl,application/vnd.ms-package.3dmanufacturing-3dmodel+xml" hidden onChange={event => { loadFile(event.target.files?.[0]); event.currentTarget.value = '' }}/>
  </div>
}

export default App
