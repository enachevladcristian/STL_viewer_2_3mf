import * as THREE from 'three'

export type OverhangCheck = {
  highlights: THREE.BufferGeometry
  triangleCount: number
  flaggedTriangles: number
  flaggedArea: number
  limitDegrees: number
  supports: THREE.BufferGeometry
}

type TriangleData = {
  positions: Float32Array
  areas: Float32Array
  triangleCount: number
  bins: number[][]
  minY: number
  maxY: number
  minZ: number
  maxZ: number
  orientationSign: number
  yCellSize: number
  zCellSize: number
}

const BIN_COUNT = 64
const DEFAULT_LAYER_HEIGHT = 0.2

function buildLayerIndex(
  geometry: THREE.BufferGeometry,
  rotation: THREE.Euler,
  scale: THREE.Vector3,
  translation: [number, number, number],
): TriangleData {
  const position = geometry.getAttribute('position')
  const index = geometry.getIndex()
  const count = index ? index.count : position.count
  const triangleCount = Math.floor(count / 3)
  const bounds = geometry.boundingBox?.clone() ?? new THREE.Box3()
  const center = bounds.getCenter(new THREE.Vector3())
  const transformedPositions = new Float32Array(triangleCount * 9)
  const areas = new Float32Array(triangleCount)
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3()
  const edge1 = new THREE.Vector3(), edge2 = new THREE.Vector3(), normal = new THREE.Vector3()
  let minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity
  let signedVolume = 0

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const offset = triangle * 3
    const ia = index ? index.getX(offset) : offset
    const ib = index ? index.getX(offset + 1) : offset + 1
    const ic = index ? index.getX(offset + 2) : offset + 2
    a.fromBufferAttribute(position, ia).sub(center).multiply(scale).applyEuler(rotation)
    b.fromBufferAttribute(position, ib).sub(center).multiply(scale).applyEuler(rotation)
    c.fromBufferAttribute(position, ic).sub(center).multiply(scale).applyEuler(rotation)
    const base = triangle * 9
    transformedPositions[base] = a.x
    transformedPositions[base + 1] = a.y
    transformedPositions[base + 2] = a.z
    transformedPositions[base + 3] = b.x
    transformedPositions[base + 4] = b.y
    transformedPositions[base + 5] = b.z
    transformedPositions[base + 6] = c.x
    transformedPositions[base + 7] = c.y
    transformedPositions[base + 8] = c.z
    edge1.subVectors(b, a)
    edge2.subVectors(c, a)
    normal.crossVectors(edge1, edge2)
    signedVolume += a.dot(normal) / 6
    areas[triangle] = normal.length() * 0.5
    minY = Math.min(minY, a.y, b.y, c.y)
    maxY = Math.max(maxY, a.y, b.y, c.y)
    minZ = Math.min(minZ, a.z, b.z, c.z)
    maxZ = Math.max(maxZ, a.z, b.z, c.z)
  }

  if (!Number.isFinite(minY)) minY = maxY = minZ = maxZ = 0
  const bedOffset = -minY + translation[1]
  for (let coordinate = 0; coordinate < transformedPositions.length; coordinate += 3) {
    transformedPositions[coordinate] += translation[0]
    transformedPositions[coordinate + 1] += bedOffset
    transformedPositions[coordinate + 2] += translation[2]
  }
  minY += bedOffset
  maxY += bedOffset
  minZ += translation[2]
  maxZ += translation[2]
  const yCellSize = Math.max((maxY - minY) / BIN_COUNT, 1e-6)
  const zCellSize = Math.max((maxZ - minZ) / BIN_COUNT, 1e-6)
  const bins = Array.from({ length: BIN_COUNT * BIN_COUNT }, () => [] as number[])
  const getYBin = (value: number) => THREE.MathUtils.clamp(Math.floor((value - minY) / yCellSize), 0, BIN_COUNT - 1)
  const getZBin = (value: number) => THREE.MathUtils.clamp(Math.floor((value - minZ) / zCellSize), 0, BIN_COUNT - 1)

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const base = triangle * 9
    const ys = [transformedPositions[base + 1], transformedPositions[base + 4], transformedPositions[base + 7]]
    const zs = [transformedPositions[base + 2], transformedPositions[base + 5], transformedPositions[base + 8]]
    const yStart = getYBin(Math.min(...ys)), yEnd = getYBin(Math.max(...ys))
    const zStart = getZBin(Math.min(...zs)), zEnd = getZBin(Math.max(...zs))
    for (let yi = yStart; yi <= yEnd; yi += 1) {
      for (let zi = zStart; zi <= zEnd; zi += 1) bins[yi * BIN_COUNT + zi].push(triangle)
    }
  }
  return { positions: transformedPositions, areas, triangleCount, bins, minY, maxY, minZ, maxZ, orientationSign: signedVolume < 0 ? -1 : 1, yCellSize, zCellSize }
}

function isInsideAt(data: TriangleData, x: number, y: number, z: number) {
  const yi = THREE.MathUtils.clamp(Math.floor((y - data.minY) / data.yCellSize), 0, BIN_COUNT - 1)
  const zi = THREE.MathUtils.clamp(Math.floor((z - data.minZ) / data.zCellSize), 0, BIN_COUNT - 1)
  const candidates = data.bins[yi * BIN_COUNT + zi]
  let intersections = 0

  // Cast along X and count crossings through the closed STL shell (odd/even rule).
  for (const triangle of candidates) {
    const base = triangle * 9
    const ax = data.positions[base], ay = data.positions[base + 1], az = data.positions[base + 2]
    const bx = data.positions[base + 3], by = data.positions[base + 4], bz = data.positions[base + 5]
    const cx = data.positions[base + 6], cy = data.positions[base + 7], cz = data.positions[base + 8]
    const denominator = (bz - cz) * (ay - cy) + (cy - by) * (az - cz)
    if (Math.abs(denominator) < 1e-12) continue
    const u = ((bz - cz) * (y - cy) + (cy - by) * (z - cz)) / denominator
    const v = ((cz - az) * (y - cy) + (ay - cy) * (z - cz)) / denominator
    const w = 1 - u - v
    if (u < -1e-8 || v < -1e-8 || w < -1e-8) continue
    const hitX = u * ax + v * bx + w * cx
    if (hitX > x + 1e-7) intersections += 1
  }
  return intersections % 2 === 1
}

function hasPreviousLayerSupport(data: TriangleData, x: number, y: number, z: number, layerHeight: number, supportRadius: number) {
  const targetY = y - layerHeight
  if (targetY < data.minY - layerHeight || targetY > data.maxY + layerHeight) return false
  const offsets = [
    [0, 0],
    [supportRadius, 0], [-supportRadius, 0], [0, supportRadius], [0, -supportRadius],
    [supportRadius * 0.707, supportRadius * 0.707], [supportRadius * 0.707, -supportRadius * 0.707],
    [-supportRadius * 0.707, supportRadius * 0.707], [-supportRadius * 0.707, -supportRadius * 0.707],
  ]
  return offsets.some(([dx, dz]) => isInsideAt(data, x + dx, targetY, z + dz))
}

/** Marks surface triangles above the bed that have no material in the preceding layer beneath them. */
export function findUnsupportedOverhangs(
  geometry: THREE.BufferGeometry,
  rotationDegrees: [number, number, number],
  limitDegrees = 45,
  scaleFactors: [number, number, number] = [1, 1, 1],
  translation: [number, number, number] = [0, 0, 0],
  layerHeight = DEFAULT_LAYER_HEIGHT,
): OverhangCheck {
  const position = geometry.getAttribute('position')
  const index = geometry.getIndex()
  const count = index ? index.count : position.count
  const center = geometry.boundingBox?.getCenter(new THREE.Vector3()) ?? new THREE.Vector3()
  const rotation = new THREE.Euler(
    THREE.MathUtils.degToRad(rotationDegrees[0]),
    THREE.MathUtils.degToRad(rotationDegrees[1]),
    THREE.MathUtils.degToRad(rotationDegrees[2]),
  )
  const scale = new THREE.Vector3(scaleFactors[0], scaleFactors[1], scaleFactors[2])
  const data = buildLayerIndex(geometry, rotation, scale, translation)
  const supportRadius = layerHeight * Math.tan(THREE.MathUtils.degToRad(limitDegrees))
  const layerTolerance = Math.max(layerHeight * 0.05, 0.01)
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3()
  const edge1 = new THREE.Vector3(), edge2 = new THREE.Vector3(), faceNormal = new THREE.Vector3()
  const highlightPositions: number[] = []
  const supportColumns = new Map<string, { x: number; z: number; top: number }>()
  let flaggedTriangles = 0
  let flaggedArea = 0
  const triangleCount = Math.floor(count / 3)
  const wholeModelIsFloating = data.minY > layerHeight + layerTolerance

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const offset = triangle * 3
    const ia = index ? index.getX(offset) : offset
    const ib = index ? index.getX(offset + 1) : offset + 1
    const ic = index ? index.getX(offset + 2) : offset + 2
    a.fromBufferAttribute(position, ia)
    b.fromBufferAttribute(position, ib)
    c.fromBufferAttribute(position, ic)
    const base = triangle * 9
    const ax = data.positions[base], ay = data.positions[base + 1], az = data.positions[base + 2]
    const bx = data.positions[base + 3], by = data.positions[base + 4], bz = data.positions[base + 5]
    const cx = data.positions[base + 6], cy = data.positions[base + 7], cz = data.positions[base + 8]
    edge1.set(bx - ax, by - ay, bz - az)
    edge2.set(cx - ax, cy - ay, cz - az)
    faceNormal.crossVectors(edge1, edge2).normalize()
    const steepDownwardOverhang = faceNormal.y * data.orientationSign < -Math.sin(THREE.MathUtils.degToRad(limitDegrees))
    const samples: [number, number, number][] = [
      [(ax + bx + cx) / 3, (ay + by + cy) / 3, (az + bz + cz) / 3],
      [ax * 0.6 + bx * 0.2 + cx * 0.2, ay * 0.6 + by * 0.2 + cy * 0.2, az * 0.6 + bz * 0.2 + cz * 0.2],
      [ax * 0.2 + bx * 0.6 + cx * 0.2, ay * 0.2 + by * 0.6 + cy * 0.2, az * 0.2 + bz * 0.6 + cz * 0.2],
      [ax * 0.2 + bx * 0.2 + cx * 0.6, ay * 0.2 + by * 0.2 + cy * 0.6, az * 0.2 + bz * 0.2 + cz * 0.6],
    ]
    const unsupportedSamples = samples.filter(([x, y, z]) => {
      if (y <= layerHeight + layerTolerance) return false
      // Treat steep downward facets as needing support even if parity tests
      // mistake nearby thread/concave geometry for a supporting layer.
      return wholeModelIsFloating || steepDownwardOverhang || !hasPreviousLayerSupport(data, x, y, z, layerHeight, supportRadius)
    })
    if (!unsupportedSamples.length) continue

    for (const [x, y, z] of unsupportedSamples) {
      const cell = 2.5
      const key = `${Math.round(x / cell)},${Math.round(z / cell)}`
      const previous = supportColumns.get(key)
      if (!previous || y > previous.top) supportColumns.set(key, { x, z, top: y })
    }
    flaggedTriangles += 1
    flaggedArea += data.areas[triangle]
    highlightPositions.push(
      a.x - center.x, a.y - center.y, a.z - center.z,
      b.x - center.x, b.y - center.y, b.z - center.z,
      c.x - center.x, c.y - center.y, c.z - center.z,
    )
  }

  const highlights = new THREE.BufferGeometry()
  highlights.setAttribute('position', new THREE.Float32BufferAttribute(highlightPositions, 3))
  highlights.computeVertexNormals()

  const supportPositions: number[] = []
  const addQuad = (a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3, d: THREE.Vector3) => {
    supportPositions.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, a.x, a.y, a.z, c.x, c.y, c.z, d.x, d.y, d.z)
  }
  const halfWidth = 0.6
  for (const column of supportColumns.values()) {
    if (column.top <= 0) continue
    const x0 = column.x - halfWidth, x1 = column.x + halfWidth
    const z0 = column.z - halfWidth, z1 = column.z + halfWidth
    const y0 = 0, y1 = column.top
    const p000 = new THREE.Vector3(x0, y0, z0), p100 = new THREE.Vector3(x1, y0, z0)
    const p110 = new THREE.Vector3(x1, y0, z1), p010 = new THREE.Vector3(x0, y0, z1)
    const p001 = new THREE.Vector3(x0, y1, z0), p101 = new THREE.Vector3(x1, y1, z0)
    const p111 = new THREE.Vector3(x1, y1, z1), p011 = new THREE.Vector3(x0, y1, z1)
    addQuad(p000, p100, p101, p001)
    addQuad(p100, p110, p111, p101)
    addQuad(p110, p010, p011, p111)
    addQuad(p010, p000, p001, p011)
    addQuad(p001, p101, p111, p011)
  }
  const supports = new THREE.BufferGeometry()
  supports.setAttribute('position', new THREE.Float32BufferAttribute(supportPositions, 3))
  supports.computeVertexNormals()
  return { highlights, triangleCount, flaggedTriangles, flaggedArea, limitDegrees, supports }
}
