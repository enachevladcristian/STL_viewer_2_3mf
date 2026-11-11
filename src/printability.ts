import * as THREE from 'three'

export type OverhangCheck = {
  highlights: THREE.BufferGeometry
  triangleCount: number
  flaggedTriangles: number
  flaggedArea: number
  thinWallTriangles: number
  limitDegrees: number
  supports: THREE.BufferGeometry
  originalTriangleCount?: number
  simplified?: boolean
  simplificationError?: number
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

function buildBoundingBoxDifference(data: TriangleData) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity
  for (let i = 0; i < data.positions.length; i += 3) {
    minX = Math.min(minX, data.positions[i]); maxX = Math.max(maxX, data.positions[i])
    minY = Math.min(minY, data.positions[i + 1]); maxY = Math.max(maxY, data.positions[i + 1])
    minZ = Math.min(minZ, data.positions[i + 2]); maxZ = Math.max(maxZ, data.positions[i + 2])
  }
  const padding = Math.max(Math.max(maxX - minX, maxY - minY, maxZ - minZ) * 0.08, 2)
  const outerMinX = minX - padding, outerMaxX = maxX + padding
  const outerMinY = 0, outerMaxY = maxY + padding
  const outerMinZ = minZ - padding, outerMaxZ = maxZ + padding
  const longest = Math.max(outerMaxX - outerMinX, outerMaxY - outerMinY, outerMaxZ - outerMinZ, 1e-6)
  const cellSize = longest / 28
  const nx = Math.max(1, Math.ceil((outerMaxX - outerMinX) / cellSize))
  const ny = Math.max(1, Math.ceil((outerMaxY - outerMinY) / cellSize))
  const nz = Math.max(1, Math.ceil((outerMaxZ - outerMinZ) / cellSize))
  const dx = (outerMaxX - outerMinX) / nx, dy = (outerMaxY - outerMinY) / ny, dz = (outerMaxZ - outerMinZ) / nz
  const occupancy = new Uint8Array(nx * ny * nz)
  const cellIndex = (i: number, j: number, k: number) => (j * nz + k) * nx + i

  for (let j = 0; j < ny; j += 1) {
    const y = outerMinY + (j + 0.5) * dy
    for (let k = 0; k < nz; k += 1) {
      const z = outerMinZ + (k + 0.5) * dz
      for (let i = 0; i < nx; i += 1) {
        const x = outerMinX + (i + 0.5) * dx
        // Nudge voxel centers off common STL triangle seams so a face diagonal
        // is not misread as two crossings by the odd/even inside test.
        if (isInsideAt(data, x, y + dy * 0.137, z + dz * 0.271)) occupancy[cellIndex(i, j, k)] = 1
      }
    }
  }

  const supportMask = new Uint8Array(occupancy.length)
  const firstModelCells = new Int16Array(nx * nz).fill(-1)
  const contactHeights = new Float32Array(nx * nz)
  for (let k = 0; k < nz; k += 1) for (let i = 0; i < nx; i += 1) {
    const column = k * nx + i
    const x = outerMinX + (i + 0.5) * dx
    const z = outerMinZ + (k + 0.5) * dz + dz * 0.271
    let firstModelCell = -1
    for (let j = 0; j < ny; j += 1) {
      if (occupancy[cellIndex(i, j, k)]) { firstModelCell = j; break }
    }
    if (firstModelCell < 0) continue
    firstModelCells[column] = firstModelCell
    let low = outerMinY
    let high = outerMinY + (firstModelCell + 0.5) * dy + dy * 0.137
    for (let iteration = 0; iteration < 12; iteration += 1) {
      const middle = (low + high) * 0.5
      if (isInsideAt(data, x, middle + dy * 0.137, z)) high = middle
      else low = middle
    }
    contactHeights[column] = high
    for (let j = 0; j < firstModelCell; j += 1) supportMask[cellIndex(i, j, k)] = 1
  }

  const positions: number[] = []
  const faces = [
    { step: [1, 0, 0], points: (_x0: number, x1: number, y0: number, y1: number, z0: number, z1: number) => [[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[x1,y0,z1]] },
    { step: [-1, 0, 0], points: (x0: number, _x1: number, y0: number, y1: number, z0: number, z1: number) => [[x0,y0,z1],[x0,y1,z1],[x0,y1,z0],[x0,y0,z0]] },
    { step: [0, 1, 0], points: (x0: number, x1: number, _y0: number, y1: number, z0: number, z1: number) => [[x0,y1,z0],[x0,y1,z1],[x1,y1,z1],[x1,y1,z0]] },
    { step: [0, -1, 0], points: (x0: number, x1: number, y0: number, _y1: number, z0: number, z1: number) => [[x0,y0,z1],[x0,y0,z0],[x1,y0,z0],[x1,y0,z1]] },
    { step: [0, 0, 1], points: (x0: number, x1: number, y0: number, y1: number, _z0: number, z1: number) => [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]] },
    { step: [0, 0, -1], points: (x0: number, x1: number, y0: number, y1: number, z0: number, _z1: number) => [[x1,y0,z0],[x0,y0,z0],[x0,y1,z0],[x1,y1,z0]] },
  ]
  for (let j = 0; j < ny; j += 1) for (let k = 0; k < nz; k += 1) for (let i = 0; i < nx; i += 1) {
    if (!supportMask[cellIndex(i, j, k)]) continue
    const x0 = outerMinX + i * dx, x1 = x0 + dx
    const y0 = outerMinY + j * dy
    let y1 = y0 + dy
    const column = k * nx + i
    if (firstModelCells[column] === j + 1) y1 = Math.max(y0, contactHeights[column])
    const z0 = outerMinZ + k * dz, z1 = z0 + dz
    for (const face of faces) {
      const ni = i + face.step[0], nj = j + face.step[1], nk = k + face.step[2]
      const outside = ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz
      if (!outside && supportMask[cellIndex(ni, nj, nk)]) continue
      const quad = face.points(x0, x1, y0, y1, z0, z1)
      const a = quad[0], b = quad[1], c = quad[2], d = quad[3]
      positions.push(...a, ...b, ...c, ...a, ...c, ...d)
    }
  }
  const difference = new THREE.BufferGeometry()
  difference.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  difference.computeVertexNormals()
  return difference
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

const MIN_WALL_THICKNESS_MM = 2

function getNormalBin(nx: number, ny: number, nz: number) {
  const ax = Math.abs(nx), ay = Math.abs(ny), az = Math.abs(nz)
  if (ax >= ay && ax >= az) return nx >= 0 ? 1 : 0
  if (ay >= ax && ay >= az) return ny >= 0 ? 3 : 2
  return nz >= 0 ? 5 : 4
}

/** Finds close, opposing surface facets that indicate a thin wall. */
function countThinWallTriangles(data: TriangleData, threshold = MIN_WALL_THICKNESS_MM) {
  const triangleCount = Math.floor(data.positions.length / 9)
  const centers = new Float32Array(triangleCount * 3)
  const normals = new Float32Array(triangleCount * 3)
  const bins = new Map<string, number[]>()
  const cellSize = threshold

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const offset = triangle * 9
    const ax = data.positions[offset], ay = data.positions[offset + 1], az = data.positions[offset + 2]
    const bx = data.positions[offset + 3], by = data.positions[offset + 4], bz = data.positions[offset + 5]
    const cx = data.positions[offset + 6], cy = data.positions[offset + 7], cz = data.positions[offset + 8]
    const ux = bx - ax, uy = by - ay, uz = bz - az
    const vx = cx - ax, vy = cy - ay, vz = cz - az
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx
    const normalLength = Math.hypot(nx, ny, nz)
    if (normalLength < 1e-10) continue
    nx /= normalLength; ny /= normalLength; nz /= normalLength
    const centerOffset = triangle * 3
    const x = centers[centerOffset] = (ax + bx + cx) / 3
    const y = centers[centerOffset + 1] = (ay + by + cy) / 3
    const z = centers[centerOffset + 2] = (az + bz + cz) / 3
    normals[centerOffset] = nx; normals[centerOffset + 1] = ny; normals[centerOffset + 2] = nz
    const bin = getNormalBin(nx, ny, nz)
    const key = `${Math.floor(x / cellSize)},${Math.floor(y / cellSize)},${Math.floor(z / cellSize)},${bin}`
    const entries = bins.get(key)
    if (entries) entries.push(triangle)
    else bins.set(key, [triangle])
  }

  const flagged = new Uint8Array(triangleCount)
  let flaggedCount = 0
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const offset = triangle * 3
    const nx = normals[offset], ny = normals[offset + 1], nz = normals[offset + 2]
    if (!nx && !ny && !nz) continue
    const x = centers[offset], y = centers[offset + 1], z = centers[offset + 2]
    const cellX = Math.floor(x / cellSize), cellY = Math.floor(y / cellSize), cellZ = Math.floor(z / cellSize)
    const oppositeBin = getNormalBin(nx, ny, nz) ^ 1
    let foundOpposingSurface = false
    for (let dxCell = -1; dxCell <= 1 && !foundOpposingSurface; dxCell += 1) {
      for (let dyCell = -1; dyCell <= 1 && !foundOpposingSurface; dyCell += 1) {
        for (let dzCell = -1; dzCell <= 1 && !foundOpposingSurface; dzCell += 1) {
          const candidates = bins.get(`${cellX + dxCell},${cellY + dyCell},${cellZ + dzCell},${oppositeBin}`)
          if (!candidates) continue
          for (const candidate of candidates) {
            if (candidate <= triangle) continue
            const candidateOffset = candidate * 3
            const qx = normals[candidateOffset], qy = normals[candidateOffset + 1], qz = normals[candidateOffset + 2]
            if (nx * qx + ny * qy + nz * qz > -0.94) continue
            const vx = centers[candidateOffset] - x
            const vy = centers[candidateOffset + 1] - y
            const vz = centers[candidateOffset + 2] - z
            const thickness = Math.abs(vx * nx + vy * ny + vz * nz)
            if (thickness < 0.05 || thickness >= threshold) continue
            const tangentSq = vx * vx + vy * vy + vz * vz - thickness * thickness
            if (tangentSq > 0.75 * 0.75) continue
            if (!flagged[triangle]) { flagged[triangle] = 1; flaggedCount += 1 }
            if (!flagged[candidate]) { flagged[candidate] = 1; flaggedCount += 1 }
            foundOpposingSurface = true
            break
          }
        }
      }
    }
  }
  return flaggedCount
}

/** Marks unsupported surfaces and reports opposing facets that form thin walls. */
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

  const supports = buildBoundingBoxDifference(data)
  const thinWallTriangles = countThinWallTriangles(data)
  return { highlights, triangleCount, flaggedTriangles, flaggedArea, thinWallTriangles, limitDegrees, supports }
}
