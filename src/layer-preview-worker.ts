import { MeshoptSimplifier } from 'meshoptimizer'
import * as THREE from 'three'

type LayerPreviewRequest = {
  positions: ArrayBuffer
  bounds: [number, number, number, number, number, number]
  rotation: [number, number, number]
  scale: [number, number, number]
  translation: [number, number, number]
  layerHeight: number
  layerNumber: number
  scanAll?: boolean
  maxLayers?: number
}
type LayerPreviewResult = {
  layerNumber: number
  layerHeight: number
  currentPositions: ArrayBuffer
  islandPositions: ArrayBuffer
  islandCount: number
  triangleCount: number
  originalTriangleCount: number
  simplified: boolean
}
type LayerScanWarning = { layerNumber: number; layerHeight: number; islandCount: number }
type LayerScanResult = { scanWarnings: LayerScanWarning[]; scannedLayers: number; triangleCount: number; originalTriangleCount: number; simplified: boolean }
type LayerPreviewMessage = { stage: string } | LayerPreviewResult | LayerScanResult | { error: string }
type WeldedMesh = { positions: Float32Array; indices: Uint32Array }
type SliceSegment = { ax: number; az: number; bx: number; bz: number; root?: number }

const TRIANGLE_BUDGET = 100_000
const FULL_SCAN_TRIANGLE_BUDGET = 25_000
const scope = self as unknown as {
  onmessage: ((event: MessageEvent<LayerPreviewRequest>) => void) | null
  postMessage: (message: LayerPreviewMessage, transfer?: Transferable[]) => void
}

function yieldToMainThread() { return new Promise<void>(resolve => setTimeout(resolve, 0)) }

function weldTriangleSoup(source: Float32Array): WeldedMesh {
  const sourceVertexCount = Math.floor(source.length / 3)
  let tableSize = 1
  while (tableSize < sourceVertexCount * 1.5) tableSize *= 2
  const table = new Int32Array(tableSize)
  const bits = new Uint32Array(source.buffer, source.byteOffset, source.length)
  const indices = new Uint32Array(sourceVertexCount)
  const mask = tableSize - 1
  let vertexCount = 0
  let indexCount = 0
  for (let offset = 0; offset + 8 < source.length; offset += 9) {
    const x0 = source[offset], y0 = source[offset + 1], z0 = source[offset + 2]
    const x1 = source[offset + 3], y1 = source[offset + 4], z1 = source[offset + 5]
    const x2 = source[offset + 6], y2 = source[offset + 7], z2 = source[offset + 8]
    const bx0 = bits[offset], by0 = bits[offset + 1], bz0 = bits[offset + 2]
    const bx1 = bits[offset + 3], by1 = bits[offset + 4], bz1 = bits[offset + 5]
    const bx2 = bits[offset + 6], by2 = bits[offset + 7], bz2 = bits[offset + 8]
    let ia = -1, ib = -1, ic = -1
    for (let corner = 0; corner < 3; corner += 1) {
      const x = corner === 0 ? x0 : corner === 1 ? x1 : x2
      const y = corner === 0 ? y0 : corner === 1 ? y1 : y2
      const z = corner === 0 ? z0 : corner === 1 ? z1 : z2
      const bx = corner === 0 ? bx0 : corner === 1 ? bx1 : bx2
      const by = corner === 0 ? by0 : corner === 1 ? by1 : by2
      const bz = corner === 0 ? bz0 : corner === 1 ? bz1 : bz2
      let slot = (Math.imul(bx, 73856093) ^ Math.imul(by, 19349663) ^ Math.imul(bz, 83492791)) & mask
      let vertex = -1
      while (table[slot] !== 0) {
        const candidate = table[slot] - 1
        const targetOffset = candidate * 3
        if (bits[targetOffset] === bx && bits[targetOffset + 1] === by && bits[targetOffset + 2] === bz) { vertex = candidate; break }
        slot = (slot + 1) & mask
      }
      if (vertex < 0) {
        vertex = vertexCount++
        const targetOffset = vertex * 3
        source[targetOffset] = x; source[targetOffset + 1] = y; source[targetOffset + 2] = z
        table[slot] = vertex + 1
      }
      if (corner === 0) ia = vertex
      else if (corner === 1) ib = vertex
      else ic = vertex
    }
    if (ia === ib || ib === ic || ic === ia) continue
    indices[indexCount++] = ia; indices[indexCount++] = ib; indices[indexCount++] = ic
  }
  return { positions: source.subarray(0, vertexCount * 3), indices: indices.subarray(0, indexCount) }
}

function makeTransform(request: LayerPreviewRequest) {
  const quaternion = new THREE.Quaternion().setFromEuler(new THREE.Euler(
    THREE.MathUtils.degToRad(request.rotation[0]),
    THREE.MathUtils.degToRad(request.rotation[1]),
    THREE.MathUtils.degToRad(request.rotation[2]),
    'XYZ',
  ))
  const scale = new THREE.Vector3(...request.scale)
  const bounds = request.bounds
  let minY = Infinity, maxY = -Infinity
  for (const x of [bounds[0], bounds[3]]) for (const y of [bounds[1], bounds[4]]) for (const z of [bounds[2], bounds[5]]) {
    const transformedY = new THREE.Vector3(x, y, z).multiply(scale).applyQuaternion(quaternion).y
    minY = Math.min(minY, transformedY)
    maxY = Math.max(maxY, transformedY)
  }
  const translation = new THREE.Vector3(request.translation[0], -minY + request.translation[1], request.translation[2])
  const matrix = new THREE.Matrix4().compose(translation, quaternion, scale)
  return { matrix, maxHeight: maxY - minY + request.translation[1] }
}

function transformTriangleSoup(positions: Float32Array, indices: Uint32Array | null, matrix: THREE.Matrix4) {
  const vertexCount = indices ? indices.length : positions.length / 3
  const transformed = new Float32Array(vertexCount * 3)
  const elements = matrix.elements
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const sourceOffset = (indices ? indices[vertex] : vertex) * 3
    const targetOffset = vertex * 3
    const x = positions[sourceOffset], y = positions[sourceOffset + 1], z = positions[sourceOffset + 2]
    transformed[targetOffset] = elements[0] * x + elements[4] * y + elements[8] * z + elements[12]
    transformed[targetOffset + 1] = elements[1] * x + elements[5] * y + elements[9] * z + elements[13]
    transformed[targetOffset + 2] = elements[2] * x + elements[6] * y + elements[10] * z + elements[14]
  }
  return transformed
}

function sliceAtHeight(positions: Float32Array, height: number): SliceSegment[] {
  const segments: SliceSegment[] = []
  for (let offset = 0; offset + 8 < positions.length; offset += 9) {
    const ax = positions[offset], ay = positions[offset + 1], az = positions[offset + 2]
    const bx = positions[offset + 3], by = positions[offset + 4], bz = positions[offset + 5]
    const cx = positions[offset + 6], cy = positions[offset + 7], cz = positions[offset + 8]
    let hitCount = 0, x0 = 0, z0 = 0, x1 = 0, z1 = 0
    for (let edge = 0; edge < 3; edge += 1) {
      const firstX = edge === 0 ? ax : edge === 1 ? bx : cx
      const firstY = edge === 0 ? ay : edge === 1 ? by : cy
      const firstZ = edge === 0 ? az : edge === 1 ? bz : cz
      const secondX = edge === 0 ? bx : edge === 1 ? cx : ax
      const secondY = edge === 0 ? by : edge === 1 ? cy : ay
      const secondZ = edge === 0 ? bz : edge === 1 ? cz : az
      if (!((firstY <= height && secondY > height) || (secondY <= height && firstY > height))) continue
      const t = (height - firstY) / (secondY - firstY)
      const x = firstX + (secondX - firstX) * t
      const z = firstZ + (secondZ - firstZ) * t
      if (hitCount > 0 && Math.abs(x0 - x) < 1e-5 && Math.abs(z0 - z) < 1e-5) continue
      if (hitCount === 0) { x0 = x; z0 = z; hitCount = 1 }
      else { x1 = x; z1 = z; hitCount = 2 }
    }
    if (hitCount === 2 && Math.hypot(x0 - x1, z0 - z1) > 1e-6) segments.push({ ax: x0, az: z0, bx: x1, bz: z1 })
  }
  return segments
}

type SliceIndex = { segments: SliceSegment[]; minZ: number; cellSize: number; buckets: number[][] }

function makeSliceIndex(segments: SliceSegment[]): SliceIndex {
  let minZ = Infinity, maxZ = -Infinity
  for (const segment of segments) { minZ = Math.min(minZ, segment.az, segment.bz); maxZ = Math.max(maxZ, segment.az, segment.bz) }
  if (!Number.isFinite(minZ)) minZ = maxZ = 0
  const bucketCount = 64
  const cellSize = Math.max((maxZ - minZ) / bucketCount, 1e-5)
  const buckets = Array.from({ length: bucketCount }, () => [] as number[])
  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index]
    if (Math.abs(segment.bz - segment.az) < 1e-9) continue
    const first = Math.max(0, Math.min(bucketCount - 1, Math.floor((Math.min(segment.az, segment.bz) - minZ) / cellSize)))
    const last = Math.max(0, Math.min(bucketCount - 1, Math.floor((Math.max(segment.az, segment.bz) - minZ) / cellSize)))
    for (let bucket = first; bucket <= last; bucket += 1) buckets[bucket].push(index)
  }
  return { segments, minZ, cellSize, buckets }
}

function pointInsideSlice(x: number, z: number, slice: SliceIndex) {
  const bucket = Math.max(0, Math.min(slice.buckets.length - 1, Math.floor((z - slice.minZ) / slice.cellSize)))
  let crossings = 0
  for (const index of slice.buckets[bucket]) {
    const segment = slice.segments[index]
    if ((segment.az > z) === (segment.bz > z)) continue
    const hitX = segment.ax + (z - segment.az) * (segment.bx - segment.ax) / (segment.bz - segment.az)
    if (hitX > x) crossings += 1
  }
  return crossings % 2 === 1
}

function tagIslandComponents(current: SliceSegment[], previous: SliceSegment[]) {
  const parent = new Int32Array(current.length)
  for (let index = 0; index < parent.length; index += 1) parent[index] = index
  const find = (value: number): number => {
    let root = value
    while (parent[root] !== root) root = parent[root]
    while (parent[value] !== value) { const next = parent[value]; parent[value] = root; value = next }
    return root
  }
  const union = (a: number, b: number) => { const first = find(a), second = find(b); if (first !== second) parent[second] = first }
  const endpoints = new Map<string, number>()
  const pointKey = (x: number, z: number) => `${Math.round(x * 100)},${Math.round(z * 100)}`
  current.forEach((segment, index) => {
    for (const key of [pointKey(segment.ax, segment.az), pointKey(segment.bx, segment.bz)]) {
      const existing = endpoints.get(key)
      if (existing === undefined) endpoints.set(key, index)
      else union(index, existing)
    }
  })
  const groups = new Map<number, number[]>()
  for (let index = 0; index < current.length; index += 1) {
    const root = find(index)
    const group = groups.get(root)
    if (group) group.push(index)
    else groups.set(root, [index])
  }
  const previousIndex = makeSliceIndex(previous)
  const islandRoots = new Set<number>()
  for (const [root, segmentIndices] of groups) {
    let supported = false
    const sampleCount = Math.min(4, segmentIndices.length)
    for (let sample = 0; sample < sampleCount && !supported; sample += 1) {
      const segment = current[segmentIndices[Math.floor(sample * segmentIndices.length / sampleCount)]]
      const dx = segment.bx - segment.ax, dz = segment.bz - segment.az
      const length = Math.hypot(dx, dz)
      if (length < 1e-7) continue
      const middleX = (segment.ax + segment.bx) * 0.5, middleZ = (segment.az + segment.bz) * 0.5
      const normalX = -dz / length * 0.2, normalZ = dx / length * 0.2
      supported = pointInsideSlice(middleX, middleZ, previousIndex) ||
        pointInsideSlice(middleX + normalX, middleZ + normalZ, previousIndex) ||
        pointInsideSlice(middleX - normalX, middleZ - normalZ, previousIndex)
    }
    if (!supported) islandRoots.add(root)
  }
  for (let index = 0; index < current.length; index += 1) current[index].root = find(index)
  return islandRoots
}

scope.onmessage = async event => {
  try {
    const request = event.data
    let positions = new Float32Array(request.positions)
    let indices: Uint32Array | null = null
    const originalTriangleCount = Math.floor(positions.length / 9)
    let simplified = false
    const transform = makeTransform(request)
    const layerHeight = Math.max(0.05, request.layerHeight)
    const requestedLayerCount = Math.max(1, Math.min(request.maxLayers ?? Math.ceil(transform.maxHeight / layerHeight), 6000))
    const scanTriangleBudget = Math.max(5_000, Math.min(FULL_SCAN_TRIANGLE_BUDGET, Math.floor(50_000_000 / requestedLayerCount)))
    const triangleBudget = request.scanAll ? scanTriangleBudget : TRIANGLE_BUDGET
    if (originalTriangleCount > triangleBudget) {
      scope.postMessage({ stage: 'Simplifying geometry for layer analysis…' })
      await yieldToMainThread()
      await MeshoptSimplifier.ready
      const welded = weldTriangleSoup(positions)
      if (welded.indices.length < 3) throw new Error('This mesh has no valid triangles to preview.')
      const targetCount = Math.min(welded.indices.length - welded.indices.length % 3, triangleBudget * 3)
      const [simplifiedIndices] = MeshoptSimplifier.simplify(welded.indices, welded.positions, 3, targetCount, 0.01)
      if (simplifiedIndices.length < 3) throw new Error('This mesh could not be simplified for a responsive layer preview.')
      positions = welded.positions
      indices = simplifiedIndices
      simplified = true
    }
    const triangleCount = indices ? Math.floor(indices.length / 3) : originalTriangleCount
    const worldPositions = transformTriangleSoup(positions, indices, transform.matrix)
    positions = new Float32Array(0)
    indices = null

    if (request.scanAll) {
      const maxLayers = requestedLayerCount
      const warnings: LayerScanWarning[] = []
      let previous: SliceSegment[] = []
      const progressInterval = Math.max(1, Math.floor(maxLayers / 100))
      for (let layer = 1; layer <= maxLayers; layer += 1) {
        const height = Math.max(layerHeight / 2, Math.min(layer * layerHeight - layerHeight / 2, transform.maxHeight - 0.001))
        const current = sliceAtHeight(worldPositions, height)
        const islandRoots = layer === 1 ? new Set<number>() : tagIslandComponents(current, previous)
        if (islandRoots.size) warnings.push({ layerNumber: layer, layerHeight: height, islandCount: islandRoots.size })
        previous = current
        if (layer % progressInterval === 0 || layer === maxLayers) {
          scope.postMessage({ stage: `Scanning layers ${layer.toLocaleString()} of ${maxLayers.toLocaleString()}…` })
          await yieldToMainThread()
        }
      }
      scope.postMessage({ scanWarnings: warnings, scannedLayers: maxLayers, triangleCount, originalTriangleCount, simplified })
      return
    }

    const layerNumber = Math.max(1, request.layerNumber)
    const currentHeight = Math.max(layerHeight / 2, Math.min(layerNumber * layerHeight - layerHeight / 2, transform.maxHeight - 0.001))
    const previousHeight = currentHeight - layerHeight
    scope.postMessage({ stage: simplified ? 'Slicing an approximate mesh…' : 'Calculating layer cross-section…' })
    await yieldToMainThread()
    const current = sliceAtHeight(worldPositions, currentHeight)
    const previous = layerNumber <= 1 ? [] : sliceAtHeight(worldPositions, previousHeight)
    const islandRoots = layerNumber <= 1 ? new Set<number>() : tagIslandComponents(current, previous)
    const currentPositions: number[] = [], islandPositions: number[] = []
    for (const segment of current) {
      const target = islandRoots.has(segment.root ?? -1) ? islandPositions : currentPositions
      target.push(segment.ax, currentHeight, segment.az, segment.bx, currentHeight, segment.bz)
    }
    const currentBuffer = new Float32Array(currentPositions).buffer
    const islandBuffer = new Float32Array(islandPositions).buffer
    const result: LayerPreviewResult = {
      layerNumber,
      layerHeight: currentHeight,
      currentPositions: currentBuffer,
      islandPositions: islandBuffer,
      islandCount: islandRoots.size,
      triangleCount,
      originalTriangleCount,
      simplified,
    }
    scope.postMessage(result, [currentBuffer, islandBuffer])
  } catch (error) {
    scope.postMessage({ error: error instanceof Error ? error.message : 'Layer preview failed.' })
  }
}
