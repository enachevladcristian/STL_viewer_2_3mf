import * as THREE from 'three'
import { MeshoptSimplifier } from 'meshoptimizer'
import { findUnsupportedOverhangs } from './printability.ts'

type CheckRequest = {
  positions: ArrayBuffer
  bounds: [number, number, number, number, number, number]
  rotation: [number, number, number]
  scale: [number, number, number]
  translation: [number, number, number]
  nozzleDiameter: number
}
type CheckProgress = { stage: string }
type CheckResponse = {
  triangleCount: number
  originalTriangleCount: number
  simplified: boolean
  simplificationError?: number
  flaggedTriangles: number
  flaggedArea: number
  thinWallTriangles: number
  undersizedFeatureTriangles: number
  minimumWallThickness: number
  minimumFeatureWidth: number
  bedContactArea: number
  minimumContactArea: number
  floating: boolean
  limitDegrees: number
  highlightPositions: ArrayBuffer
  highlightNormals: ArrayBuffer
  supportPositions: ArrayBuffer
  supportNormals: ArrayBuffer
}
type CheckFailure = { error: string }
type WeldedMesh = { positions: Float32Array; indices: Uint32Array }

const ANALYSIS_TRIANGLE_BUDGET = 100_000
const ANALYSIS_TARGET_ERROR = 0.01

const workerScope = self as unknown as {
  onmessage: ((event: MessageEvent<CheckRequest>) => void) | null
  postMessage: (message: CheckProgress | CheckResponse | CheckFailure, transfer?: Transferable[]) => void
}

function yieldToMainThread() {
  return new Promise<void>(resolve => setTimeout(resolve, 0))
}

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
    // Cache a full triangle first because welding compacts its vertices in place.
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
        if (bits[targetOffset] === bx && bits[targetOffset + 1] === by && bits[targetOffset + 2] === bz) {
          vertex = candidate
          break
        }
        slot = (slot + 1) & mask
      }
      if (vertex < 0) {
        vertex = vertexCount++
        const targetOffset = vertex * 3
        source[targetOffset] = x
        source[targetOffset + 1] = y
        source[targetOffset + 2] = z
        table[slot] = vertex + 1
      }
      if (corner === 0) ia = vertex
      else if (corner === 1) ib = vertex
      else ic = vertex
    }
    if (ia === ib || ib === ic || ic === ia) continue
    indices[indexCount++] = ia
    indices[indexCount++] = ib
    indices[indexCount++] = ic
  }
  return { positions: source.subarray(0, vertexCount * 3), indices: indices.subarray(0, indexCount) }
}

workerScope.onmessage = async event => {
  let geometry: THREE.BufferGeometry | null = null
  let result: ReturnType<typeof findUnsupportedOverhangs> | null = null
  try {
    const request = event.data
    let sourcePositions = new Float32Array(request.positions)
    const originalTriangleCount = Math.floor(sourcePositions.length / 9)
    let simplified = false
    let simplificationError: number | undefined

    if (originalTriangleCount > ANALYSIS_TRIANGLE_BUDGET) {
      workerScope.postMessage({ stage: `Simplifying ${originalTriangleCount.toLocaleString()} triangles for analysis…` })
      await yieldToMainThread()
      await MeshoptSimplifier.ready
      const welded = weldTriangleSoup(sourcePositions)
      if (welded.indices.length < 3) throw new Error('This mesh has no valid triangles to analyze.')
      const targetIndexCount = Math.min(welded.indices.length - (welded.indices.length % 3), ANALYSIS_TRIANGLE_BUDGET * 3)
      const [indices, error] = MeshoptSimplifier.simplify(
        welded.indices,
        welded.positions,
        3,
        targetIndexCount,
        ANALYSIS_TARGET_ERROR,
      )
      if (indices.length < 3 || indices.length > ANALYSIS_TRIANGLE_BUDGET * 3 || indices.length >= originalTriangleCount * 3) {
        throw new Error('This mesh could not be simplified enough for a responsive printability check.')
      }
      simplified = true
      simplificationError = error
      geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(welded.positions, 3))
      geometry.setIndex(new THREE.BufferAttribute(indices, 1))
      sourcePositions = new Float32Array(0)
      request.positions = new ArrayBuffer(0)
    } else {
      geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(sourcePositions, 3))
    }

    const [minX, minY, minZ, maxX, maxY, maxZ] = request.bounds
    geometry.boundingBox = new THREE.Box3(new THREE.Vector3(minX, minY, minZ), new THREE.Vector3(maxX, maxY, maxZ))
    workerScope.postMessage({ stage: simplified ? `Checking ${(geometry.index!.count / 3).toLocaleString()} simplified triangles and generating preview…` : 'Checking geometry and generating preview…' })
    await yieldToMainThread()

    result = findUnsupportedOverhangs(geometry, request.rotation, 45, request.scale, request.translation, 0.2, request.nozzleDiameter)
    const highlightPositions = result.highlights.getAttribute('position').array as Float32Array
    const highlightNormals = result.highlights.getAttribute('normal')!.array as Float32Array
    const supportPositions = result.supports.getAttribute('position').array as Float32Array
    const supportNormals = result.supports.getAttribute('normal')!.array as Float32Array
    const response: CheckResponse = {
      triangleCount: result.triangleCount,
      originalTriangleCount,
      simplified,
      simplificationError,
      flaggedTriangles: result.flaggedTriangles,
      flaggedArea: result.flaggedArea,
      thinWallTriangles: result.thinWallTriangles,
      undersizedFeatureTriangles: result.undersizedFeatureTriangles,
      minimumWallThickness: result.minimumWallThickness,
      minimumFeatureWidth: result.minimumFeatureWidth,
      bedContactArea: result.bedContactArea,
      minimumContactArea: result.minimumContactArea,
      floating: result.floating,
      limitDegrees: result.limitDegrees,
      highlightPositions: highlightPositions.buffer,
      highlightNormals: highlightNormals.buffer,
      supportPositions: supportPositions.buffer,
      supportNormals: supportNormals.buffer,
    }
    workerScope.postMessage(response, [
      response.highlightPositions,
      response.highlightNormals,
      response.supportPositions,
      response.supportNormals,
    ])
  } catch (error) {
    workerScope.postMessage({ error: error instanceof Error ? error.message : 'The printability check failed' })
  } finally {
    geometry?.dispose()
    result?.highlights.dispose()
    result?.supports.dispose()
  }
}
