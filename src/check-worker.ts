import * as THREE from 'three'
import { findUnsupportedOverhangs } from './printability.ts'

type CheckRequest = {
  positions: ArrayBuffer
  bounds: [number, number, number, number, number, number]
  rotation: [number, number, number]
  scale: [number, number, number]
  translation: [number, number, number]
}
type CheckResponse = {
  triangleCount: number
  flaggedTriangles: number
  flaggedArea: number
  limitDegrees: number
  highlightPositions: ArrayBuffer
  highlightNormals: ArrayBuffer
  supportPositions: ArrayBuffer
  supportNormals: ArrayBuffer
}
type CheckFailure = { error: string }

const workerScope = self as unknown as {
  onmessage: ((event: MessageEvent<CheckRequest>) => void) | null
  postMessage: (message: CheckResponse | CheckFailure, transfer?: Transferable[]) => void
}

workerScope.onmessage = event => {
  try {
    const request = event.data
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(request.positions), 3))
    const [minX, minY, minZ, maxX, maxY, maxZ] = request.bounds
    geometry.boundingBox = new THREE.Box3(new THREE.Vector3(minX, minY, minZ), new THREE.Vector3(maxX, maxY, maxZ))
    const result = findUnsupportedOverhangs(geometry, request.rotation, 45, request.scale, request.translation, 0.2)
    const highlightPositions = result.highlights.getAttribute('position').array as Float32Array
    const highlightNormals = result.highlights.getAttribute('normal')!.array as Float32Array
    const supportPositions = result.supports.getAttribute('position').array as Float32Array
    const supportNormals = result.supports.getAttribute('normal')!.array as Float32Array
    const response: CheckResponse = {
      triangleCount: result.triangleCount,
      flaggedTriangles: result.flaggedTriangles,
      flaggedArea: result.flaggedArea,
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
    geometry.dispose()
    result.highlights.dispose()
    result.supports.dispose()
  } catch (error) {
    workerScope.postMessage({ error: error instanceof Error ? error.message : 'The printability check failed' })
  }
}
