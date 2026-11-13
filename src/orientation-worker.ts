import * as THREE from 'three'

type OrientationRequest = {
  positions: ArrayBuffer
  bounds: [number, number, number, number, number, number]
  rotation: [number, number, number]
  scale: [number, number, number]
  translation: [number, number, number]
  nozzleDiameter: number
  buildVolume: { width: number; depth: number; height: number }
}
type OrientationSuggestion = {
  rotation: [number, number, number]
  unsupportedArea: number
  unsupportedPercent: number
  bedContactArea: number
  minimumContactArea: number
  dimensions: [number, number, number]
  fits: boolean
  overflow: number
}
type OrientationMessage = { stage: string } | { suggestions: OrientationSuggestion[] } | { error: string }

const scope = self as unknown as {
  onmessage: ((event: MessageEvent<OrientationRequest>) => void) | null
  postMessage: (message: OrientationMessage) => void
}
const DOWN = new THREE.Vector3(0, -1, 0)
const MAX_ANALYSIS_TRIANGLES = 60_000
const MAX_NORMAL_SAMPLES = 18_000

function normalizedDegrees(value: number): number {
  let degrees = THREE.MathUtils.radToDeg(value)
  degrees = ((degrees + 180) % 360 + 360) % 360 - 180
  return Math.abs(degrees) < 0.05 ? 0 : Number(degrees.toFixed(1))
}

function quaternionRotation(rotation: [number, number, number]) {
  return new THREE.Quaternion().setFromEuler(new THREE.Euler(
    THREE.MathUtils.degToRad(rotation[0]),
    THREE.MathUtils.degToRad(rotation[1]),
    THREE.MathUtils.degToRad(rotation[2]),
    'XYZ',
  ))
}

function collectCandidateQuaternions(
  positions: Float32Array,
  center: THREE.Vector3,
  scale: THREE.Vector3,
  currentRotation: [number, number, number],
) {
  const triangleCount = Math.floor(positions.length / 9)
  const stride = Math.max(1, Math.ceil(triangleCount / MAX_NORMAL_SAMPLES))
  const groups = new Map<string, { normal: THREE.Vector3; area: number }>()
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3()
  const edge1 = new THREE.Vector3(), edge2 = new THREE.Vector3(), normal = new THREE.Vector3()
  let signedVolume = 0

  for (let triangle = 0; triangle < triangleCount; triangle += stride) {
    const offset = triangle * 9
    a.fromArray(positions, offset).sub(center).multiply(scale)
    b.fromArray(positions, offset + 3).sub(center).multiply(scale)
    c.fromArray(positions, offset + 6).sub(center).multiply(scale)
    edge1.subVectors(b, a); edge2.subVectors(c, a); normal.crossVectors(edge1, edge2)
    const area = normal.length() * 0.5
    if (area < 1e-8) continue
    signedVolume += a.dot(normal) / 6 * stride
    normal.normalize()
    const key = `${Math.round(normal.x * 16)},${Math.round(normal.y * 16)},${Math.round(normal.z * 16)}`
    const group = groups.get(key)
    if (group) { group.normal.addScaledVector(normal, area); group.area += area }
    else groups.set(key, { normal: normal.clone().multiplyScalar(area), area })
  }

  const orientationSign = signedVolume < 0 ? -1 : 1
  const normals = [...groups.values()]
    .sort((first, second) => second.area - first.area)
    .slice(0, 6)
    .map(group => group.normal.multiplyScalar(orientationSign).normalize())
  const candidates: THREE.Quaternion[] = [quaternionRotation(currentRotation)]
  const yawRotations = [0, 90, 180, 270].map(degrees => new THREE.Quaternion().setFromAxisAngle(DOWN.clone().negate(), THREE.MathUtils.degToRad(degrees)))
  for (const normal of normals) {
    const aligned = new THREE.Quaternion().setFromUnitVectors(normal, DOWN)
    for (const yaw of yawRotations) {
      const candidate = yaw.clone().multiply(aligned).normalize()
      if (candidates.some(existing => Math.abs(existing.dot(candidate)) > 0.9995)) continue
      candidates.push(candidate)
    }
  }
  return { candidates, orientationSign }
}

function evaluateOrientation(
  positions: Float32Array,
  bounds: OrientationRequest['bounds'],
  scale: THREE.Vector3,
  rotation: THREE.Quaternion,
  orientationSign: number,
  translation: [number, number, number],
  nozzleDiameter: number,
  buildVolume: OrientationRequest['buildVolume'],
): OrientationSuggestion {
  const triangleCount = Math.floor(positions.length / 9)
  const stride = Math.max(1, Math.ceil(triangleCount / MAX_ANALYSIS_TRIANGLES))
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3()
  const e1 = new THREE.Vector3(), e2 = new THREE.Vector3(), normal = new THREE.Vector3()
  const min = new THREE.Vector3(Infinity, Infinity, Infinity)
  const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity)
  for (const x of [bounds[0], bounds[3]]) for (const y of [bounds[1], bounds[4]]) for (const z of [bounds[2], bounds[5]]) {
    const corner = new THREE.Vector3(x, y, z).multiply(scale).applyQuaternion(rotation)
    min.min(corner); max.max(corner)
  }
  let totalArea = 0

  for (let triangle = 0; triangle < triangleCount; triangle += stride) {
    const offset = triangle * 9
    a.fromArray(positions, offset).multiply(scale).applyQuaternion(rotation)
    b.fromArray(positions, offset + 3).multiply(scale).applyQuaternion(rotation)
    c.fromArray(positions, offset + 6).multiply(scale).applyQuaternion(rotation)
    min.min(a).min(b).min(c); max.max(a).max(b).max(c)
    e1.subVectors(b, a); e2.subVectors(c, a)
    totalArea += normal.crossVectors(e1, e2).length() * 0.5 * stride
  }

  const bedOffset = -min.y
  const maxY = max.y + bedOffset
  const minX = min.x + translation[0], maxX = max.x + translation[0]
  const minZ = min.z + translation[2], maxZ = max.z + translation[2]
  const width = max.x - min.x, depth = max.z - min.z, height = maxY
  const overflowX = Math.max(0, -buildVolume.width / 2 - minX, maxX - buildVolume.width / 2)
  const overflowZ = Math.max(0, -buildVolume.depth / 2 - minZ, maxZ - buildVolume.depth / 2)
  const overflowY = Math.max(0, height - buildVolume.height)
  const overflow = Math.max(overflowX, overflowY, overflowZ)

  let unsupportedArea = 0
  let bedContactArea = 0
  const sineLimit = Math.sin(THREE.MathUtils.degToRad(45))
  const layerHeight = 0.2
  for (let triangle = 0; triangle < triangleCount; triangle += stride) {
    const offset = triangle * 9
    a.fromArray(positions, offset).multiply(scale).applyQuaternion(rotation)
    b.fromArray(positions, offset + 3).multiply(scale).applyQuaternion(rotation)
    c.fromArray(positions, offset + 6).multiply(scale).applyQuaternion(rotation)
    e1.subVectors(b, a); e2.subVectors(c, a)
    const cross = normal.crossVectors(e1, e2)
    const area = cross.length() * 0.5 * stride
    if (area < 1e-10) continue
    const maxTriangleY = Math.max(a.y, b.y, c.y) + bedOffset
    const normalY = orientationSign * cross.y / cross.length()
    if (maxTriangleY > layerHeight + 0.02 && normalY < -sineLimit) unsupportedArea += area
    if (Math.max(a.y, b.y, c.y) - min.y <= 0.05) {
      bedContactArea += Math.abs((b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x)) * 0.5 * stride
    }
  }

  const quaternionEuler = new THREE.Euler().setFromQuaternion(rotation, 'XYZ')
  const dimensions: [number, number, number] = [width, height, depth]
  const minimumContactArea = (THREE.MathUtils.clamp(nozzleDiameter, 0.1, 1.2) * 10) ** 2
  return {
    rotation: [normalizedDegrees(quaternionEuler.x), normalizedDegrees(quaternionEuler.y), normalizedDegrees(quaternionEuler.z)],
    unsupportedArea,
    unsupportedPercent: totalArea > 0 ? unsupportedArea / totalArea * 100 : 0,
    bedContactArea,
    minimumContactArea,
    dimensions,
    fits: overflow <= 0.05,
    overflow,
  }
}

scope.onmessage = async event => {
  try {
    scope.postMessage({ stage: 'Sampling surface directions and testing orientations…' })
    await new Promise<void>(resolve => setTimeout(resolve, 0))
    const request = event.data
    const positions = new Float32Array(request.positions)
    if (positions.length < 9) { scope.postMessage({ error: 'This model has no triangles to orient.' }); return }
    const bounds = request.bounds
    const center = new THREE.Vector3((bounds[0] + bounds[3]) / 2, (bounds[1] + bounds[4]) / 2, (bounds[2] + bounds[5]) / 2)
    const scale = new THREE.Vector3(...request.scale)
    const orientation = collectCandidateQuaternions(positions, center, scale, request.rotation)
    const results: OrientationSuggestion[] = []
    for (let index = 0; index < orientation.candidates.length; index += 1) {
      results.push(evaluateOrientation(positions, request.bounds, scale, orientation.candidates[index], orientation.orientationSign, request.translation, request.nozzleDiameter, request.buildVolume))
      if (index % 4 === 3) await new Promise<void>(resolve => setTimeout(resolve, 0))
    }
    const fitting = results.filter(result => result.fits)
    const pool = fitting.length ? fitting : results
    const score = (item: OrientationSuggestion) => item.unsupportedPercent + Math.max(0, item.minimumContactArea - item.bedContactArea) / item.minimumContactArea * 20 + item.overflow * 100
    const sorted = pool.sort((a, b) => score(a) - score(b))
    const current = quaternionRotation(request.rotation)
    const alternatives = sorted.filter(result => Math.abs(current.dot(quaternionRotation(result.rotation))) < 0.999)
    const ranked = alternatives.length ? alternatives : sorted
    const suggestions: OrientationSuggestion[] = []
    for (const result of ranked) {
      if (suggestions.some(existing => {
        const first = quaternionRotation(existing.rotation)
        const next = quaternionRotation(result.rotation)
        return Math.abs(first.dot(next)) > 0.999
      })) continue
      suggestions.push(result)
      if (suggestions.length === 3) break
    }
    scope.postMessage({ suggestions })
  } catch (error) {
    scope.postMessage({ error: error instanceof Error ? error.message : 'Could not calculate orientation suggestions.' })
  }
}
