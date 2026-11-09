import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'

type ParseRequest = { buffer: ArrayBuffer }
type ParseSuccess = {
  positions: ArrayBuffer
  normals: ArrayBuffer
  triangles: number
  dimensions: [number, number, number]
  bounds: [number, number, number, number, number, number]
  volume: number
  area: number
}
type ParseFailure = { error: string }

const workerScope = self as unknown as {
  onmessage: ((event: MessageEvent<ParseRequest>) => void) | null
  postMessage: (message: ParseSuccess | ParseFailure, transfer?: Transferable[]) => void
}

workerScope.onmessage = event => {
  try {
    const geometry = new STLLoader().parse(event.data.buffer)
    const position = geometry.getAttribute('position')
    if (!position || position.count < 3) throw new Error('No triangles found in the STL')
    if (!geometry.getAttribute('normal')) geometry.computeVertexNormals()

    const vertices = position.array as Float32Array
    let minX = Infinity, minY = Infinity, minZ = Infinity
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity
    for (let i = 0; i < vertices.length; i += 3) {
      const x = vertices[i], y = vertices[i + 1], z = vertices[i + 2]
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) throw new Error('The STL contains invalid vertex coordinates')
      minX = Math.min(minX, x); maxX = Math.max(maxX, x)
      minY = Math.min(minY, y); maxY = Math.max(maxY, y)
      minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z)
    }
    const centerX = (minX + maxX) * 0.5
    const centerY = (minY + maxY) * 0.5
    const centerZ = (minZ + maxZ) * 0.5
    const dimensions: [number, number, number] = [maxX - minX, maxY - minY, maxZ - minZ]
    minX -= centerX; maxX -= centerX
    minY -= centerY; maxY -= centerY
    minZ -= centerZ; maxZ -= centerZ

    let signedVolume = 0
    let area = 0
    for (let i = 0; i < vertices.length; i += 9) {
      const ax = vertices[i] - centerX, ay = vertices[i + 1] - centerY, az = vertices[i + 2] - centerZ
      const bx = vertices[i + 3] - centerX, by = vertices[i + 4] - centerY, bz = vertices[i + 5] - centerZ
      const cx = vertices[i + 6] - centerX, cy = vertices[i + 7] - centerY, cz = vertices[i + 8] - centerZ
      const abx = bx - ax, aby = by - ay, abz = bz - az
      const acx = cx - ax, acy = cy - ay, acz = cz - az
      const crossX = aby * acz - abz * acy
      const crossY = abz * acx - abx * acz
      const crossZ = abx * acy - aby * acx
      area += Math.hypot(crossX, crossY, crossZ) * 0.5
      signedVolume += (ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)) / 6
    }
    for (let i = 0; i < vertices.length; i += 3) {
      vertices[i] -= centerX
      vertices[i + 1] -= centerY
      vertices[i + 2] -= centerZ
    }

    const normals = geometry.getAttribute('normal').array as Float32Array
    const reply: ParseSuccess = {
      positions: vertices.buffer,
      normals: normals.buffer,
      triangles: Math.floor(position.count / 3),
      dimensions,
      bounds: [minX, minY, minZ, maxX, maxY, maxZ],
      volume: Math.abs(signedVolume),
      area,
    }
    workerScope.postMessage(reply, [reply.positions, reply.normals])
  } catch (error) {
    workerScope.postMessage({ error: error instanceof Error ? error.message : 'Unable to parse this STL file' })
  }
}
