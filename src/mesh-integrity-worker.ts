type IntegrityIssue = 'openEdges' | 'nonManifold' | 'invertedNormals' | 'degenerateFaces'
type IntegrityRequest = { positions: ArrayBuffer; indices?: ArrayBuffer; bounds: [number, number, number, number, number, number] }
type IntegrityProgress = { stage: string }
type IntegrityResponse = {
  counts: { openEdges: number; nonManifoldEdges: number; invertedTriangles: number; degenerateTriangles: number }
  triangleCounts: { openEdges: number; nonManifold: number; invertedNormals: number; degenerateFaces: number }
  highlights: Record<IntegrityIssue, ArrayBuffer>
  sampledHighlights: boolean
}
type IntegrityFailure = { error: string }
type EdgeInfo = { count: number; orientation: number; triangles: number[] }

const MAX_HIGHLIGHT_TRIANGLES = 20_000
const workerScope = self as unknown as {
  onmessage: ((event: MessageEvent<IntegrityRequest>) => void) | null
  postMessage: (message: IntegrityProgress | IntegrityResponse | IntegrityFailure, transfer?: Transferable[]) => void
}

function trianglePositionIndices(indices: Uint32Array | null, triangle: number) {
  const corner = triangle * 3
  return indices
    ? [indices[corner], indices[corner + 1], indices[corner + 2]]
    : [corner, corner + 1, corner + 2]
}

function analyzeMesh(request: IntegrityRequest): IntegrityResponse {
  const positions = new Float32Array(request.positions)
  const indices = request.indices ? new Uint32Array(request.indices) : null
  const triangleCount = Math.floor((indices?.length ?? positions.length / 3) / 3)
  const [minX, minY, minZ, maxX, maxY, maxZ] = request.bounds
  const diagonal = Math.hypot(maxX - minX, maxY - minY, maxZ - minZ)
  const weldTolerance = Math.max(diagonal * 1e-7, 1e-7)
  const areaTolerance = Math.max(diagonal * diagonal * 1e-10, 1e-12)
  const vertexIds = new Map<string, number>()
  const faceIds: [number, number, number][] = new Array(triangleCount)
  const degenerate = new Set<number>()
  const inverted = new Set<number>()
  const openFaceIds = new Set<number>()
  const nonManifoldFaceIds = new Set<number>()
  const edges = new Map<string, EdgeInfo>()
  const vertexKey = (vertex: number) => {
    const offset = vertex * 3
    const x = positions[offset], y = positions[offset + 1], z = positions[offset + 2]
    const key = `${Math.round(x / weldTolerance)},${Math.round(y / weldTolerance)},${Math.round(z / weldTolerance)}`
    let id = vertexIds.get(key)
    if (id === undefined) { id = vertexIds.size; vertexIds.set(key, id) }
    return id
  }
  const edgeKey = (a: number, b: number) => a < b ? `${a}:${b}` : `${b}:${a}`
  const edgeDirection = (a: number, b: number) => a < b ? 1 : -1

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const [ia, ib, ic] = trianglePositionIndices(indices, triangle)
    const va = vertexKey(ia), vb = vertexKey(ib), vc = vertexKey(ic)
    faceIds[triangle] = [ia, ib, ic]
    const a = ia * 3, b = ib * 3, c = ic * 3
    const abx = positions[b] - positions[a], aby = positions[b + 1] - positions[a + 1], abz = positions[b + 2] - positions[a + 2]
    const acx = positions[c] - positions[a], acy = positions[c + 1] - positions[a + 1], acz = positions[c + 2] - positions[a + 2]
    const crossX = aby * acz - abz * acy
    const crossY = abz * acx - abx * acz
    const crossZ = abx * acy - aby * acx
    const area = Math.hypot(crossX, crossY, crossZ) * 0.5
    if (area <= areaTolerance || va === vb || vb === vc || vc === va) degenerate.add(triangle)

    const directedEdges: [number, number][] = [[va, vb], [vb, vc], [vc, va]]
    for (const [start, end] of directedEdges) {
      if (start === end) continue
      const key = edgeKey(start, end)
      const edge = edges.get(key)
      if (edge) {
        edge.count += 1
        edge.orientation += edgeDirection(start, end)
        edge.triangles.push(triangle)
      } else edges.set(key, { count: 1, orientation: edgeDirection(start, end), triangles: [triangle] })
    }
  }

  let openEdges = 0
  let nonManifoldEdges = 0
  let allEdgesPaired = edges.size > 0
  for (const edge of edges.values()) {
    if (edge.count === 1) {
      openEdges += 1
      allEdgesPaired = false
      for (const triangle of edge.triangles) openFaceIds.add(triangle)
    } else if (edge.count > 2) {
      nonManifoldEdges += 1
      allEdgesPaired = false
      for (const triangle of edge.triangles) nonManifoldFaceIds.add(triangle)
    } else if (edge.count === 2 && edge.orientation !== 0) {
      for (const triangle of edge.triangles) inverted.add(triangle)
    }
  }

  // A consistently oriented, closed shell with negative signed volume has inward-facing normals.
  if (allEdgesPaired) {
    let signedVolume = 0
    for (let triangle = 0; triangle < triangleCount; triangle += 1) {
      const [ia, ib, ic] = faceIds[triangle]
      const a = ia * 3, b = ib * 3, c = ic * 3
      signedVolume += (
        positions[a] * (positions[b + 1] * positions[c + 2] - positions[b + 2] * positions[c + 1]) +
        positions[a + 1] * (positions[b + 2] * positions[c] - positions[b] * positions[c + 2]) +
        positions[a + 2] * (positions[b] * positions[c + 1] - positions[b + 1] * positions[c])
      ) / 6
    }
    if (signedVolume < -Math.max(diagonal ** 3 * 1e-12, 1e-12)) {
      for (let triangle = 0; triangle < triangleCount; triangle += 1) inverted.add(triangle)
    }
  }

  const makeHighlight = (triangles: Set<number>) => {
    const selected = [...triangles]
    const step = Math.max(1, Math.ceil(selected.length / MAX_HIGHLIGHT_TRIANGLES))
    const output = new Float32Array(Math.ceil(selected.length / step) * 9)
    let cursor = 0
    for (let index = 0; index < selected.length; index += step) {
      const [ia, ib, ic] = faceIds[selected[index]]
      for (const vertex of [ia, ib, ic]) {
        output[cursor++] = positions[vertex * 3]
        output[cursor++] = positions[vertex * 3 + 1]
        output[cursor++] = positions[vertex * 3 + 2]
      }
    }
    return output
  }
  const highlightSets = {
    openEdges: openFaceIds,
    nonManifold: nonManifoldFaceIds,
    invertedNormals: inverted,
    degenerateFaces: degenerate,
  }
  const highlightBuffers = {
    openEdges: makeHighlight(openFaceIds).buffer,
    nonManifold: makeHighlight(nonManifoldFaceIds).buffer,
    invertedNormals: makeHighlight(inverted).buffer,
    degenerateFaces: makeHighlight(degenerate).buffer,
  }
  const sampledHighlights = Object.values(highlightSets).some(set => set.size > MAX_HIGHLIGHT_TRIANGLES)
  return {
    counts: { openEdges, nonManifoldEdges, invertedTriangles: inverted.size, degenerateTriangles: degenerate.size },
    triangleCounts: { openEdges: openFaceIds.size, nonManifold: nonManifoldFaceIds.size, invertedNormals: inverted.size, degenerateFaces: degenerate.size },
    highlights: highlightBuffers,
    sampledHighlights,
  }
}

workerScope.onmessage = async event => {
  try {
    workerScope.postMessage({ stage: 'Checking mesh topology and face orientation…' })
    await new Promise<void>(resolve => setTimeout(resolve, 0))
    const result = analyzeMesh(event.data)
    const transfer = Object.values(result.highlights)
    workerScope.postMessage(result, transfer)
  } catch (error) {
    workerScope.postMessage({ error: error instanceof Error ? error.message : 'The mesh integrity check failed' })
  }
}
