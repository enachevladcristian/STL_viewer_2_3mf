import * as THREE from 'three'
import { DOMParser as WorkerDOMParser } from '@xmldom/xmldom'
import { ThreeMFLoader } from 'three/examples/jsm/loaders/3MFLoader.js'
import { unzipSync } from 'three/examples/jsm/libs/fflate.module.js'

type Request = { buffer: ArrayBuffer }
type Parameter = { key: string; value: string; source: string }
type Success = {
  positions: ArrayBuffer
  normals: ArrayBuffer
  triangles: number
  dimensions: [number, number, number]
  bounds: [number, number, number, number, number, number]
  volume: number
  area: number
  parameters: Parameter[]
  unit: string
}
type Failure = { error: string }

const scope = self as unknown as {
  onmessage: ((event: MessageEvent<Request>) => void) | null
  postMessage: (message: Success | Failure, transfer?: Transferable[]) => void
}

function installXmlParser() {
  const query = (root: Element | Document, selector: string) => {
    const rootNode = root as Node
    let current: Element[] = [rootNode.nodeType === 9 ? (root as Document).documentElement : root as Element]
    let firstTag = true
    for (const tag of selector.trim().split(/\s+/)) {
      const next: Element[] = []
      for (const parent of current) {
        if (firstTag && rootNode.nodeType === 9 && (parent.tagName === tag || parent.localName === tag)) next.push(parent)
        next.push(...Array.from(parent.getElementsByTagName(tag)))
      }
      current = next
      firstTag = false
    }
    return current
  }
  const sample = new WorkerDOMParser().parseFromString('<root/>', 'application/xml')
  const elementPrototype = Object.getPrototypeOf(sample.documentElement)
  const documentPrototype = Object.getPrototypeOf(sample)
  Object.defineProperty(elementPrototype, 'querySelectorAll', { configurable: true, value: function (this: Element, selector: string) { return query(this, selector) } })
  Object.defineProperty(elementPrototype, 'querySelector', { configurable: true, value: function (this: Element, selector: string) { return query(this, selector)[0] ?? null } })
  Object.defineProperty(documentPrototype, 'querySelectorAll', { configurable: true, value: function (this: Document, selector: string) { return query(this, selector) } })
  Object.defineProperty(documentPrototype, 'querySelector', { configurable: true, value: function (this: Document, selector: string) { return query(this, selector)[0] ?? null } })
  Object.defineProperty(globalThis, 'DOMParser', { value: WorkerDOMParser, configurable: true, writable: true })
}
installXmlParser()

const MM_PER_UNIT: Record<string, number> = {
  micron: 0.001, micrometer: 0.001, millimeter: 1, centimeter: 10,
  meter: 1000, inch: 25.4, foot: 304.8,
}
const settingPattern = /layer|nozzle|temperature|temp|bed|infill|support|speed|wall|perimeter|retract|filament|material|quality|adhesion|printer|build.?plate|extruder|flow|density|shell|brim|raft|resolution|cooling|fan|acceleration|travel/i

function addParameter(list: Parameter[], key: string, value: string, source: string) {
  const normalizedKey = key.trim().replace(/[_-]+/g, ' ')
  const normalizedValue = value.trim()
  if (!normalizedKey || !normalizedValue || normalizedValue.length > 500 || !settingPattern.test(`${key} ${source}`)) return
  if (/^(true|false|null|undefined)$/i.test(normalizedValue) && !settingPattern.test(key)) return
  if (list.some(item => item.key.toLowerCase() === normalizedKey.toLowerCase() && item.value === normalizedValue)) return
  list.push({ key: normalizedKey, value: normalizedValue, source })
}

function extractParameters(files: Record<string, Uint8Array>, modelXml: string) {
  const decoder = new TextDecoder()
  const parameters: Parameter[] = []
  const candidates = Object.entries(files).filter(([path]) => /^(metadata\/|3d\/.*\.model$)/i.test(path)
    && /\.(config|json|xml|ini|txt|model)$/i.test(path))
  const modelEntries = candidates.filter(([path]) => /\.model$/i.test(path))
  for (const [path, bytes] of candidates) {
    const text = path.toLowerCase().endsWith('.model') ? modelXml : decoder.decode(bytes)
    const source = path.split('/').pop() || path
    if (/\.json$/i.test(path) || /^[\s\n]*[\{\[]/.test(text)) {
      try {
        const flatten = (value: unknown, prefix = '') => {
          if (value && typeof value === 'object' && !Array.isArray(value)) {
            for (const [key, child] of Object.entries(value as Record<string, unknown>)) flatten(child, prefix ? `${prefix}.${key}` : key)
          } else if (Array.isArray(value)) {
            if (value.length < 12 && value.every(item => ['string', 'number', 'boolean'].includes(typeof item))) addParameter(parameters, prefix, value.join(', '), source)
          } else if (value !== null && value !== undefined) addParameter(parameters, prefix, String(value), source)
        }
        flatten(JSON.parse(text))
      } catch { /* Ignore non-JSON metadata. */ }
    } else {
      try {
        const document = new DOMParser().parseFromString(text, 'application/xml')
        if (!document.querySelector('parsererror')) {
          for (const element of Array.from(document.getElementsByTagName('*'))) {
            const tag = element.tagName.split(':').pop() || element.tagName
            const textValue = element.textContent?.trim() || ''
            const keyAttr = element.getAttribute('key') || element.getAttribute('name') || element.getAttribute('id')
            const valueAttr = element.getAttribute('value')
            if (keyAttr && valueAttr) addParameter(parameters, keyAttr, valueAttr, source)
            else if (keyAttr && textValue && element.children.length === 0) addParameter(parameters, keyAttr, textValue, source)
            else if (textValue && element.children.length === 0 && !['metadata', 'option', 'setting', 'config'].includes(tag.toLowerCase())) addParameter(parameters, tag, textValue, source)
          }
        }
      } catch { /* Continue with simple key/value parsing below. */ }
      for (const line of text.split(/\r?\n/)) {
        const match = line.match(/^\s*([\w.-]+)\s*=\s*(.*?)\s*$/)
        if (match) addParameter(parameters, match[1], match[2].replace(/^['"]|['"]$/g, ''), source)
      }
    }
  }
  // The ThreeMF loader also supports embedded model metadata; retain it even if
  // the XML settings files do not use a slicer-specific config convention.
  if (modelEntries.length === 0 && modelXml) {
    try {
      const document = new DOMParser().parseFromString(modelXml, 'application/xml')
      for (const node of Array.from(document.getElementsByTagName('*'))) {
        if ((node.tagName.split(':').pop() || node.tagName).toLowerCase() === 'metadata') {
          const key = node.getAttribute('name') || node.getAttribute('key') || ''
          if (key) addParameter(parameters, key, node.textContent || '', '3D model metadata')
        }
      }
    } catch { /* Optional metadata only. */ }
  }
  const score = (parameter: Parameter) => /layer|nozzle|temperature|bed|infill|support|speed|filament|material|printer|build.?plate/i.test(parameter.key) ? 0 : 1
  return parameters.sort((a, b) => score(a) - score(b) || a.key.localeCompare(b.key)).slice(0, 80)
}

scope.onmessage = event => {
  try {
    // 3MFLoader's optional texture path creates an HTML image. The worker only
    // needs mesh geometry, so provide a inert image placeholder in worker scope.
    const workerGlobal = globalThis as typeof globalThis & { document?: { createElementNS: () => object } }
    if (!workerGlobal.document) {
      Object.defineProperty(workerGlobal, 'document', { value: { createElementNS: () => ({ addEventListener() {}, removeEventListener() {}, set src(_value: string) {} }) }, configurable: true })
    }
    const files = unzipSync(new Uint8Array(event.data.buffer))
    const decoder = new TextDecoder()
    const modelEntry = Object.entries(files).find(([path]) => /^3d\/.*\.model$/i.test(path))
    if (!modelEntry) throw new Error('This 3MF archive does not contain a 3D model.')
    const modelXml = decoder.decode(modelEntry[1])
    const xml = new DOMParser().parseFromString(modelXml, 'application/xml')
    if (xml.querySelector('parsererror')) throw new Error('The 3MF model data is invalid.')
    const modelNode = xml.documentElement
    const unit = modelNode.getAttribute('unit') || 'millimeter'
    const unitScale = MM_PER_UNIT[unit.toLowerCase()]
    if (!unitScale) throw new Error(`The 3MF uses an unsupported unit: ${unit}.`)

    const root = new ThreeMFLoader().parse(event.data.buffer)
    // 3MF coordinates are Z-up; the viewer's build plate uses Y-up.
    root.rotation.x = -Math.PI / 2
    root.updateMatrixWorld(true)
    const vertices: number[] = []
    const point = new THREE.Vector3()
    root.traverse(object => {
      if (!(object instanceof THREE.Mesh)) return
      const position = object.geometry.getAttribute('position')
      const index = object.geometry.getIndex()
      if (!position) return
      const matrix = object.matrixWorld.clone()
      for (let i = 0; i < (index ? index.count : position.count); i += 3) {
        for (let corner = 0; corner < 3; corner += 1) {
          const vertexIndex = index ? index.getX(i + corner) : i + corner
          point.fromBufferAttribute(position, vertexIndex).applyMatrix4(matrix).multiplyScalar(unitScale)
          vertices.push(point.x, point.y, point.z)
        }
      }
    })
    if (vertices.length < 9) throw new Error('No printable mesh triangles were found in the 3MF.')

    let minX = Infinity, minY = Infinity, minZ = Infinity
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity
    for (let i = 0; i < vertices.length; i += 3) {
      const x = vertices[i], y = vertices[i + 1], z = vertices[i + 2]
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) throw new Error('The 3MF contains invalid vertex coordinates.')
      minX = Math.min(minX, x); maxX = Math.max(maxX, x)
      minY = Math.min(minY, y); maxY = Math.max(maxY, y)
      minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z)
    }
    const centerX = (minX + maxX) * 0.5, centerY = (minY + maxY) * 0.5, centerZ = (minZ + maxZ) * 0.5
    const dimensions: [number, number, number] = [maxX - minX, maxY - minY, maxZ - minZ]
    minX -= centerX; maxX -= centerX; minY -= centerY; maxY -= centerY; minZ -= centerZ; maxZ -= centerZ

    const positions = new Float32Array(vertices.length)
    let area = 0, signedVolume = 0
    for (let i = 0; i < vertices.length; i += 9) {
      const ax = vertices[i] - centerX, ay = vertices[i + 1] - centerY, az = vertices[i + 2] - centerZ
      const bx = vertices[i + 3] - centerX, by = vertices[i + 4] - centerY, bz = vertices[i + 5] - centerZ
      const cx = vertices[i + 6] - centerX, cy = vertices[i + 7] - centerY, cz = vertices[i + 8] - centerZ
      const abx = bx - ax, aby = by - ay, abz = bz - az, acx = cx - ax, acy = cy - ay, acz = cz - az
      area += Math.hypot(aby * acz - abz * acy, abz * acx - abx * acz, abx * acy - aby * acx) * 0.5
      signedVolume += (ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)) / 6
      for (let j = 0; j < 9; j += 3) {
        positions[i + j] = vertices[i + j] - centerX
        positions[i + j + 1] = vertices[i + j + 1] - centerY
        positions[i + j + 2] = vertices[i + j + 2] - centerZ
      }
    }
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.computeVertexNormals()
    const normals = (geometry.getAttribute('normal').array as Float32Array).slice()
    const result: Success = {
      positions: positions.buffer,
      normals: normals.buffer,
      triangles: Math.floor(positions.length / 9),
      dimensions,
      bounds: [minX, minY, minZ, maxX, maxY, maxZ],
      volume: Math.abs(signedVolume),
      area,
      parameters: extractParameters(files, modelXml),
      unit,
    }
    scope.postMessage(result, [result.positions, result.normals])
  } catch (error) {
    scope.postMessage({ error: error instanceof Error ? error.message : 'Unable to read this 3MF file.' })
  }
}
