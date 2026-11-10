import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Box, Check, ChevronDown, CloudUpload, Download, Expand, FileBox, FileUp, Grid2X2, Info, Move,
  Maximize, MousePointer2, Pencil, Rotate3D, RotateCcw, Ruler, Scan, Settings2,
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
  source: File
}

type CheckWorkerResponse = {
  stage?: string
  originalTriangleCount?: number
  simplified?: boolean
  simplificationError?: number
  triangleCount?: number
  flaggedTriangles?: number
  flaggedArea?: number
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
  const minX = -110 - bounds.min.x
  const maxX = 110 - bounds.max.x
  const minZ = -110 - bounds.min.z
  const maxZ = 110 - bounds.max.z
  const x = minX <= maxX ? THREE.MathUtils.clamp(position[0], minX, maxX) : 0
  const z = minZ <= maxZ ? THREE.MathUtils.clamp(position[2], minZ, maxZ) : 0
  return [x, position[1], z]
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

function makeBinarySTL(geometry: THREE.BufferGeometry) {
  const position = geometry.getAttribute('position')
  const index = geometry.getIndex()
  const triangleCount = Math.floor((index?.count ?? position.count) / 3)
  const buffer = new ArrayBuffer(84 + triangleCount * 50)
  const bytes = new Uint8Array(buffer)
  new TextEncoder().encode('STL Viewer export from 3MF model').forEach((byte, i) => { bytes[i] = byte })
  const view = new DataView(buffer)
  view.setUint32(80, triangleCount, true)
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3()
  const normal = new THREE.Vector3(), edge1 = new THREE.Vector3(), edge2 = new THREE.Vector3()
  let offset = 84
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const base = triangle * 3
    const ia = index ? index.getX(base) : base
    const ib = index ? index.getX(base + 1) : base + 1
    const ic = index ? index.getX(base + 2) : base + 2
    a.fromBufferAttribute(position, ia); b.fromBufferAttribute(position, ib); c.fromBufferAttribute(position, ic)
    normal.crossVectors(edge1.subVectors(b, a), edge2.subVectors(c, a)).normalize()
    for (const value of [normal.x, normal.y, normal.z, a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z]) {
      view.setFloat32(offset, value, true)
      offset += 4
    }
    view.setUint16(offset, 0, true)
    offset += 2
  }
  return new Blob([buffer], { type: 'model/stl' })
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
  model, showGrid, showAxes, showDimensions, rotation, position, scale, rotateMode, placeOnBedMode, axisTransformMode, analysisOverlay, supportOverlay, onRotationChange, onPlaceSurface, onFreeRotateEnd, onAxisTransformStart, onAxisTransform, onAxisTransformEnd, onError, resetRef, viewPresetRef,
}: {
  model: ModelInfo | null
  showGrid: boolean
  showAxes: boolean
  showDimensions: boolean
  rotation: Rotation
  position: AxisValues
  scale: AxisValues
  analysisOverlay: THREE.BufferGeometry | null
  supportOverlay: THREE.BufferGeometry | null
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
    grid?: THREE.GridHelper
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
      new THREE.BoxGeometry(220, 0.18, 220),
      new THREE.MeshStandardMaterial({ color: '#dfe5ed', roughness: 0.92, metalness: 0.02 }),
    )
    bed.position.y = -0.16
    scene.add(bed)
    const grid = new THREE.GridHelper(220, 44, 0x9baabe, 0xcbd4e0)
    grid.position.y = -0.065
    scene.add(grid)
    const bedOutline = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-110, -0.06, -110), new THREE.Vector3(110, -0.06, -110),
        new THREE.Vector3(110, -0.06, 110), new THREE.Vector3(-110, -0.06, 110),
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
    const extent = Math.max(...model.dimensions, 220)
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

function App() {
  const [model, setModel] = useState<ModelInfo | null>(null)
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
  const [showGrid, setShowGrid] = useState(true)
  const [showDimensions, setShowDimensions] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isChecking, setIsChecking] = useState(false)
  const [checkStatus, setCheckStatus] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const projectNameInputRef = useRef<HTMLInputElement>(null)
  const cameraControls = useRef<(() => void) | null>(null)
  const viewPresetControls = useRef<((view: ViewPreset) => void) | null>(null)
  const loadWorkerRef = useRef<Worker | null>(null)
  const checkWorkerRef = useRef<Worker | null>(null)
  const cancelCheck = useCallback(() => {
    checkWorkerRef.current?.terminate()
    checkWorkerRef.current = null
    setIsChecking(false)
    setCheckStatus('')
  }, [])

  useEffect(() => () => {
    loadWorkerRef.current?.terminate()
    checkWorkerRef.current?.terminate()
  }, [])

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
          dimensions: result.dimensions, volume: result.volume ?? 0, area: result.area ?? 0, geometry, source: file,
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
        source: file,
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
  }, [cancelCheck])

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
      worker.postMessage({ positions: positions.buffer, bounds: boundsTuple, rotation, scale, translation: position }, [positions.buffer])
    } catch {
      checkWorkerRef.current = null
      worker.terminate()
      setIsChecking(false)
      setCheckStatus('')
      setError('Not enough memory to run the printability check on this model.')
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
  const downloadModel = () => {
    if (!model) return
    const safeName = projectName.trim()
      .replace(/\.stl$/i, '')
      .replace(/[<>:"/\\|?*\u0000-\u001F]/g, '-')
      .replace(/\s+/g, ' ')
      .trim() || 'model'
    const fileName = `${safeName}.stl`
    const stlFile = model.format === 'STL' ? model.source : makeBinarySTL(model.geometry)
    const url = URL.createObjectURL(stlFile)
    const link = document.createElement('a')
    link.href = url
    link.download = fileName
    link.style.display = 'none'
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), 10_000)
    setNotice(`Downloading ${fileName}`)
    window.setTimeout(() => setNotice(''), 2800)
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
            <button className="download-project-button" title="Download STL using the project name" onClick={downloadModel} disabled={!model}><Download size={15}/><span>Download STL</span></button>
            <span className="tool-separator"/>
            <button className={`canvas-tool ${showGrid ? 'selected' : ''}`} title="Toggle grid" onClick={() => setShowGrid(value => !value)}><Grid2X2 size={16}/></button>
            <span className="tool-separator"/>
            <button className="canvas-tool" title="Fit model to view" onClick={resetView}><Maximize size={16}/></button>
            <span className="tool-separator"/>
            <button className="canvas-tool fullscreen-tool" title="Fullscreen" onClick={() => document.querySelector('.canvas-panel')?.requestFullscreen?.()}><Expand size={16}/></button>
          </div>

        </div>

        <div className={`canvas-body${(rotateMenuOpen || movePanelOpen || scalePanelOpen || checkPanelOpen) && model ? ' with-rotate-panel' : ''}`}>
          <Viewport model={model} showGrid={showGrid} showAxes={rotateMenuOpen || rotateMode || movePanelOpen || scalePanelOpen || axisDragMode !== null} showDimensions={showDimensions} rotation={rotation} position={position} scale={scale} rotateMode={rotateMode} placeOnBedMode={placeOnBedMode} axisTransformMode={axisDragMode ?? (movePanelOpen ? 'move' : scalePanelOpen ? 'scale' : null)} analysisOverlay={analysisResult?.highlights ?? null} supportOverlay={analysisResult?.supports ?? null} onRotationChange={onRotationChange} onPlaceSurface={onPlaceSurface} onFreeRotateEnd={onFreeRotateEnd} onAxisTransformStart={onAxisTransformStart} onAxisTransform={onAxisTransform} onAxisTransformEnd={onAxisTransformEnd} onError={setError} resetRef={cameraControls} viewPresetRef={viewPresetControls}/>
          {model && <div className="left-tool-rail">
            <button className={`left-tool-button ${rotateMenuOpen || rotateMode || placeOnBedMode ? 'selected' : ''}`} title="Open rotate tools" aria-label="Open rotate tools" aria-expanded={rotateMenuOpen} onClick={toggleRotatePanel}><Rotate3D size={18}/><span>Rotate</span></button>
            <button className={`left-tool-button ${movePanelOpen || axisDragMode === 'move' ? 'selected' : ''}`} title="Move model" aria-label="Move model" aria-expanded={movePanelOpen} onClick={toggleMovePanel}><Move size={18}/><span>Move</span></button>
            <button className={`left-tool-button ${scalePanelOpen || axisDragMode === 'scale' ? 'selected' : ''}`} title="Scale model" aria-label="Scale model" aria-expanded={scalePanelOpen} onClick={toggleScalePanel}><Maximize size={18}/><span>Scale</span></button>
            <button className={`left-tool-button ${checkPanelOpen || analysisResult ? 'selected' : ''}`} title="Check printability" aria-label="Check printability" aria-pressed={checkPanelOpen || Boolean(analysisResult)} onClick={toggleCheckPanel}><Scan size={18}/><span>Check</span></button>
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
            <div className="rotate-menu-header"><div><Scan size={16}/><strong>Printability check</strong></div><button title="Close printability panel" aria-label="Close printability panel" onClick={() => setCheckPanelOpen(false)}><X size={15}/></button></div>
            <div className="analysis-profile"><span className="analysis-profile-dot"/>FDM profile <span>·</span> current orientation</div>
            {!analysisResult ? <>
              <p className="analysis-intro">Checks surfaces above the bed against the previous 0.2 mm layer. Green shows only the box-minus-model volume that rises from the bed to the model.</p>
              <button className="run-analysis-button" onClick={runPrintabilityCheck} disabled={isChecking}><Scan size={15}/>{isChecking ? (checkStatus || 'Checking geometry…') : 'Run check'}</button>
            </> : <>
              <div className={`analysis-summary ${analysisResult.flaggedTriangles ? 'warning' : 'pass'}`}>
                <strong>{analysisResult.flaggedTriangles ? 'Potential supports needed' : 'No unsupported surfaces found'}</strong>
                <span>{analysisResult.flaggedTriangles.toLocaleString()} of {analysisResult.triangleCount.toLocaleString()} analyzed triangles are flagged for support</span>
                {analysisResult.simplified && <span className="analysis-approx-note">Quick approximate analysis: simplified from {analysisResult.originalTriangleCount?.toLocaleString()} triangles. The displayed model and export are unchanged.</span>}
              </div>
              {analysisResult.flaggedTriangles > 0 && <div className="analysis-legend"><i/> Red: unsupported</div>}
              <div className="analysis-legend"><i className="support-legend-swatch"/> Green: bed-to-model difference (voxelized)</div>
              <p className="analysis-intro">This envelope visualizes the volume difference; it is not print-ready support material.</p>
              <div className="analysis-result-row"><span>Unsupported surface area</span><strong>{analysisResult.flaggedArea.toFixed(1)} mm²</strong></div>
              <button className="run-analysis-button secondary" onClick={runPrintabilityCheck} disabled={isChecking}><Scan size={14}/>{isChecking ? (checkStatus || 'Checking geometry…') : 'Run again'}</button>
            </>}
            <div className="analysis-divider"/>
            <div className="analysis-manual"><strong>Other FDM rules</strong><p>Wall thickness (0.8 mm), details (0.6 × 2 mm), bridges (10 mm), holes (Ø2 mm), clearance (0.5 mm), minimum features (2 mm), and pins (3 mm) still need manual review. They are not counted as passing.</p></div>
            {analysisResult && <button className="clear-analysis-button" title="Remove check highlights and preview" onClick={() => { cancelCheck(); setAnalysisResult(null) }}><X size={14}/> Clear highlights</button>}
          </div>}
          {isDragging && <div className="drag-overlay"><div className="drag-card"><CloudUpload size={30}/><strong>Drop an STL or 3MF to open</strong><span>Release to start viewing</span></div></div>}
          {!model && !isDragging && <div className="welcome-card">
            <div className="welcome-art"><div className="art-orbit orbit-one"/><div className="art-orbit orbit-two"/><div className="art-cube"><div className="cube-top"/><div className="cube-left"/><div className="cube-right"/></div><span className="art-spark spark-a">✳</span><span className="art-spark spark-b">✦</span></div>
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
        <div className="inspector-heading"><div><span className="panel-kicker">WORKSPACE</span><h2>Inspector</h2></div><button className="icon-button settings-button" title="Inspector settings"><Settings2 size={17}/></button></div>
        <div className="inspector-content">
          <div className="section-label">MODEL</div>
          {model ? <>
            <div className="model-card"><div className="model-card-copy"><strong title={model.name}>{model.name}</strong><span>{formatSize(model.size)}</span></div><button className="more-button" title="Remove model" aria-label="Remove model" onClick={() => { cancelCheck(); setModel(null); setProjectName('Untitled project'); setRotation([0, 0, 0]); setPosition([0, 0, 0]); setScale([1, 1, 1]); setAxisDragMode(null); setRotateMode(false); setPlaceOnBedMode(false); setMovePanelOpen(false); setScalePanelOpen(false); setCheckPanelOpen(false); setAnalysisResult(null); setError('') }}><X size={15}/></button></div>
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
          </> : <>
            <div className="empty-model"><div className="empty-icon"><Box size={20}/></div><strong>No model open</strong><p>Upload an STL or 3MF file to inspect its mesh and available print settings.</p><button onClick={chooseFile}><Upload size={14}/> Open file</button></div>
            <div className="section-label section-gap">SUPPORTED FORMATS</div><div className="format-row"><span className="format-chip">STL</span><div><strong>STereoLithography</strong><span>Binary and ASCII meshes</span></div><Check size={15}/></div><div className="format-row"><span className="format-chip">3MF</span><div><strong>3D Manufacturing Format</strong><span>Mesh and embedded print settings</span></div><Check size={15}/></div>
            <div className="privacy-note"><ShieldCheck size={16}/><span>Your files never leave your device. Processing happens locally in your browser.</span></div>
          </>}
        </div>
        <button className="upload-button bottom-open-button" onClick={chooseFile} disabled={isProcessing} title="Open an STL or 3MF model" aria-label="Open STL or 3MF model"><Upload size={18}/><span>{isProcessing ? 'Processing model…' : 'Open STL'}</span></button>
      </aside>
    </main>
    <input ref={inputRef} type="file" accept=".stl,.3mf,model/stl,application/vnd.ms-package.3dmanufacturing-3dmodel+xml" hidden onChange={event => { loadFile(event.target.files?.[0]); event.currentTarget.value = '' }}/>
  </div>
}

export default App
