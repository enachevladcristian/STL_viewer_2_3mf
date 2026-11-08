import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Box, Check, CloudUpload, Download, Expand, FileBox, FileUp, Grid2X2, Info,
  Maximize, MousePointer2, Pencil, Rotate3D, RotateCcw, Scan, Settings2,
  ShieldCheck, Sparkles, Triangle, Upload, X,
} from 'lucide-react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'

type ModelInfo = {
  name: string
  size: number
  triangles: number
  dimensions: [number, number, number]
  volume: number
  area: number
  geometry: THREE.BufferGeometry
  source: ArrayBuffer
}

type Rotation = [number, number, number]
const formatSize = (bytes: number) => bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / (1024 * 1024)).toFixed(2)} MB`
const formatMeasure = (n: number) => `${n.toFixed(2)} mm`

function getStats(geometry: THREE.BufferGeometry) {
  const position = geometry.getAttribute('position')
  let volume = 0
  let area = 0
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3()
  for (let i = 0; i < position.count; i += 3) {
    a.fromBufferAttribute(position, i)
    b.fromBufferAttribute(position, i + 1)
    c.fromBufferAttribute(position, i + 2)
    volume += a.dot(b.clone().cross(c)) / 6
    area += b.clone().sub(a).cross(c.clone().sub(a)).length() / 2
  }
  return { volume: Math.abs(volume), area }
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
  model, wireframe, showGrid, showAxes, rotation, rotateMode, onRotationChange, onError, resetRef,
}: {
  model: ModelInfo | null
  wireframe: boolean
  showGrid: boolean
  showAxes: boolean
  rotation: Rotation
  rotateMode: boolean
  onRotationChange: (value: Rotation) => void
  onError: (message: string) => void
  resetRef: React.MutableRefObject<(() => void) | null>
}) {
  const hostRef = useRef<HTMLDivElement>(null)
  const rotationRef = useRef<Rotation>(rotation)
  const engineRef = useRef<{
    renderer: THREE.WebGLRenderer
    scene: THREE.Scene
    camera: THREE.PerspectiveCamera
    controls: OrbitControls
    mesh?: THREE.Mesh
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
    if (!model) return

    const geometry = model.geometry.clone()
    geometry.computeBoundingBox()
    const bounds = geometry.boundingBox!
    const center = bounds.getCenter(new THREE.Vector3())
    geometry.translate(-center.x, -center.y, -center.z)
    geometry.computeBoundingBox()
    geometry.computeVertexNormals()
    const material = new THREE.MeshStandardMaterial({ color: '#c4d4e8', roughness: 0.34, metalness: 0.12, flatShading: true, wireframe })
    const mesh = new THREE.Mesh(geometry, material)
    mesh.add(makeAxisLabels(Math.max(...model.dimensions) * 0.58))
    engine.scene.add(mesh)
    engine.mesh = mesh

    const modelBounds = geometry.boundingBox!.clone()
    mesh.position.y = -modelBounds.min.y
    const extent = Math.max(...model.dimensions, 220)
    engine.camera.position.set(extent * 1.35, extent * 1.05, extent * 1.65)
    engine.camera.near = Math.max(extent / 1000, 0.001)
    engine.camera.far = extent * 100
    engine.camera.updateProjectionMatrix()
    engine.controls.target.set(0, 0, 0)
    engine.controls.minDistance = extent * 0.08
    engine.controls.maxDistance = extent * 20
    engine.controls.update()
    const resetCamera = () => {
      engine.camera.position.set(extent * 1.35, extent * 1.05, extent * 1.65)
      engine.controls.target.set(0, 0, 0)
      engine.controls.update()
    }
    resetRef.current = resetCamera
  }, [model, wireframe, resetRef])

  useEffect(() => {
    rotationRef.current = rotation
    const mesh = engineRef.current?.mesh
    if (!mesh || !model) return
    mesh.rotation.set(THREE.MathUtils.degToRad(rotation[0]), THREE.MathUtils.degToRad(rotation[1]), THREE.MathUtils.degToRad(rotation[2]))
    mesh.position.set(0, 0, 0)
    mesh.updateMatrixWorld(true)
    const bounds = mesh.geometry.boundingBox!.clone()
    bounds.applyMatrix4(new THREE.Matrix4().makeRotationFromEuler(mesh.rotation))
    mesh.position.y = -bounds.min.y
    mesh.updateMatrixWorld(true)
  }, [model, rotation, wireframe])

  useEffect(() => {
    const axes = engineRef.current?.mesh?.getObjectByName('model-axis-arrows')
    if (axes) axes.visible = showAxes
  }, [model, showAxes, wireframe])

  useEffect(() => {
    const engine = engineRef.current
    if (!engine) return
    engine.controls.enabled = !rotateMode
    const canvas = engine.renderer.domElement
    if (!showAxes || !model) return () => { engine.controls.enabled = true }

    const raycaster = new THREE.Raycaster()
    raycaster.params.Line.threshold = Math.max(Math.max(...model.dimensions) * 0.035, 0.5)
    const pointer = new THREE.Vector2()
    let drag: { pointerId: number; x: number; y: number; axis: number | null } | null = null
    const onPointerDown = (event: PointerEvent) => {
      const mesh = engine.mesh
      if (!mesh) return
      const rect = canvas.getBoundingClientRect()
      pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1)
      raycaster.setFromCamera(pointer, engine.camera)
      const axisGroup = mesh.getObjectByName('model-axis-arrows')
      const axisHit = axisGroup
        ? raycaster.intersectObject(axisGroup, true).find(hit => Number.isInteger(hit.object.userData.rotationAxis))
        : undefined
      const axis = axisHit ? Number(axisHit.object.userData.rotationAxis) : null
      const hitModel = rotateMode && raycaster.intersectObject(mesh, false).length > 0
      if (axis === null && !hitModel) return
      event.preventDefault()
      event.stopImmediatePropagation()
      canvas.setPointerCapture(event.pointerId)
      drag = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, axis }
    }
    const onPointerMove = (event: PointerEvent) => {
      if (!drag || event.pointerId !== drag.pointerId) return
      event.stopImmediatePropagation()
      const dx = event.clientX - drag.x
      const dy = event.clientY - drag.y
      drag.x = event.clientX
      drag.y = event.clientY
      let next: Rotation
      if (drag.axis !== null) {
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
      drag = null
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId)
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
  }, [model, onRotationChange, rotateMode, showAxes])

  useEffect(() => {
    const engine = engineRef.current
    if (engine?.grid) engine.grid.visible = showGrid
  }, [showGrid])

  return <div className={`viewport${showAxes ? ' rotate-axes-active' : ''}${rotateMode ? ' rotate-mode' : ''}`} ref={hostRef} />
}

function App() {
  const [model, setModel] = useState<ModelInfo | null>(null)
  const [projectName, setProjectName] = useState('Untitled project')
  const [rotation, setRotation] = useState<Rotation>([0, 0, 0])
  const [rotateMenuOpen, setRotateMenuOpen] = useState(false)
  const [rotateMode, setRotateMode] = useState(false)
  const [wireframe, setWireframe] = useState(false)
  const [showGrid, setShowGrid] = useState(true)
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const projectNameInputRef = useRef<HTMLInputElement>(null)
  const cameraControls = useRef<(() => void) | null>(null)

  const loadFile = useCallback((file?: File) => {
    if (!file) return
    setError('')
    setNotice('')
    if (!file.name.toLowerCase().endsWith('.stl')) {
      setError('That file type is not supported yet. Choose an .stl file.')
      return
    }
    if (!file.size) {
      setError('This file appears to be empty. Please choose another STL.')
      return
    }
    const reader = new FileReader()
    reader.onerror = () => setError('We could not read this file. Please try another STL.')
    reader.onload = () => {
      try {
        const source = reader.result as ArrayBuffer
        const geometry = new STLLoader().parse(source)
        if (!geometry.getAttribute('position')?.count) throw new Error('No triangles found')
        geometry.computeBoundingBox()
        const box = geometry.boundingBox!
        const dimensions: [number, number, number] = [box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z]
        const stats = getStats(geometry)
        setModel({ name: file.name, size: file.size, triangles: Math.floor(geometry.getAttribute('position').count / 3), dimensions, volume: stats.volume, area: stats.area, geometry, source })
        setProjectName(file.name.replace(/\.stl$/i, '') || 'Untitled project')
        setRotation([0, 0, 0])
        setRotateMode(false)
        setNotice('Model loaded successfully')
        window.setTimeout(() => setNotice(''), 2800)
      } catch {
        setError('This STL could not be parsed. It may be damaged or contain no valid triangles.')
      }
    }
    reader.readAsArrayBuffer(file)
  }, [])

  const chooseFile = () => inputRef.current?.click()
  const resetView = () => cameraControls.current?.()
  const onRotationChange = useCallback((next: Rotation) => setRotation(next), [])
  const adjustRotation = (axis: number, amount: number) => setRotation(current => current.map((value, index) => index === axis ? value + amount : value) as Rotation)
  const downloadModel = () => {
    if (!model) return
    const safeName = projectName.trim()
      .replace(/\.stl$/i, '')
      .replace(/[<>:"/\\|?*\u0000-\u001F]/g, '-')
      .replace(/\s+/g, ' ')
      .trim() || 'model'
    const fileName = `${safeName}.stl`
    const url = URL.createObjectURL(new Blob([model.source.slice(0)], { type: 'model/stl' }))
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
    <header className="topbar">
      <div className="topbar-actions"><button className="upload-button" onClick={chooseFile}><Upload size={16}/><span>Open STL</span></button></div>
      <input ref={inputRef} type="file" accept=".stl,model/stl" hidden onChange={event => { loadFile(event.target.files?.[0]); event.currentTarget.value = '' }}/>
    </header>

    <main className="workspace">
      <section className="canvas-panel">
        <div className="canvas-header">
          <div className="file-crumb">
            <span className="crumb-icon"><FileBox size={16}/></span>
            <input ref={projectNameInputRef} className="project-name-input" aria-label="Project name" title="Edit project name" value={projectName} onChange={event => setProjectName(event.target.value)} onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur() }}/>
            <button className="rename-project-button" type="button" title="Rename project" aria-label="Rename project" onClick={() => projectNameInputRef.current?.focus()}><Pencil size={13}/></button>
            {model && <span className="file-ext">.stl</span>}
            {model && <span className="crumb-divider"/>}
            {model && <span className="local-badge"><ShieldCheck size={12}/> Local file</span>}
          </div>
          <div className="canvas-actions">
            <button className="download-project-button" title="Download STL using the project name" onClick={downloadModel} disabled={!model}><Download size={15}/><span>Download STL</span></button>
            <span className="tool-separator"/>
            <button className={`canvas-tool ${showGrid ? 'selected' : ''}`} title="Toggle grid" onClick={() => setShowGrid(value => !value)}><Grid2X2 size={16}/></button>
            <span className="tool-separator"/>
            <button className="canvas-tool" title="Fit model to view" onClick={resetView}><Maximize size={16}/></button>
            <button className="canvas-tool" title="Reset camera" onClick={resetView}><RotateCcw size={15}/></button>
            <span className="tool-separator"/>
            <button className={`canvas-tool ${wireframe ? 'selected' : ''}`} title="Toggle wireframe" onClick={() => setWireframe(value => !value)}><Scan size={17}/></button>
            <button className={`canvas-tool rotate-tool ${rotateMenuOpen || rotateMode ? 'selected' : ''}`} title="Rotate model" aria-label="Open rotate tools" aria-expanded={rotateMenuOpen} disabled={!model} onClick={() => setRotateMenuOpen(value => !value)}><Rotate3D size={17}/></button>
            <button className="canvas-tool fullscreen-tool" title="Fullscreen" onClick={() => document.querySelector('.canvas-panel')?.requestFullscreen?.()}><Expand size={16}/></button>
          </div>
          {rotateMenuOpen && model && <div className="rotate-menu" role="dialog" aria-label="Rotate model">
            <div className="rotate-menu-header"><div><Rotate3D size={16}/><strong>Rotate model</strong></div><button title="Close rotate menu" aria-label="Close rotate menu" onClick={() => setRotateMenuOpen(false)}><X size={15}/></button></div>
            <button className="free-rotate-toggle" aria-pressed={rotateMode} onClick={() => setRotateMode(value => !value)}><span><strong>Free rotate</strong><small>{rotateMode ? 'Drag the model in the viewport' : 'Drag directly on the model'}</small></span><span className={`toggle ${rotateMode ? 'on' : ''}`}><i/></span></button>
            <div className="rotate-menu-divider"/>
            <div className="rotate-menu-label">ROTATE BY 45°</div>
            {(['X', 'Y', 'Z'] as const).map((axis, index) => <div className="rotate-axis-row" key={axis}><span className={`rotate-axis-tag axis-${index === 0 ? 'red' : index === 1 ? 'green' : 'blue'}`}>{axis}</span><span className="rotate-angle">{Math.round(rotation[index])}°</span><button aria-label={`Rotate ${axis} negative 45 degrees`} onClick={() => adjustRotation(index, -45)}>−45°</button><button aria-label={`Rotate ${axis} positive 45 degrees`} onClick={() => adjustRotation(index, 45)}>+45°</button></div>)}
            <button className="reset-rotation" title="Return the model to its initial orientation and centered bed position" onClick={() => setRotation([0, 0, 0])}>Reset position</button>
          </div>}
        </div>

        <div className="canvas-body">
          <Viewport model={model} wireframe={wireframe} showGrid={showGrid} showAxes={rotateMenuOpen || rotateMode} rotation={rotation} rotateMode={rotateMode} onRotationChange={onRotationChange} onError={setError} resetRef={cameraControls}/>
          {isDragging && <div className="drag-overlay"><div className="drag-card"><CloudUpload size={30}/><strong>Drop your STL to open</strong><span>Release to start viewing</span></div></div>}
          {!model && !isDragging && <div className="welcome-card">
            <div className="welcome-art"><div className="art-orbit orbit-one"/><div className="art-orbit orbit-two"/><div className="art-cube"><div className="cube-top"/><div className="cube-left"/><div className="cube-right"/></div><span className="art-spark spark-a">✳</span><span className="art-spark spark-b">✦</span></div>
            <div className="welcome-copy"><span className="eyebrow"><Sparkles size={12}/> YOUR 3D WORKSPACE</span><h1>Bring your model<br/>into <em>focus.</em></h1><p>Open an STL file to explore every angle, inspect dimensions, and get to know your design.</p><button className="primary-cta" onClick={chooseFile}><FileUp size={17}/> Choose an STL file <span>↗</span></button><div className="drop-hint"><span className="hint-line"/>or drag and drop it here<span className="hint-line"/></div></div>
          </div>}
          {model && <div className="canvas-hint"><MousePointer2 size={13}/><span>{rotateMode ? 'Drag the model to rotate' : rotateMenuOpen ? 'Drag a colored axis to rotate' : 'Drag to orbit'}</span><i/><span>{rotateMode ? 'Turn off Free rotate in menu' : rotateMenuOpen ? 'Or use 45° controls' : 'Scroll to zoom'}</span></div>}
          <div className="view-cube" aria-label="3D axis orientation"><span className="axis-y">Y</span><span className="axis-x">X</span><span className="axis-z">Z</span><div className="cube-center"><Box size={14}/></div></div>
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
            <div className="model-card"><div className="model-thumb"><div className="thumb-shape"><Box size={25}/></div></div><div className="model-card-copy"><strong title={model.name}>{model.name}</strong><span>{formatSize(model.size)} <b>·</b> STL mesh</span></div><button className="more-button" title="Remove model" onClick={() => { setModel(null); setProjectName('Untitled project'); setRotation([0, 0, 0]); setRotateMode(false); setError('') }}><X size={15}/></button></div>
            <div className="section-label section-gap">DIMENSIONS <span>mm</span></div>
            <div className="dimension-grid"><div className="dim-item"><span className="dim-axis axis-red">X</span><strong>{formatMeasure(model.dimensions[0])}</strong></div><div className="dim-item"><span className="dim-axis axis-green">Y</span><strong>{formatMeasure(model.dimensions[1])}</strong></div><div className="dim-item"><span className="dim-axis axis-blue">Z</span><strong>{formatMeasure(model.dimensions[2])}</strong></div></div>
            <div className="section-label section-gap">MESH STATISTICS</div>
            <div className="stat-list"><div><span><Triangle size={14}/> Triangles</span><strong>{model.triangles.toLocaleString()}</strong></div><div><span><Scan size={14}/> Surface area</span><strong>{model.area.toFixed(2)} mm²</strong></div><div><span><Box size={14}/> Volume</span><strong>{model.volume.toFixed(2)} mm³</strong></div></div>
            <div className="health-card"><div className="health-icon"><Check size={14}/></div><div><strong>Mesh loaded</strong><span>Ready to inspect in 3D</span></div><span className="health-dot"/></div>
          </> : <>
            <div className="empty-model"><div className="empty-icon"><Box size={20}/></div><strong>No model open</strong><p>Upload an STL file to see its dimensions and mesh details.</p><button onClick={chooseFile}><Upload size={14}/> Open file</button></div>
            <div className="section-label section-gap">SUPPORTED FORMAT</div><div className="format-row"><span className="format-chip">STL</span><div><strong>STereoLithography</strong><span>Binary and ASCII formats</span></div><Check size={15}/></div>
            <div className="privacy-note"><ShieldCheck size={16}/><span>Your files never leave your device. Processing happens locally in your browser.</span></div>
          </>}
        </div>
      </aside>
    </main>
    <div className="mobile-upload" onClick={chooseFile}><Upload size={16}/> Open STL</div>
  </div>
}

export default App
