import React, { useCallback, useMemo } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  MarkerType,
  useReactFlow,
} from '@xyflow/react'
import PipelineNode from './PipelineNode'
import SourceNode from './SourceNode'

// ---------------------------------------------------------------------------
// Port type compatibility
// ---------------------------------------------------------------------------
// Any file:* connects to any file:* (they are all paths at runtime).
// Directories only connect to directories.
// Primitives must match exactly.
function isCompatibleConnection(srcPort, tgtPort) {
  const srcType = srcPort?.type ?? 'file'
  const tgtType = tgtPort?.type ?? 'file'

  if (srcType === 'directory' || tgtType === 'directory') {
    return srcType === 'directory' && tgtType === 'directory'
  }
  // Both are file variants — always compatible
  if (srcType === 'file' || tgtType === 'file') return true
  return srcType === tgtType
}

// ---------------------------------------------------------------------------
// PipelineCanvas
// ---------------------------------------------------------------------------
export default function PipelineCanvas({
  nodes,
  edges,
  onNodesChange,
  onEdgesChange,
  onAddNode,
  onAddEdge,
  registry,
  onParamChange,
}) {
  const { screenToFlowPosition } = useReactFlow()

  // Build a fast lookup: nodeType -> definition
  const registryMap = useMemo(() => {
    const m = {}
    registry.forEach((d) => { m[d.id] = d })
    return m
  }, [registry])

  // Build stable nodeTypes — only depends on registry (NOT on node statuses)
  const nodeTypes = useMemo(() => {
    const types = {
      file_input: (props) => (
        <SourceNode
          {...props}
          data={{
            ...props.data,
            nodeSubtype: 'file',
            onParamChange,
          }}
        />
      ),
      dir_input: (props) => (
        <SourceNode
          {...props}
          data={{
            ...props.data,
            nodeSubtype: 'directory',
            onParamChange,
          }}
        />
      ),
    }

    registry.forEach((def) => {
      types[def.id] = (props) => (
        <PipelineNode
          {...props}
          data={{
            ...props.data,
            nodeDef: def,
            onParamChange,
          }}
        />
      )
    })

    return types
  }, [registry, onParamChange])

  // Connection validation
  const isValidConnection = useCallback(
    (connection) => {
      const srcNode = nodes.find((n) => n.id === connection.source)
      const tgtNode = nodes.find((n) => n.id === connection.target)
      if (!srcNode || !tgtNode) return false

      // Source nodes are always valid sources
      if (srcNode.type === 'file_input' || srcNode.type === 'dir_input') return true

      const srcDef = registryMap[srcNode.type]
      const tgtDef = registryMap[tgtNode.type]
      if (!srcDef || !tgtDef) return false

      const srcPort = (srcDef.ports?.outputs ?? []).find((p) => p.id === connection.sourceHandle)
      const tgtPort = (tgtDef.ports?.inputs ?? []).find((p) => p.id === connection.targetHandle)

      return isCompatibleConnection(srcPort, tgtPort)
    },
    [nodes, registryMap],
  )

  const onConnect = useCallback(
    (params) => {
      if (!isValidConnection(params)) return
      onAddEdge(
        addEdge(
          {
            ...params,
            type: 'smoothstep',
            animated: false,
            style: { stroke: '#475569', strokeWidth: 2 },
            markerEnd: { type: MarkerType.ArrowClosed, color: '#475569' },
          },
          [],
        )[0],
      )
    },
    [isValidConnection, onAddEdge],
  )

  // Drag-from-palette → drop onto canvas
  const onDrop = useCallback(
    (event) => {
      event.preventDefault()
      const nodeType = event.dataTransfer.getData('application/hybridgt-node-type')
      if (!nodeType) return

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      })

      const newNode = {
        id: `${nodeType}_${Date.now()}`,
        type: nodeType,
        position,
        data: { params: {} },
      }

      // Pre-fill param defaults from registry
      const def = registryMap[nodeType]
      if (def?.params) {
        const defaults = {}
        def.params.forEach((p) => {
          if (p.default !== undefined) defaults[p.id] = p.default
        })
        newNode.data.params = defaults
      }

      onAddNode(newNode)
    },
    [screenToFlowPosition, registryMap, onAddNode],
  )

  const onDragOver = useCallback((event) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }, [])

  // MiniMap node color based on status
  const miniMapNodeColor = useCallback((n) => {
    const s = n.data?.status
    if (s === 'running') return '#f59e0b'
    if (s === 'success') return '#22c55e'
    if (s === 'error')   return '#ef4444'
    return '#334155'
  }, [])

  return (
    <div className="w-full h-full" onDrop={onDrop} onDragOver={onDragOver}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        isValidConnection={isValidConnection}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        minZoom={0.15}
        maxZoom={2}
        deleteKeyCode="Delete"
        snapToGrid
        snapGrid={[16, 16]}
        defaultEdgeOptions={{
          type: 'smoothstep',
          style: { stroke: '#475569', strokeWidth: 2 },
          markerEnd: { type: MarkerType.ArrowClosed, color: '#475569' },
        }}
      >
        <Background color="#1e293b" gap={20} size={1} />
        <Controls
          style={{
            background: '#1e293b',
            border: '1px solid #334155',
            borderRadius: 6,
          }}
        />
        <MiniMap
          nodeColor={miniMapNodeColor}
          maskColor="rgba(15,23,42,0.7)"
          style={{
            background: '#1e293b',
            border: '1px solid #334155',
            borderRadius: 6,
          }}
        />
      </ReactFlow>
    </div>
  )
}
