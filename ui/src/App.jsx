import React, { useState, useEffect, useRef, useCallback } from 'react'
import {
  ReactFlowProvider,
  applyNodeChanges,
  applyEdgeChanges,
} from '@xyflow/react'
import PipelineCanvas from './components/PipelineCanvas'
import NodePalette from './components/NodePalette'
import StatusPanel from './components/StatusPanel'
import { fetchNodes, fetchPipelines, loadPipeline, savePipeline, openRunSocket } from './api'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function buildGraphPayload(nodes, edges) {
  return {
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.type,
      position: n.position,
      params: n.data?.params ?? {},
    })),
    edges: edges.map((e) => ({
      source: e.source,
      sourceHandle: e.sourceHandle,
      target: e.target,
      targetHandle: e.targetHandle,
    })),
  }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
export default function App() {
  // Registry from backend
  const [registry, setRegistry] = useState([])
  const [savedPipelines, setSavedPipelines] = useState([])

  // React Flow state
  const [nodes, setNodes] = useState([])
  const [edges, setEdges] = useState([])

  // Execution state
  const [nodeStatuses, setNodeStatuses] = useState({})
  const [logs, setLogs] = useState([])
  const [isRunning, setIsRunning] = useState(false)

  // UI state
  const [pipelineName, setPipelineName] = useState('default')
  const [statusMsg, setStatusMsg] = useState('')

  const wsRef = useRef(null)

  // -------------------------------------------------------------------------
  // On mount: load registry and default pipeline
  // -------------------------------------------------------------------------
  useEffect(() => {
    fetchNodes()
      .then(setRegistry)
      .catch((e) => setStatusMsg(`Registry error: ${e.message}`))

    fetchPipelines()
      .then(setSavedPipelines)
      .catch(() => {})

    loadPipeline('default')
      .then((graph) => {
        setNodes(graph.nodes.map(hydrateNode))
        setEdges(graph.edges)
      })
      .catch(() => {
        // No default pipeline yet — start blank
      })
  }, [])

  // -------------------------------------------------------------------------
  // Stable onParamChange callback — updates node.data.params
  // -------------------------------------------------------------------------
  const onParamChange = useCallback((nodeId, paramId, value) => {
    setNodes((prev) =>
      prev.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, params: { ...(n.data.params ?? {}), [paramId]: value } } }
          : n,
      ),
    )
  }, [])

  // -------------------------------------------------------------------------
  // Propagate nodeStatuses into node.data.status (keeps nodeTypes stable)
  // -------------------------------------------------------------------------
  useEffect(() => {
    setNodes((prev) =>
      prev.map((n) => {
        const s = nodeStatuses[n.id] ?? 'idle'
        if (n.data?.status === s) return n
        return { ...n, data: { ...n.data, status: s } }
      }),
    )
  }, [nodeStatuses])

  // -------------------------------------------------------------------------
  // React Flow change handlers
  // -------------------------------------------------------------------------
  const onNodesChange = useCallback(
    (changes) => {
      if (Array.isArray(changes) && changes.length > 0 && typeof changes[0] === 'object' && changes[0].type) {
        // React Flow internal change objects
        setNodes((ns) => applyNodeChanges(changes, ns))
      } else if (typeof changes === 'function') {
        setNodes(changes)
      } else {
        // Direct array set from drop / load
        setNodes(changes)
      }
    },
    [],
  )

  const onEdgesChange = useCallback(
    (changes) => {
      if (Array.isArray(changes) && changes.length > 0 && typeof changes[0] === 'object' && changes[0].type) {
        setEdges((es) => applyEdgeChanges(changes, es))
      } else if (typeof changes === 'function') {
        setEdges(changes)
      } else {
        setEdges(changes)
      }
    },
    [],
  )

  // -------------------------------------------------------------------------
  // Run pipeline
  // -------------------------------------------------------------------------
  const handleRun = useCallback(() => {
    if (isRunning) return

    setIsRunning(true)
    setLogs([])

    // Reset all statuses
    const resetStatuses = {}
    nodes.forEach((n) => { resetStatuses[n.id] = 'idle' })
    setNodeStatuses(resetStatuses)
    setStatusMsg('Connecting...')

    const payload = buildGraphPayload(nodes, edges)

    const ws = openRunSocket(
      (msg) => {
        switch (msg.type) {
          case 'pipeline_start':
            setStatusMsg(`Run ${msg.run_id} started`)
            break

          case 'node_start':
            setNodeStatuses((prev) => ({ ...prev, [msg.node_id]: 'running' }))
            setLogs((prev) => [
              ...prev,
              { nodeId: msg.node_id, text: `> ${msg.command}` },
            ])
            break

          case 'log':
            setLogs((prev) => [...prev, { nodeId: msg.node_id, text: msg.text }])
            break

          case 'node_complete':
            setNodeStatuses((prev) => ({ ...prev, [msg.node_id]: 'success' }))
            break

          case 'node_error':
            setNodeStatuses((prev) => ({ ...prev, [msg.node_id]: 'error' }))
            setLogs((prev) => [
              ...prev,
              { nodeId: msg.node_id, text: `ERROR: ${msg.message}` },
            ])
            setIsRunning(false)
            setStatusMsg(`Failed at node ${msg.node_id}`)
            break

          case 'pipeline_complete':
            setIsRunning(false)
            setStatusMsg(`Pipeline complete ✓`)
            break

          case 'pipeline_error':
            setIsRunning(false)
            setStatusMsg(`Pipeline error: ${msg.message}`)
            setLogs((prev) => [
              ...prev,
              { nodeId: 'pipeline', text: `ERROR: ${msg.message}` },
            ])
            break

          default:
            break
        }
      },
      () => {
        setIsRunning(false)
      },
    )

    ws.onopen = () => ws.send(JSON.stringify(payload))
    wsRef.current = ws
  }, [nodes, edges, isRunning])

  // -------------------------------------------------------------------------
  // Save / load
  // -------------------------------------------------------------------------
  const handleSave = useCallback(async () => {
    try {
      const graph = { nodes, edges }
      await savePipeline(pipelineName, graph)
      setStatusMsg(`Saved "${pipelineName}"`)
      const names = await fetchPipelines()
      setSavedPipelines(names)
    } catch (e) {
      setStatusMsg(`Save failed: ${e.message}`)
    }
  }, [nodes, edges, pipelineName])

  const handleLoad = useCallback(
    async (name) => {
      try {
        const graph = await loadPipeline(name)
        setNodes(graph.nodes.map(hydrateNode))
        setEdges(graph.edges)
        setPipelineName(name)
        setNodeStatuses({})
        setLogs([])
        setStatusMsg(`Loaded "${name}"`)
      } catch (e) {
        setStatusMsg(`Load failed: ${e.message}`)
      }
    },
    [],
  )

  const handleNew = useCallback(() => {
    setNodes([])
    setEdges([])
    setNodeStatuses({})
    setLogs([])
    setPipelineName('untitled')
    setStatusMsg('New pipeline')
  }, [])

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <ReactFlowProvider>
      <div className="flex h-screen w-screen overflow-hidden bg-slate-900">

        {/* Left Sidebar: Node Palette */}
        <NodePalette registry={registry} />

        {/* Main area */}
        <div className="flex flex-col flex-1 min-w-0">

          {/* Toolbar */}
          <div className="flex items-center gap-2 px-4 py-2 bg-slate-800 border-b border-slate-700 flex-shrink-0">
            {/* Logo / title */}
            <span className="font-bold text-white text-sm tracking-wide mr-2">
              HybridGT
              <span className="text-slate-500 font-normal ml-1.5 text-xs">Pipeline Editor</span>
            </span>

            {/* New */}
            <button
              onClick={handleNew}
              className="bg-slate-700 hover:bg-slate-600 text-white px-2.5 py-1 rounded text-xs"
            >
              New
            </button>

            {/* Load dropdown */}
            {savedPipelines.length > 0 && (
              <select
                className="bg-slate-700 text-white text-xs rounded px-2 py-1 border border-slate-600"
                defaultValue=""
                onChange={(e) => { if (e.target.value) handleLoad(e.target.value) }}
              >
                <option value="">Load...</option>
                {savedPipelines.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            )}

            {/* Pipeline name */}
            <input
              className="bg-slate-700 text-white rounded px-2.5 py-1 text-xs w-36
                         border border-slate-600 focus:border-slate-400 outline-none"
              value={pipelineName}
              onChange={(e) => setPipelineName(e.target.value)}
              placeholder="Pipeline name"
            />

            {/* Save */}
            <button
              onClick={handleSave}
              className="bg-slate-600 hover:bg-slate-500 text-white px-2.5 py-1 rounded text-xs"
            >
              Save
            </button>

            {/* Status message */}
            {statusMsg && (
              <span className="text-xs text-slate-400 ml-1 truncate max-w-xs">{statusMsg}</span>
            )}

            {/* Run button — far right */}
            <button
              onClick={handleRun}
              disabled={isRunning}
              className="ml-auto bg-green-600 hover:bg-green-500 disabled:bg-slate-600
                         disabled:cursor-not-allowed text-white px-4 py-1.5 rounded text-sm
                         font-bold transition-colors flex items-center gap-2"
            >
              {isRunning && (
                <span className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
              )}
              {isRunning ? 'Running…' : '▶  Run'}
            </button>
          </div>

          {/* Canvas */}
          <div className="flex-1 min-h-0">
            <PipelineCanvas
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              registry={registry}
              onParamChange={onParamChange}
            />
          </div>

          {/* Status / log panel */}
          <StatusPanel logs={logs} nodeStatuses={nodeStatuses} />
        </div>
      </div>
    </ReactFlowProvider>
  )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
/** Ensure a node loaded from JSON has a proper data object. */
function hydrateNode(n) {
  return {
    ...n,
    data: {
      params: {},
      status: 'idle',
      ...n.data,
    },
  }
}
