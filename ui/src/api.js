const BASE = '/api'

export async function fetchNodes() {
  const r = await fetch(`${BASE}/nodes`)
  if (!r.ok) throw new Error('Failed to load node registry')
  const data = await r.json()
  return data.nodes
}

export async function fetchPipelines() {
  const r = await fetch(`${BASE}/pipelines`)
  if (!r.ok) throw new Error('Failed to list pipelines')
  return (await r.json()).pipelines
}

export async function loadPipeline(name) {
  const r = await fetch(`${BASE}/pipelines/${encodeURIComponent(name)}`)
  if (!r.ok) throw new Error(`Pipeline '${name}' not found`)
  return r.json()
}

export async function savePipeline(name, graph) {
  const r = await fetch(`${BASE}/pipelines/${encodeURIComponent(name)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(graph),
  })
  if (!r.ok) throw new Error('Save failed')
  return r.json()
}

/**
 * Opens a WebSocket to /ws/run.
 * @param {(msg: object) => void} onMessage
 * @param {() => void} onClose
 * @returns WebSocket instance
 */
export function openRunSocket(onMessage, onClose) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${location.host}/ws/run`)
  ws.onmessage = (e) => {
    try {
      onMessage(JSON.parse(e.data))
    } catch (_) {}
  }
  ws.onclose = onClose
  ws.onerror = () => onClose()
  return ws
}
