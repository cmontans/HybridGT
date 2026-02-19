import React, { useState } from 'react'

const CATEGORY_ORDER = [
  'Source',
  'Preprocessing',
  'Prediction',
  'Analysis',
  'Clustering',
  'Generation',
]

const BUILT_IN_NODES = [
  { id: 'file_input', name: 'File Input',      color: '#94a3b8', category: 'Source', description: 'A file path input (GeoJSON, Shapefile, GeoPackage, etc.)' },
  { id: 'dir_input',  name: 'Directory Input', color: '#f97316', category: 'Source', description: 'A directory path input' },
]

export default function NodePalette({ registry }) {
  const [search, setSearch] = useState('')
  const [collapsed, setCollapsed] = useState({})

  const query = search.toLowerCase()

  // Build grouped map
  const grouped = { Source: [...BUILT_IN_NODES] }

  registry
    .filter(
      (n) =>
        !query ||
        n.name.toLowerCase().includes(query) ||
        n.category.toLowerCase().includes(query) ||
        (n.description || '').toLowerCase().includes(query),
    )
    .forEach((node) => {
      if (!grouped[node.category]) grouped[node.category] = []
      grouped[node.category].push(node)
    })

  // Filter built-ins by search too
  if (query) {
    grouped['Source'] = BUILT_IN_NODES.filter(
      (n) =>
        n.name.toLowerCase().includes(query) ||
        n.description.toLowerCase().includes(query),
    )
    if (grouped['Source'].length === 0) delete grouped['Source']
  }

  const categories = [
    ...CATEGORY_ORDER.filter((c) => grouped[c]?.length),
    ...Object.keys(grouped).filter((c) => !CATEGORY_ORDER.includes(c) && grouped[c]?.length),
  ]

  const toggleCategory = (cat) =>
    setCollapsed((prev) => ({ ...prev, [cat]: !prev[cat] }))

  const onDragStart = (e, nodeType) => {
    e.dataTransfer.setData('application/hybridgt-node-type', nodeType)
    e.dataTransfer.effectAllowed = 'move'
  }

  return (
    <div className="flex flex-col w-60 flex-shrink-0 bg-slate-900 border-r border-slate-700 overflow-y-auto">
      {/* Header */}
      <div className="px-3 pt-3 pb-2 border-b border-slate-700">
        <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">
          Node Palette
        </p>
        <input
          type="text"
          placeholder="Search nodes..."
          className="w-full bg-slate-800 text-white text-xs rounded px-2.5 py-1.5
                     border border-slate-700 focus:border-slate-500 outline-none
                     placeholder-slate-500"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      {/* Node list */}
      <div className="flex-1 overflow-y-auto py-1">
        {categories.length === 0 && (
          <p className="text-xs text-slate-500 px-3 py-4 text-center">No nodes found</p>
        )}

        {categories.map((cat) => (
          <div key={cat} className="mb-0.5">
            {/* Category header */}
            <button
              onClick={() => toggleCategory(cat)}
              className="w-full flex items-center justify-between px-3 py-1.5
                         text-xs font-semibold text-slate-500 uppercase tracking-widest
                         hover:text-slate-300 transition-colors"
            >
              <span>{cat}</span>
              <span className="text-slate-600">{collapsed[cat] ? '▶' : '▼'}</span>
            </button>

            {!collapsed[cat] &&
              grouped[cat].map((node) => (
                <div
                  key={node.id}
                  draggable
                  onDragStart={(e) => onDragStart(e, node.id)}
                  title={node.description || node.name}
                  className="mx-2 mb-0.5 px-2.5 py-2 rounded cursor-grab active:cursor-grabbing
                             bg-slate-800 hover:bg-slate-700/80 border border-slate-700
                             hover:border-slate-500 transition-all flex items-center gap-2
                             select-none"
                >
                  <div
                    className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: node.color }}
                  />
                  <span className="text-xs text-slate-200 truncate">{node.name}</span>
                </div>
              ))}
          </div>
        ))}
      </div>

      {/* Footer hint */}
      <div className="px-3 py-2 border-t border-slate-700">
        <p className="text-xs text-slate-600 text-center">Drag nodes onto the canvas</p>
      </div>
    </div>
  )
}
