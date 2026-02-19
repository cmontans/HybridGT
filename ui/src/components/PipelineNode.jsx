import React, { memo, useCallback } from 'react'
import { Handle, Position } from '@xyflow/react'

// Port type → color
const PORT_COLORS = {
  geojson:   '#22c55e',
  csv:       '#3b82f6',
  pkl:       '#a855f7',
  png:       '#eab308',
  directory: '#f97316',
  float:     '#06b6d4',
  int:       '#06b6d4',
  boolean:   '#ef4444',
  file:      '#94a3b8',
  string:    '#94a3b8',
}

function portColor(port) {
  if (port.produces) return PORT_COLORS[port.produces] ?? PORT_COLORS.file
  if (port.type === 'directory') return PORT_COLORS.directory
  if (port.type === 'file' && port.accepts?.length === 1)
    return PORT_COLORS[port.accepts[0]] ?? PORT_COLORS.file
  return PORT_COLORS[port.type] ?? PORT_COLORS.file
}

const STATUS_BORDER = {
  idle:    'border-slate-600',
  running: 'border-amber-400',
  success: 'border-green-500',
  error:   'border-red-500',
}

const STATUS_SHADOW = {
  running: '0 0 12px 2px rgba(251,191,36,0.3)',
  success: '0 0 10px 2px rgba(34,197,94,0.2)',
  error:   '0 0 12px 2px rgba(239,68,68,0.3)',
  idle:    'none',
}

const PipelineNode = memo(function PipelineNode({ id, data }) {
  const nodeDef = data.nodeDef
  const status = data.status || 'idle'

  const inputPorts  = nodeDef?.ports?.inputs  ?? []
  const outputPorts = nodeDef?.ports?.outputs ?? []
  const params      = nodeDef?.params         ?? []

  const onParamChange = useCallback(
    (paramId, value) => data.onParamChange?.(id, paramId, value),
    [id, data],
  )

  if (!nodeDef) return null

  return (
    <div
      className={`relative rounded-lg border-2 min-w-[230px] bg-slate-800 ${STATUS_BORDER[status]}`}
      style={{ boxShadow: STATUS_SHADOW[status] }}
    >
      {/* Running pulse indicator */}
      {status === 'running' && (
        <div className="absolute top-2 right-2 w-2 h-2 bg-amber-400 rounded-full animate-pulse z-10" />
      )}

      {/* Header */}
      <div
        className="rounded-t-lg px-3 py-2 flex items-center gap-2"
        style={{ backgroundColor: nodeDef.color + '28' }}
      >
        <div
          className="w-2.5 h-2.5 rounded-full flex-shrink-0"
          style={{ backgroundColor: nodeDef.color }}
        />
        <span className="font-semibold text-white text-sm truncate leading-tight">
          {nodeDef.name}
        </span>
        <span
          className="ml-auto text-xs px-1.5 py-0.5 rounded flex-shrink-0"
          style={{ backgroundColor: nodeDef.color + '44', color: nodeDef.color }}
        >
          {nodeDef.category}
        </span>
      </div>

      {/* Ports area */}
      <div className="py-1.5">
        {/* Input ports */}
        {inputPorts.map((port) => (
          <div key={port.id} className="relative flex items-center" style={{ minHeight: 28 }}>
            <Handle
              type="target"
              position={Position.Left}
              id={port.id}
              style={{
                left: -7,
                top: '50%',
                transform: 'translateY(-50%)',
                width: 14,
                height: 14,
                borderRadius: '50%',
                background: portColor(port),
                border: '2px solid #0f172a',
                zIndex: 10,
              }}
            />
            <span className="text-xs text-slate-400 pl-4 truncate">
              {port.label}
              {port.required && <span className="text-red-400 ml-0.5">*</span>}
            </span>
          </div>
        ))}

        {/* Output ports */}
        {outputPorts.map((port) => (
          <div key={port.id} className="relative flex items-center justify-end" style={{ minHeight: 28 }}>
            <span className="text-xs text-slate-300 pr-4 truncate">{port.label}</span>
            <Handle
              type="source"
              position={Position.Right}
              id={port.id}
              style={{
                right: -7,
                top: '50%',
                transform: 'translateY(-50%)',
                width: 14,
                height: 14,
                borderRadius: '50%',
                background: portColor(port),
                border: '2px solid #0f172a',
                zIndex: 10,
              }}
            />
          </div>
        ))}
      </div>

      {/* Params */}
      {params.length > 0 && (
        <div className="border-t border-slate-700 px-3 py-2 space-y-1.5">
          {params.map((param) => {
            const val = data.params?.[param.id] ?? param.default ?? ''
            return (
              <div key={param.id} className="flex items-center gap-2">
                <label className="text-xs text-slate-400 truncate" style={{ minWidth: 80 }}>
                  {param.label}
                </label>

                {param.type === 'boolean' ? (
                  <input
                    type="checkbox"
                    className="ml-auto accent-violet-500 cursor-pointer"
                    checked={!!val}
                    onChange={(e) => onParamChange(param.id, e.target.checked)}
                  />
                ) : param.type === 'float' || param.type === 'int' ? (
                  <input
                    type="number"
                    step={param.type === 'float' ? 0.01 : 1}
                    className="ml-auto bg-slate-700 text-white text-xs rounded px-2 py-0.5
                               border border-slate-600 focus:border-slate-400 outline-none w-24"
                    value={val}
                    onChange={(e) =>
                      onParamChange(
                        param.id,
                        param.type === 'int' ? parseInt(e.target.value, 10) : parseFloat(e.target.value),
                      )
                    }
                  />
                ) : (
                  <input
                    type="text"
                    className="ml-auto bg-slate-700 text-white text-xs rounded px-2 py-0.5
                               border border-slate-600 focus:border-slate-400 outline-none w-32"
                    placeholder={param.optional ? 'optional' : String(param.default ?? '')}
                    value={val === null ? '' : val}
                    onChange={(e) => onParamChange(param.id, e.target.value || null)}
                  />
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
})

export default PipelineNode
