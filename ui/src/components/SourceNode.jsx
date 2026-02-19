import React, { memo, useCallback } from 'react'
import { Handle, Position } from '@xyflow/react'

const COLORS = {
  file: '#94a3b8',
  directory: '#f97316',
}

const SourceNode = memo(function SourceNode({ id, data }) {
  const isDir = data.nodeSubtype === 'directory'
  const color = isDir ? COLORS.directory : COLORS.file
  const label = isDir ? 'Directory Input' : 'File Input'
  const placeholder = isDir ? '/path/to/directory' : '/path/to/file.geojson'

  const onChange = useCallback(
    (e) => data.onParamChange?.(id, 'path', e.target.value),
    [id, data],
  )

  return (
    <div
      className="relative rounded-lg border-2 min-w-[220px]"
      style={{ backgroundColor: '#1e293b', borderColor: color + '88' }}
    >
      {/* Header */}
      <div
        className="rounded-t-lg px-3 py-2 flex items-center gap-2"
        style={{ backgroundColor: color + '22' }}
      >
        <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
        <span className="font-semibold text-white text-sm">{label}</span>
      </div>

      {/* Path input */}
      <div className="px-3 py-2.5">
        <input
          type="text"
          className="w-full bg-slate-700 text-white text-xs rounded px-2 py-1.5
                     border border-slate-600 focus:border-slate-400 outline-none
                     placeholder-slate-500"
          placeholder={placeholder}
          value={data.params?.path ?? ''}
          onChange={onChange}
        />
      </div>

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="value"
        style={{
          right: -7,
          top: '50%',
          width: 14,
          height: 14,
          borderRadius: '50%',
          background: color,
          border: '2px solid #0f172a',
        }}
      />
    </div>
  )
})

export default SourceNode
