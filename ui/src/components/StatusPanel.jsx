import React, { useEffect, useRef } from 'react'

const STATUS_PILL = {
  running: 'bg-amber-900/60 text-amber-300 border border-amber-700',
  success: 'bg-green-900/60 text-green-300 border border-green-700',
  error:   'bg-red-900/60 text-red-300 border border-red-700',
}

export default function StatusPanel({ logs, nodeStatuses }) {
  const logEndRef = useRef(null)

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const activeStatuses = Object.entries(nodeStatuses).filter(([, s]) => s !== 'idle')

  return (
    <div className="h-44 bg-slate-950 border-t border-slate-700 flex flex-col">
      {/* Tab bar */}
      <div className="flex items-center gap-2 px-4 py-1.5 bg-slate-900 border-b border-slate-700 flex-shrink-0">
        <span className="text-xs font-bold text-slate-400 uppercase tracking-wider mr-2">
          Output
        </span>
        {activeStatuses.slice(0, 10).map(([id, s]) => (
          <span key={id} className={`text-xs px-2 py-0.5 rounded font-mono ${STATUS_PILL[s]}`}>
            {id}
          </span>
        ))}
      </div>

      {/* Scrolling log */}
      <div className="flex-1 overflow-y-auto px-4 py-1.5 font-mono text-xs">
        {logs.length === 0 && (
          <p className="text-slate-600 italic mt-2">
            Run the pipeline to see output here...
          </p>
        )}

        {logs.map((entry, i) => {
          const isError   = entry.text.startsWith('ERROR') || entry.text.startsWith('Error')
          const isCommand = entry.text.startsWith('>')
          const isWarning = entry.text.toLowerCase().startsWith('warning')

          let textColor = 'text-slate-400'
          if (isError)   textColor = 'text-red-400'
          else if (isCommand) textColor = 'text-blue-400'
          else if (isWarning) textColor = 'text-amber-400'

          return (
            <div key={i} className={`leading-5 ${textColor}`}>
              <span className="text-slate-600 mr-2 select-none">[{entry.nodeId}]</span>
              {entry.text}
            </div>
          )
        })}
        <div ref={logEndRef} />
      </div>
    </div>
  )
}
