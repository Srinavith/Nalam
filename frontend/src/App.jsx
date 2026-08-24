import { useEffect, useRef, useState } from 'react'

const FIELDS = [
  ['patient_name', 'Patient name'],
  ['age', 'Age'],
  ['sex', 'Sex'],
  ['patient_id', 'UHID'],
  ['visit_date', 'Date'],
  ['doctor', 'Doctor'],
  ['diagnosis', 'Diagnosis'],
  ['advice', 'Advice'],
]
const VITALS = ['bp', 'pulse', 'temp', 'spo2', 'weight']

async function api(path, options) {
  const res = await fetch('/api' + path, options)
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText)
  return res.status === 204 ? null : res.json()
}


function ScribblePad() {
  const canvasRef = useRef(null)
  const drawing = useRef(false)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)

  function clear() {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    ctx.fillStyle = '#fff'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    setResult(null)
  }

  useEffect(clear, [])

  function pos(e) {
    const r = canvasRef.current.getBoundingClientRect()
    return [e.clientX - r.left, e.clientY - r.top]
  }

  function start(e) {
    e.preventDefault()
    drawing.current = true
    const ctx = canvasRef.current.getContext('2d')
    ctx.lineWidth = 3
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.strokeStyle = '#000'
    ctx.beginPath()
    ctx.moveTo(...pos(e))
    canvasRef.current.setPointerCapture(e.pointerId)
  }

  function move(e) {
    if (!drawing.current) return
    e.preventDefault()
    const ctx = canvasRef.current.getContext('2d')
    ctx.lineTo(...pos(e))
    ctx.stroke()
  }

  const stop = () => { drawing.current = false }

  async function recognize() {
    setBusy(true)
    setResult(null)
    try {
      const res = await fetch('/api/recognize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: canvasRef.current.toDataURL('image/png') }),
      })
      const data = await res.json()
      setResult(res.ok ? data : { error: data.error || res.statusText })
    } catch (err) {
      setResult({ error: err.message })
    }
    setBusy(false)
  }

  return (
    <div>
      <h2>Scribble pad</h2>
      <p>Write one word or line, then press Read.</p>
      <canvas
        ref={canvasRef}
        width={620}
        height={160}
        className="pad"
        onPointerDown={start}
        onPointerMove={move}
        onPointerUp={stop}
        onPointerLeave={stop}
      />
      <p>
        <button onClick={recognize} disabled={busy}>{busy ? 'Reading...' : 'Read'}</button>{' '}
        <button onClick={clear}>Clear</button>
      </p>
      {result && (
        <p>
          {result.error && <span>Error: {result.error}</span>}
          {!result.error && !result.text && <span>{result.note || 'Nothing recognised.'}</span>}
          {result.text && (
            <>
              Read: <b>{result.text}</b>
              {result.changed && <> &rarr; nearest formulary entry: <b>{result.corrected}</b></>}
            </>
          )}
        </p>
      )}
    </div>
  )
}

export default function App() {
  const [reports, setReports] = useState([])
  const [draft, setDraft] = useState(null)
  const [status, setStatus] = useState('')
  const [stats, setStats] = useState(null)

  const refresh = () => api('/reports').then(setReports).catch((e) => setStatus(e.message))
  useEffect(() => { refresh(); api('/stats').then(setStats).catch(() => {}) }, [])

  async function upload(e) {
    const file = e.target.files[0]
    if (!file) return
    setStatus('Reading ' + file.name + '...')
    const body = new FormData()
    body.append('file', file)
    try {
      setDraft(await api('/reports', { method: 'POST', body }))
      setStatus('Done.')
      refresh()
    } catch (err) { setStatus('Error: ' + err.message) }
    e.target.value = ''
  }

  async function save(reviewed) {
    setStatus('Saving...')
    try {
      setDraft(await api('/reports/' + draft.id, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...draft, reviewed }),
      }))
      setStatus('Saved.')
      refresh()
    } catch (err) { setStatus('Error: ' + err.message) }
  }

  async function remove(id) {
    await api('/reports/' + id, { method: 'DELETE' }).catch((e) => setStatus(e.message))
    if (draft && draft.id === id) setDraft(null)
    refresh()
  }

  const set = (k, v) => setDraft({ ...draft, [k]: v })
  const setVital = (k, v) => setDraft({ ...draft, vitals: { ...draft.vitals, [k]: v } })
  const setMed = (i, k, v) =>
    setDraft({ ...draft, medications: draft.medications.map((m, j) => (j === i ? { ...m, [k]: v } : m)) })

  return (
    <div>
      <h1>NALAM</h1>
      <p>OCR-based medical record parsing</p>

      <p>
        <input type="file" accept="image/*" onChange={upload} />
        {status && <span> {status}</span>}
      </p>

      {stats && (
        <table className="stats">
          <thead>
            <tr><th></th><th>Printed reports</th><th>Handwritten (real prescriptions)</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>Field accuracy</td>
              <td>{Math.round(stats.printed.field_accuracy * 100)}%</td>
              <td>{Math.round(stats.handwritten.exact_match * 100)}% exact match</td>
            </tr>
            <tr>
              <td>Character error rate</td>
              <td>&mdash;</td>
              <td>{Math.round(stats.handwritten.cer * 100)}%
                {' '}(vs {Math.round(stats.handwritten.baseline_cer * 100)}% for plain OCR)</td>
            </tr>
            <tr>
              <td>Measured on</td>
              <td>{stats.printed.sample}</td>
              <td>{stats.handwritten.sample}</td>
            </tr>
          </tbody>
        </table>
      )}

      <ScribblePad />

      <h2>Reports ({reports.length})</h2>
      <table>
        <thead>
          <tr>
            <th>Patient</th><th>Date</th><th>Confidence</th><th>Status</th><th></th>
          </tr>
        </thead>
        <tbody>
          {reports.map((r) => (
            <tr key={r.id}>
              <td><a href="#review" onClick={() => api('/reports/' + r.id).then(setDraft)}>
                {r.patient_name || r.filename}</a></td>
              <td>{r.visit_date || r.created_at.slice(0, 10)}</td>
              <td>{Math.round((r.confidence || 0) * 100)}%</td>
              <td>{r.reviewed ? 'reviewed' : 'pending'}</td>
              <td><button onClick={() => remove(r.id)}>delete</button></td>
            </tr>
          ))}
        </tbody>
      </table>
      {reports.length === 0 && <p>None yet.</p>}

      <p>
        <a href="/api/export.csv">Export CSV</a>{' | '}
        <a href="/api/export.csv?reviewed=1">Export reviewed only</a>
      </p>

      {draft && (
        <div id="review">
          <hr />
          <h2>{draft.filename}</h2>

          <table className="review">
            <tbody>
              <tr>
                <td className="scan">
                  <img src={'/api/reports/' + draft.id + '/image'} alt="uploaded report" />
                </td>
                <td>
                  {FIELDS.map(([key, label]) => (
                    <p key={key}>
                      <label>{label}<br />
                        <input value={draft[key] || ''} onChange={(e) => set(key, e.target.value)} />
                      </label>
                    </p>
                  ))}

                  <p>Vitals<br />
                    {VITALS.map((key) => (
                      <label key={key}>{key}{' '}
                        <input size="7" value={(draft.vitals || {})[key] || ''}
                          onChange={(e) => setVital(key, e.target.value)} />{' '}
                      </label>
                    ))}
                  </p>

                  <p>Medications</p>
                  {(draft.medications || []).map((m, i) => (
                    <p key={i}>
                      {['name', 'dose', 'frequency', 'duration'].map((k) => (
                        <input key={k} placeholder={k} value={m[k] || ''} size="12"
                          onChange={(e) => setMed(i, k, e.target.value)} />
                      ))}
                    </p>
                  ))}
                  <p>
                    <button onClick={() => set('medications',
                      [...(draft.medications || []), { name: '', dose: '', frequency: '', duration: '' }])}>
                      add row
                    </button>
                  </p>

                  <p>
                    <button onClick={() => save(true)}>Save and mark reviewed</button>{' '}
                    <button onClick={() => save(false)}>Save draft</button>
                  </p>
                </td>
              </tr>
            </tbody>
          </table>

          <details>
            <summary>Raw OCR text</summary>
            <pre>{draft.raw_text}</pre>
          </details>
        </div>
      )}
    </div>
  )
}
