import { useEffect, useMemo, useState } from 'react'

const API = import.meta.env.VITE_API_BASE || ''

const VERDICTS = {
  non_compliant: { label: 'Non-compliant', mark: '✕' },
  needs_review: { label: 'Needs review', mark: '!' },
  compliant: { label: 'Compliant', mark: '✓' },
  no_flag: { label: 'No flag raised', mark: '–' },
  not_addressed: { label: 'Not addressed', mark: '·' },
}
const ORDER = ['non_compliant', 'needs_review', 'compliant', 'no_flag', 'not_addressed']

async function getJSON(path, options) {
  const res = await fetch(API + path, options)
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'The proposal could not be audited. Check that it is at least a sentence long.')
  return body
}

function Passage({ id }) {
  const [p, setP] = useState(null)
  useEffect(() => { getJSON(`/api/passages/${encodeURIComponent(id)}`).then(setP).catch(() => setP(false)) }, [id])
  if (p === null) return <p className="muted">Loading regulation text…</p>
  if (p === false) return <p className="muted">Regulation text unavailable.</p>
  return (
    <div className="passage">
      <div className="passage-head">{p.citation} · {p.section_title}</div>
      {p.context && <p className="muted">{p.context}</p>}
      <p>{p.text}</p>
    </div>
  )
}

function Finding({ f }) {
  const [open, setOpen] = useState(false)
  const v = VERDICTS[f.verdict]
  return (
    <article className={`finding v-${f.verdict}`}>
      <header>
        <span className="chip"><span aria-hidden="true">{v.mark}</span> {v.label}</span>
        <span className="clause-no">Clause {f.clause_no}</span>
      </header>
      <p className="clause">{f.clause}</p>
      {f.citation && (
        <blockquote>
          <cite>{f.citation}</cite>
          “{f.quote}”
        </blockquote>
      )}
      {f.rationale && <p>{f.rationale}</p>}
      {f.recommendation && <p><strong>Recommendation.</strong> {f.recommendation}</p>}
      {f.verifier_note && <p className="note">{f.verifier_note}</p>}
      {f.passage_id && (
        <>
          <button className="link" onClick={() => setOpen(!open)} aria-expanded={open}>
            {open ? 'Hide' : 'Read'} the full passage
          </button>
          {open && <Passage id={f.passage_id} />}
        </>
      )}
    </article>
  )
}

export default function App() {
  const [health, setHealth] = useState(null)
  const [samples, setSamples] = useState([])
  const [text, setText] = useState('')
  const [title, setTitle] = useState('Proposal')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [filter, setFilter] = useState('all')

  useEffect(() => {
    getJSON('/api/health').then(setHealth).catch(() => setError('The audit service is not reachable.'))
    getJSON('/api/samples').then(setSamples).catch(() => {})
  }, [])

  async function audit() {
    setBusy(true); setError(''); setResult(null); setFilter('all')
    try { setResult(await getJSON('/api/audit', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, title }) })) }
    catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }

  function download() {
    const url = URL.createObjectURL(new Blob([result.report_markdown], { type: 'text/markdown' }))
    const a = Object.assign(document.createElement('a'), { href: url, download: 'audit-report.md' })
    a.click(); URL.revokeObjectURL(url)
  }

  const shown = useMemo(() => {
    if (!result) return []
    const list = filter === 'all' ? result.findings : result.findings.filter(f => f.verdict === filter)
    return [...list].sort((a, b) => ORDER.indexOf(a.verdict) - ORDER.indexOf(b.verdict) || a.clause_no - b.clause_no)
  }, [result, filter])

  return (
    <main>
      <header className="top">
        <h1>Compliance Audit Agent</h1>
        <p className="lede">
          Checks an internal proposal clause by clause against FDA drug manufacturing rules
          (21 CFR Parts 11, 210 and 211) and cites the exact paragraph behind each finding.
        </p>
        {health && (
          <p className="status">
            {health.passages} regulation passages indexed · Analyst: {health.analyst}
          </p>
        )}
      </header>

      <div className="layout">
        <section className="panel">
          <h2>Proposal</h2>
          <div className="samples">
            <span className="muted">Try a sample:</span>
            {samples.map(s => (
              <button key={s.id} className="pill" onClick={() => { setText(s.text); setTitle(s.title); setResult(null) }}>{s.title}</button>
            ))}
          </div>
          <label className="sr" htmlFor="proposal">Proposal text</label>
          <textarea id="proposal" value={text} onChange={e => setText(e.target.value)}
            placeholder="Paste a proposal. Put each commitment on its own numbered line or in its own paragraph." />
          <div className="actions">
            <button className="primary" onClick={audit} disabled={busy || text.trim().length < 40}>
              {busy ? 'Auditing…' : 'Run audit'}
            </button>
            <span className="muted">Sample proposals are fictional and written for testing.</span>
          </div>
          {error && <p className="error" role="alert">{error}</p>}
        </section>

        <section className="panel">
          <h2>Findings</h2>
          {!result && !busy && <p className="muted">Findings appear here, most serious first, each with the regulation text it rests on.</p>}
          {busy && <p className="muted">Extracting clauses, retrieving regulations, checking citations…</p>}
          {result && (
            <>
              <div className="summary">
                <button className={`count ${filter === 'all' ? 'on' : ''}`} onClick={() => setFilter('all')}>
                  <b>{result.findings.length}</b> clauses
                </button>
                {ORDER.filter(k => result.counts[k]).map(k => (
                  <button key={k} className={`count v-${k} ${filter === k ? 'on' : ''}`} onClick={() => setFilter(k)}>
                    <b>{result.counts[k]}</b> {VERDICTS[k].label.toLowerCase()}
                  </button>
                ))}
              </div>
              {result.analyst_kind === 'baseline' && (
                <p className="banner">
                  Running without a language model. The keyword baseline can only flag clauses for review;
                  it cannot rule a clause compliant or non-compliant.
                </p>
              )}
              {result.citations_removed > 0 && (
                <p className="banner">
                  {result.citations_removed} citation{result.citations_removed > 1 ? 's were' : ' was'} removed
                  because the quoted text was not found in the regulation.
                </p>
              )}
              <button className="link" onClick={download}>Download the audit report (.md)</button>
              {shown.map(f => <Finding key={f.clause_no} f={f} />)}
            </>
          )}
        </section>
      </div>
      <footer>A screening aid, not legal advice. Regulation text: eCFR, Title 21.</footer>
    </main>
  )
}
