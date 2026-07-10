import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Brain, Database, Gauge, Play, Sparkles } from 'lucide-react';
import './styles.css';

const API = 'http://127.0.0.1:8000';
const defaultText = 'MshaAllah! Ye product boht hi achha hy, bilkul slow nhi hai? 100% recommended.';

function App() {
  const [active, setActive] = useState('preprocess');
  const [status, setStatus] = useState('checking');
  const [text, setText] = useState(defaultText);
  const [steps, setSteps] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [prediction, setPrediction] = useState(null);

  async function api(path, options = {}) {
    const res = await fetch(API + path, options);
    if (!res.ok) throw new Error('API error: ' + res.status);
    return res.json();
  }

  async function loadStatus() {
    try {
      const data = await api('/status');
      setStatus(data.status === 'running' ? 'Running' : 'Offline');
    } catch {
      setStatus('Backend Offline');
    }
  }

  async function runPreprocess() {
    try {
      const data = await api('/preprocess', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text })
      });
      setSteps(data.steps || []);
    } catch (e) { alert('Backend not running. Start run_project.bat first.'); }
  }

  async function loadMetrics() {
    try {
      const data = await api('/metrics');
      setMetrics(data.metrics || []);
    } catch { alert('Backend not running.'); }
  }

  async function runPrediction() {
    try {
      const data = await api('/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text })
      });
      setPrediction(data);
    } catch { alert('Backend not running.'); }
  }

  useEffect(() => { loadStatus(); runPreprocess(); loadMetrics(); }, []);

  return (
    <main className="app">
      <header className="hero">
        <div className="logo"><Brain size={34}/></div>
        <div>
          <h1>Roman Urdu Sentiment Intelligence Pipeline</h1>
          <p>Deep Learning + NLP Project • Roman Urdu Sentiment Analysis with ML + Neural Network</p>
        </div>
        <div className="status"><span></span>Status: {status}</div>
      </header>

      <nav className="tabs">
        <button onClick={() => setActive('corpus')} className={active==='corpus'?'active':''}><Database size={16}/> Corpus & Vocab</button>
        <button onClick={() => setActive('preprocess')} className={active==='preprocess'?'active':''}><Sparkles size={16}/> Preprocessing Pipeline</button>
        <button onClick={() => {setActive('training'); loadMetrics();}} className={active==='training'?'active':''}><Brain size={16}/> ML + DL Training Center</button>
        <button onClick={() => {setActive('performance'); loadMetrics();}} className={active==='performance'?'active':''}><Gauge size={16}/> Performance Arena</button>
        <button onClick={() => setActive('playground')} className={active==='playground'?'active':''}><Play size={16}/> Sentiment Playground</button>
      </nav>

      {active === 'preprocess' && <section className="grid two">
        <div className="card">
          <h2>Preprocessing Input Tester</h2>
          <p>Type any noisy Roman Urdu review containing spelling variants, punctuation and stopwords.</p>
          <label>Type Roman Urdu Review</label>
          <textarea value={text} onChange={e=>setText(e.target.value)} />
          <button className="primary" onClick={runPreprocess}>Run Preprocessing</button>
          <div className="tip"><b>Educational Tip:</b><br/>Variants like <b>MshaAllah, boht, achha, nhi</b> are standardized into <b>mashallah, bohat, acha, nahi</b>. Then stopwords like <b>ye, hy, hi</b> are removed.</div>
        </div>
        <div className="card">
          <h2>Pipeline Execution Steps</h2>
          <div className="timeline">
            {steps.map((s, i) => <div className="step" key={i}>
              <div className={i===steps.length-1?'num green':'num'}>{s.step}</div>
              <div><h4>{s.title}</h4><p className={i===steps.length-1?'final':''}>"{s.text}"</p></div>
            </div>)}
          </div>
        </div>
      </section>}

      {active === 'corpus' && <section className="card">
        <h2>Corpus & Vocabulary Explorer</h2>
        <p>This project uses Roman Urdu comments/reviews with three sentiment labels: Positive, Negative and Neutral.</p>
        <div className="miniGrid">
          <div><h3>Positive</h3><p>bohat acha, behtareen, recommended, zabardast</p></div>
          <div><h3>Negative</h3><p>kharab, slow, bekar, recommend nahi, waste</p></div>
          <div><h3>Neutral</h3><p>normal, average, receive ho gaya, theek</p></div>
        </div>
      </section>}

      {active === 'training' && <section className="card">
        <h2>ML + Deep Learning Training Center</h2>
        <p>Backend trains classical NLP baselines and a Deep Neural Network/MLP model using TF-IDF unigram + bigram features.</p>
        <button className="primary" onClick={async()=>{await api('/train'); loadMetrics();}}>Retrain Models</button>
        <MetricTable metrics={metrics}/>
      </section>}

      {active === 'performance' && <section className="card">
        <h2>Performance Arena</h2>
        <p>Compare Machine Learning models with the Deep Neural Network model.</p>
        <MetricTable metrics={metrics}/>
      </section>}

      {active === 'playground' && <section className="grid two">
        <div className="card">
          <h2>Sentiment Playground</h2>
          <textarea value={text} onChange={e=>setText(e.target.value)} />
          <button className="primary" onClick={runPrediction}>Predict Sentiment</button>
        </div>
        <div className="card">
          <h2>Prediction Output</h2>
          {prediction ? <>
            <p><b>Clean Text:</b> {prediction.clean_text}</p>
            <p><b>Final Consensus:</b> <span className="badge">{prediction.consensus}</span></p>
            <MetricTable metrics={prediction.predictions.map(p=>({model:p.model, accuracy:p.confidence, f1_score:'-', type:p.label}))}/>
          </> : <p>Run prediction to view model outputs.</p>}
        </div>
      </section>}

      <footer>© 2026 Ahmed Sarfraz • Roman Urdu Deep Learning + NLP University Assignment</footer>
    </main>
  );
}

function MetricTable({metrics}) {
  return <table className="table"><thead><tr><th>Model</th><th>Accuracy / Confidence</th><th>F1 Score</th><th>Type / Label</th></tr></thead><tbody>
    {(metrics||[]).map((m,i)=><tr key={i}><td>{m.model}</td><td>{m.accuracy}%</td><td>{m.f1_score}{m.f1_score==='-'?'':'%'}</td><td>{m.type}</td></tr>)}
  </tbody></table>
}

createRoot(document.getElementById('root')).render(<App />);
