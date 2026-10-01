import { useEffect, useState } from 'react';
import { getHealth } from './api.js';
import Classify from './features/Classify.jsx';
import Detect from './features/Detect.jsx';
import Search from './features/Search.jsx';
import Chat from './features/Chat.jsx';

const TABS = [
  { id: 'classify', label: 'Phân loại ảnh', model: 'classifier', Component: Classify },
  { id: 'detect', label: 'Phát hiện đối tượng', model: 'detector', Component: Detect },
  { id: 'search', label: 'Tìm kiếm ảnh', model: 'retrieval', Component: Search },
  { id: 'chat', label: 'Chatbot RAG', model: 'llm', Component: Chat },
];

export default function App() {
  const [tab, setTab] = useState('classify');
  const [health, setHealth] = useState(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: 'down', models: {} }));
  }, []);

  const current = TABS.find((t) => t.id === tab);
  const ready = health?.models?.[current.model];

  return (
    <div className="app">
      <header>
        <h1>AI Web Apps</h1>
        <p className="muted">
          Backend: {health ? (health.status === 'ok' ? `đang chạy (${health.device})` : 'không kết nối') : 'đang kiểm tra…'}
        </p>
      </header>
      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? 'active' : ''}
                  onClick={() => setTab(t.id)}>
            {t.label}{health && !health.models?.[t.model] ? ' (tắt)' : ''}
          </button>
        ))}
      </nav>
      <main>
        {health && !ready && <p className="error">Mô hình “{current.model}” chưa được nạp ở backend.</p>}
        <current.Component />
      </main>
    </div>
  );
}
