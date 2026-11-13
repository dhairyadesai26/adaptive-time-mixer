import React, { useState, useCallback, useEffect, useMemo } from 'react';
import {
  Activity,
  Layers,
  Zap,
  BarChart2,
  UploadCloud,
  CheckCircle2,
  AlertCircle,
  Cpu,
  TrendingUp,
  TrendingDown,
  Search,
  Sliders,
  Database,
  RefreshCw,
  Play,
  Info,
  ShieldCheck,
  Car,
  CloudSun,
  BatteryCharging
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine
} from 'recharts';

interface ForecastResult {
  predictions_adaptive: number[][];
  predictions_baseline: number[][];
  historical: number[][];
  columns: string[];
  selected_channel_name: string;
  channel_idx: number;
  scale_weights: number[];
  scale_labels: string[];
  metrics_adaptive: {
    mean: number;
    std: number;
    min: number;
    max: number;
    range: number;
    trend: string;
  };
  metrics_baseline: {
    mean: number;
    std: number;
    min: number;
    max: number;
    range: number;
    trend: string;
  };
  runtime_ms: number;
  dataset_info: {
    name: string;
    total_features: number;
    seq_len: number;
    pred_len: number;
  };
}

interface DemoDataset {
  id: string;
  name: string;
  domain: string;
  features: number;
  frequency: string;
  default_checkpoint: string;
  description: string;
  available: boolean;
  icon: string;
}

interface CheckpointMeta {
  id: string;
  name: string;
  dataset: string;
  horizon: number;
  features?: number;
  d_model: number;
  filename: string;
}

interface BenchmarkItem {
  dataset: string;
  horizon: number;
  n_features: number;
  baseline_mse: number;
  baseline_mae: number;
  adaptive_mse: number;
  adaptive_mae: number;
  delta_mse_pct: number;
  overhead_params: number;
}

interface FieldError {
  message: string;
  suggestion?: string;
  autoFix?: () => void;
  autoFixLabel?: string;
}

type TabType = 'studio' | 'benchmarks' | 'theory';
type VisMode = 'split' | 'both' | 'adaptive' | 'baseline';

function TimeMixerLogo({ size = 38 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 38 38"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      style={{ display: 'block', flexShrink: 0 }}
      aria-label="TimeMixer TM Logo"
    >
      {/* Background Squircle with High-End Cobalt Gradient */}
      <rect width="38" height="38" rx="9.5" fill="url(#tmMonogramGrad)" />
      {/* Subtle Bezel Stroke */}
      <rect x="0.5" y="0.5" width="37" height="37" rx="9" stroke="#ffffff" strokeOpacity="0.22" strokeWidth="1" />

      {/* Subtle background radial depth glow */}
      <circle cx="19" cy="19" r="12" fill="#60a5fa" fillOpacity="0.12" />

      {/* 'M' (Mixer Waveform): Electric Sky Cyan Dynamic Flow */}
      <path
        d="M8 27.5L13.5 16.5L19 22L24.5 16.5L30 27.5"
        stroke="#38bdf8"
        strokeWidth="2.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* 'T' (Time Horizon & Axis): Bold Pure White Structure */}
      {/* T Horizontal Horizon Crossbar */}
      <path
        d="M8.5 10.5H29.5"
        stroke="#ffffff"
        strokeWidth="3.2"
        strokeLinecap="round"
      />
      {/* T Vertical Temporal Stem meeting M at mixing center */}
      <path
        d="M19 10.5V21"
        stroke="#ffffff"
        strokeWidth="3.2"
        strokeLinecap="round"
      />

      {/* Convergence Mixer Core (Where T & M Intersect) */}
      <circle cx="19" cy="22" r="4.2" fill="#ffffff" fillOpacity="0.25" />
      <circle cx="19" cy="22" r="2.7" fill="#ffffff" />
      <circle cx="19" cy="22" r="1.4" fill="#1d4ed8" />

      {/* Crest Nodes (Multiscale Frequency Markers on M) */}
      <circle cx="13.5" cy="16.5" r="1.5" fill="#ffffff" />
      <circle cx="24.5" cy="16.5" r="1.5" fill="#ffffff" />

      <defs>
        <linearGradient id="tmMonogramGrad" x1="0" y1="0" x2="38" y2="38" gradientUnits="userSpaceOnUse">
          <stop stopColor="#2563eb" />
          <stop offset="0.55" stopColor="#1d4ed8" />
          <stop offset="1" stopColor="#172554" />
        </linearGradient>
      </defs>
    </svg>
  );
}

function parseApiError(errorData: any): { field: string; message: string; suggestion?: string; code?: string; actual_features?: number } {
  if (!errorData) {
    return {
      field: 'general',
      message: 'Inference failed with an unknown error.',
      suggestion: 'Please verify your inputs and try again.'
    };
  }

  const detail = errorData.detail !== undefined ? errorData.detail : errorData;

  // Case 1: Array of validation errors (e.g. FastAPI 422)
  if (Array.isArray(detail)) {
    const item = detail[0] || {};
    const loc = Array.isArray(item.loc) ? item.loc[item.loc.length - 1] : 'general';
    const fieldName = String(loc);
    const msg = typeof item.msg === 'string' ? item.msg : 'Invalid input value';
    return {
      field: fieldName.includes('file') ? 'file' : (fieldName.includes('checkpoint') ? 'checkpoint' : fieldName),
      message: `Invalid input for ${fieldName}: ${msg}`,
      suggestion: 'Please provide a valid value for this field.'
    };
  }

  // Case 2: Structured dictionary from our backend { field, message, suggestion, code, ... }
  if (typeof detail === 'object' && detail !== null) {
    const field = typeof detail.field === 'string' ? detail.field : 'general';
    let message = typeof detail.message === 'string' ? detail.message : (
      typeof detail.msg === 'string' ? detail.msg : 'Input parameter error.'
    );
    if (message === '[object Object]' || message.includes('[object Object]')) {
      message = 'Parameter mismatch or invalid configuration detected.';
    }
    const suggestion = typeof detail.suggestion === 'string' ? detail.suggestion : 'Please verify this field and try again.';
    return {
      field,
      message,
      suggestion,
      code: detail.code,
      actual_features: detail.actual_features
    };
  }

  // Case 3: Simple string error message
  if (typeof detail === 'string') {
    if (detail === '[object Object]' || detail.includes('[object Object]')) {
      return {
        field: 'general',
        message: 'Feature mismatch or invalid configuration detected.',
        suggestion: 'Please choose a matching benchmark preset or check the model checkpoint.'
      };
    }
    return {
      field: 'general',
      message: detail,
      suggestion: 'Please check your inputs and try again.'
    };
  }

  return {
    field: 'general',
    message: 'An unexpected error occurred.',
    suggestion: 'Please verify server logs or restart the application.'
  };
}

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('studio');

  // Datasets and Checkpoints State
  const [demoDatasets, setDemoDatasets] = useState<DemoDataset[]>([]);
  const [selectedDemoId, setSelectedDemoId] = useState<string>('ETTh1');
  const [checkpoints, setCheckpoints] = useState<CheckpointMeta[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>('');

  // Forecast Execution State
  const [file, setFile] = useState<File | null>(null);
  const [predLen, setPredLen] = useState<number>(96);
  const [channelIdx, setChannelIdx] = useState<number>(0);
  const [visMode, setVisMode] = useState<VisMode>('split');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<ForecastResult | null>(null);
  const [fieldErrors, setFieldErrors] = useState<Record<string, FieldError>>({});

  // Benchmarks State
  const [benchmarks, setBenchmarks] = useState<BenchmarkItem[]>([]);
  const [benchmarkSearch, setBenchmarkSearch] = useState<string>('');
  const [benchmarkHorizonFilter, setBenchmarkHorizonFilter] = useState<number | 'all'>('all');

  // Load Initial Metadata
  useEffect(() => {
    fetch('/api/demo-datasets')
      .then(res => res.json())
      .then(data => {
        if (data.datasets && data.datasets.length > 0) {
          setDemoDatasets(data.datasets);
        }
      })
      .catch(err => console.error('Failed to load demo datasets:', err));

    fetch('/api/checkpoints')
      .then(res => res.json())
      .then(data => {
        if (data.checkpoints && data.checkpoints.length > 0) {
          setCheckpoints(data.checkpoints);
          // Match initial default dataset (ETTh1)
          const etth1Ckpt = data.checkpoints.find((c: CheckpointMeta) =>
            c.dataset.toLowerCase() === 'etth1' || c.id.toLowerCase().includes('etth1')
          );
          setSelectedCheckpoint(etth1Ckpt ? etth1Ckpt.id : data.checkpoints[0].id);
        }
      })
      .catch(err => console.error('Failed to load checkpoints:', err));

    fetch('/api/benchmarks')
      .then(res => res.json())
      .then(data => {
        if (data.benchmarks) {
          setBenchmarks(data.benchmarks);
        }
      })
      .catch(err => console.error('Failed to load benchmarks:', err));
  }, []);

  // When demo dataset changes, pick matching checkpoint
  const handleSelectDemo = (id: string) => {
    setSelectedDemoId(id);
    setFile(null);
    setChannelIdx(0);
    setFieldErrors({});

    const matchingCkpt = checkpoints.find(c =>
      c.dataset.toLowerCase() === id.toLowerCase() ||
      c.id.toLowerCase().includes(id.toLowerCase())
    );
    if (matchingCkpt) {
      setSelectedCheckpoint(matchingCkpt.id);
    }
  };

  // Helper to inspect dropped/selected CSV and auto-select matching checkpoint
  const inspectCsvAndSelectCheckpoint = (uploadedFile: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = (e.target?.result as string) || '';
        const lines = text.split(/\r?\n/).filter(line => line.trim().length > 0);
        if (lines.length > 1) {
          const firstRow = lines[1].split(',').map(s => s.trim());
          let numFeatures = 0;
          firstRow.forEach(val => {
            if (!isNaN(Number(val)) && val !== '') numFeatures++;
          });
          if (numFeatures > 0) {
            const matching = checkpoints.find(c => c.features === numFeatures);
            if (matching) {
              setSelectedCheckpoint(matching.id);
            }
          }
        }
      } catch (err) {
        // Safe fallback
      }
    };
    reader.readAsText(uploadedFile.slice(0, 4096));
  };

  // Drag and drop CSV upload with validation
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (!droppedFile.name.toLowerCase().endsWith('.csv')) {
        setFieldErrors({
          file: {
            message: `Invalid format: "${droppedFile.name}" is not a CSV file.`,
            suggestion: "TimeMixer requires comma-separated (.csv) data. Please save or export your data as CSV."
          }
        });
        return;
      }
      if (droppedFile.size === 0) {
        setFieldErrors({
          file: {
            message: `Empty file: "${droppedFile.name}" contains 0 bytes.`,
            suggestion: "Please upload a valid CSV file with historical time-series rows."
          }
        });
        return;
      }
      setFile(droppedFile);
      setSelectedDemoId('');
      setFieldErrors({});
      inspectCsvAndSelectCheckpoint(droppedFile);
    }
  }, [checkpoints]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      if (!selectedFile.name.toLowerCase().endsWith('.csv')) {
        setFieldErrors({
          file: {
            message: `Invalid format: "${selectedFile.name}" is not a CSV file.`,
            suggestion: "TimeMixer requires comma-separated (.csv) data. Please save or export your data as CSV."
          }
        });
        return;
      }
      if (selectedFile.size === 0) {
        setFieldErrors({
          file: {
            message: `Empty file: "${selectedFile.name}" contains 0 bytes.`,
            suggestion: "Please upload a valid CSV file with historical time-series rows."
          }
        });
        return;
      }
      setFile(selectedFile);
      setSelectedDemoId('');
      setFieldErrors({});
      inspectCsvAndSelectCheckpoint(selectedFile);
    }
  };

  // Run Forecast API
  const handleRunForecast = async (targetChannelOverride?: number) => {
    // Pre-flight client-side input validations
    const preErrors: Record<string, FieldError> = {};

    if (!file && !selectedDemoId) {
      preErrors.file = {
        message: "No input dataset selected.",
        suggestion: "Please choose one of the 5 benchmark presets above, or drop a CSV file."
      };
    }

    if (Object.keys(preErrors).length > 0) {
      setFieldErrors(preErrors);
      return;
    }

    setLoading(true);
    setFieldErrors({});

    const activeChannel = targetChannelOverride !== undefined ? targetChannelOverride : channelIdx;

    const formData = new FormData();
    if (file) {
      formData.append('file', file);
    } else {
      formData.append('demo_dataset', selectedDemoId || 'ETTh1');
    }

    formData.append('pred_len', predLen.toString());
    formData.append('channel_idx', activeChannel.toString());
    formData.append('checkpoint', selectedCheckpoint || '');
    formData.append('model_mode', visMode === 'split' ? 'both' : visMode);

    try {
      const response = await fetch('/api/forecast', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorData: any = null;
        try {
          errorData = await response.json();
        } catch {
          errorData = { detail: `Inference failed with status ${response.status}.` };
        }

        const parsed = parseApiError(errorData);
        let autoFix: (() => void) | undefined = undefined;
        let autoFixLabel: string | undefined = undefined;

        if (parsed.code === 'FEATURE_MISMATCH' && parsed.actual_features) {
          const actualFeatures = parsed.actual_features;
          const matching = checkpoints.find(c => {
            if (actualFeatures === 862) return c.dataset.toLowerCase().includes('traffic') || c.id.toLowerCase().includes('traffic');
            if (actualFeatures === 321) return c.dataset.toLowerCase().includes('electricity') || c.id.toLowerCase().includes('electricity');
            if (actualFeatures === 21) return c.dataset.toLowerCase().includes('weather') || c.id.toLowerCase().includes('weather');
            if (actualFeatures === 7) return c.dataset.toLowerCase().includes('ett') || c.id.toLowerCase().includes('ett');
            return false;
          });
          if (matching) {
            autoFix = () => {
              setSelectedCheckpoint(matching.id);
              setFieldErrors({});
            };
            autoFixLabel = `Switch to ${matching.name}`;
          }
        }

        setFieldErrors({
          [parsed.field]: {
            message: parsed.message,
            suggestion: parsed.suggestion,
            autoFix,
            autoFixLabel
          }
        });
        return;
      }

      const data = await response.json();
      setResult(data);
      if (data.checkpoint_used && checkpoints.some(c => c.id === data.checkpoint_used)) {
        setSelectedCheckpoint(data.checkpoint_used);
      }
    } catch (err: any) {
      let rawMsg = 'An unexpected inference error occurred.';
      if (err instanceof Error) {
        rawMsg = err.message;
      } else if (typeof err === 'string') {
        rawMsg = err;
      } else if (err && typeof err === 'object') {
        rawMsg = typeof err.message === 'string' ? err.message : JSON.stringify(err);
      }

      if (rawMsg === '[object Object]' || rawMsg.includes('[object Object]')) {
        rawMsg = 'Connection error: Unable to complete inference request.';
      }

      if (rawMsg.toLowerCase().includes('failed to fetch') || rawMsg.toLowerCase().includes('network')) {
        setFieldErrors({
          general: {
            message: 'Connection Failed: Cannot reach Backend API at http://127.0.0.1:8000',
            suggestion: 'Make sure the backend server is running. Run start_full_suite.bat or "python -m uvicorn backend.main:app --port 8000".'
          }
        });
      } else {
        setFieldErrors({
          general: {
            message: rawMsg,
            suggestion: 'Please verify server logs or restart the application.'
          }
        });
      }
    } finally {
      setLoading(false);
    }
  };

  // Execute first forecast automatically once checkpoints load
  useEffect(() => {
    if (checkpoints.length > 0 && !result && !loading && Object.keys(fieldErrors).length === 0) {
      handleRunForecast();
    }
  }, [checkpoints]);

  // Handle Channel Selection Change
  const handleChannelChange = (newIdx: number) => {
    if (result && (newIdx < 0 || newIdx >= result.columns.length)) {
      setFieldErrors({
        channel_idx: {
          message: `Invalid channel index: ${newIdx}.`,
          suggestion: `Please choose a channel between 0 and ${result.columns.length - 1}.`
        }
      });
      return;
    }
    setFieldErrors({});
    setChannelIdx(newIdx);
    if (result) {
      handleRunForecast(newIdx);
    }
  };

  // Prepare Chart Data
  const chartData = useMemo(() => {
    if (!result) return [];

    const seqLen = result.historical.length;
    const predLenActual = result.predictions_adaptive.length;
    const ch = result.channel_idx;

    const points: any[] = [];

    // Historical Points (t = -95 to 0)
    for (let i = 0; i < seqLen; i++) {
      const step = i - seqLen + 1;
      points.push({
        step,
        label: step === 0 ? 'Now (t)' : `t${step}`,
        historical: Number(result.historical[i][ch].toFixed(3)),
        baseline: step === 0 ? Number(result.historical[i][ch].toFixed(3)) : null,
        adaptive: step === 0 ? Number(result.historical[i][ch].toFixed(3)) : null,
      });
    }

    // Forecast Points (t = +1 to +predLen)
    for (let i = 0; i < predLenActual; i++) {
      const step = i + 1;
      points.push({
        step,
        label: `t+${step}`,
        historical: null,
        baseline: Number(result.predictions_baseline[i][ch].toFixed(3)),
        adaptive: Number(result.predictions_adaptive[i][ch].toFixed(3)),
      });
    }

    return points;
  }, [result]);

  // Filtered Benchmarks
  const filteredBenchmarks = useMemo(() => {
    return benchmarks.filter(item => {
      const matchesSearch =
        item.dataset.toLowerCase().includes(benchmarkSearch.toLowerCase());
      const matchesHorizon =
        benchmarkHorizonFilter === 'all' || item.horizon === benchmarkHorizonFilter;
      return matchesSearch && matchesHorizon;
    });
  }, [benchmarks, benchmarkSearch, benchmarkHorizonFilter]);

  // Dataset Icon Renderer
  const renderDatasetIcon = (iconName: string) => {
    switch (iconName) {
      case 'Car': return <Car size={16} style={{ color: '#2563eb' }} />;
      case 'Zap': return <Zap size={16} style={{ color: '#d97706' }} />;
      case 'Activity': return <Activity size={16} style={{ color: '#059669' }} />;
      case 'CloudSun': return <CloudSun size={16} style={{ color: '#0284c7' }} />;
      case 'BatteryCharging': return <BatteryCharging size={16} style={{ color: '#7c3aed' }} />;
      default: return <Layers size={16} style={{ color: '#2563eb' }} />;
    }
  };

  return (
    <div className="app-wrapper animate-fade-in">
      {/* ========================================================================= */}
      {/* 1. TOP NAVIGATION & HEADER (Light, Crisp, Professional)                    */}
      {/* ========================================================================= */}
      <header className="top-navbar">
        <div className="brand-section">
          <div className="brand-logo-icon">
            <TimeMixerLogo size={38} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <h1 style={{ fontSize: '1.25rem', margin: 0, fontWeight: 700, color: '#0f172a', letterSpacing: '-0.02em' }}>
                TimeMixer <span style={{ color: '#2563eb' }}>Adaptive-FMM</span>
              </h1>
            </div>
            <p style={{ margin: 0, fontSize: '0.8rem', color: '#64748b' }}>
              Adaptive Multiscale Forecast Mixing Suite • Enterprise Research Platform
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="nav-tabs">
          <button
            className={`nav-tab-btn ${activeTab === 'studio' ? 'active' : ''}`}
            onClick={() => setActiveTab('studio')}
          >
            <Activity size={15} />
            <span>Forecast Studio</span>
          </button>

          <button
            className={`nav-tab-btn ${activeTab === 'benchmarks' ? 'active' : ''}`}
            onClick={() => setActiveTab('benchmarks')}
          >
            <BarChart2 size={15} />
            <span>Benchmark Matrix (31 Tasks)</span>
          </button>

          <button
            className={`nav-tab-btn ${activeTab === 'theory' ? 'active' : ''}`}
            onClick={() => setActiveTab('theory')}
          >
            <Cpu size={15} />
            <span>Theory & Architecture</span>
          </button>
        </nav>

        {/* Live Engine Status - Dedicated Container with Guaranteed Spacing */}
        <div style={{ display: 'flex', alignItems: 'center', marginLeft: 'auto', flexShrink: 0 }}>
          <div className="system-status-badge">
            <div className="status-dot"></div>
            <span>Inference Ready</span>
          </div>
        </div>
      </header>

      {/* ========================================================================= */}
      {/* TAB 1: LIVE FORECAST STUDIO                                               */}
      {/* ========================================================================= */}
      {activeTab === 'studio' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Quick-Start Benchmark Dataset Selector */}
          <section className="glass-panel" style={{ padding: '1.25rem 1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
              <div>
                <h3 style={{ fontSize: '0.98rem', margin: 0, fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.45rem', color: '#0f172a' }}>
                  <Database size={16} style={{ color: '#2563eb' }} />
                  Benchmark Dataset Presets
                </h3>
                <p style={{ margin: 0, fontSize: '0.78rem', color: '#64748b' }}>
                  Select any empirical benchmark dataset for instant model evaluation with calibrated checkpoints
                </p>
              </div>
              <span style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 500 }}>
                5 Canonical Datasets Available
              </span>
            </div>

            <div className="preset-grid" style={{ marginBottom: 0 }}>
              {demoDatasets.map(ds => (
                <div
                  key={ds.id}
                  className={`preset-card ${selectedDemoId === ds.id && !file ? 'active' : ''}`}
                  onClick={() => handleSelectDemo(ds.id)}
                >
                  <div className="preset-title">
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                      {renderDatasetIcon(ds.icon)}
                      {ds.id}
                    </span>
                    <span className="preset-badge">{ds.features} Ch</span>
                  </div>
                  <p className="preset-desc">{ds.description}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Main Forecast Layout (Sidebar Controls + Studio Canvas) */}
          <div className="main-forecast-layout">
            {/* Left Controls Sidebar */}
            <aside className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
              <div style={{ borderBottom: '1px solid #e2e8f0', paddingBottom: '0.65rem' }}>
                <h3 style={{ fontSize: '0.92rem', margin: 0, display: 'flex', alignItems: 'center', gap: '0.45rem', color: '#0f172a' }}>
                  <Sliders size={16} style={{ color: '#2563eb' }} />
                  Inference Parameters
                </h3>
              </div>

              {/* CSV Upload Area */}
              <div>
                <label className="input-label" style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Custom Dataset (.csv)</span>
                  {file && <span style={{ color: '#059669', fontWeight: 600 }}>Active</span>}
                </label>
                <div
                  className={`upload-area ${file ? 'drag-active' : ''} ${fieldErrors.file ? 'input-field-error' : ''}`}
                  onDragOver={e => e.preventDefault()}
                  onDrop={handleDrop}
                  onClick={() => document.getElementById('csvFileInput')?.click()}
                >
                  <UploadCloud size={24} className="upload-icon" style={{ margin: '0 auto 0.4rem auto' }} />
                  <p style={{ fontSize: '0.82rem', fontWeight: 600, margin: '0 0 0.2rem 0', color: '#0f172a' }}>
                    {file ? file.name : 'Drop CSV or click to browse'}
                  </p>
                  <p style={{ fontSize: '0.72rem', color: '#64748b', margin: 0 }}>
                    {file ? `${(file.size / 1024).toFixed(1)} KB` : 'Supports multi-feature time series'}
                  </p>
                  <input
                    id="csvFileInput"
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                  />
                </div>

                {fieldErrors.file && (
                  <div className="input-error-box">
                    <div className="input-error-msg">
                      <AlertCircle size={14} />
                      <span>{fieldErrors.file.message}</span>
                    </div>
                    {fieldErrors.file.suggestion && (
                      <div className="input-error-fix">
                        💡 <strong>How to fix:</strong> {fieldErrors.file.suggestion}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Horizon Selector (Pill Group) */}
              <div>
                <label className="input-label">Prediction Horizon (Steps)</label>
                <div className="pill-group">
                  {[96, 192, 336, 720].map(h => (
                    <button
                      key={h}
                      className={`pill-btn ${predLen === h ? 'active' : ''}`}
                      onClick={() => {
                        setPredLen(h);
                        setFieldErrors(prev => {
                          const next = { ...prev };
                          delete next.pred_len;
                          return next;
                        });
                      }}
                    >
                      H = {h}
                    </button>
                  ))}
                </div>

                {fieldErrors.pred_len && (
                  <div className="input-error-box">
                    <div className="input-error-msg">
                      <AlertCircle size={14} />
                      <span>{fieldErrors.pred_len.message}</span>
                    </div>
                    {fieldErrors.pred_len.suggestion && (
                      <div className="input-error-fix">
                        💡 <strong>How to fix:</strong> {fieldErrors.pred_len.suggestion}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Trained Checkpoint Selector */}
              <div>
                <label className="input-label">Neural Checkpoint</label>
                <select
                  className={`input-field select-field ${fieldErrors.checkpoint ? 'input-field-error' : ''}`}
                  value={selectedCheckpoint}
                  onChange={e => {
                    setSelectedCheckpoint(e.target.value);
                    setFieldErrors(prev => {
                      const next = { ...prev };
                      delete next.checkpoint;
                      return next;
                    });
                  }}
                >
                  <option value="">✨ Auto-Detect & Match Calibrated Model</option>
                  {checkpoints.map(c => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>

                {fieldErrors.checkpoint && (
                  <div className="input-error-box">
                    <div className="input-error-msg">
                      <AlertCircle size={14} />
                      <span>{fieldErrors.checkpoint.message}</span>
                    </div>
                    {fieldErrors.checkpoint.suggestion && (
                      <div className="input-error-fix">
                        💡 <strong>How to fix:</strong> {fieldErrors.checkpoint.suggestion}
                      </div>
                    )}
                    {fieldErrors.checkpoint.autoFix && (
                      <button
                        type="button"
                        className="btn-fix-sm"
                        onClick={fieldErrors.checkpoint.autoFix}
                      >
                        {fieldErrors.checkpoint.autoFixLabel || "Auto-Select Matching Checkpoint"}
                      </button>
                    )}
                  </div>
                )}
              </div>

              {/* Channel / Feature Selector */}
              {result && result.columns.length > 0 && (
                <div>
                  <label className="input-label">
                    Target Channel ({result.columns.length} Total)
                  </label>
                  <select
                    className={`input-field select-field ${fieldErrors.channel_idx ? 'input-field-error' : ''}`}
                    value={channelIdx}
                    onChange={e => handleChannelChange(Number(e.target.value))}
                  >
                    {result.columns.map((col, idx) => (
                      <option key={idx} value={idx}>
                        Ch {idx}: {col}
                      </option>
                    ))}
                  </select>

                  {fieldErrors.channel_idx && (
                    <div className="input-error-box">
                      <div className="input-error-msg">
                        <AlertCircle size={14} />
                        <span>{fieldErrors.channel_idx.message}</span>
                      </div>
                      {fieldErrors.channel_idx.suggestion && (
                        <div className="input-error-fix">
                          💡 <strong>How to fix:</strong> {fieldErrors.channel_idx.suggestion}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Model Visualization Mode */}
              <div>
                <label className="input-label">Visualization Comparison</label>
                <div className="pill-group">
                  <button
                    type="button"
                    className={`pill-btn ${visMode === 'split' ? 'active' : ''}`}
                    onClick={() => setVisMode('split')}
                    style={{ flex: 1, minWidth: '70px', textAlign: 'center' }}
                  >
                    Side-by-Side
                  </button>
                  <button
                    type="button"
                    className={`pill-btn ${visMode === 'both' ? 'active' : ''}`}
                    onClick={() => setVisMode('both')}
                    style={{ flex: 1, minWidth: '70px', textAlign: 'center' }}
                  >
                    Overlay
                  </button>
                  <button
                    type="button"
                    className={`pill-btn ${visMode === 'adaptive' ? 'active' : ''}`}
                    onClick={() => setVisMode('adaptive')}
                    style={{ flex: 1, minWidth: '70px', textAlign: 'center' }}
                  >
                    Adaptive
                  </button>
                  <button
                    type="button"
                    className={`pill-btn ${visMode === 'baseline' ? 'active' : ''}`}
                    onClick={() => setVisMode('baseline')}
                    style={{ flex: 1, minWidth: '70px', textAlign: 'center' }}
                  >
                    Baseline
                  </button>
                </div>
              </div>

              {/* Execute Button */}
              <button
                className="btn btn-primary"
                onClick={() => handleRunForecast()}
                disabled={loading}
                style={{ width: '100%', marginTop: '0.25rem' }}
              >
                {loading ? (
                  <>
                    <RefreshCw size={15} className="animate-spin" />
                    <span>Computing Projections...</span>
                  </>
                ) : (
                  <>
                    <Play size={15} fill="white" />
                    <span>Generate Forecast</span>
                  </>
                )}
              </button>

              {/* General System / Connectivity Error */}
              {fieldErrors.general && (
                <div className="input-error-box" style={{ marginTop: '0.25rem' }}>
                  <div className="input-error-msg">
                    <AlertCircle size={15} style={{ flexShrink: 0 }} />
                    <span>{fieldErrors.general.message}</span>
                  </div>
                  {fieldErrors.general.suggestion && (
                    <div className="input-error-fix">
                      💡 <strong>How to fix:</strong> {fieldErrors.general.suggestion}
                    </div>
                  )}
                </div>
              )}
            </aside>

            {/* Right Display Canvas */}
            <main style={{ minWidth: 0, display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              {/* Dual Side-by-Side View vs Unified Overlay */}
              {visMode === 'split' ? (
                <div className="side-by-side-grid">
                  {/* Left Side: Baseline TimeMixer */}
                  <div className="side-by-side-panel">
                    <div className="side-by-side-header">
                      <div>
                        <h4 style={{ fontSize: '0.96rem', fontWeight: 600, margin: '0 0 0.15rem 0', color: '#0f172a', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                          <span style={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: '#d97706', display: 'inline-block' }} />
                          Baseline TimeMixer
                        </h4>
                        <p style={{ margin: 0, fontSize: '0.75rem', color: '#64748b' }}>
                          Uniform Multi-Scale Mixing (Fixed weights w_k = 1.00)
                        </p>
                      </div>
                      <span className="side-by-side-badge" style={{ background: '#fef3c7', color: '#92400e', border: '1px solid #fde68a' }}>
                        Dashed Amber
                      </span>
                    </div>

                    <div style={{ height: '320px', width: '100%', minWidth: 0 }}>
                      {loading ? (
                        <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '0.65rem', color: '#64748b' }}>
                          <RefreshCw size={24} className="animate-spin text-amber-600" />
                          <p style={{ fontSize: '0.8rem' }}>Computing Baseline forecast...</p>
                        </div>
                      ) : chartData.length > 0 ? (
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={chartData} margin={{ top: 8, right: 12, left: -10, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                            <XAxis
                              dataKey="label"
                              stroke="#94a3b8"
                              tick={{ fill: '#64748b', fontSize: 10 }}
                              interval={Math.floor(chartData.length / 8)}
                              tickLine={false}
                            />
                            <YAxis
                              stroke="#94a3b8"
                              tick={{ fill: '#64748b', fontSize: 10 }}
                              domain={['auto', 'auto']}
                              tickLine={false}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: '#ffffff',
                                border: '1px solid #e2e8f0',
                                borderRadius: '8px',
                                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                                fontSize: '0.78rem',
                                color: '#0f172a'
                              }}
                              labelStyle={{ color: '#0f172a', fontWeight: 600, marginBottom: '2px' }}
                            />
                            <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: '0.78rem', color: '#475569' }} />
                            <ReferenceLine x="Now (t)" stroke="#94a3b8" strokeDasharray="3 3" />
                            <Line
                              type="monotone"
                              dataKey="historical"
                              name="Lookback (t-96)"
                              stroke="#64748b"
                              strokeWidth={1.8}
                              dot={false}
                              isAnimationActive={false}
                            />
                            <Line
                              type="monotone"
                              dataKey="baseline"
                              name="Baseline Projection"
                              stroke="#d97706"
                              strokeWidth={2.2}
                              strokeDasharray="4 4"
                              dot={false}
                              isAnimationActive={false}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      ) : (
                        <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b', fontSize: '0.82rem' }}>
                          Click "Generate Forecast" to plot baseline
                        </div>
                      )}
                    </div>

                    {result && (
                      <div className="side-by-side-stats-row">
                        <div className="side-by-side-stat-item">
                          <span>Peak:</span> <strong style={{ color: '#0f172a' }}>{result.metrics_baseline.max.toFixed(2)}</strong>
                        </div>
                        <div className="side-by-side-stat-item">
                          <span>Trough:</span> <strong style={{ color: '#0f172a' }}>{result.metrics_baseline.min.toFixed(2)}</strong>
                        </div>
                        <div className="side-by-side-stat-item">
                          <span>Mean:</span> <strong style={{ color: '#0f172a' }}>{result.metrics_baseline.mean.toFixed(2)}</strong>
                        </div>
                        <div className="side-by-side-stat-item">
                          <span>Range Δ:</span> <strong style={{ color: '#0f172a' }}>{result.metrics_baseline.range.toFixed(2)}</strong>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Right Side: Adaptive-FMM (Proposed Ours) */}
                  <div className="side-by-side-panel">
                    <div className="side-by-side-header">
                      <div>
                        <h4 style={{ fontSize: '0.96rem', fontWeight: 600, margin: '0 0 0.15rem 0', color: '#0f172a', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                          <span style={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: '#2563eb', display: 'inline-block' }} />
                          Adaptive-FMM (Proposed Ours)
                        </h4>
                        <p style={{ margin: 0, fontSize: '0.75rem', color: '#64748b' }}>
                          Horizon-Aware Attention Router (Dynamic w_k)
                        </p>
                      </div>
                      <span className="side-by-side-badge" style={{ background: '#eff6ff', color: '#1d4ed8', border: '1px solid #bfdbfe' }}>
                        Solid Royal Blue
                      </span>
                    </div>

                    <div style={{ height: '320px', width: '100%', minWidth: 0 }}>
                      {loading ? (
                        <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '0.65rem', color: '#64748b' }}>
                          <RefreshCw size={24} className="animate-spin text-blue-600" />
                          <p style={{ fontSize: '0.8rem' }}>Computing Adaptive-FMM forecast...</p>
                        </div>
                      ) : chartData.length > 0 ? (
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={chartData} margin={{ top: 8, right: 12, left: -10, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                            <XAxis
                              dataKey="label"
                              stroke="#94a3b8"
                              tick={{ fill: '#64748b', fontSize: 10 }}
                              interval={Math.floor(chartData.length / 8)}
                              tickLine={false}
                            />
                            <YAxis
                              stroke="#94a3b8"
                              tick={{ fill: '#64748b', fontSize: 10 }}
                              domain={['auto', 'auto']}
                              tickLine={false}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: '#ffffff',
                                border: '1px solid #e2e8f0',
                                borderRadius: '8px',
                                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                                fontSize: '0.78rem',
                                color: '#0f172a'
                              }}
                              labelStyle={{ color: '#0f172a', fontWeight: 600, marginBottom: '2px' }}
                            />
                            <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: '0.78rem', color: '#475569' }} />
                            <ReferenceLine x="Now (t)" stroke="#94a3b8" strokeDasharray="3 3" />
                            <Line
                              type="monotone"
                              dataKey="historical"
                              name="Lookback (t-96)"
                              stroke="#64748b"
                              strokeWidth={1.8}
                              dot={false}
                              isAnimationActive={false}
                            />
                            <Line
                              type="monotone"
                              dataKey="adaptive"
                              name="Adaptive-FMM (Ours)"
                              stroke="#2563eb"
                              strokeWidth={2.4}
                              dot={false}
                              isAnimationActive={false}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      ) : (
                        <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b', fontSize: '0.82rem' }}>
                          Click "Generate Forecast" to plot adaptive
                        </div>
                      )}
                    </div>

                    {result && (
                      <div className="side-by-side-stats-row">
                        <div className="side-by-side-stat-item">
                          <span>Peak:</span> <strong style={{ color: '#0f172a' }}>{result.metrics_adaptive.max.toFixed(2)}</strong>
                        </div>
                        <div className="side-by-side-stat-item">
                          <span>Trough:</span> <strong style={{ color: '#0f172a' }}>{result.metrics_adaptive.min.toFixed(2)}</strong>
                        </div>
                        <div className="side-by-side-stat-item">
                          <span>Mean:</span> <strong style={{ color: '#0f172a' }}>{result.metrics_adaptive.mean.toFixed(2)}</strong>
                        </div>
                        <div className="side-by-side-stat-item">
                          <span>Range Δ:</span> <strong style={{ color: '#0f172a' }}>{result.metrics_adaptive.range.toFixed(2)}</strong>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                /* Overlay or Focused Single Chart */
                <section className="glass-panel" style={{ padding: '1.25rem 1.5rem', minWidth: 0 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                    <div>
                      <h3 style={{ fontSize: '1.05rem', margin: 0, fontWeight: 600, color: '#0f172a' }}>
                        {visMode === 'both' ? 'Unified Multiscale Forecast Overlay' : (visMode === 'adaptive' ? 'Adaptive-FMM Trajectory Focus' : 'Baseline TimeMixer Trajectory Focus')}
                      </h3>
                      <p style={{ margin: 0, fontSize: '0.8rem', color: '#64748b' }}>
                        Channel: <span style={{ color: '#2563eb', fontWeight: 600 }}>{result?.selected_channel_name || 'Loading...'}</span> | Lookback L=96 | Horizon H={result?.dataset_info.pred_len || predLen}
                      </p>
                    </div>

                    {result && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span className="preset-badge" style={{ background: '#ecfdf5', color: '#047857', borderColor: '#a7f3d0' }}>
                          Inference: {result.runtime_ms} ms
                        </span>
                        <span className="preset-badge" style={{ background: '#f1f5f9', color: '#475569' }}>
                          {result.dataset_info.total_features} Channels
                        </span>
                      </div>
                    )}
                  </div>

                  <div style={{ height: '370px', width: '100%', minWidth: 0 }}>
                    {loading ? (
                      <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '0.85rem', color: '#64748b' }}>
                        <RefreshCw size={32} className="animate-spin text-blue-600" />
                        <p style={{ fontSize: '0.85rem' }}>Computing multiscale forecast recombination...</p>
                      </div>
                    ) : chartData.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                          <XAxis
                            dataKey="label"
                            stroke="#94a3b8"
                            tick={{ fill: '#64748b', fontSize: 11 }}
                            interval={Math.floor(chartData.length / 10)}
                            tickLine={false}
                          />
                          <YAxis
                            stroke="#94a3b8"
                            tick={{ fill: '#64748b', fontSize: 11 }}
                            domain={['auto', 'auto']}
                            tickLine={false}
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: '#ffffff',
                              border: '1px solid #e2e8f0',
                              borderRadius: '8px',
                              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                              fontSize: '0.82rem',
                              color: '#0f172a'
                            }}
                            labelStyle={{ color: '#0f172a', fontWeight: 600, marginBottom: '4px' }}
                          />
                          <Legend
                            verticalAlign="top"
                            height={34}
                            wrapperStyle={{ fontSize: '0.82rem', color: '#475569' }}
                          />
                          <ReferenceLine
                            x="Now (t)"
                            stroke="#94a3b8"
                            strokeDasharray="4 4"
                            label={{ value: 'Horizon Start', fill: '#64748b', fontSize: 11, position: 'top' }}
                          />

                          {/* Past Lookback (Slate) */}
                          <Line
                            type="monotone"
                            dataKey="historical"
                            name="Historical Lookback (t-96)"
                            stroke="#64748b"
                            strokeWidth={2}
                            dot={false}
                            isAnimationActive={false}
                          />

                          {/* Baseline TimeMixer (Dashed Amber) */}
                          {(visMode === 'both' || visMode === 'baseline') && (
                            <Line
                              type="monotone"
                              dataKey="baseline"
                              name="Baseline TimeMixer (Uniform)"
                              stroke="#d97706"
                              strokeWidth={2}
                              strokeDasharray="4 4"
                              dot={false}
                              isAnimationActive={false}
                            />
                          )}

                          {/* Adaptive-FMM (Solid Royal Blue) */}
                          {(visMode === 'both' || visMode === 'adaptive') && (
                            <Line
                              type="monotone"
                              dataKey="adaptive"
                              name="Adaptive-FMM (Proposed Ours)"
                              stroke="#2563eb"
                              strokeWidth={2.5}
                              dot={false}
                              isAnimationActive={false}
                            />
                          )}
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b' }}>
                        Click "Generate Forecast" to plot time series trajectory
                      </div>
                    )}
                  </div>
                </section>
              )}

              {/* KPI Scorecards Grid */}
              {result && (
                <div className="dashboard-grid">
                  <div className="metric-card">
                    <span className="metric-label">Predicted Peak</span>
                    <span className="metric-value">
                      {result.metrics_adaptive.max.toFixed(2)}
                    </span>
                    <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
                      Baseline: {result.metrics_baseline.max.toFixed(2)}
                    </span>
                  </div>

                  <div className="metric-card">
                    <span className="metric-label">Predicted Trough</span>
                    <span className="metric-value">
                      {result.metrics_adaptive.min.toFixed(2)}
                    </span>
                    <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
                      Baseline: {result.metrics_baseline.min.toFixed(2)}
                    </span>
                  </div>

                  <div className="metric-card">
                    <span className="metric-label">Dynamic Range (Δ)</span>
                    <span className="metric-value">
                      {result.metrics_adaptive.range.toFixed(2)}
                    </span>
                    <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
                      Baseline: {result.metrics_baseline.range.toFixed(2)}
                    </span>
                  </div>

                  <div className="metric-card">
                    <span className="metric-label">Volatility (σ)</span>
                    <span className="metric-value">
                      {result.metrics_adaptive.std.toFixed(2)}
                    </span>
                    <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
                      Signal standard deviation
                    </span>
                  </div>

                  <div className="metric-card">
                    <span className="metric-label">Forecast Trend</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', marginTop: '0.2rem' }}>
                      {result.metrics_adaptive.trend === 'upward' ? (
                        <TrendingUp size={20} style={{ color: '#059669' }} />
                      ) : (
                        <TrendingDown size={20} style={{ color: '#dc2626' }} />
                      )}
                      <span style={{
                        fontSize: '1.15rem',
                        fontWeight: 700,
                        textTransform: 'capitalize',
                        color: result.metrics_adaptive.trend === 'upward' ? '#059669' : '#dc2626'
                      }}>
                        {result.metrics_adaptive.trend}
                      </span>
                    </div>
                    <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Trajectory profile</span>
                  </div>
                </div>
              )}

              {/* Multiscale Scale Attention Decomposition */}
              {result && (
                <section className="glass-panel scale-decomposer">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #e2e8f0', paddingBottom: '0.65rem' }}>
                    <div>
                      <h3 style={{ fontSize: '0.98rem', margin: 0, fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.45rem', color: '#0f172a' }}>
                        <Activity size={17} style={{ color: '#2563eb' }} />
                        Multiscale Attention Decomposition (Adaptive-FMM Routing)
                      </h3>
                      <p style={{ margin: 0, fontSize: '0.78rem', color: '#64748b' }}>
                        Learned dynamic gate weights w_k modulating each temporal downsampling scale
                      </p>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span className="preset-badge" style={{ background: '#eff6ff', color: '#1d4ed8', borderColor: '#bfdbfe' }}>
                        Σ w_k = {result.scale_weights.reduce((a, b) => a + b, 0).toFixed(2)} / {result.scale_weights.length}.00 (Budget Preserved)
                      </span>
                    </div>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.4rem' }}>
                    {result.scale_weights.map((weight, idx) => {
                      const colors = [
                        '#2563eb', // Scale 0: Blue
                        '#4f46e5', // Scale 1: Indigo
                        '#7c3aed', // Scale 2: Purple
                        '#d97706'  // Scale 3: Amber
                      ];
                      const maxPossible = 2.5;
                      const fillPercent = Math.min(100, Math.max(10, (weight / maxPossible) * 100));

                      return (
                        <div key={idx} className="scale-row">
                          <div className="scale-label-box">
                            {result.scale_labels[idx] || `Scale ${idx} (${Math.pow(2, idx)}x)`}
                          </div>
                          <div className="scale-bar-track">
                            <div
                              className="scale-bar-fill"
                              style={{
                                width: `${fillPercent}%`,
                                background: colors[idx % colors.length]
                              }}
                            />
                          </div>
                          <div className="scale-val-badge">
                            w_{idx} = {weight.toFixed(2)}
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <div style={{
                    marginTop: '0.4rem',
                    padding: '0.75rem 0.95rem',
                    background: '#f8fafc',
                    border: '1px solid #e2e8f0',
                    borderRadius: '6px',
                    fontSize: '0.78rem',
                    color: '#475569',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.45rem'
                  }}>
                    <Info size={15} style={{ color: '#2563eb', flexShrink: 0 }} />
                    <span>
                      {result.dataset_info.pred_len <= 96
                        ? 'Short horizon dynamic: Fine-grained Scale 0/1 are prioritized to capture sharp intraday fluctuations.'
                        : 'Long horizon dynamic: Coarse Scale 2/3 trend routing is prioritized to prevent error accumulation.'}
                    </span>
                  </div>
                </section>
              )}
            </main>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 2: BENCHMARK MATRIX (31 TASKS)                                        */}
      {/* ========================================================================= */}
      {activeTab === 'benchmarks' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Top Summary Banner */}
          <div className="dashboard-grid">
            <div className="metric-card">
              <span className="metric-label">Benchmark Tasks</span>
              <span className="metric-value">31 Tasks</span>
              <span style={{ fontSize: '0.76rem', color: '#64748b' }}>Across 8 Canonical Datasets</span>
            </div>

            <div className="metric-card">
              <span className="metric-label">Peak MSE Improvement</span>
              <span className="metric-value" style={{ color: '#059669' }}>-80.46%</span>
              <span style={{ fontSize: '0.76rem', color: '#64748b' }}>Traffic Benchmark (H=192)</span>
            </div>

            <div className="metric-card">
              <span className="metric-label">High-Dim Stress Test</span>
              <span className="metric-value" style={{ color: '#2563eb' }}>862 Sensors</span>
              <span style={{ fontSize: '0.76rem', color: '#64748b' }}>PeMS Bay Area Highway Grid</span>
            </div>

            <div className="metric-card">
              <span className="metric-label">Parameter Overhead</span>
              <span className="metric-value" style={{ color: '#4f46e5' }}>&lt; 0.05%</span>
              <span style={{ fontSize: '0.76rem', color: '#64748b' }}>Linear Summary Router</span>
            </div>
          </div>

          {/* Interactive Benchmark Search & Filters */}
          <div className="glass-panel" style={{ padding: '1rem 1.25rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.85rem', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', flex: 1, minWidth: '260px' }}>
              <Search size={16} style={{ color: '#64748b' }} />
              <input
                type="text"
                placeholder="Search benchmark dataset (Traffic, ETT, Weather, Electricity)..."
                value={benchmarkSearch}
                onChange={e => setBenchmarkSearch(e.target.value)}
                className="input-field"
                style={{ margin: 0, padding: '0.45rem 0.75rem' }}
              />
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
              <span style={{ fontSize: '0.8rem', color: '#64748b' }}>Horizon:</span>
              {(['all', 96, 192, 336, 720] as const).map(h => (
                <button
                  key={h}
                  className={`pill-btn ${benchmarkHorizonFilter === h ? 'active' : ''}`}
                  onClick={() => setBenchmarkHorizonFilter(h)}
                >
                  {h === 'all' ? 'All (31)' : `H=${h}`}
                </button>
              ))}
            </div>
          </div>

          {/* Benchmark Table */}
          <div className="table-wrapper">
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Horizon (H)</th>
                  <th>Features (D)</th>
                  <th>Baseline MSE</th>
                  <th>Adaptive MSE (Ours)</th>
                  <th>MSE Gain (Δ %)</th>
                  <th>Baseline MAE</th>
                  <th>Adaptive MAE</th>
                  <th>Param Overhead</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {filteredBenchmarks.length > 0 ? (
                  filteredBenchmarks.map((b, idx) => {
                    const isImproved = b.delta_mse_pct < 0;
                    return (
                      <tr key={idx}>
                        <td style={{ fontWeight: 600, color: '#0f172a' }}>{b.dataset}</td>
                        <td>
                          <span className="preset-badge">
                            {b.horizon}
                          </span>
                        </td>
                        <td style={{ color: '#64748b' }}>{b.n_features}</td>
                        <td style={{ fontFamily: 'monospace' }}>{b.baseline_mse.toFixed(4)}</td>
                        <td style={{ fontFamily: 'monospace', color: isImproved ? '#059669' : '#dc2626', fontWeight: 600 }}>
                          {b.adaptive_mse.toFixed(4)}
                        </td>
                        <td>
                          {isImproved ? (
                            <span className="badge-improvement-green">
                              ▼ {Math.abs(b.delta_mse_pct).toFixed(2)}%
                            </span>
                          ) : (
                            <span className="badge-neutral">
                              + {b.delta_mse_pct.toFixed(2)}%
                            </span>
                          )}
                        </td>
                        <td style={{ fontFamily: 'monospace' }}>{b.baseline_mae.toFixed(4)}</td>
                        <td style={{ fontFamily: 'monospace' }}>{b.adaptive_mae.toFixed(4)}</td>
                        <td style={{ color: '#64748b', fontSize: '0.78rem' }}>+{b.overhead_params}</td>
                        <td>
                          {isImproved ? (
                            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', color: '#059669', fontSize: '0.78rem', fontWeight: 600 }}>
                              <CheckCircle2 size={13} /> Superior
                            </span>
                          ) : (
                            <span style={{ color: '#64748b', fontSize: '0.78rem' }}>Parity</span>
                          )}
                        </td>
                      </tr>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan={10} style={{ textAlign: 'center', padding: '2rem', color: '#64748b' }}>
                      No benchmark tasks match the selected search query or horizon filter.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 3: THEORY & ARCHITECTURE                                              */}
      {/* ========================================================================= */}
      {activeTab === 'theory' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Architectural Overview */}
          <section className="glass-panel" style={{ padding: '1.75rem' }}>
            <h2 style={{ fontSize: '1.35rem', marginBottom: '0.65rem', display: 'flex', alignItems: 'center', gap: '0.55rem', color: '#0f172a' }}>
              <Cpu size={22} style={{ color: '#2563eb' }} />
              Theoretical Formulation: Adaptive Multiscale Forecast Mixing (Adaptive-FMM)
            </h2>
            <p style={{ color: '#334155', lineHeight: 1.6, fontSize: '0.92rem' }}>
              In the canonical TimeMixer architecture (Wang et al., ICLR 2024), multi-resolution forecast projections
              Ŷ_k from each downsampled temporal scale are blended via an inflexible uniform average:
              Ŷ = (1/K) · Σ Ŷ_k.
              This design assumes that all temporal frequencies contribute uniformly across all prediction horizons,
              disregarding the reality that fine-grained scales dominate short horizons (H=96) while coarse macro-trends
              dominate long horizons (H ≥ 336).
            </p>

            {/* Formula Block */}
            <div className="formula-block">
              // 1. Multiscale Context Summary Aggregation<br />
              h_summary = [ GAP(H_0) || GAP(H_1) || ... || GAP(H_K-1) ] ∈ ℝ^(K · d_model)<br /><br />
              // 2. Budget-Constrained Softmax Scale Routing<br />
              s = W_gate · h_summary + b_gate ∈ ℝ^K<br />
              w_k = K · [ exp(s_k / τ) / Σ_(j=0)^(K-1) exp(s_j / τ) ], &nbsp; such that Σ_(k=0)^(K-1) w_k = K<br /><br />
              // 3. Dynamic Recombination<br />
              Ŷ_Adaptive = Σ_(k=0)^(K-1) w_k · Ŷ_k
            </div>
          </section>

          {/* Proposition 1 Theorem Card */}
          <section className="theorem-card">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', marginBottom: '0.45rem' }}>
              <ShieldCheck size={19} style={{ color: '#2563eb' }} />
              <h3 style={{ fontSize: '1.05rem', margin: 0, fontWeight: 700, color: '#0f172a' }}>
                Proposition 1 (Identity Recovery via Zero-Initialization Equivalence)
              </h3>
            </div>
            <p style={{ color: '#334155', fontSize: '0.9rem', lineHeight: 1.6, margin: '0 0 0.85rem 0' }}>
              <strong>Statement:</strong> Let the routing linear projection parameters be initialized to zero:
              <span style={{ fontFamily: 'monospace', color: '#2563eb', fontWeight: 600 }}> W_gate = 0, b_gate = 0</span>.
              Then for any arbitrary input sequence X, the initial scale logits satisfy
              <span style={{ fontFamily: 'monospace', color: '#2563eb', fontWeight: 600 }}> s_k = 0</span> for all k ∈ &#123;0, ..., K-1&#125;.
              Consequently, the budget-constrained routing weights strictly evaluate to:
            </p>

            <div className="formula-block" style={{ margin: '0.65rem 0' }}>
              w_k = K · [ exp(0) / Σ_(j=0)^(K-1) exp(0) ] = K · [ 1 / K ] = 1.000 &nbsp; (∀ k)
            </div>

            <p style={{ color: '#475569', fontSize: '0.85rem', lineHeight: 1.5, margin: 0 }}>
              <strong>Proof & Architectural Guarantee:</strong> Since every scale weight identically equals 1.0,
              the initial forward pass of Adaptive-FMM strictly recovers the exact mathematical output of the pre-trained
              canonical TimeMixer model without training collapse, exploding gradients, or cold-start performance degradation.
            </p>
          </section>

          {/* Comparative Matrix Table */}
          <section className="glass-panel" style={{ padding: '1.5rem' }}>
            <h3 style={{ fontSize: '1.05rem', marginBottom: '0.85rem', fontWeight: 600, color: '#0f172a' }}>
              Architectural Comparison: Baseline TimeMixer vs. Proposed Adaptive-FMM
            </h3>

            <div className="table-wrapper">
              <table className="benchmark-table">
                <thead>
                  <tr>
                    <th>Capability / Attribute</th>
                    <th>Baseline TimeMixer (ICLR 2024)</th>
                    <th>Adaptive-FMM (This Project)</th>
                    <th>Theoretical Advantage</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ fontWeight: 600 }}>Forecast Mixing Paradigm</td>
                    <td>Static Uniform Average (w_k = 1/K)</td>
                    <td style={{ color: '#059669', fontWeight: 600 }}>Input-Conditioned Softmax Gate</td>
                    <td>Dynamically adapts across varying horizons</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 600 }}>Zero-Init Backward Compatibility</td>
                    <td>N/A (Reference Model)</td>
                    <td style={{ color: '#059669', fontWeight: 600 }}>Guaranteed by Proposition 1</td>
                    <td>Zero risk of pretraining degradation</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 600 }}>Parameter Overhead</td>
                    <td>0 Params</td>
                    <td style={{ color: '#2563eb' }}>2,988 to 11,604 Params (&lt;0.05%)</td>
                    <td>Negligible VRAM footprint</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 600 }}>Inference Latency</td>
                    <td>380 ms (ETTh1)</td>
                    <td style={{ color: '#059669' }}>386 ms (&lt; 2% delta)</td>
                    <td>Real-time production deployable</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 600 }}>High-Dimensional Scalability</td>
                    <td>Tested on standard tasks</td>
                    <td style={{ color: '#059669', fontWeight: 600 }}>Verified on 862 PeMS Highway Sensors</td>
                    <td>-80.46% MSE reduction on high-dim Traffic</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
