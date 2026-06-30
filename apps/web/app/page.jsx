"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BrainCircuit,
  CheckCircle2,
  Database,
  GitCompareArrows,
  LoaderCircle,
  Play,
  RefreshCw,
  Save,
  Send,
  Server,
  SlidersHorizontal,
  XCircle
} from "lucide-react";

const DEFAULT_API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

const FALLBACK_MODELS = [
  {
    model_id: "random-tiny-byte",
    description: "Random tiny byte-level GPT.",
    state: "random-untrained",
    parameters: 0,
    context_length: 64,
    tokenizer: "byte"
  }
];

const TABS = [
  { id: "chat", label: "Chat", icon: Send },
  { id: "training", label: "Training", icon: Activity },
  { id: "checkpoints", label: "Checkpoints", icon: Save }
];

export default function Home() {
  const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_API_BASE_URL);
  const [apiStatus, setApiStatus] = useState("checking");
  const [statusMessage, setStatusMessage] = useState("Checking API");
  const [activeTab, setActiveTab] = useState("chat");
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [checkpoints, setCheckpoints] = useState([]);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const [message, setMessage] = useState("Every effort moves you");
  const [chatModelId, setChatModelId] = useState("random-tiny-byte");
  const [maxNewTokens, setMaxNewTokens] = useState(24);
  const [temperature, setTemperature] = useState(0);
  const [chatResult, setChatResult] = useState(null);
  const [chatError, setChatError] = useState("");
  const [isChatting, setIsChatting] = useState(false);

  const [leftModelId, setLeftModelId] = useState("random-tiny-byte");
  const [rightModelId, setRightModelId] = useState("trained-tiny-byte");
  const [compareResults, setCompareResults] = useState(null);
  const [isComparing, setIsComparing] = useState(false);

  const [datasetId, setDatasetId] = useState("every-effort");
  const [outputModelId, setOutputModelId] = useState("trained-tiny-byte");
  const [trainingSteps, setTrainingSteps] = useState(80);
  const [evalEvery, setEvalEvery] = useState(10);
  const [loadWhenComplete, setLoadWhenComplete] = useState(true);
  const [trainingJob, setTrainingJob] = useState(null);
  const [trainingError, setTrainingError] = useState("");
  const [isStartingTraining, setIsStartingTraining] = useState(false);

  const [loadingCheckpointId, setLoadingCheckpointId] = useState("");
  const [checkpointError, setCheckpointError] = useState("");

  const normalizedApiBaseUrl = useMemo(
    () => apiBaseUrl.replace(/\/+$/, ""),
    [apiBaseUrl]
  );

  const modelOptions = useMemo(() => {
    const base = models.length > 0 ? models : FALLBACK_MODELS;
    const byId = new Map(base.map((model) => [model.model_id, model]));
    [chatModelId, leftModelId, rightModelId].forEach((modelId) => {
      if (modelId && !byId.has(modelId)) {
        byId.set(modelId, {
          model_id: modelId,
          description: "Not loaded yet.",
          state: "not-loaded",
          parameters: 0,
          context_length: 0,
          tokenizer: "byte"
        });
      }
    });
    return Array.from(byId.values());
  }, [chatModelId, leftModelId, models, rightModelId]);

  const lastProgress = trainingJob?.progress?.at(-1);
  const progressPercent = lastProgress
    ? Math.min(100, Math.round((lastProgress.step / lastProgress.max_steps) * 100))
    : 0;
  const trainingIsActive =
    trainingJob?.status === "queued" || trainingJob?.status === "running";

  useEffect(() => {
    const stored = window.localStorage.getItem("llm-abc-api-base-url");
    if (stored) {
      setApiBaseUrl(stored);
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem("llm-abc-api-base-url", apiBaseUrl);
  }, [apiBaseUrl]);

  useEffect(() => {
    refreshAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [normalizedApiBaseUrl]);

  useEffect(() => {
    if (!trainingIsActive || !trainingJob?.job_id) {
      return undefined;
    }

    const timeoutId = window.setTimeout(async () => {
      try {
        const job = await requestJson(`/training/jobs/${trainingJob.job_id}`);
        setTrainingJob(job);
        if (job.status === "succeeded") {
          await refreshAll();
          if (job.result?.loaded_model?.model_id) {
            setChatModelId(job.result.loaded_model.model_id);
            setRightModelId(job.result.loaded_model.model_id);
          }
        }
      } catch (error) {
        setTrainingError(error.message);
      }
    }, 1000);

    return () => window.clearTimeout(timeoutId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainingIsActive, trainingJob?.job_id, trainingJob?.updated_at]);

  async function requestJson(path, options = {}) {
    const response = await fetch(`${normalizedApiBaseUrl}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {})
      }
    });
    const text = await response.text();
    const data = text ? JSON.parse(text) : null;

    if (!response.ok) {
      throw new Error(data?.detail || response.statusText);
    }
    return data;
  }

  async function refreshAll() {
    setIsRefreshing(true);
    setStatusMessage("Checking API");

    try {
      await requestJson("/health");
      setApiStatus("online");
      setStatusMessage("API online");
    } catch (error) {
      setApiStatus("offline");
      setStatusMessage(error.message);
      setIsRefreshing(false);
      return;
    }

    const [modelsResult, datasetsResult, checkpointsResult] =
      await Promise.allSettled([
        requestJson("/models"),
        requestJson("/training/datasets"),
        requestJson("/checkpoints")
      ]);

    if (modelsResult.status === "fulfilled") {
      setModels(modelsResult.value);
    }
    if (datasetsResult.status === "fulfilled") {
      setDatasets(datasetsResult.value);
      if (datasetsResult.value[0]?.dataset_id) {
        setDatasetId((current) => current || datasetsResult.value[0].dataset_id);
      }
    }
    if (checkpointsResult.status === "fulfilled") {
      setCheckpoints(checkpointsResult.value);
    }

    setIsRefreshing(false);
  }

  async function sendChat() {
    setIsChatting(true);
    setChatError("");
    setChatResult(null);

    try {
      const result = await requestJson("/chat", {
        method: "POST",
        body: JSON.stringify({
          model_id: chatModelId,
          message,
          max_new_tokens: Number(maxNewTokens),
          temperature: Number(temperature),
          include_prompt: false
        })
      });
      setChatResult(result);
    } catch (error) {
      setChatError(error.message);
    } finally {
      setIsChatting(false);
    }
  }

  async function compareModels() {
    setIsComparing(true);
    setCompareResults(null);

    const payload = {
      message,
      max_new_tokens: Number(maxNewTokens),
      temperature: Number(temperature),
      include_prompt: false
    };

    const [left, right] = await Promise.allSettled([
      requestJson("/chat", {
        method: "POST",
        body: JSON.stringify({ ...payload, model_id: leftModelId })
      }),
      requestJson("/chat", {
        method: "POST",
        body: JSON.stringify({ ...payload, model_id: rightModelId })
      })
    ]);

    setCompareResults({
      left: resultFromSettled(left),
      right: resultFromSettled(right)
    });
    setIsComparing(false);
  }

  async function startTraining() {
    setIsStartingTraining(true);
    setTrainingError("");
    setTrainingJob(null);
    setActiveTab("training");

    try {
      const job = await requestJson("/training/jobs", {
        method: "POST",
        body: JSON.stringify({
          dataset_id: datasetId,
          base_model_id: "random-tiny-byte",
          output_model_id: outputModelId,
          max_steps: Number(trainingSteps),
          eval_every: Number(evalEvery),
          batch_size: 4,
          block_size: 32,
          learning_rate: 0.003,
          sample_prompt: message,
          load_when_complete: loadWhenComplete
        })
      });
      setTrainingJob(job);
    } catch (error) {
      setTrainingError(error.message);
    } finally {
      setIsStartingTraining(false);
    }
  }

  async function loadCheckpoint(checkpoint) {
    setLoadingCheckpointId(checkpoint.checkpoint_id);
    setCheckpointError("");

    try {
      const loaded = await requestJson("/models/load", {
        method: "POST",
        body: JSON.stringify({
          checkpoint_id: checkpoint.checkpoint_id,
          model_id: checkpoint.model_id
        })
      });
      await refreshAll();
      setChatModelId(loaded.model_id);
      setRightModelId(loaded.model_id);
      setActiveTab("chat");
    } catch (error) {
      setCheckpointError(error.message);
    } finally {
      setLoadingCheckpointId("");
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <BrainCircuit aria-hidden="true" />
          <div>
            <h1>LLM ABC Console</h1>
            <p>Minimal Web UI learning console</p>
          </div>
        </div>

        <div className="connection-bar">
          <label className="api-field">
            <span>API</span>
            <input
              value={apiBaseUrl}
              onChange={(event) => setApiBaseUrl(event.target.value)}
              aria-label="API base URL"
            />
          </label>
          <span className={`status-pill ${apiStatus}`}>
            {apiStatus === "online" ? (
              <CheckCircle2 aria-hidden="true" />
            ) : apiStatus === "offline" ? (
              <XCircle aria-hidden="true" />
            ) : (
              <LoaderCircle aria-hidden="true" className="spin" />
            )}
            {statusMessage}
          </span>
          <button
            className="icon-button"
            type="button"
            onClick={refreshAll}
            disabled={isRefreshing}
            title="Refresh API data"
            aria-label="Refresh API data"
          >
            <RefreshCw aria-hidden="true" className={isRefreshing ? "spin" : ""} />
          </button>
        </div>
      </header>

      <nav className="tabs" aria-label="Console views">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              type="button"
              className={activeTab === tab.id ? "tab active" : "tab"}
              onClick={() => setActiveTab(tab.id)}
            >
              <Icon aria-hidden="true" />
              {tab.label}
            </button>
          );
        })}
      </nav>

      <div className="workspace">
        <section className="main-surface">
          {activeTab === "chat" && (
            <ChatView
              chatError={chatError}
              chatModelId={chatModelId}
              chatResult={chatResult}
              compareModels={compareModels}
              compareResults={compareResults}
              isChatting={isChatting}
              isComparing={isComparing}
              leftModelId={leftModelId}
              maxNewTokens={maxNewTokens}
              message={message}
              modelOptions={modelOptions}
              rightModelId={rightModelId}
              sendChat={sendChat}
              setChatModelId={setChatModelId}
              setLeftModelId={setLeftModelId}
              setMaxNewTokens={setMaxNewTokens}
              setMessage={setMessage}
              setRightModelId={setRightModelId}
              setTemperature={setTemperature}
              temperature={temperature}
            />
          )}

          {activeTab === "training" && (
            <TrainingView
              datasetId={datasetId}
              datasets={datasets}
              evalEvery={evalEvery}
              isStartingTraining={isStartingTraining}
              lastProgress={lastProgress}
              loadWhenComplete={loadWhenComplete}
              outputModelId={outputModelId}
              progressPercent={progressPercent}
              setDatasetId={setDatasetId}
              setEvalEvery={setEvalEvery}
              setLoadWhenComplete={setLoadWhenComplete}
              setOutputModelId={setOutputModelId}
              setTrainingSteps={setTrainingSteps}
              startTraining={startTraining}
              trainingError={trainingError}
              trainingJob={trainingJob}
              trainingSteps={trainingSteps}
            />
          )}

          {activeTab === "checkpoints" && (
            <CheckpointView
              checkpointError={checkpointError}
              checkpoints={checkpoints}
              loadCheckpoint={loadCheckpoint}
              loadingCheckpointId={loadingCheckpointId}
              refreshAll={refreshAll}
            />
          )}
        </section>

        <aside className="model-rail">
          <section className="rail-section">
            <div className="section-title">
              <Server aria-hidden="true" />
              <h2>Models</h2>
            </div>
            <div className="stack">
              {modelOptions.map((model) => (
                <ModelRow key={model.model_id} model={model} />
              ))}
            </div>
          </section>

          <section className="rail-section">
            <div className="section-title">
              <Database aria-hidden="true" />
              <h2>Datasets</h2>
            </div>
            <div className="stack">
              {(datasets.length ? datasets : [{ dataset_id: "every-effort", exists: false }]).map(
                (dataset) => (
                  <div className="data-row" key={dataset.dataset_id}>
                    <span>{dataset.dataset_id}</span>
                    <span className={dataset.exists ? "state good" : "state muted"}>
                      {dataset.exists ? "ready" : "missing"}
                    </span>
                  </div>
                )
              )}
            </div>
          </section>
        </aside>
      </div>
    </main>
  );
}

function ChatView({
  chatError,
  chatModelId,
  chatResult,
  compareModels,
  compareResults,
  isChatting,
  isComparing,
  leftModelId,
  maxNewTokens,
  message,
  modelOptions,
  rightModelId,
  sendChat,
  setChatModelId,
  setLeftModelId,
  setMaxNewTokens,
  setMessage,
  setRightModelId,
  setTemperature,
  temperature
}) {
  return (
    <div className="view-stack">
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Chat</h2>
            <p>Prompt the selected model.</p>
          </div>
          <button
            className="primary-button"
            type="button"
            onClick={sendChat}
            disabled={isChatting || !message.trim()}
          >
            {isChatting ? (
              <LoaderCircle aria-hidden="true" className="spin" />
            ) : (
              <Send aria-hidden="true" />
            )}
            Send
          </button>
        </div>

        <div className="form-grid">
          <label className="field wide">
            <span>Prompt</span>
            <textarea
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              rows={4}
            />
          </label>

          <label className="field">
            <span>Model</span>
            <select
              value={chatModelId}
              onChange={(event) => setChatModelId(event.target.value)}
            >
              {modelOptions.map((model) => (
                <option key={model.model_id} value={model.model_id}>
                  {model.model_id}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Max tokens</span>
            <input
              type="number"
              min="1"
              max="200"
              value={maxNewTokens}
              onChange={(event) => setMaxNewTokens(event.target.value)}
            />
          </label>

          <label className="field">
            <span>Temperature</span>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={temperature}
              onChange={(event) => setTemperature(event.target.value)}
            />
            <strong>{Number(temperature).toFixed(1)}</strong>
          </label>
        </div>

        {chatError && <div className="error-line">{chatError}</div>}

        {chatResult && (
          <div className="output-box">
            <div className="metrics">
              <span>prompt tokens {chatResult.prompt_tokens}</span>
              <span>generated {chatResult.tokens_generated}</span>
            </div>
            <pre>{formatText(chatResult.reply)}</pre>
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Compare</h2>
            <p>Run the same prompt against two models.</p>
          </div>
          <button
            className="secondary-button"
            type="button"
            onClick={compareModels}
            disabled={isComparing || !message.trim()}
          >
            {isComparing ? (
              <LoaderCircle aria-hidden="true" className="spin" />
            ) : (
              <GitCompareArrows aria-hidden="true" />
            )}
            Compare
          </button>
        </div>

        <div className="compare-controls">
          <label className="field">
            <span>Left model</span>
            <select
              value={leftModelId}
              onChange={(event) => setLeftModelId(event.target.value)}
            >
              {modelOptions.map((model) => (
                <option key={model.model_id} value={model.model_id}>
                  {model.model_id}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Right model</span>
            <select
              value={rightModelId}
              onChange={(event) => setRightModelId(event.target.value)}
            >
              {modelOptions.map((model) => (
                <option key={model.model_id} value={model.model_id}>
                  {model.model_id}
                </option>
              ))}
            </select>
          </label>
        </div>

        {compareResults && (
          <div className="comparison-grid">
            <ComparisonColumn title={leftModelId} result={compareResults.left} />
            <ComparisonColumn title={rightModelId} result={compareResults.right} />
          </div>
        )}
      </section>
    </div>
  );
}

function TrainingView({
  datasetId,
  datasets,
  evalEvery,
  isStartingTraining,
  lastProgress,
  loadWhenComplete,
  outputModelId,
  progressPercent,
  setDatasetId,
  setEvalEvery,
  setLoadWhenComplete,
  setOutputModelId,
  setTrainingSteps,
  startTraining,
  trainingError,
  trainingJob,
  trainingSteps
}) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h2>Training</h2>
          <p>Train the tiny model and save a checkpoint.</p>
        </div>
        <button
          className="primary-button"
          type="button"
          onClick={startTraining}
          disabled={isStartingTraining || trainingJob?.status === "running"}
        >
          {isStartingTraining ? (
            <LoaderCircle aria-hidden="true" className="spin" />
          ) : (
            <Play aria-hidden="true" />
          )}
          Start
        </button>
      </div>

      <div className="form-grid">
        <label className="field">
          <span>Dataset</span>
          <select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>
            {(datasets.length ? datasets : [{ dataset_id: "every-effort" }]).map(
              (dataset) => (
                <option key={dataset.dataset_id} value={dataset.dataset_id}>
                  {dataset.dataset_id}
                </option>
              )
            )}
          </select>
        </label>

        <label className="field">
          <span>Output model</span>
          <input
            value={outputModelId}
            onChange={(event) => setOutputModelId(event.target.value)}
          />
        </label>

        <label className="field">
          <span>Steps</span>
          <input
            type="number"
            min="1"
            max="2000"
            value={trainingSteps}
            onChange={(event) => setTrainingSteps(event.target.value)}
          />
        </label>

        <label className="field">
          <span>Eval every</span>
          <input
            type="number"
            min="1"
            max="500"
            value={evalEvery}
            onChange={(event) => setEvalEvery(event.target.value)}
          />
        </label>

        <label className="toggle-row wide">
          <input
            type="checkbox"
            checked={loadWhenComplete}
            onChange={(event) => setLoadWhenComplete(event.target.checked)}
          />
          <span>Load checkpoint when complete</span>
        </label>
      </div>

      {trainingError && <div className="error-line">{trainingError}</div>}

      {trainingJob && (
        <div className="training-status">
          <div className="job-header">
            <span className={`state ${stateClass(trainingJob.status)}`}>
              {trainingJob.status}
            </span>
            <span>{trainingJob.job_id}</span>
          </div>

          <div className="progress-track" aria-label="Training progress">
            <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
          </div>

          <div className="metrics">
            <span>step {lastProgress?.step || 0}</span>
            <span>loss {lastProgress?.loss ?? "-"}</span>
            <span>tokens {lastProgress?.tokens_seen || 0}</span>
          </div>

          {trainingJob.progress?.length > 0 && (
            <div className="progress-list">
              {trainingJob.progress.map((event) => (
                <div key={`${event.step}-${event.tokens_seen}`} className="progress-row">
                  <span>step {event.step}</span>
                  <span>loss {event.loss}</span>
                  <span>{event.tokens_seen} tokens</span>
                </div>
              ))}
            </div>
          )}

          {trainingJob.result?.training_summary && (
            <div className="comparison-grid">
              <OutputColumn
                title="Before"
                text={trainingJob.result.training_summary.before_sample}
              />
              <OutputColumn
                title="After"
                text={trainingJob.result.training_summary.sample_text}
              />
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function CheckpointView({
  checkpointError,
  checkpoints,
  loadCheckpoint,
  loadingCheckpointId,
  refreshAll
}) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h2>Checkpoints</h2>
          <p>Load saved full model snapshots.</p>
        </div>
        <button className="secondary-button" type="button" onClick={refreshAll}>
          <RefreshCw aria-hidden="true" />
          Refresh
        </button>
      </div>

      {checkpointError && <div className="error-line">{checkpointError}</div>}

      <div className="checkpoint-list">
        {checkpoints.length === 0 && (
          <div className="empty-state">
            <Save aria-hidden="true" />
            <span>No checkpoints yet</span>
          </div>
        )}

        {checkpoints.map((checkpoint) => (
          <article className="checkpoint-row" key={checkpoint.checkpoint_id}>
            <div>
              <h3>{checkpoint.model_id}</h3>
              <p>{checkpoint.checkpoint_id}</p>
              <div className="metrics">
                <span>base {checkpoint.base_model_id}</span>
                <span>loss {checkpoint.training_summary?.final_loss ?? "-"}</span>
                <span>tokens {checkpoint.training_summary?.tokens_seen ?? "-"}</span>
              </div>
            </div>
            <button
              className="secondary-button"
              type="button"
              onClick={() => loadCheckpoint(checkpoint)}
              disabled={loadingCheckpointId === checkpoint.checkpoint_id}
            >
              {loadingCheckpointId === checkpoint.checkpoint_id ? (
                <LoaderCircle aria-hidden="true" className="spin" />
              ) : (
                <SlidersHorizontal aria-hidden="true" />
              )}
              Load
            </button>
          </article>
        ))}
      </div>
    </section>
  );
}

function ModelRow({ model }) {
  return (
    <div className="model-row">
      <div>
        <strong>{model.model_id}</strong>
        <span>{model.state}</span>
      </div>
      <div className="metrics compact">
        <span>{formatNumber(model.parameters)} params</span>
        <span>{model.context_length || "-"} ctx</span>
      </div>
    </div>
  );
}

function ComparisonColumn({ title, result }) {
  if (result.error) {
    return (
      <div className="output-box">
        <h3>{title}</h3>
        <div className="error-line">{result.error}</div>
      </div>
    );
  }

  return (
    <div className="output-box">
      <h3>{title}</h3>
      <div className="metrics">
        <span>prompt tokens {result.data.prompt_tokens}</span>
        <span>generated {result.data.tokens_generated}</span>
      </div>
      <pre>{formatText(result.data.reply)}</pre>
    </div>
  );
}

function OutputColumn({ title, text }) {
  return (
    <div className="output-box">
      <h3>{title}</h3>
      <pre>{formatText(text)}</pre>
    </div>
  );
}

function resultFromSettled(result) {
  if (result.status === "fulfilled") {
    return { data: result.value };
  }
  return { error: result.reason.message };
}

function formatText(value) {
  return JSON.stringify(value ?? "");
}

function formatNumber(value) {
  if (!value) {
    return "-";
  }
  return new Intl.NumberFormat("en-US").format(value);
}

function stateClass(status) {
  if (status === "succeeded") {
    return "good";
  }
  if (status === "failed") {
    return "bad";
  }
  return "active-state";
}
