# CivicMind

**Case-based reasoning for civic incident prediction in a city.**

CivicMind is a memory-augmented decision support system for urban civic management. Given a situation description (weather, time, active conditions, ongoing events), it retrieves similar past incidents from a semantic memory store, traverses a causal-temporal knowledge graph to predict likely next events, and recommends interventions with full citation trails.

**Core philosophy:** *Between the input and output is memory, not a model.*

---

## System Architecture

```
                         ┌─────────────────────────┐
                         │  scripts/database.py     │  Synthetic data generator
                         │  (dataset gen)           │  10,000 realistic civic episodes
                         └────────┬─────────────────┘
                                  │
                         ┌────────v──────────┐
                         │  episodes.csv     │
                         │  28 columns       │
                         └───┬────────────┬──┘
                             │            │
                    ┌────────v─┐    ┌─────v────────┐
                    │build_    │    │ build_graph  │
                    │embeddings│    │ .py          │
                    │  .py     │    │ NetworkX     │
                    │FAISS L2  │    │ DiGraph      │
                    │index     │    │ 10K nodes    │
                    └───┬──────┘    │ 3.7K edges   │
                        │           └──────┬───────┘
                         v                  v
                ┌─────────────────────────────────┐
                │      INFERENCE ENGINE            │
                │      engine/inference.py         │
                │                                  │
                │  1. Encode situation → FAISS     │
                │  2. Walk graph from matches      │
                │  3. Compute triple-source risk    │
                │  4. Rank actions with citations   │
                │  5. Build reasoning trace         │
                └──────────────┬──────────────────┘
                               │
                               v
                 ┌─────────────────────────────────┐
                 │      WEB DASHBOARD               │
                 │      app/main.py                 │
                 │      FastAPI + HTML/JS           │
                 │      http://localhost:8000       │
                 ├─────────────────────────────────┤
                 │   POST /predict → inference     │
                 │   POST /summarize → Gemini LLM  │
                 └────────┬────────────┬───────────┘
                          │            │
              ┌───────────v─┐    ┌─────v──────────┐
              │  Render     │    │  Vercel         │
              │  (backend)  │    │  (frontend)     │
              └─────────────┘    └─────────────────┘
```

### Three Memory Layers

| Layer | Format | Size | Purpose |
|---|---|---|---|
| **Episodic** | `episodes.csv` | 2 MB | Raw incident data, 10,000 rows × 28 columns |
| **Semantic** | FAISS `IndexFlatL2` | 15 MB | 384-dim embeddings for similarity retrieval |
| **Graph** | NetworkX DiGraph (GraphML) | 3.1 MB | Causal (CONCEPT) + Temporal (PRECEDED) edges |

---

## Files

| File | Lines | Purpose |
|--- | --- | ---|
| `scripts/database.py` | 273 | Generates 10,000 synthetic civic episodes for Mumbai |
| `dataset/civicmind_episodes.csv` | 10,001 | The generated dataset |
| `memory/build_embeddings.py` | 55 | Embeds episodes with all-MiniLM-L6-v2 → FAISS index |
| `memory/build_graph.py` | 67 | Builds causal + temporal knowledge graph from CSV |
| `dataset/civicmind_memory.index` | 15 MB | FAISS L2 index (384-dim) |
| `dataset/episode_lookup.pkl` | 4.7 MB | Pickled DataFrame with memory_text field |
| `dataset/civicmind_graph.graphml` | 3.1 MB | NetworkX DiGraph (10,013 nodes, 3,701 edges) |
| `scripts/transition_stats.py` | 85 | Temporal transition analysis (confidence, delay stats) |
| `engine/inference.py` | 343 | Core inference engine with passive consolidation |
| `app/main.py` | 132 | FastAPI server: dashboard + /predict + /summarize APIs |
| `app/templates/index.html` | 448 | Dashboard UI: form, results, AI summary, reasoning trace |
| `app/static/civicmind-logo.svg` | 12 | CivicMind logo |
| `requirements.txt` | 10 | Python dependencies |
| `frontend/index.html` | 448 | Standalone frontend for Vercel deployment |
| `frontend/static/civicmind-logo.svg` | 12 | Logo for frontend deployment |
| `frontend/vercel.json` | 6 | Vercel rewrites config (proxies API to Render) |
| `render.yaml` | 9 | Render web service deployment config |
| `.env.example` | 1 | Template for GEMINI_API_KEY |

---

## Setup

```bash
cd Civic_Mind
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### API Key (for AI Summaries)

The dashboard can generate LLM summaries of predictions using Google Gemini. To enable this:

1. Get a free API key from https://aistudio.google.com/apikey
2. Copy `.env.example` to `.env` and add your key:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and set `GEMINI_API_KEY=your_key_here`.

The `/summarize` endpoint is optional — the dashboard works without it.

---

## How to Run

All commands run from the project root. Generate data first (or use the pre-generated CSV):

```bash
# Generate 10,000 synthetic episodes
python scripts/database.py

# Build FAISS embedding index (takes ~30s)
python memory/build_embeddings.py

# Build knowledge graph
python memory/build_graph.py
```

### Temporal transition analysis
```bash
python scripts/transition_stats.py
```
Outputs transition confidence scores, support counts, average delays, and standard deviations between event types from the PRECEDED graph edges.

### Launch the web dashboard
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in a browser. The sidebar lets you configure a situation (timestamp, area, weather, conditions, event), and the results panel displays top matches, risk scores, recommended actions, and a reasoning trace. After a prediction, click **✨ Generate AI Summary** to get a natural-language overview (requires `GEMINI_API_KEY`).

### Run inference on a situation
```bash
python engine/inference.py
```
The engine comes with a sample situation (Heavy Rain + Festival + Construction at Transit Hub) and outputs:
- **Risk scores** — per-event-type combined probability (retrieval × transition × concept fusion)
- **Recommended actions** — ranked interventions from matched episodes, cited by source
- **Reasoning trace** — the complete graph paths used for each prediction

To run on your own situation:

```python
from engine.inference import CivicMindInference

engine = CivicMindInference(k_matches=20)

result = engine.predict({
    "timestamp": "2026-06-17T07:42:00",
    "ward": "Transit Hub",
    "active_conditions": ["heavy_rain", "road_construction"],
    "weather_forecast": "rain_6h",
    "event_name": "City Music Festival",
    "expected_crowd": 12000,
})

print(result["risk_scores"])
print(result["actions"])
print(result["trace"])
```

---

## How the Inference Engine Works

### Input Normalization
Situation fields are mapped to internal ontologies:
- `weather_forecast` codes → 4 weather categories (Clear, Cloudy, Light Rain, Heavy Rain)
- `active_conditions` strings → graph concept node names via lookup table
- Timestamps → boolean peak_hour flag

### Step 1 — Semantic Retrieval (FAISS)
The situation is formatted as a text paragraph matching the embedding training format, encoded to 384-dim, and searched against the FAISS index for the top-20 most similar past episodes. L2 distance is converted to similarity via `1 / (1 + distance)`.

### Step 2 — Triple-Source Risk Scoring
Three independent scores are computed and fused:

| Signal | Source | Weight | Formula |
|---|---|---|---|
| **Retrieval** | FAISS matches | 0.5 | Σ sim for event / total sim |
| **Transition** | PRECEDED edges from matches | 0.3 | Σ sim × recency weight, normalized |
| **Concept** | CONCEPT edges from conditions | 0.2 | Σ edge weight, normalized |

Combined score = `0.5 × retrieval + 0.3 × transition + 0.2 × concept`

### Step 3 — Action Ranking
Interventions from matched episodes are ranked by `similarity × outcome_score`, deduplicated, and cited with source episode IDs. "No Action" interventions are excluded.

### Step 4 — Reasoning Trace
Two path types are constructed:
- **CONCEPT paths**: condition → event with conditional probability `P(event | condition)`
- **PRECEDED paths**: episode → next_event with similarity and time gap

---

## The Knowledge Graph

Built by `memory/build_graph.py`:

| Component | Count | Description |
|---|---|---|
| Episode nodes | 10,000 | One per CSV row with event_type, area_type, weather, severity, timestamp |
| Concept nodes | 13 | 9 event types + 4 cause values (Heavy Rain, Road Construction, Festival Crowd, High Traffic Volume) |
| PRECEDED edges | 3,695 | Episode → Episode, same area_type, within 6h, with time_gap_hours |
| CONCEPT edges | 6 | Cause value → Event type, weighted by co-occurrence frequency |

Analyzed by `scripts/transition_stats.py` which computes transition confidence, support, avg delay, and std deviation.

---

## Design Decisions

- **No ML training**: The system never updates model weights. All learning is adding cases to the episodic store. Predictions come from retrieval + graph traversal, not parameterized inference.
- **Full provenance**: Every prediction cites its source episodes. The system can always show its work.
- **Graceful degradation**: Low-similarity matches produce low-confidence predictions. The concept prior provides a fallback when retrieval is sparse.
- **Synthetic data**: The 10,000-episode dataset is generated with realistic probability distributions (weather-weighted, context-aware event determination). The generation logic is fully deterministic given the random seed, making the system reproducible.

---

## Deployment

The app is split into a static frontend (Vercel) and a Python backend (Render).

### Render (Backend)

1. Push the repo to GitHub
2. Go to https://dashboard.render.com → New Web Service → connect your repo
3. Render auto-detects `render.yaml`. Set the `GEMINI_API_KEY` environment variable in the dashboard.
4. Deploy. You'll get a URL like `https://civicmind-backend.onrender.com`

### Vercel (Frontend)

1. Edit `frontend/vercel.json` and replace `REPLACE_ME_WITH_YOUR_RENDER_URL` with your actual Render URL
2. Import the repo into Vercel, set the root directory to `frontend/`
3. Deploy

Vercel rewrites proxy `/predict` and `/summarize` requests to the Render backend, so the browser sees same-origin requests.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.14 |
| Embeddings | sentence-transformers / all-MiniLM-L6-v2 |
| Vector Search | FAISS (IndexFlatL2) |
| Graph | NetworkX |
| Data | pandas, numpy |
| ML Backend | PyTorch (CUDA-capable) |
| Dashboard | FastAPI, Uvicorn, HTML/JS |
| LLM Summaries | google-genai (Gemini 2.0 Flash Lite) |
| Config | python-dotenv, .env |

---

## Project Status

Fully functional retrieval and inference pipeline. Next steps:

- [x] Web dashboard (FastAPI + HTML/JS) for interactive situation input
- [x] LLM summary layer (Gemini 2.0 Flash Lite) with error handling
- [x] Consolidation loop: increment retrieval_count, update confidence_score after each inference
- [ ] Real city data integration
- [ ] Cross-validation of fusion weights (0.5/0.3/0.2)
