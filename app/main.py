import os
import sys
import json
from dotenv import load_dotenv
import numpy as np
import google.genai as genai
from google.genai.types import GenerateContentConfig
from google.genai import errors as genai_errors
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.inference import CivicMindInference

app = FastAPI(title="CivicMind Dashboard")

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

engine = CivicMindInference(k_matches=20)

class SituationInput(BaseModel):
    timestamp: str
    ward: str = "Residential"
    weather_forecast: str = "clear"
    active_conditions: list[str] = []
    event_name: str | None = None
    expected_crowd: int | None = None

class SummaryInput(BaseModel):
    query_text: str
    risk_scores: list[dict]
    actions: list[dict]
    trace: list[dict]

@app.get("/", response_class=HTMLResponse)
async def index():
    path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(path) as f:
        return HTMLResponse(f.read())

@app.post("/predict")
async def predict(situation: SituationInput):
    data = situation.model_dump(exclude_none=True)
    result = engine.predict(data)
    body = {
        "query_text": result["query_text"],
        "risk_scores": result["risk_scores"],
        "actions": result["actions"],
        "trace": result["trace"],
        "top_matches": result["top_matches"],
    }
    return JSONResponse(content=json.loads(json.dumps(body, cls=NumpyEncoder)))

@app.post("/summarize")
async def summarize(data: SummaryInput):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY not set. Set it as an environment variable.")

    prompt = _build_summary_prompt(data)
    try:
        response = _client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt,
            config=GenerateContentConfig(temperature=0.3),
        )
    except genai_errors.ClientError as e:
        detail = str(e)
        if "RESOURCE_EXHAUSTED" in detail:
            raise HTTPException(status_code=429, detail=f"Gemini API quota exhausted. Try again later or use a different API key. ({e})")
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")
    return {"summary": response.text.strip()}

def _build_summary_prompt(data: SummaryInput) -> str:
    risks = "\n".join(
        f"  - {r['event']}: combined={r['combined_score']:.3f} "
        f"(retrieval={r['retrieval_score']:.3f}, transition={r['transition_score']:.3f}, concept={r['concept_score']:.3f})"
        for r in data.risk_scores
    )
    actions = "\n".join(
        f"  - {a['action']} (confidence={a['confidence']:.3f}, target={a['target_event']}, sim={a['similarity']:.3f}, outcome={a['outcome_score']:.2f})"
        for a in data.actions
    )
    trace = "\n".join(
        f"  - {t['type']}: {t['from']} → {t['to']} (confidence={t.get('confidence', '')}, time_gap={t.get('time_gap_hours', '')})"
        for t in data.trace
    )

    return f"""You are a civic incident analyst. Given the following situation analysis from a case-based reasoning engine, write a concise 2-3 sentence summary. Cover: what the situation is, the highest-risk events, and the most important recommended actions.

Situation:
{data.query_text}

Risk Scores:
{risks or '  (none)'}

Recommended Actions:
{actions or '  (none)'}

Reasoning Trace:
{trace or '  (none)'}

Summary:"""

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
