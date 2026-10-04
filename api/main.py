import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# src qovlugunu import yoluna elave edirik
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from predict import Predictor  # noqa: E402

MAX_CHARS = 2000
LANGS = ("en", "az")

state = {}


@asynccontextmanager
async def lifespan(app):
    # Server isleyende her iki dilin modeli bir defe yuklenir
    state["predictors"] = {lang: Predictor(lang) for lang in LANGS}
    yield
    state.clear()


app = FastAPI(title="Sentiment API", version="0.2.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)
    lang: Literal["en", "az"] = "en"


class PredictResponse(BaseModel):
    label: str
    confidence: float
    known_words: int
    total_words: int
    reliable: bool
    lang: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "predictors" in state,
        "languages": list(state.get("predictors", {})),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Metn bos ola bilmez.")

    result = state["predictors"][request.lang].predict(text)
    total = result["total_words"]
    result["reliable"] = total > 0 and result["known_words"] / total >= 0.5
    return result