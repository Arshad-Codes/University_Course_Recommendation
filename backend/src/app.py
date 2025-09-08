import os
import warnings
warnings.filterwarnings("ignore")

from typing import List, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse
from sentence_transformers import SentenceTransformer, util
import torch
import boto3
from pathlib import Path


# -----------------------
# Config
# -----------------------
MODEL_DIR = os.getenv("MODEL_DIR", "./model")  # folder with your saved model (SentenceTransformer.save)
HF_FALLBACK_MODEL = os.getenv("HF_FALLBACK_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_DIR = os.getenv("DATA_DIR", "./data")

CAREER_CSV = os.path.join(DATA_DIR, "Career Dataset.csv")
PRIVATE_CSV = os.path.join(DATA_DIR, "private_universities.csv")
GOV_CSV = os.path.join(DATA_DIR, "government_universities.csv")

TOPK = int(os.getenv("TOPK", "5"))

# -----------------------
# FastAPI init + CORS
# -----------------------
app = FastAPI(title="AL Course/Career Recommender", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Prometheus metrics
# -----------------------
REQS = Counter("inference_requests_total", "Total inference requests", ["route"])
LAT  = Histogram("inference_latency_seconds", "Inference latency", ["route"])

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(generate_latest(), media_type="text/plain")

@app.get("/healthz")
def healthz():
    return {"ok": True}

# -----------------------
# Schemas
# -----------------------
class RecommendRequest(BaseModel):
    skill_input: str = Field(..., description="Free text describing skills/interests")
    gov_interest: Optional[str] = Field(default="no", description="'yes'/'no' to include gov uni recs")
    z_score: Optional[float] = Field(default=None, description="Student Z-score for gov filter (optional)")

class PrivateUniRec(BaseModel):
    University: str
    Degree: str
    Relevant_Field: str
    Link: Optional[str] = None
    Similarity_Score: float

class GovUniRec(BaseModel):
    University: str
    Degree: str
    Z_score_Program: float
    Stream: str
    Similarity_Score: float

class RecommendResponse(BaseModel):
    matched_skill_text: str
    matched_career: str
    matched_score: float
    private_universities: List[PrivateUniRec]
    government_universities: Optional[List[GovUniRec]] = None

# -----------------------
# Load model + data (startup)
# -----------------------
@app.on_event("startup")
def _load_everything():
    global model, device
    global career_df, private_unis, gov_unis_df
    global skill_embeddings, private_field_embeddings

    device = "cuda" if torch.cuda.is_available() else "cpu"



    
    
        # ----------------------------
    # Download model from S3 if specified
    # ----------------------------
    s3_uri = os.getenv("MODEL_S3_URI")
    if s3_uri:
        bucket_name = s3_uri.replace("s3://", "").split("/")[0]
        prefix = "/".join(s3_uri.replace("s3://", "").split("/")[1:])
        local_model_dir = Path(MODEL_DIR)
        local_model_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading model from {s3_uri} to {local_model_dir} ...")
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))

        # list objects in prefix
        resp = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if "Contents" not in resp:
            raise RuntimeError(f"No files found in {s3_uri}")

        for obj in resp["Contents"]:
            key = obj["Key"]
            if key.endswith("/"):  # skip folder marker
                continue
            local_path = local_model_dir / Path(key).name
            s3.download_file(bucket_name, key, str(local_path))
            print(f"Downloaded {key} -> {local_path}")

    
        # Try downloaded/local model folder first, else fallback to HF
    if os.path.isdir(MODEL_DIR) and any(os.scandir(MODEL_DIR)):
        model_path = MODEL_DIR
    else:
        model_path = HF_FALLBACK_MODEL
        print(f"Warning: MODEL_DIR '{MODEL_DIR}' not found or empty, falling back to {HF_FALLBACK_MODEL}")
    
    # Load SentenceTransformer once

    print(f"Loading model: {model_path}")
    model = SentenceTransformer(model_path, device=device)

    # Load CSVs
    def _safe_read(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required data file: {path}")
        df = pd.read_csv(path)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].fillna("").astype(str)
        return df

    career_df = _safe_read(CAREER_CSV)
    private_unis = _safe_read(PRIVATE_CSV)
    gov_unis_df = _safe_read(GOV_CSV)

    # Precompute embeddings (static corpora)
    if "Skill" not in career_df.columns or "Career" not in career_df.columns:
        raise ValueError("Career Dataset must contain 'Skill' and 'Career' columns")

    print("Encoding career skills...")
    skill_embeddings = model.encode(
        career_df["Skill"].tolist(),
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=device
    )

    print("Encoding private uni fields...")
    private_fields = private_unis.get("Relevant_Field", pd.Series([""]*len(private_unis))).tolist()
    private_field_embeddings = model.encode(
        private_fields,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=device
    )

    print("Startup completed.")

# -----------------------
# Helpers
# -----------------------
def _top_match_from_semantic_search(query_emb, corpus_emb, top_k=1):
    results = util.semantic_search(query_emb, corpus_emb, top_k=top_k)[0]
    return results  # list of {corpus_id, score}

# -----------------------
# Main endpoint
# -----------------------
@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    route = "/recommend"
    REQS.labels(route=route).inc()
    with LAT.labels(route=route).time():
        text = (req.skill_input or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="skill_input is required")

        # Encode query once
        query_emb = model.encode([text], convert_to_tensor=True, normalize_embeddings=True, device=device)

        # --- Career match ---
        career_results = _top_match_from_semantic_search(query_emb, skill_embeddings, top_k=TOPK)
        if not career_results:
            matched_career = career_df.iloc[0]["Career"]
            matched_skill_text = career_df.iloc[0]["Skill"]
            matched_score = 0.0
        else:
            top_match = career_results[0]
            row = career_df.iloc[top_match["corpus_id"]]
            matched_career = row["Career"]
            matched_skill_text = row["Skill"]
            matched_score = float(top_match["score"])

        # --- Private universities (top 3) ---
        private_results = _top_match_from_semantic_search(query_emb, private_field_embeddings, top_k=min(len(private_unis), 50))
        private_recs = []
        for res in private_results[:3]:
            row = private_unis.iloc[res["corpus_id"]]
            private_recs.append(PrivateUniRec(
                University=row.get("University",""),
                Degree=row.get("Degree",""),
                Relevant_Field=row.get("Relevant_Field",""),
                Link=row.get("Link",""),
                Similarity_Score=round(float(res["score"]), 2)
            ))

        # --- Government universities (optional) ---
        gov_recs = None
        if (req.gov_interest or "no").lower() in {"yes","y"}:
            if req.z_score is None:
                gov_recs = []
            else:
                # Filter by z_score threshold <= student's z
                filt = gov_unis_df[gov_unis_df["Z_score"].astype(float) <= float(req.z_score)].copy()
                if filt.empty:
                    # fallback to any
                    r0 = gov_unis_df.iloc[0]
                    gov_recs = [GovUniRec(
                        University=r0.get("Selected_University",""),
                        Degree=r0.get("Course",""),
                        Z_score_Program=float(r0.get("Z_score", 0.0)),
                        Stream=r0.get("Stream",""),
                        Similarity_Score=0.0
                    )]
                else:
                    # Use stream column for a coarse semantic match
                    gov_fields = filt["Stream"].fillna("").astype(str).tolist()
                    gov_field_embs = model.encode(
                        gov_fields,
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                        device=device
                    )
                    gov_results = _top_match_from_semantic_search(query_emb, gov_field_embs, top_k=min(len(filt), 50))
                    gov_recs = []
                    for res in gov_results[:3]:
                        row = filt.iloc[res["corpus_id"]]
                        gov_recs.append(GovUniRec(
                            University=row.get("Selected_University",""),
                            Degree=row.get("Course",""),
                            Z_score_Program=float(row.get("Z_score", 0.0)),
                            Stream=row.get("Stream",""),
                            Similarity_Score=round(float(res["score"]), 2)
                        ))

        return RecommendResponse(
            matched_skill_text=matched_skill_text,
            matched_career=matched_career,
            matched_score=round(matched_score, 2),
            private_universities=private_recs,
            government_universities=gov_recs
        )
