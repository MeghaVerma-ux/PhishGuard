
from pathlib import Path
import logging
import json
import math
import re
import ipaddress
from urllib.parse import urlsplit, parse_qsl
import joblib
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, HttpUrl

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "model.joblib"
HTML_PATH = BASE_DIR / "static" / "index.html"
SAMPLE_PATH = BASE_DIR / "data" / "sample_input.json"
IMPORTANCE_PATH = BASE_DIR / "results" / "feature_importance.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PhishGuard - Phishing Detection",
    description="Ensemble Learning Based Website Classification System",
    version="1.1.0"
)

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
feature_names = list(bundle["feature_names"])
best_model_name = bundle["best_model_name"]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float]


class ScanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str


def entropy(value):
    if not value:
        return 0.0

    probabilities = [
        value.count(char) / len(value)
        for char in set(value)
    ]

    return -sum(p * math.log2(p) for p in probabilities if p > 0)


def extract_url_features(raw_url):
    raw_url = raw_url.strip()

    if not raw_url:
        raise HTTPException(status_code=422, detail="Please enter a URL.")

    if len(raw_url) > 2048:
        raise HTTPException(status_code=422, detail="URL is too long.")

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw_url):
        raw_url = "http://" + raw_url

    try:
        parsed = urlsplit(raw_url)
        hostname = parsed.hostname or ""
        if not hostname:
            raise ValueError("Invalid hostname")
    except Exception:
        raise HTTPException(status_code=422, detail="Please enter a valid website URL.")

    domain = hostname.lower()
    path = parsed.path or ""
    query = parsed.query or ""
    full_url = raw_url

    try:
        ipaddress.ip_address(domain)
        is_ip = 1
    except ValueError:
        is_ip = 0

    labels = domain.split(".")
    tld = labels[-1] if len(labels) > 1 else ""
    domain_name = labels[-2] if len(labels) > 1 else domain

    segments = [part for part in path.split("/") if part]
    first_dir = segments[0] if segments else ""

    query_pairs = parse_qsl(query, keep_blank_values=True)
    query_values = [value for _, value in query_pairs]
    query_value_avg = (
        sum(len(value) for value in query_values) / len(query_values)
        if query_values else 0
    )

    total_chars = max(len(full_url), 1)
    letters = sum(char.isalpha() for char in full_url)
    digits = sum(char.isdigit() for char in full_url)
    special = sum(not char.isalnum() for char in full_url)

    suspicious_tlds = {
        "zip", "mov", "top", "xyz", "click", "country",
        "gq", "tk", "work", "support", "fit", "rest",
        "kim", "loan", "party", "review"
    }

    suspicious_words = [
        "login", "secure", "verify", "account",
        "update", "bank", "cloud"
    ]

    common_brands = [
        "paypal", "google", "microsoft", "apple", "amazon",
        "facebook", "instagram", "netflix", "whatsapp",
        "linkedin", "twitter", "telegram", "bank"
    ]

    features = {name: None for name in feature_names}

    computed = {
        "url_length": len(full_url),
        "domain_length": len(domain),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "first_dir_length": len(first_dir),
        "tld_length": len(tld),
        "tld_length_domain": len(tld),
        "url_depth": len(segments),
        "query_length": len(query),
        "path_segments_count": len(segments),
        "num_digits": digits,
        "num_letters": letters,
        "num_special_chars": special,
        "num_dots": full_url.count("."),
        "num_hyphens": full_url.count("-"),
        "num_at": full_url.count("@"),
        "num_percent": full_url.count("%"),
        "num_equals": full_url.count("="),
        "num_question": full_url.count("?"),
        "num_ampersand": full_url.count("&"),
        "num_hash": full_url.count("#"),
        "num_underscore": full_url.count("_"),
        "num_special": special,
        "num_slash": full_url.count("/"),
        "num_params": len(query_pairs),
        "entropy_url": entropy(full_url),
        "entropy_hostname": entropy(hostname),
        "entropy_domain": entropy(domain_name),
        "entropy_path": entropy(path),
        "query_entropy": entropy(query),
        "ratio_digits": digits / total_chars,
        "ratio_letters": letters / total_chars,
        "ratio_special_chars": special / total_chars,
        "uppercase_ratio": sum(c.isupper() for c in full_url) / total_chars,
        "lowercase_ratio": sum(c.islower() for c in full_url) / total_chars,
        "is_ip_address": is_ip,
        "starts_with_ip": is_ip,
        "is_suspicious_tld": int(tld.lower() in suspicious_tlds),
        "uses_https": int(parsed.scheme.lower() == "https"),
        "has_www": int(domain.startswith("www.")),
        "unusual_double_slash": int("//" in path),
        "multiple_http": int(full_url.lower().count("http") > 1),
        "contains_port_number": int(parsed.port is not None),
        "path_has_encoded_chars": int("%" in path),
        "query_has_base64": int(bool(re.search(r"[A-Za-z0-9+/]{20,}={0,2}", query))),
        "contains_login": int("login" in full_url.lower()),
        "contains_secure": int("secure" in full_url.lower()),
        "contains_verify": int("verify" in full_url.lower()),
        "contains_account": int("account" in full_url.lower()),
        "contains_update": int("update" in full_url.lower()),
        "contains_bank": int("bank" in full_url.lower()),
        "contains_cloud": int("cloud" in full_url.lower()),
        "contains_brand": int(any(brand in domain for brand in common_brands)),
        "query_key_count": len(query_pairs),
        "query_value_length_avg": query_value_avg,
        "success": 1,
        "whois_missing": 1,
    }

    for name, value in computed.items():
        if name in features:
            features[name] = value

    # DNS and WHOIS attributes are deliberately left unknown.
    # The trained pipeline's SimpleImputer handles these missing values.

    return features


def run_prediction(received):
    values = [received.get(name, np.nan) for name in feature_names]

    input_df = pd.DataFrame([values], columns=feature_names)
    prediction = int(pipeline.predict(input_df)[0])

    confidence = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(input_df)[0]
        confidence = round(float(max(probabilities)) * 100, 2)

    return prediction, confidence


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": best_model_name,
        "features_required": len(feature_names)
    }


@app.get("/")
def home():
    if not HTML_PATH.exists() or HTML_PATH.stat().st_size == 0:
        raise HTTPException(status_code=404, detail="Website file missing or empty")
    return FileResponse(HTML_PATH)


@app.get("/sample")
def sample():
    try:
        if not SAMPLE_PATH.exists():
            raise HTTPException(status_code=404, detail="Sample input file not found")

        with open(SAMPLE_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Sample loading failed")
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/feature-importance")
def get_feature_importance():
    try:
        if not IMPORTANCE_PATH.exists():
            raise HTTPException(status_code=404, detail="Feature importance file not found")

        df = pd.read_csv(IMPORTANCE_PATH).head(10)
        records = df.replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient="records")

        return {
            "model": best_model_name,
            "method": "Permutation Importance",
            "features": records
        }

    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Feature importance loading failed")
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/predict")
def predict(request: PredictRequest):
    received = request.features

    missing = set(feature_names) - set(received)
    extra = set(received) - set(feature_names)

    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {list(missing)[:10]}")

    if extra:
        raise HTTPException(status_code=422, detail=f"Unexpected features: {list(extra)[:10]}")

    try:
        values = [received[name] for name in feature_names]

        if not np.isfinite(values).all():
            raise HTTPException(status_code=422, detail="Features must contain finite numbers")

        prediction, confidence = run_prediction(received)

        return {
            "prediction": prediction,
            "predicted_label": prediction,
            "result": f"Class {prediction}",
            "confidence": confidence,
            "model": best_model_name,
            "message": "Prediction generated successfully"
        }

    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/scan")
def scan_website(request: ScanRequest):
    try:
        features = extract_url_features(request.url)
        prediction, confidence = run_prediction(features)

        unknown_features = [
            name for name, value in features.items() if value is None
        ]

        return {
            "url": request.url,
            "prediction": prediction,
            "predicted_label": prediction,
            "result": f"Class {prediction}",
            "confidence": confidence,
            "model": best_model_name,
            "features_extracted": len(feature_names) - len(unknown_features),
            "features_required": len(feature_names),
            "unknown_features": unknown_features,
            "message": "URL scanned and prediction generated."
        }

    except HTTPException:
        raise
    except Exception as error:
        logger.exception("URL scan failed")
        raise HTTPException(status_code=400, detail=str(error))
