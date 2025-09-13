# app.py
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
from urllib.parse import urlparse
import os
import requests
import gdown
import traceback
import time

# ===============================
# Config
# ===============================
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# Environment variable names expected on Render
ENV_FILE_IDS = {
    "phishing_detector_model.pkl": "MODEL_FILE_ID",
    "scaler.pkl": "SCALER_FILE_ID",
    "feature_columns.pkl": "FEATURES_FILE_ID",
    "phishtank.csv": "PHISHTANK_FILE_ID",
}

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)

# Globals for model/data (set after load)
rf_model = None
scaler = None
TRAIN_FEATURES = None
phishtank_urls = set()

# ===============================
# Helper: download via gdown
# ===============================
def download_from_drive_with_gdown(file_id: str, dest_path: str) -> bool:
    """
    Download a file from Google Drive using gdown (handles large files and confirmation tokens).
    Returns True if download succeeded or file already exists, False otherwise.
    """
    if not file_id:
        print(f"⚠️ Missing file ID for {dest_path}.")
        return False

    if os.path.exists(dest_path):
        print(f"✅ {dest_path} already exists.")
        return True

    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        print(f"⬇️ Downloading {dest_path} from Google Drive (id={file_id})...")
        # gdown.download returns the output path or None on failure
        out = gdown.download(url, dest_path, quiet=False, fuzzy=True)
        if out and os.path.exists(dest_path):
            print(f"✅ Downloaded {dest_path}")
            return True
        else:
            print(f"❌ gdown failed to download {dest_path}.")
            return False
    except Exception as e:
        print(f"❌ Exception while downloading {dest_path}: {e}")
        traceback.print_exc()
        return False

# ===============================
# Download files if missing
# ===============================
for filename, env_name in ENV_FILE_IDS.items():
    file_id = os.getenv(env_name)
    dest_path = os.path.join(DATASET_DIR, filename)
    if file_id:
        success = download_from_drive_with_gdown(file_id, dest_path)
        if not success:
            print(f"⚠️ Warning: could not download {filename} (env {env_name}).")
    else:
        print(f"⚠️ Environment variable {env_name} not set. {filename} will not be downloaded automatically.")

# ===============================
# Load model + artifacts (safe)
# ===============================
def safe_load_joblib(path):
    try:
        start = time.time()
        obj = joblib.load(path)
        print(f"✅ Loaded `{path}` in {time.time()-start:.2f}s")
        return obj
    except Exception as e:
        print(f"❌ Failed to load `{path}`: {e}")
        traceback.print_exc()
        return None

# Load model, scaler, features
rf_model = safe_load_joblib(os.path.join(DATASET_DIR, "phishing_detector_model.pkl"))
scaler = safe_load_joblib(os.path.join(DATASET_DIR, "scaler.pkl"))
TRAIN_FEATURES = safe_load_joblib(os.path.join(DATASET_DIR, "feature_columns.pkl"))

# Load phishtank CSV into set
def load_phishtank_blocklist(file_path=os.path.join(DATASET_DIR, "phishtank.csv")):
    try:
        df = pd.read_csv(file_path, dtype=str)
        if "url" in df.columns:
            urls = set(df["url"].str.strip().str.lower())
            print(f"✅ Loaded {len(urls)} URLs from PhishTank.")
            return urls
        else:
            print("⚠️ 'url' column not found in phishtank.csv. Returning empty set.")
            return set()
    except FileNotFoundError:
        print(f"⚠️ PhishTank CSV not found at {file_path}.")
        return set()
    except Exception as e:
        print(f"❌ Error loading PhishTank CSV: {e}")
        traceback.print_exc()
        return set()

phishtank_urls = load_phishtank_blocklist()

# ===============================
# Google Safe Browsing helper
# ===============================
def check_with_google_safebrowsing(url, api_key=GOOGLE_API_KEY):
    if not api_key:
        return False
    endpoint = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
    body = {
        "client": {"clientId": "PhishingDetector", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }
    try:
        resp = requests.post(f"{endpoint}?key={api_key}", json=body, timeout=6)
        res = resp.json()
        return bool(res.get("matches"))
    except Exception as e:
        print(f"⚠️ SafeBrowsing check failed: {e}")
        return False

# ===============================
# Feature extraction (same as your earlier)
# ===============================
def extract_url_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path

    feats = {
        "UrlLength": len(url),
        "HostnameLength": len(domain),
        "NumDots": url.count("."),
        "NumDash": url.count("-"),
        "NumNumericChars": sum(c.isdigit() for c in url),
        "NoHttps": int(not url.lower().startswith("https")),
        "AtSymbol": int("@" in url),
        "DoubleSlashInPath": int("//" in path),
        "SuspiciousSubdomain": int(domain.count(".") > 2),
        "ContainsBrand": int(any(b in url.lower() for b in ["paypal", "bank", "amazon"])),
    }

    if isinstance(TRAIN_FEATURES, (list, tuple)):
        return {col: feats.get(col, 0) for col in TRAIN_FEATURES}
    else:
        # fallback: return feats keys present
        return feats

# ===============================
# FastAPI app
# ===============================
app = FastAPI(title="Phishing Detection API")

@app.get("/")
def home():
    status = {
        "message": "🛡️ Phishing Detection API is running!",
        "model_loaded": rf_model is not None,
        "scaler_loaded": scaler is not None,
        "features_loaded": TRAIN_FEATURES is not None,
        "phishtank_loaded": len(phishtank_urls) > 0
    }
    return status

@app.get("/check_url")
def check_url(url: str = Query(...)):
    url = url.strip().lower()

    # If model artifacts missing, return helpful error instead of crashing
    if rf_model is None or scaler is None or TRAIN_FEATURES is None:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Model or artifacts not loaded on server. Check logs.",
                "model_loaded": rf_model is not None,
                "scaler_loaded": scaler is not None,
                "features_loaded": TRAIN_FEATURES is not None
            }
        )

    # 1) PhishTank check
    if url in phishtank_urls:
        return JSONResponse(content={
            "url": url,
            "decision": "❌ Block",
            "reason": "PhishTank blocklist",
            "score": None
        })

    # 2) Google Safe Browsing check (best-effort)
    try:
        if check_with_google_safebrowsing(url):
            return JSONResponse(content={
                "url": url,
                "decision": "❌ Block",
                "reason": "Google Safe Browsing",
                "score": None
            })
    except Exception:
        # don't fail hard if Google API has issues
        pass

    # 3) ML model prediction
    try:
        feats = extract_url_features(url)
        X = pd.DataFrame([feats])
        X_scaled = scaler.transform(X)
        ml_proba = float(rf_model.predict_proba(X_scaled)[0][1])

        if ml_proba > 0.75:
            decision = "❌ Block"
            reason = "High ML score"
        elif ml_proba > 0.6:
            decision = "⚠️ Suspicious"
            reason = "Medium ML score"
        else:
            decision = "✅ Proceed"
            reason = "Low risk"

        return JSONResponse(content={
            "url": url,
            "score": ml_proba,
            "decision": decision,
            "reason": reason
        })
    except Exception as e:
        print("❌ Prediction error:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "Prediction failed", "detail": str(e)})

# Optional: health-check endpoint used by Render or monitoring
@app.get("/health")
def health():
    ok = rf_model is not None and scaler is not None and TRAIN_FEATURES is not None
    return {"healthy": ok}
