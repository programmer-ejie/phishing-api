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

# Globals for model/data
rf_model = None
scaler = None
TRAIN_FEATURES = None
phishtank_urls = set()

# ===============================
# Download via gdown
# ===============================
def download_from_drive_with_gdown(file_id: str, dest_path: str) -> bool:
    if not file_id:
        print(f"⚠️ Missing file ID for {dest_path}.")
        return False

    if os.path.exists(dest_path):
        print(f"✅ {dest_path} already exists.")
        return True

    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        print(f"⬇️ Downloading {dest_path} from Google Drive (id={file_id})...")
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

# Download required files
for filename, env_name in ENV_FILE_IDS.items():
    file_id = os.getenv(env_name)
    dest_path = os.path.join(DATASET_DIR, filename)
    if file_id:
        success = download_from_drive_with_gdown(file_id, dest_path)
        if not success:
            print(f"⚠️ Warning: could not download {filename} (env {env_name}).")
    else:
        print(f"⚠️ Environment variable {env_name} not set. {filename} not downloaded automatically.")

# ===============================
# Safe load
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

rf_model = safe_load_joblib(os.path.join(DATASET_DIR, "phishing_detector_model.pkl"))
scaler = safe_load_joblib(os.path.join(DATASET_DIR, "scaler.pkl"))
TRAIN_FEATURES = safe_load_joblib(os.path.join(DATASET_DIR, "feature_columns.pkl"))

# ===============================
# Load PhishTank
# ===============================
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
# Google Safe Browsing
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
# Feature extraction
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
        return feats

# ===============================
# Pattern-based detection
# ===============================
def pattern_based_check(url):
    patterns = {
        "Has IP": urlparse(url).netloc.replace(".", "").isdigit(),
        "Has @": "@" in url,
        "Has -": "-" in urlparse(url).netloc,
        "Suspicious Subdomain": urlparse(url).netloc.count(".") > 2,
    }
    return patterns

# ===============================
# Unified Hybrid Check
# ===============================
def hybrid_check(url, api_key=GOOGLE_API_KEY):
    url = url.strip().lower()

    # --- Checks ---
    phishtank_flag = url in phishtank_urls
    api_flag = check_with_google_safebrowsing(url, api_key) if api_key else False

    feats = extract_url_features(url)
    features_df = pd.DataFrame([feats])
    try:
        X_scaled = scaler.transform(features_df)
        ml_proba = rf_model.predict_proba(X_scaled)[0][1]
    except Exception:
        ml_proba = None

    pattern_flags = pattern_based_check(url)
    pattern_flag = any(pattern_flags.values())

    # --- Risk assessment ---
    risk_level = "Low"
    reason_list = []

    if phishtank_flag:
        risk_level = "High"
        reason_list.append("Phishing (PhishTank)")
    if api_flag:
        risk_level = "High"
        reason_list.append("Phishing (Google Safe Browsing)")

    if ml_proba is not None:
        if ml_proba > 0.75:
            risk_level = "High"
            reason_list.append(f"High ML score ({ml_proba:.2f})")
        elif ml_proba > 0.6:
            if pattern_flag:
                risk_level = "High"
                reason_list.append(f"ML score {ml_proba:.2f} + Suspicious Pattern")
            else:
                risk_level = "Medium"
                reason_list.append(f"Medium ML score ({ml_proba:.2f})")
        else:
            reason_list.append(f"Low ML score ({ml_proba:.2f})")

    if pattern_flag and risk_level == "Low":
        risk_level = "Medium"
        labels = [k for k, v in pattern_flags.items() if v]
        reason_list.append("Suspicious Pattern: " + ", ".join(labels))

    # --- Final status ---
    if risk_level == "High":
        status = "❌ Block"
    elif risk_level == "Medium":
        status = "⚠️ Suspicious"
    else:
        status = "✅ Proceed"

    return {
        "url": url,
        "score": float(ml_proba) if ml_proba is not None else None,
        "risk_level": risk_level,
        "status": status,
        "reason": " | ".join(reason_list) if reason_list else "No strong signals"
    }

# ===============================
# FastAPI app
# ===============================
app = FastAPI(title="Phishing Detection API")

@app.get("/")
def home():
    return {
        "message": "🛡️ Phishing Detection API is running!",
        "model_loaded": rf_model is not None,
        "scaler_loaded": scaler is not None,
        "features_loaded": TRAIN_FEATURES is not None,
        "phishtank_loaded": len(phishtank_urls) > 0
    }

@app.get("/check_url")
def check_url(url: str = Query(...)):
    # Fail fast if model missing
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

    # Use unified checker
    try:
        result = hybrid_check(url)
        return JSONResponse(content=result)
    except Exception as e:
        print("❌ Hybrid check error:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "Hybrid check failed", "detail": str(e)})

@app.get("/health")
def health():
    ok = rf_model is not None and scaler is not None and TRAIN_FEATURES is not None
    return {"healthy": ok}
