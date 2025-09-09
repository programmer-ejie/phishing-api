from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
from urllib.parse import urlparse
import os
import requests

# ===============================
# 1. Load Models & Data
# ===============================
DATASET_DIR = "dataset"

rf_model = joblib.load(os.path.join(DATASET_DIR, "phishing_detector_model.pkl"))
scaler = joblib.load(os.path.join(DATASET_DIR, "scaler.pkl"))
TRAIN_FEATURES = joblib.load(os.path.join(DATASET_DIR, "feature_columns.pkl"))

# Load PhishTank blocklist
def load_phishtank_blocklist(file_path=os.path.join(DATASET_DIR, "phishtank.csv")):
    try:
        df = pd.read_csv(file_path, dtype=str)
        if "url" in df.columns:
            urls = set(df["url"].str.strip().str.lower())
            print(f"✅ Loaded {len(urls)} URLs from PhishTank.")
            return urls
        return set()
    except Exception as e:
        print(f"⚠️ Could not load PhishTank CSV: {e}")
        return set()

phishtank_urls = load_phishtank_blocklist()

# ===============================
# 2. Google Safe Browsing
# ===============================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)

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
        response = requests.post(f"{endpoint}?key={api_key}", json=body, timeout=5)
        result = response.json()
        return bool(result.get("matches"))
    except Exception:
        return False

# ===============================
# 3. Feature Extraction
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

    return {col: feats.get(col, 0) for col in TRAIN_FEATURES}

# ===============================
# 4. FastAPI App
# ===============================
app = FastAPI()

@app.get("/")
def home():
    return {"message": "🛡️ Phishing Detection API is running!"}

@app.get("/check_url")
def check_url(url: str = Query(...)):
    url = url.strip().lower()

    # 1. PhishTank blocklist
    if url in phishtank_urls:
        return {"url": url, "decision": "❌ Block", "reason": "PhishTank blocklist"}

    # 2. Google Safe Browsing
    if check_with_google_safebrowsing(url):
        return {"url": url, "decision": "❌ Block", "reason": "Google Safe Browsing"}

    # 3. ML model
    feats = extract_url_features(url)
    X = pd.DataFrame([feats])
    X_scaled = scaler.transform(X)
    ml_proba = rf_model.predict_proba(X_scaled)[0][1]

    if ml_proba > 0.75:
        decision = "❌ Block"
    elif ml_proba > 0.6:
        decision = "⚠️ Suspicious"
    else:
        decision = "✅ Proceed"

    return JSONResponse(content={
        "url": url,
        "score": float(ml_proba),
        "decision": decision
    })
S