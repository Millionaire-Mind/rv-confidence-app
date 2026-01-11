from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import json
import os
from datetime import datetime, timezone

# --------------------------------------------------
# IMPORTANT:
# Disable FastAPI's built-in OpenAPI route so OUR
# /openapi.json is the one that gets served.
# --------------------------------------------------
app = FastAPI(
    title="RV Buyer & Owner Confidence Assistant (Local Dev)",
    description="High-trust decision intelligence tools for RV buyers and owners.",
    version="0.7.3",
    openapi_url=None,   # <-- critical
    docs_url=None,      # optional: turn off Swagger UI
    redoc_url=None,     # optional: turn off ReDoc
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

# --------------------------------------------------
# Utilities
# --------------------------------------------------
def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def load_json(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --------------------------------------------------
# Data
# --------------------------------------------------
manufacturer_warranties = load_json("manufacturer_warranties.json")

# --------------------------------------------------
# Models
# --------------------------------------------------
class RVIdentity(BaseModel):
    manufacturer: str
    model: str
    year: Optional[int] = None
    trim: Optional[str] = None

class ManufacturerIntelligenceRequest(BaseModel):
    manufacturer: str
    model: Optional[str] = None
    year: Optional[int] = None
    trim: Optional[str] = None
    focus: Optional[str] = "full"

class RVCompareRequest(BaseModel):
    rv_a: RVIdentity
    rv_b: RVIdentity
    focus: Optional[str] = "specs"

class CostDepreciationRequest(BaseModel):
    rv: RVIdentity
    purchase_price_usd: Optional[float] = None

class FeeLineItem(BaseModel):
    name: str
    amount_usd: float

class DealRiskScanRequest(BaseModel):
    quoted_unit_price_usd: Optional[float] = None
    fees: Optional[List[FeeLineItem]] = None

class MaintenanceRepairTriageRequest(BaseModel):
    system: str
    symptoms_text: str
    propane_smell: Optional[bool] = False

# --------------------------------------------------
# Non-tool endpoints (hidden from schema)
# --------------------------------------------------
@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok", "service": "rv-confidence", "time_utc": utc_now_iso()}

@app.get("/debug/env", include_in_schema=False)
def debug_env() -> Dict[str, Any]:
    return {
        "PUBLIC_BASE_URL": os.getenv("PUBLIC_BASE_URL"),
        "PORT": os.getenv("PORT"),
    }

# --------------------------------------------------
# Tool endpoints (these ARE in schema)
# --------------------------------------------------
@app.post("/tools/manufacturer_intelligence")
def manufacturer_intelligence(req: ManufacturerIntelligenceRequest):
    data = manufacturer_warranties.get(req.manufacturer)
    if not data:
        return {"status": "not_found", "manufacturer": req.manufacturer}
    return {
        "manufacturer": req.manufacturer,
        "warranty": data,
        "data_confidence": {"can_be_used_as_fact": True},
    }

@app.post("/tools/rv_compare")
def rv_compare(req: RVCompareRequest):
    return {"status": "ok", "rv_a": req.rv_a.model_dump(), "rv_b": req.rv_b.model_dump()}

@app.post("/tools/cost_depreciation_estimate")
def cost_depreciation_estimate(req: CostDepreciationRequest):
    return {"status": "ok", "rv": req.rv.model_dump(), "note": "estimate only"}

@app.post("/tools/deal_risk_scan")
def deal_risk_scan(req: DealRiskScanRequest):
    return {"status": "ok", "quoted_unit_price_usd": req.quoted_unit_price_usd, "fees": [f.model_dump() for f in (req.fees or [])]}

@app.post("/tools/maintenance_repair_triage")
def maintenance_repair_triage(req: MaintenanceRepairTriageRequest):
    if req.propane_smell:
        return {"status": "ok", "safety_level": "stop_now", "reason": "Propane smell detected. Stop and get professional help."}
    return {"status": "ok", "safety_level": "caution", "reason": "No high-risk red flag selected."}

# --------------------------------------------------
# OUR OpenAPI endpoint (Actions will import this)
# --------------------------------------------------
@app.get("/openapi.json", include_in_schema=False)
def openapi_json():
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    public = os.getenv("PUBLIC_BASE_URL")
    if public:
        schema["servers"] = [{"url": public}]

    return JSONResponse(schema)

@app.get("/debug/openapi", include_in_schema=False)
def debug_openapi() -> Dict[str, Any]:
    schema = openapi_json().body.decode("utf-8")
    return {
        "PUBLIC_BASE_URL": os.getenv("PUBLIC_BASE_URL"),
        "note": "This endpoint confirms the custom /openapi.json is active. Check /openapi.json for servers.",
        "openapi_json_starts_with": schema[:120],
    }
