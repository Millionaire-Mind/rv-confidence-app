from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import json
import os
from datetime import datetime, timezone

# --------------------------------------------------
# App metadata
# --------------------------------------------------
app = FastAPI(
    title="RV Buyer & Owner Confidence Assistant (Local Dev)",
    description="High-trust decision intelligence tools for RV buyers and owners.",
    version="0.7.2",
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
# Non-tool endpoints (hidden)
# --------------------------------------------------
@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok", "time": utc_now_iso()}

@app.get("/debug/env", include_in_schema=False)
def debug_env():
    return {
        "PUBLIC_BASE_URL": os.getenv("PUBLIC_BASE_URL"),
        "PORT": os.getenv("PORT"),
    }

@app.get("/debug/openapi", include_in_schema=False)
def debug_openapi():
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    return {
        "has_servers": "servers" in schema,
        "servers": schema.get("servers"),
        "public": os.getenv("PUBLIC_BASE_URL"),
    }

# --------------------------------------------------
# Tool endpoints
# --------------------------------------------------
@app.post("/tools/manufacturer_intelligence")
def manufacturer_intelligence(req: ManufacturerIntelligenceRequest):
    data = manufacturer_warranties.get(req.manufacturer)
    if not data:
        return {"status": "not_found"}
    return {
        "manufacturer": req.manufacturer,
        "warranty": data,
        "data_confidence": {"can_be_used_as_fact": True}
    }

@app.post("/tools/rv_compare")
def rv_compare(req: RVCompareRequest):
    return {"rv_a": req.rv_a, "rv_b": req.rv_b}

@app.post("/tools/cost_depreciation_estimate")
def cost_estimate(req: CostDepreciationRequest):
    return {"rv": req.rv, "note": "estimate only"}

@app.post("/tools/deal_risk_scan")
def deal_scan(req: DealRiskScanRequest):
    return {"quoted": req.quoted_unit_price_usd, "fees": req.fees}

@app.post("/tools/maintenance_repair_triage")
def triage(req: MaintenanceRepairTriageRequest):
    if req.propane_smell:
        return {"status": "stop_now", "reason": "propane smell"}
    return {"status": "ok"}

# --------------------------------------------------
# HARD OVERRIDE OpenAPI so GPT always sees servers
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
