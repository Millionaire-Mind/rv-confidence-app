from fastapi import FastAPI, HTTPException
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import json
import os
from datetime import datetime, timezone
from fastapi.openapi.utils import get_openapi

# ----------------------------
# App metadata
# ----------------------------
app = FastAPI(
    title="RV Buyer & Owner Confidence Assistant (Local Dev)",
    description="High-trust decision intelligence tools for RV buyers and owners.",
    version="0.7.1",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

# ----------------------------
# Utility
# ----------------------------

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------
# Load datasets
# ----------------------------
manufacturer_warranties = load_json("manufacturer_warranties.json")
rv_models = load_json("rv_models.json")
depreciation_model = load_json("depreciation_model.json")
construction_profiles = load_json("construction_profiles.json")
verified_construction = load_json("verified_construction.json")

# ----------------------------
# Models
# ----------------------------

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

class CostEstimateProfile(BaseModel):
    miles_per_year: Optional[int] = None
    storage: Optional[str] = None
    maintenance_records: Optional[str] = "unknown"

class CostDepreciationRequest(BaseModel):
    rv: RVIdentity
    purchase_price_usd: Optional[float] = None
    rv_category: Optional[str] = None
    profile: Optional[CostEstimateProfile] = None
    focus: Optional[str] = "full"

class FeeLineItem(BaseModel):
    name: str
    amount_usd: float
    category: Optional[str] = "other"
    disclosed_as_optional: Optional[bool] = None

class FinancingTerms(BaseModel):
    apr_percent: Optional[float] = None
    term_months: Optional[int] = None
    down_payment_usd: Optional[float] = None
    total_amount_financed_usd: Optional[float] = None
    lender_name: Optional[str] = None

class TradeInInfo(BaseModel):
    has_trade_in: bool = False
    offered_trade_in_value_usd: Optional[float] = None
    payoff_amount_usd: Optional[float] = None

class DealRiskScanRequest(BaseModel):
    rv: Optional[RVIdentity] = None
    quoted_unit_price_usd: Optional[float] = None
    fees: Optional[List[FeeLineItem]] = None
    financing: Optional[FinancingTerms] = None
    trade_in: Optional[TradeInInfo] = None
    buyer_priorities: Optional[List[str]] = None

class TriageRedFlags(BaseModel):
    propane_smell: Optional[bool] = False
    smoke_or_burning_smell: Optional[bool] = False
    carbon_monoxide_alarm: Optional[bool] = False
    active_water_near_electrical: Optional[bool] = False
    brake_or_steering_issue: Optional[bool] = False
    engine_overheating: Optional[bool] = False
    sparking_or_arcing: Optional[bool] = False

class MaintenanceTriageContext(BaseModel):
    rv_category: Optional[str] = None
    connected_to_shore_power: Optional[bool] = None
    generator_running: Optional[bool] = None
    propane_on: Optional[bool] = None
    recent_weather_freezing: Optional[bool] = None
    recently_serviced: Optional[bool] = None

class MaintenanceRepairTriageRequest(BaseModel):
    rv: Optional[RVIdentity] = None
    system: str
    symptoms_text: str
    red_flags: Optional[TriageRedFlags] = None
    context: Optional[MaintenanceTriageContext] = None

# ----------------------------
# Non-tool endpoints (hidden)
# ----------------------------

@app.get("/health", include_in_schema=False)
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "rv-confidence", "time_utc": utc_now_iso()}

@app.get("/debug/openapi", include_in_schema=False)
def debug_openapi() -> Dict[str, Any]:
    schema = app.openapi()
    return {
        "PUBLIC_BASE_URL": os.getenv("PUBLIC_BASE_URL"),
        "has_servers": "servers" in schema,
        "servers": schema.get("servers"),
        "keys_at_top": [k for k in schema.keys() if k in ("openapi", "info", "servers", "paths")],
    }

@app.get("/tools", include_in_schema=False)
def list_tools() -> Dict[str, Any]:
    return {"status": "ok"}

# ----------------------------
# Tool endpoints (POST only)
# ----------------------------

@app.post("/tools/manufacturer_intelligence")
def manufacturer_intelligence(req: ManufacturerIntelligenceRequest):
    data = manufacturer_warranties.get(req.manufacturer)
    if not data:
        return {"status": "not_found", "manufacturer": req.manufacturer}
    return {
        "manufacturer": req.manufacturer,
        "warranty": data,
        "data_confidence": {"can_be_used_as_fact": True}
    }

@app.post("/tools/rv_compare")
def rv_compare(req: RVCompareRequest):
    return {"status": "ok", "rv_a": req.rv_a, "rv_b": req.rv_b}

@app.post("/tools/cost_depreciation_estimate")
def cost_depreciation(req: CostDepreciationRequest):
    return {
        "status": "ok",
        "rv": req.rv,
        "data_confidence": {"can_be_used_as_fact": False},
    }

@app.post("/tools/deal_risk_scan")
def deal_risk_scan(req: DealRiskScanRequest):
    return {
        "status": "ok",
        "summary": {"quoted_unit_price_usd": req.quoted_unit_price_usd},
    }

@app.post("/tools/maintenance_repair_triage")
def maintenance_repair_triage(req: MaintenanceRepairTriageRequest):
    if req.red_flags and req.red_flags.propane_smell:
        return {
            "status": "stop_now",
            "reason": "Propane smell detected",
        }
    return {"status": "ok"}

# ----------------------------
# OpenAPI override (inject servers)
# ----------------------------

def custom_openapi():
    public_base = os.getenv("PUBLIC_BASE_URL")

    # If schema is already cached, ensure servers is injected if missing.
    if app.openapi_schema:
        if public_base and "servers" not in app.openapi_schema:
            app.openapi_schema["servers"] = [{"url": public_base}]
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )

    if public_base:
        schema["servers"] = [{"url": public_base}]

    app.openapi_schema = schema
    return app.openapi_schema
