from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, conint, confloat, constr


# ============================
# Helpers
# ============================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


# ============================
# Paths / Data
# ============================

# repo root: rv-confidence-app/
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# These are small local datasets you created (verified sources only when present)
MANUFACTURER_WARRANTIES_PATH = DATA_DIR / "manufacturer_warranties.json"
VERIFIED_CONSTRUCTION_PATH = DATA_DIR / "verified_construction.json"
CONSTRUCTION_PROFILES_PATH = DATA_DIR / "construction_profiles.json"
RV_MODELS_PATH = DATA_DIR / "rv_models.json"
DEPRECIATION_MODEL_PATH = DATA_DIR / "depreciation_model.json"

manufacturer_warranties_db: Dict[str, Any] = load_json(MANUFACTURER_WARRANTIES_PATH, {})
verified_construction_db: Dict[str, Any] = load_json(VERIFIED_CONSTRUCTION_PATH, {})
construction_profiles_db: Dict[str, Any] = load_json(CONSTRUCTION_PROFILES_PATH, {})
rv_models_db: Dict[str, Any] = load_json(RV_MODELS_PATH, {})
depreciation_model_db: Dict[str, Any] = load_json(DEPRECIATION_MODEL_PATH, {})


# ============================
# Pydantic Models
# ============================

# ---- Shared ----

class DataConfidence(BaseModel):
    level: Literal["stub", "partial", "verified"] = "stub"
    meaning: str
    can_be_used_as_fact: bool = False


class RVIdentity(BaseModel):
    manufacturer: constr(min_length=1, max_length=80)
    model: constr(min_length=1, max_length=120)
    year: Optional[conint(ge=1980, le=2100)] = None
    trim: Optional[constr(max_length=120)] = None


# ---- Manufacturer Intelligence ----

ManufacturerFocus = Literal[
    "warranty",
    "construction",
    "quality_overview",
    "strengths_weaknesses",
    "who_its_for",
    "full",
]


class ManufacturerIntelligenceRequest(BaseModel):
    manufacturer: constr(min_length=1, max_length=80)
    model: Optional[constr(max_length=120)] = None
    year: Optional[conint(ge=1980, le=2100)] = None
    trim: Optional[constr(max_length=120)] = None
    focus: Optional[ManufacturerFocus] = "full"


class WarrantyBlock(BaseModel):
    status: Literal["verified", "unverified"] = "unverified"
    summary: str
    source_url: Optional[str] = None
    last_verified: Optional[str] = None
    coverage_notes: List[str] = Field(default_factory=list)
    how_to_verify: Optional[List[str]] = None


class VerifiedConstructionForRequest(BaseModel):
    status: Literal["verified", "unverified"] = "unverified"
    scope_used: Literal["default", "manufacturer", "model", "trim"] = "default"
    summary: str
    source_url: Optional[str] = None
    last_verified: Optional[str] = None
    construction_tags: List[str] = Field(default_factory=list)
    how_to_verify: Optional[List[str]] = None


class ConstructionTypeTradeoffs(BaseModel):
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)


class ConstructionTypeInfo(BaseModel):
    neutral_description: str
    tradeoffs: ConstructionTypeTradeoffs


class ConstructionBlock(BaseModel):
    status: Literal["educational_with_optional_verification"] = "educational_with_optional_verification"
    note: str
    construction_types: Dict[str, ConstructionTypeInfo]
    verified_construction_for_this_request: VerifiedConstructionForRequest


class QualityOverviewBlock(BaseModel):
    status: Literal["stub"] = "stub"
    summary: str = "Not implemented yet. This will be neutral and source-backed (no hype)."


class ManufacturerIntelligenceResponse(BaseModel):
    manufacturer: str
    model: Optional[str] = None
    year: Optional[int] = None
    trim: Optional[str] = None
    data_confidence: DataConfidence
    trust_disclosures: List[str] = Field(default_factory=list)
    warranty: Optional[WarrantyBlock] = None
    construction: Optional[ConstructionBlock] = None
    quality_overview: Optional[QualityOverviewBlock] = None


# ---- RV Compare ----

class RVCompareRequest(BaseModel):
    rv_a: RVIdentity
    rv_b: RVIdentity
    focus: Optional[Literal["specs"]] = "specs"


class VerifiedSpecs(BaseModel):
    status: Literal["verified", "unverified"] = "unverified"
    source_url: Optional[str] = None
    last_verified: Optional[str] = None
    specs: Dict[str, Any] = Field(default_factory=dict)


class RVCompareKeyDiff(BaseModel):
    status: Literal["computed_from_verified_specs"] = "computed_from_verified_specs"
    differences: List[Dict[str, Any]] = Field(default_factory=list)
    same_on_both: List[Dict[str, Any]] = Field(default_factory=list)
    note: str


class RVCompareResponse(BaseModel):
    comparison_intent: str = "specs"
    rv_a: Dict[str, Any]
    rv_b: Dict[str, Any]
    trust_disclosures: List[str] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    comparison: Dict[str, Any]
    data_confidence: DataConfidence


# ---- Cost / Depreciation ----

CostFocus = Literal["depreciation_only", "ownership_only", "full"]

StorageType = Literal["garage", "covered", "outdoor_shade", "outdoor_sun"]
MaintenanceRecords = Literal["unknown", "poor", "average", "excellent"]


class CostEstimateProfile(BaseModel):
    miles_per_year: Optional[conint(ge=0, le=60000)] = None
    storage: Optional[StorageType] = None
    maintenance_records: Optional[MaintenanceRecords] = "unknown"


class CostDepreciationRequest(BaseModel):
    rv: RVIdentity
    purchase_price_usd: Optional[confloat(gt=0, le=5000000)] = None
    rv_category: Optional[Literal["motorhome", "towable"]] = None
    profile: Optional[CostEstimateProfile] = None
    focus: Optional[CostFocus] = "full"


class DepreciationEstimate(BaseModel):
    status: Literal["estimate"] = "estimate"
    percent_ranges: Dict[str, Any]
    dollar_ranges: Optional[Dict[str, Any]] = None
    note: str


class CostDepreciationResponse(BaseModel):
    status: Literal["ok"] = "ok"
    rv: Dict[str, Any]
    data_confidence: DataConfidence
    model_meta: Dict[str, Any]
    applied_adjustments: List[Dict[str, Any]] = Field(default_factory=list)
    disclosures: List[str] = Field(default_factory=list)
    what_would_make_this_more_precise: List[str] = Field(default_factory=list)
    depreciation_estimate: DepreciationEstimate


# ---- Deal Risk Scan ----

FeeCategory = Literal[
    "tax",
    "title_registration",
    "doc_fee",
    "prep",
    "delivery",
    "warranty",
    "financing",
    "addon",
    "other",
]


class FeeLineItem(BaseModel):
    name: constr(min_length=1, max_length=120)
    amount_usd: confloat(ge=0, le=500000)
    category: Optional[FeeCategory] = "other"
    disclosed_as_optional: Optional[bool] = None


class FinancingTerms(BaseModel):
    apr_percent: Optional[confloat(ge=0, le=40)] = None
    term_months: Optional[conint(ge=0, le=360)] = None
    down_payment_usd: Optional[confloat(ge=0, le=5000000)] = None
    total_amount_financed_usd: Optional[confloat(ge=0, le=5000000)] = None
    lender_name: Optional[constr(max_length=120)] = None


class TradeInInfo(BaseModel):
    has_trade_in: bool = False
    offered_trade_in_value_usd: Optional[confloat(ge=0, le=5000000)] = None
    payoff_amount_usd: Optional[confloat(ge=0, le=5000000)] = None


BuyerPriority = Literal[
    "lowest_total_cost",
    "lowest_monthly_payment",
    "lowest_upfront",
    "warranty_coverage",
    "simple_ownership",
    "resale_value",
]


class DealRiskScanRequest(BaseModel):
    rv: Optional[RVIdentity] = None
    quoted_unit_price_usd: Optional[confloat(ge=0, le=5000000)] = None
    fees: Optional[List[FeeLineItem]] = None
    financing: Optional[FinancingTerms] = None
    trade_in: Optional[TradeInInfo] = None
    buyer_priorities: Optional[List[BuyerPriority]] = None


class DealRiskScanResponse(BaseModel):
    status: Literal["ok"] = "ok"
    data_confidence: DataConfidence
    summary: Dict[str, Any]
    clarifying_questions: List[str]
    flags: List[Dict[str, Any]]
    scripts_to_use_with_dealer: List[Dict[str, str]]
    checkpoint: Dict[str, str]
    trust_disclosures: List[str]


# ---- Maintenance / Repair Triage ----

MaintenanceSystem = Literal[
    "propane",
    "electrical_12v",
    "electrical_120v",
    "water_pump_plumbing",
    "water_heater",
    "toilet_holding_tanks",
    "hvac_ac",
    "hvac_furnace",
    "slide_out",
    "roof_leaks",
    "appliances_general",
    "engine_chassis",
]


class TriageRedFlags(BaseModel):
    propane_smell: Optional[bool] = False
    smoke_or_burning_smell: Optional[bool] = False
    carbon_monoxide_alarm: Optional[bool] = False
    active_water_near_electrical: Optional[bool] = False
    brake_or_steering_issue: Optional[bool] = False
    engine_overheating: Optional[bool] = False
    sparking_or_arcing: Optional[bool] = False


class MaintenanceTriageContext(BaseModel):
    rv_category: Optional[Literal["motorhome", "towable"]] = None
    connected_to_shore_power: Optional[bool] = None
    generator_running: Optional[bool] = None
    propane_on: Optional[bool] = None
    recent_weather_freezing: Optional[bool] = None
    recently_serviced: Optional[bool] = None


class MaintenanceRepairTriageRequest(BaseModel):
    rv: Optional[RVIdentity] = None
    system: MaintenanceSystem
    symptoms_text: constr(min_length=3, max_length=2000)
    red_flags: Optional[TriageRedFlags] = None
    context: Optional[MaintenanceTriageContext] = None


SafetyLevel = Literal["stop_now", "caution", "normal"]


class MaintenanceRepairTriageResponse(BaseModel):
    status: Literal["ok"] = "ok"
    safety_level: SafetyLevel
    reason: str
    stop_triggers: List[str] = Field(default_factory=list)
    immediate_actions: List[str] = Field(default_factory=list)
    what_to_tell_the_pro: List[str] = Field(default_factory=list)
    trust_disclosures: List[str] = Field(default_factory=list)


# ============================
# FastAPI App
# ============================

app = FastAPI(
    title="RV Buyer & Owner Confidence Assistant (Local Dev)",
    version="0.7.4",
    description="High-trust decision intelligence tools for RV buyers and owners.",
    openapi_url="/openapi.json",
)


# ============================
# Non-tool endpoints (hidden)
# ============================

@app.get("/health", include_in_schema=False)
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "rv-confidence", "time_utc": utc_now_iso()}


@app.get("/tools", include_in_schema=False)
def list_tools() -> Dict[str, Any]:
    return {
        "tools": [
            {
                "name": "manufacturer_intelligence",
                "path": "/tools/manufacturer_intelligence",
                "purpose": "Neutral manufacturer/model/trim intelligence including warranty and construction method.",
                "read_only": True,
                "open_world": True,
            },
            {
                "name": "rv_compare",
                "path": "/tools/rv_compare",
                "purpose": "Compare two RVs using verified sources only; unknowns are stated explicitly.",
                "read_only": True,
                "open_world": True,
            },
            {
                "name": "cost_depreciation_estimate",
                "path": "/tools/cost_depreciation_estimate",
                "purpose": "Estimate depreciation/ownership ranges with disclosed assumptions. Not a guarantee.",
                "read_only": True,
                "open_world": True,
            },
            {
                "name": "deal_risk_scan",
                "path": "/tools/deal_risk_scan",
                "purpose": "Flags clarity risks and cost drivers in a quote. Does not accuse dealers/sellers. Works with partial info.",
                "read_only": True,
                "open_world": True,
            },
            {
                "name": "maintenance_repair_triage",
                "path": "/tools/maintenance_repair_triage",
                "purpose": "Safety-first triage for post-purchase issues. Stops on high-risk conditions.",
                "read_only": True,
                "open_world": True,
            },
        ]
    }


# ============================
# Tool endpoints
# ============================

@app.post("/tools/manufacturer_intelligence", response_model=ManufacturerIntelligenceResponse)
def manufacturer_intelligence(req: ManufacturerIntelligenceRequest) -> ManufacturerIntelligenceResponse:
    trust_disclosures = [
        "This is educational information, not legal or financial advice.",
        "No dealer recommendations. No sponsored content in v1.",
        "If something is not verified with a source, we will say so explicitly.",
    ]

    mfr_key = req.manufacturer.strip()
    focus = req.focus or "full"

    # Warranty lookup (by manufacturer)
    warranty_entry = manufacturer_warranties_db.get(mfr_key)

    warranty_block: Optional[WarrantyBlock] = None
    if focus in ("warranty", "full") or focus is None:
        if warranty_entry and warranty_entry.get("source_url"):
            warranty_block = WarrantyBlock(
                status="verified",
                summary=warranty_entry.get("warranty_summary", "Verified warranty summary not provided."),
                source_url=warranty_entry.get("source_url"),
                last_verified=warranty_entry.get("last_verified"),
                coverage_notes=warranty_entry.get("coverage_notes", []),
                how_to_verify=None,
            )
            warranty_conf = DataConfidence(
                level="partial",
                meaning="Warranty terms are factual only when marked verified with a source_url. Construction content is educational definitions, not brand-specific claims.",
                can_be_used_as_fact=True,
            )
        else:
            warranty_block = WarrantyBlock(
                status="unverified",
                summary="Not verified yet. I can’t state warranty terms as fact until we add an official source URL and verified summary.",
                source_url=None,
                last_verified=None,
                coverage_notes=[],
                how_to_verify=[
                    "Add the official manufacturer warranty webpage or PDF URL into data/manufacturer_warranties.json",
                    "Replace the placeholder warranty_summary with a short neutral summary of the warranty term and what it covers",
                    "Set last_verified to today’s date when confirmed",
                ],
            )
            warranty_conf = DataConfidence(
                level="stub",
                meaning="Warranty terms are factual only when marked verified with a source_url. Construction content is educational definitions, not brand-specific claims.",
                can_be_used_as_fact=False,
            )

    # Construction (educational + optional verification per model/trim)
    construction_block: Optional[ConstructionBlock] = None
    construction_conf = DataConfidence(
        level="stub",
        meaning="Sections marked verified are source-backed. Unverified sections are educational guidance or placeholders and must not be treated as factual claims.",
        can_be_used_as_fact=False,
    )

    if focus in ("construction", "full"):
        # Neutral definitions
        construction_types: Dict[str, ConstructionTypeInfo] = {
            "stick_and_tin": ConstructionTypeInfo(
                neutral_description="Wood or aluminum framing with corrugated metal (aluminum) exterior siding; common in value-oriented travel trailers.",
                tradeoffs=ConstructionTypeTradeoffs(
                    pros=["Often lower purchase price", "Repairs can be simpler for some exterior damage"],
                    cons=[
                        "Seams and fasteners can increase water-intrusion risk if not maintained",
                        "May have lower insulation performance depending on build",
                    ],
                ),
            ),
            "laminated_fiberglass": ConstructionTypeInfo(
                neutral_description="Layered wall construction with fiberglass exterior bonded to underlying materials; common in mid to higher tiers.",
                tradeoffs=ConstructionTypeTradeoffs(
                    pros=["Often better exterior finish", "Can improve insulation and rigidity depending on design"],
                    cons=["Repairs can be more complex if delamination or water damage occurs", "May cost more upfront"],
                ),
            ),
        }

        # Optional verified construction for manufacturer/model/trim (only if present in data)
        model_key = (req.model or "").strip()
        trim_key = (req.trim or "").strip()

        verified_entry: Optional[Dict[str, Any]] = None
        scope_used: Literal["default", "manufacturer", "model", "trim"] = "default"

        # priority: trim -> model -> manufacturer
        if model_key and trim_key:
            verified_entry = (
                verified_construction_db.get(mfr_key, {})
                .get(model_key, {})
                .get(trim_key)
            )
            if verified_entry:
                scope_used = "trim"

        if not verified_entry and model_key:
            verified_entry = (
                verified_construction_db.get(mfr_key, {})
                .get(model_key, {})
                .get("_model")
            )
            if verified_entry:
                scope_used = "model"

        if not verified_entry:
            verified_entry = (
                verified_construction_db.get(mfr_key, {})
                .get("_manufacturer")
            )
            if verified_entry:
                scope_used = "manufacturer"

        if verified_entry and verified_entry.get("source_url"):
            verified_block = VerifiedConstructionForRequest(
                status="verified",
                scope_used=scope_used,
                summary=verified_entry.get("construction_summary", "Verified construction summary not provided."),
                source_url=verified_entry.get("source_url"),
                last_verified=verified_entry.get("last_verified"),
                construction_tags=verified_entry.get("construction_tags", []),
                how_to_verify=None,
            )
            construction_conf = DataConfidence(
                level="partial",
                meaning="Sections marked verified are source-backed. Unverified sections are educational guidance or placeholders and must not be treated as factual claims.",
                can_be_used_as_fact=True,
            )
        else:
            verified_block = VerifiedConstructionForRequest(
                status="unverified",
                scope_used="default",
                summary="Not verified yet. I can explain construction types generally, but I won’t claim this manufacturer/model’s construction as fact until we add a verified source.",
                source_url=None,
                last_verified=None,
                construction_tags=[],
                how_to_verify=[
                    "Add an official manufacturer spec page, brochure PDF, or documentation URL into data/verified_construction.json",
                    "Add a short neutral construction_summary (no marketing language) and set last_verified",
                ],
            )

        construction_block = ConstructionBlock(
            note="Construction guidance includes (1) neutral definitions of common build types and (2) optional verified facts about a specific manufacturer/model only when a source-backed entry exists.",
            construction_types=construction_types,
            verified_construction_for_this_request=verified_block,
        )

    # Quality overview stub (kept intentionally minimal)
    quality_overview_block: Optional[QualityOverviewBlock] = None
    quality_conf = DataConfidence(
        level="stub",
        meaning="Requested section is not implemented as source-backed yet; do not treat as factual claims.",
        can_be_used_as_fact=False,
    )
    if focus == "quality_overview":
        quality_overview_block = QualityOverviewBlock()

    # Choose confidence summary (prioritize verified/partial if any section verified)
    # Basic merge: if any section can_be_used_as_fact true -> partial else stub.
    any_fact = False
    meaning = "Sections are educational unless verified with source_url."
    if warranty_block and warranty_block.status == "verified":
        any_fact = True
        meaning = "Warranty terms are factual only when marked verified with a source_url. Construction content is educational definitions, not brand-specific claims."
    if construction_block and construction_block.verified_construction_for_this_request.status == "verified":
        any_fact = True
        meaning = "Sections marked verified are source-backed. Unverified sections are educational guidance or placeholders and must not be treated as factual claims."

    overall_conf = DataConfidence(
        level="partial" if any_fact else "stub",
        meaning=meaning,
        can_be_used_as_fact=any_fact,
    )

    return ManufacturerIntelligenceResponse(
        manufacturer=req.manufacturer,
        model=req.model,
        year=req.year,
        trim=req.trim,
        data_confidence=overall_conf,
        trust_disclosures=trust_disclosures,
        warranty=warranty_block if focus in ("warranty", "full") else None,
        construction=construction_block if focus in ("construction", "full") else None,
        quality_overview=quality_overview_block if focus == "quality_overview" else None,
    )


@app.post("/tools/rv_compare", response_model=RVCompareResponse)
def rv_compare(req: RVCompareRequest) -> RVCompareResponse:
    trust_disclosures = [
        "This comparison is factual only where source-backed model entries exist in the dataset.",
        "If a spec is missing or the model is not found, we will say so rather than guessing.",
        "This is educational information, not legal or financial advice.",
        "No dealer recommendations. No sponsored content in v1.",
    ]

    def lookup_specs(identity: RVIdentity) -> VerifiedSpecs:
        # rv_models_db structure assumed:
        # { "Manufacturer": { "Model": { "Year": { "Trim": {...} } } } }
        m = identity.manufacturer
        mo = identity.model
        y = str(identity.year) if identity.year else None
        t = identity.trim or None

        entry = rv_models_db.get(m, {}).get(mo, {})
        if y:
            entry = entry.get(y, {})
        # If trim exists, use it; else allow "_model" fallback if you store it that way
        if isinstance(entry, dict) and t:
            entry = entry.get(t)
        elif isinstance(entry, dict) and "_model" in entry:
            entry = entry.get("_model")

        if entry and isinstance(entry, dict) and entry.get("source_url"):
            return VerifiedSpecs(
                status="verified",
                source_url=entry.get("source_url"),
                last_verified=entry.get("last_verified"),
                specs=entry.get("specs", {}),
            )

        return VerifiedSpecs(
            status="unverified",
            source_url=None,
            last_verified=None,
            specs={
                "rv_type": "Unknown",
            },
        )

    specs_a = lookup_specs(req.rv_a)
    specs_b = lookup_specs(req.rv_b)

    unknowns: List[str] = []
    if specs_a.status != "verified":
        unknowns.append("RV A specs not verified (missing or placeholder source_url).")
    if specs_b.status != "verified":
        unknowns.append("RV B specs not verified (missing or placeholder source_url).")

    # Compute key differences only if both are verified
    key_differences: Optional[RVCompareKeyDiff] = None
    if specs_a.status == "verified" and specs_b.status == "verified":
        diffs: List[Dict[str, Any]] = []
        same: List[Dict[str, Any]] = []

        common_fields = set(specs_a.specs.keys()).intersection(set(specs_b.specs.keys()))
        for f in sorted(common_fields):
            a_val = specs_a.specs.get(f)
            b_val = specs_b.specs.get(f)
            if a_val != b_val:
                diffs.append({"field": f, "label": f.replace("_", " ").title(), "rv_a": a_val, "rv_b": b_val})
            else:
                same.append({"field": f, "label": f.replace("_", " ").title(), "value": a_val})

        key_differences = RVCompareKeyDiff(
            differences=diffs,
            same_on_both=same,
            note="Computed only from fields present on both RVs with verified spec sources.",
        )

    comparison: Dict[str, Any] = {
        "specs": {
            "rv_a": specs_a.model_dump(),
            "rv_b": specs_b.model_dump(),
            "note": "Specs are factual only when the status is verified and a source_url is provided.",
        }
    }
    if key_differences:
        comparison["key_differences"] = key_differences.model_dump()

    overall_conf = DataConfidence(
        level="partial" if (specs_a.status == "verified" and specs_b.status == "verified") else "stub",
        meaning="This tool is factual only where verified sources exist. Missing data remains unknown; we do not guess.",
        can_be_used_as_fact=(specs_a.status == "verified" and specs_b.status == "verified"),
    )

    return RVCompareResponse(
        rv_a={"input": req.rv_a.model_dump(), "data_status": "found" if specs_a.specs else "not_found"},
        rv_b={"input": req.rv_b.model_dump(), "data_status": "found" if specs_b.specs else "not_found"},
        trust_disclosures=trust_disclosures,
        unknowns=unknowns,
        comparison=comparison,
        data_confidence=overall_conf,
    )


@app.post("/tools/cost_depreciation_estimate", response_model=CostDepreciationResponse)
def cost_depreciation_estimate(req: CostDepreciationRequest) -> CostDepreciationResponse:
    disclosures = [
        "Estimates only. Not financial advice. Actual resale value varies widely by condition, mileage, region, demand, and maintenance history.",
        "This tool does not determine a fair price, negotiate, or recommend dealers.",
        "Resale value depends heavily on condition, maintenance, demand, and local market.",
    ]

    profile = req.profile or CostEstimateProfile()

    # Basic ranges (v1)
    # These are intentionally conservative and not “market guide” claims.
    year_1 = (0.12, 0.22)
    year_3 = (0.25, 0.40)
    year_5 = (0.35, 0.55)

    # Tiny adjustment examples (documented)
    applied_adjustments: List[Dict[str, Any]] = []
    if profile.miles_per_year is not None and profile.miles_per_year > 15000:
        year_3 = (min(0.60, year_3[0] + 0.03), min(0.70, year_3[1] + 0.03))
        year_5 = (min(0.80, year_5[0] + 0.04), min(0.85, year_5[1] + 0.04))
        applied_adjustments.append({"reason": "Higher annual mileage", "effect": "Slightly higher depreciation ranges"})

    if profile.storage in ("outdoor_sun", "outdoor_shade"):
        year_5 = (min(0.85, year_5[0] + 0.03), min(0.90, year_5[1] + 0.03))
        applied_adjustments.append({"reason": "Outdoor storage", "effect": "Slightly higher long-term depreciation risk"})

    if profile.maintenance_records in ("poor",):
        year_3 = (min(0.65, year_3[0] + 0.04), min(0.75, year_3[1] + 0.04))
        year_5 = (min(0.90, year_5[0] + 0.05), min(0.95, year_5[1] + 0.05))
        applied_adjustments.append({"reason": "Poor maintenance record quality", "effect": "Higher depreciation risk"})

    dollar_ranges = None
    if req.purchase_price_usd:
        p = float(req.purchase_price_usd)
        dollar_ranges = {
            "year_1_loss_usd_range": [round(p * year_1[0], 2), round(p * year_1[1], 2)],
            "year_3_total_loss_usd_range": [round(p * year_3[0], 2), round(p * year_3[1], 2)],
            "year_5_total_loss_usd_range": [round(p * year_5[0], 2), round(p * year_5[1], 2)],
        }

    what_more = [
        "Provide purchase_price_usd to get dollar ranges",
        "Provide miles_per_year and storage type",
        "Provide maintenance record quality (unknown/poor/average/excellent)",
    ]

    overall_conf = DataConfidence(
        level="partial",
        meaning="Estimates are based on a disclosed model and your inputs. Not guarantees. Confidence improves with more complete inputs.",
        can_be_used_as_fact=False,
    )

    dep = DepreciationEstimate(
        percent_ranges={
            "year_1_loss_pct_range": [year_1[0], year_1[1]],
            "year_3_total_loss_pct_range": [year_3[0], year_3[1]],
            "year_5_total_loss_pct_range": [year_5[0], year_5[1]],
        },
        dollar_ranges=dollar_ranges,
        note="Percent ranges are estimates from the model. Dollar ranges require purchase_price_usd.",
    )

    return CostDepreciationResponse(
        rv={
            "manufacturer": req.rv.manufacturer,
            "model": req.rv.model,
            "year": req.rv.year,
            "trim": req.rv.trim,
            "category_used": req.rv_category or "unknown",
            "category_source": "input" if req.rv_category else "unknown",
        },
        data_confidence=overall_conf,
        model_meta={"model_version": "v1", "last_updated": "2026-01-10"},
        applied_adjustments=applied_adjustments,
        disclosures=disclosures,
        what_would_make_this_more_precise=what_more,
        depreciation_estimate=dep,
    )


@app.post("/tools/deal_risk_scan", response_model=DealRiskScanResponse)
def deal_risk_scan(req: DealRiskScanRequest) -> DealRiskScanResponse:
    """
    Deal clarity scan. MUST work even when RV identity (manufacturer/model/trim) is unknown.
    Flags clarity risks and cost drivers based on provided inputs. Does not prove wrongdoing.
    """

    fees = req.fees or []
    total_fees = sum((f.amount_usd for f in fees), 0.0)

    has_financing = req.financing is not None and (
        req.financing.apr_percent is not None
        or req.financing.term_months is not None
        or req.financing.total_amount_financed_usd is not None
    )

    has_trade_in = req.trade_in is not None and (
        req.trade_in.has_trade_in
        or req.trade_in.offered_trade_in_value_usd is not None
        or req.trade_in.payoff_amount_usd is not None
    )

    buyer_priorities = req.buyer_priorities or []

    clarifying_questions: List[str] = []
    flags: List[Dict[str, Any]] = []

    # Clarifying questions (never block)
    if req.quoted_unit_price_usd is None:
        clarifying_questions.append("What is the quoted unit price (before taxes/fees)?")

    if not fees:
        clarifying_questions.append(
            "Can you list the taxes/fees/add-ons line by line (name + amount), even if some are $0?"
        )

    if req.rv is None:
        clarifying_questions.append(
            "Optional (improves accuracy): What RV is this? (manufacturer + model + year + trim or a VIN/spec link)"
        )

    if has_financing:
        if req.financing.apr_percent is None:
            clarifying_questions.append("What is the APR (%)?")
        if req.financing.term_months is None:
            clarifying_questions.append("What is the loan term (months)?")
        if req.financing.total_amount_financed_usd is None:
            clarifying_questions.append("What is the total amount financed (USD)?")

    if has_trade_in:
        if req.trade_in.offered_trade_in_value_usd is None:
            clarifying_questions.append("What trade-in value are they offering (USD)?")
        if req.trade_in.payoff_amount_usd is None:
            clarifying_questions.append("What is your payoff amount on the trade-in (USD)?")

    # Flags (neutral; do not accuse)
    for f in fees:
        name_lower = (f.name or "").lower()

        if f.disclosed_as_optional is None and f.category in ("addon", "warranty", "other"):
            flags.append(
                {
                    "type": "clarity_risk",
                    "severity": "medium",
                    "description": f"'{f.name}' is not clearly marked as optional vs required. Ask for an out-the-door quote with and without it.",
                }
            )

        if f.category in ("doc_fee", "prep", "delivery", "addon", "warranty"):
            flags.append(
                {
                    "type": "cost_driver",
                    "severity": "low",
                    "description": f"{f.name} — ask what it covers, whether it’s required, and whether it can be removed or reduced.",
                }
            )

        if ("protection" in name_lower or "package" in name_lower or "prep" in name_lower) and f.disclosed_as_optional is None:
            flags.append(
                {
                    "type": "clarity_risk",
                    "severity": "medium",
                    "description": f"'{f.name}' sounds like an add-on package. Ask for a baseline quote with all optional items removed.",
                }
            )

    if req.rv is None:
        flags.append(
            {
                "type": "verification_limit",
                "severity": "medium",
                "description": "RV identity not provided, so this scan cannot verify whether the price matches market ranges. It can still flag fee/financing clarity risks.",
            }
        )

    suggested_next_move = "pause_and_clarify" if clarifying_questions else "compare_out_the_door"
    reason = (
        "Key details are missing. Getting a clean itemized breakdown reduces surprise and makes comparisons fair."
        if suggested_next_move == "pause_and_clarify"
        else "You have enough detail to compare out-the-door totals and identify cost drivers."
    )

    confidence_level = "partial" if (req.quoted_unit_price_usd is not None or fees or has_financing or has_trade_in) else "stub"

    return DealRiskScanResponse(
        data_confidence=DataConfidence(
            level=confidence_level,  # type: ignore[arg-type]
            meaning="This scan flags clarity risks and cost drivers from provided inputs. It does not prove wrongdoing. Market-range checks require RV identity or a source link.",
            can_be_used_as_fact=False,
        ),
        summary={
            "quoted_unit_price_usd": req.quoted_unit_price_usd,
            "total_fees_usd": round(total_fees, 2) if fees else None,
            "fee_line_count": len(fees),
            "has_financing_info": has_financing,
            "has_trade_in": has_trade_in,
            "buyer_priorities": buyer_priorities,
        },
        clarifying_questions=clarifying_questions,
        flags=flags,
        scripts_to_use_with_dealer=[
            {
                "purpose": "Ask for a clean, comparable out-the-door number",
                "script": "Can you give me an itemized out-the-door total showing: unit price, taxes, required government fees, dealer fees, and optional add-ons—each labeled clearly?",
            },
            {
                "purpose": "Remove optional items to set a baseline",
                "script": "Please show me the out-the-door total with all optional add-ons removed, so I can decide on add-ons separately.",
            },
            {
                "purpose": "Clarify financing in plain language",
                "script": "Can you confirm the APR, term length, total amount financed, and whether there is any prepayment penalty?",
            },
        ],
        checkpoint={"suggested_next_move": suggested_next_move, "reason": reason},
        trust_disclosures=[
            "Educational use only; not legal or financial advice.",
            "This tool does not accuse any dealer or seller of misconduct.",
            "The goal is informed consent: clear numbers, clear optionality, and understandable terms.",
        ],
    )


@app.post("/tools/maintenance_repair_triage", response_model=MaintenanceRepairTriageResponse)
def maintenance_repair_triage(req: MaintenanceRepairTriageRequest) -> MaintenanceRepairTriageResponse:
    red = req.red_flags or TriageRedFlags()

    stop_triggers: List[str] = []
    if red.propane_smell:
        stop_triggers.append("propane_smell")
    if red.smoke_or_burning_smell:
        stop_triggers.append("smoke_or_burning_smell")
    if red.carbon_monoxide_alarm:
        stop_triggers.append("carbon_monoxide_alarm")
    if red.sparking_or_arcing:
        stop_triggers.append("sparking_or_arcing")
    if red.active_water_near_electrical:
        stop_triggers.append("active_water_near_electrical")
    if red.brake_or_steering_issue:
        stop_triggers.append("brake_or_steering_issue")
    if red.engine_overheating:
        stop_triggers.append("engine_overheating")

    if stop_triggers:
        return MaintenanceRepairTriageResponse(
            safety_level="stop_now",
            reason="One or more high-risk safety red flags were selected. This tool will not provide DIY troubleshooting steps for these conditions.",
            stop_triggers=stop_triggers,
            immediate_actions=[
                "Stop using the RV system involved right now.",
                "Move to fresh air if there is any smell of gas, smoke, or a CO alarm.",
                "If safe to do so, turn off the relevant supply (for example: propane at the tank, shore power/breakers, generator).",
                "Contact a qualified RV technician or emergency services if you believe there is immediate danger.",
            ],
            what_to_tell_the_pro=[
                "Your RV category (motorhome/towable) and system involved",
                "Exactly what you smelled/heard/observed and when it started",
                "Whether shore power / generator / propane was on at the time",
                "Any alarms or error codes",
            ],
            trust_disclosures=[
                "Safety-first guidance only; not professional repair advice.",
                "If there is any immediate danger, prioritize safety and professional help.",
            ],
        )

    return MaintenanceRepairTriageResponse(
        safety_level="caution",
        reason="No immediate high-risk red flags were selected. This is general guidance only; if symptoms worsen, seek a qualified technician.",
        immediate_actions=[
            "Describe when the symptom happens (startup, under load, only when on shore power, only when on propane, etc.).",
            "If you can do so safely, take photos and notes of any error codes or indicator lights.",
            "If you smell anything unusual or an alarm activates, stop and seek professional help.",
        ],
        what_to_tell_the_pro=[
            "RV make/model/year (if known) and system involved",
            "Symptom description and when it occurs",
            "Any recent service, modifications, or environmental conditions (freezing, heavy rain, etc.)",
        ],
        trust_disclosures=[
            "Educational guidance only; not professional mechanical advice.",
            "If safety risk appears, stop and seek professional help.",
        ],
    )


# ============================
# Debug endpoints (hidden)
# ============================

@app.get("/debug/env", include_in_schema=False)
def debug_env() -> Dict[str, Any]:
    return {"PUBLIC_BASE_URL": os.getenv("PUBLIC_BASE_URL"), "PORT": os.getenv("PORT")}


@app.get("/debug/openapi", include_in_schema=False)
def debug_openapi() -> Dict[str, Any]:
    schema = app.openapi()
    return {
        "PUBLIC_BASE_URL": os.getenv("PUBLIC_BASE_URL"),
        "has_servers": bool(schema.get("servers")),
        "servers": schema.get("servers"),
        "keys_at_top": list(schema.keys()),
    }


# ============================
# OpenAPI customization (critical for Actions)
# ============================

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # IMPORTANT: Actions need a valid servers[0].url
    # Set on Render as: PUBLIC_BASE_URL=https://rv-confidence-api.onrender.com
    public_base = os.getenv("PUBLIC_BASE_URL")
    if public_base:
        schema["servers"] = [{"url": public_base}]

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi
