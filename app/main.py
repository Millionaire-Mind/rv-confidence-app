from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, conint, confloat, constr


# --------------------------------------------------
# IMPORTANT:
# Disable FastAPI's built-in OpenAPI route so OUR
# /openapi.json is what GPT Actions imports.
# --------------------------------------------------
app = FastAPI(
    title="RV Buyer & Owner Confidence Assistant (Local Dev)",
    description="High-trust decision intelligence tools for RV buyers and owners.",
    version="0.7.4",
    openapi_url=None,   # critical
    docs_url=None,      # optional: keep off in production
    redoc_url=None,     # optional: keep off in production
)


# --------------------------------------------------
# Paths / data
# Repo structure:
#   rv-confidence-app/
#     app/main.py
#     data/*.json
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(name: str) -> Any:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm_key(s: str) -> str:
    return (s or "").strip().lower()


# Load datasets (safe defaults)
manufacturer_warranties = load_json("manufacturer_warranties.json")
verified_construction = load_json("verified_construction.json")
rv_models = load_json("rv_models.json")
depreciation_model = load_json("depreciation_model.json")


# --------------------------------------------------
# Models for tool inputs
# --------------------------------------------------
RVCategory = Literal["motorhome", "towable"]

BuyerPriority = Literal[
    "lowest_total_cost",
    "lowest_monthly_payment",
    "lowest_upfront",
    "warranty_coverage",
    "simple_ownership",
    "resale_value",
]

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


class RVIdentity(BaseModel):
    manufacturer: constr(min_length=1, max_length=80)
    model: constr(min_length=1, max_length=120)
    year: Optional[conint(ge=1980, le=2100)] = None
    trim: Optional[constr(min_length=1, max_length=120)] = None


class ManufacturerIntelligenceRequest(BaseModel):
    manufacturer: constr(min_length=1, max_length=80)
    model: Optional[constr(min_length=1, max_length=120)] = None
    year: Optional[conint(ge=1980, le=2100)] = None
    trim: Optional[constr(min_length=1, max_length=120)] = None
    focus: Optional[
        Literal[
            "warranty",
            "construction",
            "quality_overview",
            "strengths_weaknesses",
            "who_its_for",
            "full",
        ]
    ] = "full"


class RVCompareRequest(BaseModel):
    rv_a: RVIdentity
    rv_b: RVIdentity
    focus: Optional[Literal["specs"]] = "specs"


class CostEstimateProfile(BaseModel):
    miles_per_year: Optional[conint(ge=0, le=60000)] = None
    storage: Optional[Literal["garage", "covered", "outdoor_shade", "outdoor_sun"]] = None
    maintenance_records: Optional[Literal["unknown", "poor", "average", "excellent"]] = "unknown"


class CostDepreciationRequest(BaseModel):
    rv: RVIdentity
    purchase_price_usd: Optional[confloat(gt=0, le=5_000_000)] = None
    rv_category: Optional[RVCategory] = None
    profile: Optional[CostEstimateProfile] = None
    focus: Optional[Literal["depreciation_only", "ownership_only", "full"]] = "full"


class FeeLineItem(BaseModel):
    name: constr(min_length=1, max_length=120)
    amount_usd: confloat(ge=0, le=500_000)
    category: Optional[FeeCategory] = "other"
    disclosed_as_optional: Optional[bool] = None


class FinancingTerms(BaseModel):
    apr_percent: Optional[confloat(ge=0, le=40)] = None
    term_months: Optional[conint(ge=0, le=360)] = None
    down_payment_usd: Optional[confloat(ge=0, le=5_000_000)] = None
    total_amount_financed_usd: Optional[confloat(ge=0, le=5_000_000)] = None
    lender_name: Optional[constr(min_length=0, max_length=120)] = None


class TradeInInfo(BaseModel):
    has_trade_in: bool = False
    offered_trade_in_value_usd: Optional[confloat(ge=0, le=5_000_000)] = None
    payoff_amount_usd: Optional[confloat(ge=0, le=5_000_000)] = None


class DealRiskScanRequest(BaseModel):
    rv: Optional[RVIdentity] = None
    quoted_unit_price_usd: Optional[confloat(ge=0, le=5_000_000)] = None
    fees: Optional[List[FeeLineItem]] = None
    financing: Optional[FinancingTerms] = None
    trade_in: Optional[TradeInInfo] = None
    buyer_priorities: Optional[List[BuyerPriority]] = None


class TriageRedFlags(BaseModel):
    propane_smell: Optional[bool] = False
    smoke_or_burning_smell: Optional[bool] = False
    carbon_monoxide_alarm: Optional[bool] = False
    active_water_near_electrical: Optional[bool] = False
    brake_or_steering_issue: Optional[bool] = False
    engine_overheating: Optional[bool] = False
    sparking_or_arcing: Optional[bool] = False


class MaintenanceTriageContext(BaseModel):
    rv_category: Optional[RVCategory] = None
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


# --------------------------------------------------
# Non-tool endpoints (hidden from schema)
# --------------------------------------------------
@app.get("/health", include_in_schema=False)
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "rv-confidence", "time_utc": utc_now_iso()}


@app.get("/debug/env", include_in_schema=False)
def debug_env() -> Dict[str, Any]:
    return {"PUBLIC_BASE_URL": os.getenv("PUBLIC_BASE_URL"), "PORT": os.getenv("PORT")}


# --------------------------------------------------
# Tool 1: manufacturer_intelligence
# --------------------------------------------------
@app.post("/tools/manufacturer_intelligence")
def manufacturer_intelligence(req: ManufacturerIntelligenceRequest) -> Dict[str, Any]:
    trust_disclosures = [
        "This is educational information, not legal or financial advice.",
        "No dealer recommendations. No sponsored content in v1.",
        "If something is not verified with a source, we will say so explicitly.",
    ]

    output: Dict[str, Any] = {
        "manufacturer": req.manufacturer,
        "model": req.model,
        "year": req.year,
        "trim": req.trim,
        "trust_disclosures": trust_disclosures,
    }

    focus = req.focus or "full"
    m_key = norm_key(req.manufacturer)

    # Warranty (manufacturer-level only)
    if focus in ("warranty", "full"):
        entry = None
        for k, v in manufacturer_warranties.items():
            if norm_key(k) == m_key:
                entry = v
                break

        if entry and entry.get("source_url"):
            output["data_confidence"] = {
                "level": "partial",
                "meaning": "Sections marked verified are source-backed.",
                "can_be_used_as_fact": True,
            }
            output["warranty"] = {
                "status": "verified",
                "summary": entry.get("warranty_summary"),
                "source_url": entry.get("source_url"),
                "last_verified": entry.get("last_verified"),
                "coverage_notes": entry.get("coverage_notes", []),
                "how_to_verify": None,
            }
        else:
            output["data_confidence"] = {
                "level": "stub",
                "meaning": "Warranty terms are factual only when marked verified with a source_url.",
                "can_be_used_as_fact": False,
            }
            output["warranty"] = {
                "status": "unverified",
                "summary": "Not verified yet. I can’t state warranty terms as fact until we add an official source URL and verified summary.",
                "source_url": None,
                "last_verified": None,
                "coverage_notes": [],
                "how_to_verify": [
                    "Add the official manufacturer warranty webpage or PDF URL into data/manufacturer_warranties.json",
                    "Replace the placeholder warranty_summary with a short neutral summary of the warranty term and what it covers",
                    "Set last_verified to today’s date when confirmed",
                ],
            }

    # Construction (educational definitions + optional verified model facts)
    if focus in ("construction", "full"):
        construction_types = {
            "stick_and_tin": {
                "neutral_description": "Wood or aluminum framing with corrugated metal (aluminum) exterior siding; common in value-oriented travel trailers.",
                "tradeoffs": {
                    "pros": ["Often lower purchase price", "Repairs can be simpler for some exterior damage"],
                    "cons": [
                        "Seams and fasteners can increase water-intrusion risk if not maintained",
                        "May have lower insulation performance depending on build",
                    ],
                },
            },
            "laminated_fiberglass": {
                "neutral_description": "Layered wall construction with fiberglass exterior bonded to underlying materials; common in mid to higher tiers.",
                "tradeoffs": {
                    "pros": ["Often better exterior finish", "Can improve insulation and rigidity depending on design"],
                    "cons": [
                        "Repairs can be more complex if delamination or water damage occurs",
                        "May cost more upfront",
                    ],
                },
            },
        }

        verified_block = {
            "status": "unverified",
            "scope_used": "default",
            "summary": "Not verified yet. I can explain construction types generally, but I won’t claim this manufacturer/model’s construction as fact until we add a verified source.",
            "source_url": None,
            "last_verified": None,
            "construction_tags": [],
            "how_to_verify": [
                "Add an official manufacturer spec page, brochure PDF, or documentation URL into data/verified_construction.json",
                "Add a short neutral construction_summary (no marketing language) and set last_verified",
            ],
        }

        if req.model:
            lookup_key = f"{m_key}:{norm_key(req.model)}"
            entry = None
            for k, v in verified_construction.items():
                if norm_key(k) == lookup_key:
                    entry = v
                    break
            if entry and entry.get("source_url"):
                verified_block = {
                    "status": "verified",
                    "scope_used": entry.get("scope_used", "model"),
                    "summary": entry.get("construction_summary"),
                    "source_url": entry.get("source_url"),
                    "last_verified": entry.get("last_verified"),
                    "construction_tags": entry.get("construction_tags", []),
                    "how_to_verify": None,
                }

        output["construction"] = {
            "status": "educational_with_optional_verification",
            "note": "Construction guidance includes (1) neutral definitions of common build types and (2) optional verified facts about a specific manufacturer/model only when a source-backed entry exists.",
            "construction_types": construction_types,
            "verified_construction_for_this_request": verified_block,
        }

    # Placeholder: quality_overview
    if focus == "quality_overview":
        output["data_confidence"] = {
            "level": "stub",
            "meaning": "Requested section is not implemented as source-backed yet; do not treat as factual claims.",
            "can_be_used_as_fact": False,
        }
        output["quality_overview"] = {"status": "stub", "summary": "Not implemented yet. This will be neutral and source-backed (no hype)."}

    return {"tool": "manufacturer_intelligence", "input": req.model_dump(), "output": output}


# --------------------------------------------------
# Tool 2: rv_compare (verified-only comparisons)
# --------------------------------------------------
@app.post("/tools/rv_compare")
def rv_compare(req: RVCompareRequest) -> Dict[str, Any]:
    trust_disclosures = [
        "This comparison is factual only where source-backed model entries exist in the dataset.",
        "If a spec is missing or the model is not found, we will say so rather than guessing.",
        "This is educational information, not legal or financial advice.",
        "No dealer recommendations. No sponsored content in v1.",
    ]

    def lookup(rv: RVIdentity) -> Optional[Dict[str, Any]]:
        parts = [norm_key(rv.manufacturer), norm_key(rv.model)]
        if rv.year is not None:
            parts.append(str(rv.year))
        if rv.trim:
            parts.append(norm_key(rv.trim))
        want = ":".join(parts)
        for k, v in rv_models.items():
            if norm_key(k) == want:
                return v
        return None

    a = lookup(req.rv_a)
    b = lookup(req.rv_b)

    def pack(entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not entry:
            return {"status": "not_found"}
        if entry.get("source_url"):
            return {
                "status": "verified",
                "source_url": entry.get("source_url"),
                "last_verified": entry.get("last_verified"),
                "specs": entry.get("specs", {}),
            }
        return {"status": "unverified", "source_url": None, "last_verified": None, "specs": entry.get("specs", {})}

    rv_a_block = pack(a)
    rv_b_block = pack(b)

    unknowns: List[str] = []
    if rv_a_block.get("status") != "verified":
        unknowns.append("RV A specs are not verified.")
    if rv_b_block.get("status") != "verified":
        unknowns.append("RV B specs are not verified.")

    comparison = {"specs": {"rv_a": rv_a_block, "rv_b": rv_b_block, "note": "Specs are factual only when status is verified and a source_url is provided."}}

    # Compute key differences only if both verified
    if rv_a_block.get("status") == "verified" and rv_b_block.get("status") == "verified":
        a_specs = rv_a_block.get("specs", {})
        b_specs = rv_b_block.get("specs", {})
        fields = [
            ("sleeping_capacity", "Sleeping capacity"),
            ("black_tank_gal", "Black tank (gal)"),
            ("fresh_water_gal", "Fresh water (gal)"),
            ("gray_water_gal", "Gray water (gal)"),
            ("gvwr_lb", "GVWR (lb)"),
            ("gcwr_lb", "GCWR (lb)"),
            ("exterior_length_overall", "Exterior length overall"),
            ("exterior_width", "Exterior width"),
            ("exterior_height_with_ac", "Exterior height (with A/C)"),
        ]
        diffs, same = [], []
        for f, label in fields:
            if f in a_specs and f in b_specs:
                if a_specs[f] != b_specs[f]:
                    diffs.append({"field": f, "label": label, "rv_a": a_specs[f], "rv_b": b_specs[f]})
                else:
                    same.append({"field": f, "label": label, "value": a_specs[f]})
        comparison["key_differences"] = {
            "status": "computed_from_verified_specs",
            "differences": diffs,
            "same_on_both": same,
            "note": "Computed only from fields present on both RVs with verified spec sources.",
        }

    output = {
        "comparison_intent": req.focus or "specs",
        "rv_a": {"input": req.rv_a.model_dump(), "data_status": rv_a_block.get("status")},
        "rv_b": {"input": req.rv_b.model_dump(), "data_status": rv_b_block.get("status")},
        "trust_disclosures": trust_disclosures,
        "unknowns": unknowns,
        "comparison": comparison,
        "data_confidence": {"level": "partial" if not unknowns else "stub", "meaning": "Verified-only comparisons; missing data stays unknown.", "can_be_used_as_fact": len(unknowns) == 0},
    }

    return {"tool": "rv_compare", "input": req.model_dump(), "output": output}


# --------------------------------------------------
# Tool 3: cost_depreciation_estimate (disclosed ranges)
# --------------------------------------------------
@app.post("/tools/cost_depreciation_estimate")
def cost_depreciation_estimate(req: CostDepreciationRequest) -> Dict[str, Any]:
    category_used = req.rv_category or "motorhome"
    curves = depreciation_model.get("curves", {}) if isinstance(depreciation_model, dict) else {}

    defaults = {
        "motorhome": {"year_1_loss_pct_range": [0.12, 0.22], "year_3_total_loss_pct_range": [0.25, 0.40], "year_5_total_loss_pct_range": [0.35, 0.55]},
        "towable": {"year_1_loss_pct_range": [0.10, 0.20], "year_3_total_loss_pct_range": [0.22, 0.36], "year_5_total_loss_pct_range": [0.32, 0.50]},
    }

    curve = curves.get(category_used) or defaults.get(category_used) or defaults["motorhome"]

    y1 = curve["year_1_loss_pct_range"]
    y3 = curve["year_3_total_loss_pct_range"]
    y5 = curve["year_5_total_loss_pct_range"]

    dollar_ranges = None
    if req.purchase_price_usd:
        price = float(req.purchase_price_usd)
        dollar_ranges = {
            "year_1_loss_usd_range": [round(price * y1[0], 2), round(price * y1[1], 2)],
            "year_3_total_loss_usd_range": [round(price * y3[0], 2), round(price * y3[1], 2)],
            "year_5_total_loss_usd_range": [round(price * y5[0], 2), round(price * y5[1], 2)],
        }

    output = {
        "status": "ok",
        "rv": {**req.rv.model_dump(), "category_used": category_used, "category_source": ("input" if req.rv_category else "default")},
        "data_confidence": {"level": "partial", "meaning": "Estimates are disclosed ranges, not guarantees.", "can_be_used_as_fact": False},
        "model_meta": {"model_version": depreciation_model.get("model_version", "v1") if isinstance(depreciation_model, dict) else "v1", "last_updated": depreciation_model.get("last_updated", utc_now_iso()[:10]) if isinstance(depreciation_model, dict) else utc_now_iso()[:10]},
        "disclosures": [
            "Estimates only. Not financial advice. Actual resale value varies by condition, mileage, region, demand, and maintenance history.",
            "This tool does not negotiate, recommend dealers, or guarantee prices.",
        ],
        "depreciation_estimate": {
            "status": "estimate",
            "percent_ranges": {"year_1_loss_pct_range": y1, "year_3_total_loss_pct_range": y3, "year_5_total_loss_pct_range": y5},
            "dollar_ranges": dollar_ranges,
            "note": "Dollar ranges require purchase_price_usd." if not dollar_ranges else "Dollar ranges computed from purchase_price_usd.",
        },
    }

    return {"tool": "cost_depreciation_estimate", "input": req.model_dump(), "output": output}


# --------------------------------------------------
# Tool 4: deal_risk_scan (clarity risk scan)
# --------------------------------------------------
@app.post("/tools/deal_risk_scan")
def deal_risk_scan(req: DealRiskScanRequest) -> Dict[str, Any]:
    fees = req.fees or []
    total_fees = sum(float(f.amount_usd) for f in fees) if fees else None

    clarifying_questions = []
    if req.quoted_unit_price_usd is None:
        clarifying_questions.append("What is the quoted unit price (before taxes/fees)?")
    if not fees:
        clarifying_questions.append("Can you list the taxes/fees/add-ons line by line (name + amount)?")

    flags = []
    for f in fees:
        name_l = f.name.lower()

        # Heuristic: add-ons not clearly optional
        if (f.category == "addon" or "protection" in name_l or "package" in name_l) and f.disclosed_as_optional is None:
            flags.append(
                {
                    "type": "clarity_risk",
                    "severity": "medium",
                    "title": "Optional add-on not clearly labeled",
                    "detail": f"'{f.name}' might be optional. Ask for an out-the-door total with all optional items removed, then add back what you actually want.",
                }
            )

        if "prep" in name_l or "pdi" in name_l:
            flags.append(
                {
                    "type": "cost_driver",
                    "severity": "low",
                    "title": "Prep/PDI fee present",
                    "detail": f"'{f.name}' can vary by dealer. Ask what it covers and whether it’s required.",
                }
            )

        if "doc" in name_l or "documentation" in name_l:
            flags.append(
                {
                    "type": "cost_driver",
                    "severity": "low",
                    "title": "Doc fee present",
                    "detail": "Doc fees are common but vary. Ask whether it’s capped in your state and whether it’s negotiable.",
                }
            )

    checkpoint = {
        "suggested_next_move": "pause_and_clarify" if clarifying_questions else "review_and_compare",
        "reason": "Key details are missing. A clean itemized out-the-door number reduces surprise and makes comparisons fair."
        if clarifying_questions
        else "You have enough detail to compare out-the-door totals.",
    }

    output = {
        "status": "ok",
        "data_confidence": {"level": "partial" if not clarifying_questions else "stub", "meaning": "Flags clarity risks and cost drivers; does not prove wrongdoing.", "can_be_used_as_fact": False},
        "summary": {
            "quoted_unit_price_usd": req.quoted_unit_price_usd,
            "total_fees_usd": total_fees,
            "fee_line_count": len(fees),
            "has_financing_info": req.financing is not None,
            "has_trade_in": req.trade_in is not None and req.trade_in.has_trade_in,
            "buyer_priorities": req.buyer_priorities or [],
        },
        "clarifying_questions": clarifying_questions,
        "flags": flags,
        "scripts_to_use_with_dealer": [
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
        "checkpoint": checkpoint,
        "trust_disclosures": [
            "Educational use only; not legal or financial advice.",
            "This tool does not accuse any dealer of misconduct.",
            "The goal is informed consent: clear numbers, clear optionality, and understandable terms.",
        ],
    }

    return {"tool": "deal_risk_scan", "input": req.model_dump(), "output": output}


# --------------------------------------------------
# Tool 5: maintenance_repair_triage (safety-first)
# --------------------------------------------------
@app.post("/tools/maintenance_repair_triage")
def maintenance_repair_triage(req: MaintenanceRepairTriageRequest) -> Dict[str, Any]:
    rf = req.red_flags or TriageRedFlags()

    stop_triggers = []
    if rf.propane_smell:
        stop_triggers.append("propane_smell")
    if rf.carbon_monoxide_alarm:
        stop_triggers.append("carbon_monoxide_alarm")
    if rf.smoke_or_burning_smell:
        stop_triggers.append("smoke_or_burning_smell")
    if rf.sparking_or_arcing:
        stop_triggers.append("sparking_or_arcing")
    if rf.active_water_near_electrical:
        stop_triggers.append("active_water_near_electrical")
    if rf.brake_or_steering_issue:
        stop_triggers.append("brake_or_steering_issue")
    if rf.engine_overheating:
        stop_triggers.append("engine_overheating")

    if stop_triggers:
        output = {
            "status": "ok",
            "safety_level": "stop_now",
            "reason": "One or more high-risk safety red flags were selected. This tool will not provide DIY troubleshooting steps for these conditions.",
            "stop_triggers": stop_triggers,
            "immediate_actions": [
                "Stop using the RV system involved right now.",
                "Move to fresh air if there is any smell of gas, smoke, or a CO alarm.",
                "If safe to do so, turn off the relevant supply (for example: propane at the tank, shore power/breakers, generator).",
                "Contact a qualified RV technician or emergency services if you believe there is immediate danger.",
            ],
            "what_to_tell_the_pro": [
                "Your RV category (motorhome/towable) and system involved",
                "Exactly what you smelled/heard/observed and when it started",
                "Whether shore power / generator / propane was on at the time",
                "Any alarms or error codes",
            ],
            "trust_disclosures": [
                "Safety-first guidance only; not professional repair advice.",
                "If there is any immediate danger, prioritize safety and professional help.",
            ],
        }
        return {"tool": "maintenance_repair_triage", "input": req.model_dump(), "output": output}

    output = {
        "status": "ok",
        "safety_level": "caution",
        "reason": "No immediate high-risk red flags selected. Guidance is general and safety-first.",
        "next_steps": [
            "Identify when the symptom occurs (parked vs moving, shore power vs battery/generator, propane on vs off).",
            "Observe what you can safely without disassembly (visible leaks, tripped breakers, blown fuses).",
            "If the problem persists or you are unsure, schedule a qualified RV technician.",
        ],
        "questions_to_narrow_it_down": [
            "When did the symptom start and did anything change recently (service, weather, storage)?",
            "Is the RV connected to shore power or running on battery/generator?",
            "Does the symptom happen consistently or intermittently?",
        ],
        "trust_disclosures": [
            "General guidance only; not professional repair advice.",
            "If any danger signs appear (gas smell, smoke, CO alarm, arcing), stop and seek professional help immediately.",
        ],
    }
    return {"tool": "maintenance_repair_triage", "input": req.model_dump(), "output": output}


# --------------------------------------------------
# OUR OpenAPI endpoint (Actions imports this)
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
