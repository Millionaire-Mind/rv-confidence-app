from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal, Any, Dict, List, Tuple
from datetime import datetime, timezone
import json
from pathlib import Path

app = FastAPI(
    title="RV Buyer & Owner Confidence Assistant (Local Dev)",
    version="0.7.0",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

WARRANTY_FILE = DATA_DIR / "manufacturer_warranties.json"
CONSTRUCTION_FILE = DATA_DIR / "construction_profiles.json"
VERIFIED_CONSTRUCTION_FILE = DATA_DIR / "verified_construction.json"
RV_MODELS_FILE = DATA_DIR / "rv_models.json"
DEPRECIATION_MODEL_FILE = DATA_DIR / "depreciation_model.json"


# -----------------------------
# Helpers (safe loading)
# -----------------------------
def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_placeholder(text: Optional[str]) -> bool:
    if not text:
        return True
    t = text.strip().lower()
    return ("placeholder" in t) or ("paste_" in t) or ("do not use as fact" in t)


def _norm_key(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return " ".join(s.strip().split())


def _find_rv_model(models_db: Dict[str, Any], manufacturer: str, model: str, year: Optional[int], trim: Optional[str]) -> Optional[Dict[str, Any]]:
    items = models_db.get("models", []) if isinstance(models_db, dict) else []
    mfg = _norm_key(manufacturer)
    mdl = _norm_key(model)
    trm = _norm_key(trim) if trim else None

    for item in items:
        if not isinstance(item, dict):
            continue
        if _norm_key(item.get("manufacturer")) != mfg:
            continue
        if _norm_key(item.get("model")) != mdl:
            continue
        if year is not None and item.get("year") != year:
            continue
        if trm is not None and _norm_key(item.get("trim")) != trm:
            continue
        return item
    return None


def _add_ranges(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])


def _clamp_range(r: Tuple[float, float], lo: float = 0.0, hi: float = 0.95) -> Tuple[float, float]:
    return (max(lo, min(hi, r[0])), max(lo, min(hi, r[1])))


def _money_range_from_pct(purchase_price: float, loss_pct_range: Tuple[float, float]) -> Tuple[float, float]:
    return (purchase_price * loss_pct_range[0], purchase_price * loss_pct_range[1])


def _remaining_value_range(purchase_price: float, total_loss_pct_range: Tuple[float, float]) -> Tuple[float, float]:
    low_remaining = purchase_price * (1 - total_loss_pct_range[1])
    high_remaining = purchase_price * (1 - total_loss_pct_range[0])
    return (low_remaining, high_remaining)


def _format_usd(x: float) -> str:
    return f"${x:,.0f}"


# -----------------------------
# Health + Tool registry
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "rv-confidence",
        "time_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/tools")
def list_tools():
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
                "purpose": "Compares two RVs using source-backed data when available, otherwise flags unknowns and provides educational guidance.",
                "read_only": True,
                "open_world": True,
            },
            {
                "name": "cost_depreciation_estimate",
                "path": "/tools/cost_depreciation_estimate",
                "purpose": "Estimates depreciation and ownership cost ranges using a disclosed model and user inputs. Estimates only; not financial advice.",
                "read_only": True,
                "open_world": True,
            },
            {
                "name": "deal_risk_scan",
                "path": "/tools/deal_risk_scan",
                "purpose": "Traffic-light risk scan of an RV deal quote. Educational: flags unclear or potentially costly items and provides questions to ask.",
                "read_only": True,
                "open_world": True,
            },
            {
                "name": "maintenance_repair_triage",
                "path": "/tools/maintenance_repair_triage",
                "purpose": "Post-purchase maintenance & repair triage. Safety-first: classifies urgency and provides safe checks + when to call a professional.",
                "read_only": True,
                "open_world": True,
            }
        ]
    }


# -----------------------------
# Shared schemas
# -----------------------------
class RVIdentity(BaseModel):
    manufacturer: str = Field(..., min_length=1, max_length=80)
    model: str = Field(..., min_length=1, max_length=120)
    year: Optional[int] = Field(None, ge=1980, le=2100)
    trim: Optional[str] = Field(None, max_length=120)


# -----------------------------
# TOOL #1 — Manufacturer Intelligence
# -----------------------------
class ManufacturerIntelligenceRequest(BaseModel):
    manufacturer: str = Field(..., min_length=1, max_length=80)
    model: Optional[str] = Field(None, max_length=120)
    year: Optional[int] = Field(None, ge=1980, le=2100)
    trim: Optional[str] = Field(None, max_length=120)
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


@app.post("/tools/manufacturer_intelligence")
def manufacturer_intelligence(req: ManufacturerIntelligenceRequest):
    manufacturer = _norm_key(req.manufacturer)
    model = _norm_key(req.model) if req.model else None
    trim = _norm_key(req.trim) if req.trim else None

    warranties = _load_json(WARRANTY_FILE)
    constructions = _load_json(CONSTRUCTION_FILE)
    verified_construction = _load_json(VERIFIED_CONSTRUCTION_FILE)

    # Warranty
    w = warranties.get(manufacturer) if isinstance(warranties, dict) else None
    if w and isinstance(w, dict):
        w_source_url = w.get("source_url")
        w_summary = w.get("warranty_summary")
        w_last_verified = w.get("last_verified")
        w_coverage_notes = w.get("coverage_notes") or []
        warranty_verified = (not _is_placeholder(w_source_url)) and (not _is_placeholder(w_summary))
    else:
        w_source_url = None
        w_summary = None
        w_last_verified = None
        w_coverage_notes = []
        warranty_verified = False

    warranty_output = {
        "status": "verified" if warranty_verified else "unverified",
        "summary": w_summary if warranty_verified else "Not verified yet. I can’t state warranty terms as fact until we add an official source URL and verified summary.",
        "source_url": w_source_url if warranty_verified else None,
        "last_verified": w_last_verified if warranty_verified else None,
        "coverage_notes": w_coverage_notes if warranty_verified else [],
        "how_to_verify": None if warranty_verified else [
            "Add the official manufacturer warranty webpage or PDF URL into data/manufacturer_warranties.json",
            "Replace the placeholder warranty_summary with a short neutral summary of the warranty term and what it covers",
            "Set last_verified to today’s date when confirmed"
        ]
    }

    # Construction
    construction_types = constructions.get("construction_types", {}) if isinstance(constructions, dict) else {}
    vc = verified_construction.get(manufacturer) if isinstance(verified_construction, dict) else None

    vc_entry = None
    vc_scope = None
    if vc and isinstance(vc, dict):
        if model and vc.get(model) and isinstance(vc.get(model), dict):
            vc_entry = vc.get(model)
            vc_scope = "model"
        elif vc.get("default") and isinstance(vc.get("default"), dict):
            vc_entry = vc.get("default")
            vc_scope = "default"

    if vc_entry and isinstance(vc_entry, dict):
        c_source_url = vc_entry.get("source_url")
        c_summary = vc_entry.get("construction_summary")
        c_last_verified = vc_entry.get("last_verified")
        c_tags = vc_entry.get("construction_tags") or []
        construction_verified = (not _is_placeholder(c_source_url)) and (not _is_placeholder(c_summary))
    else:
        c_source_url = None
        c_summary = None
        c_last_verified = None
        c_tags = []
        construction_verified = False
        vc_scope = None

    verified_construction_output = {
        "status": "verified" if construction_verified else "unverified",
        "scope_used": vc_scope,
        "summary": c_summary if construction_verified else "Not verified yet. I can explain construction types generally, but I won’t claim this manufacturer/model’s construction as fact until we add a verified source.",
        "source_url": c_source_url if construction_verified else None,
        "last_verified": c_last_verified if construction_verified else None,
        "construction_tags": c_tags if construction_verified else [],
        "how_to_verify": None if construction_verified else [
            "Add an official manufacturer spec page, brochure PDF, or documentation URL into data/verified_construction.json",
            "Add a short neutral construction_summary (no marketing language) and set last_verified"
        ]
    }

    construction_output = {
        "status": "educational_with_optional_verification",
        "note": "Construction guidance includes (1) neutral definitions of common build types and (2) optional verified facts about a specific manufacturer/model only when a source-backed entry exists.",
        "construction_types": construction_types,
        "verified_construction_for_this_request": verified_construction_output
    }

    focus = req.focus or "full"
    if focus == "warranty":
        requested_verified = warranty_verified
        requested_meaning = "Warranty section is source-backed only when marked verified."
    elif focus == "construction":
        requested_verified = construction_verified
        requested_meaning = "Construction is factual only when the verified construction block is marked verified."
    elif focus == "full":
        requested_verified = bool(warranty_verified or construction_verified)
        requested_meaning = "In full mode, any section marked verified is source-backed; unverified sections are educational or placeholders."
    else:
        requested_verified = False
        requested_meaning = "Requested section is not implemented as source-backed yet; do not treat as factual claims."

    data_confidence = {
        "level": "partial" if requested_verified else "stub",
        "meaning": requested_meaning,
        "can_be_used_as_fact": bool(requested_verified)
    }

    output: Dict[str, Any] = {
        "manufacturer": manufacturer,
        "model": model,
        "year": req.year,
        "trim": trim,
        "data_confidence": data_confidence,
        "trust_disclosures": [
            "This is educational information, not legal or financial advice.",
            "No dealer recommendations. No sponsored content in v1.",
            "If something is not verified with a source, we will say so explicitly."
        ],
    }

    if focus in ("warranty", "full"):
        output["warranty"] = warranty_output
    if focus in ("construction", "full"):
        output["construction"] = construction_output
    if focus in ("quality_overview", "full"):
        output["quality_overview"] = {"status": "stub", "summary": "Not implemented yet. This will be neutral and source-backed (no hype)."}
    if focus in ("strengths_weaknesses", "full"):
        output["strengths_and_weaknesses"] = {"status": "stub", "strengths": [], "weaknesses": [], "evidence_notes": ["We will only state claims we can support with sources, and we will label uncertainty."]}
    if focus in ("who_its_for", "full"):
        output["buyer_fit"] = {"status": "stub", "best_for": [], "not_ideal_for": []}

    return {"tool": "manufacturer_intelligence", "input": req.model_dump(), "output": output}


# -----------------------------
# TOOL #2 — RV Compare (kept intentionally minimal)
# -----------------------------
class RVCompareRequest(BaseModel):
    rv_a: RVIdentity
    rv_b: RVIdentity
    focus: Optional[Literal["specs"]] = "specs"


@app.post("/tools/rv_compare")
def rv_compare(req: RVCompareRequest):
    models_db = _load_json(RV_MODELS_FILE)

    a = req.rv_a
    b = req.rv_b

    a_match = _find_rv_model(models_db, a.manufacturer, a.model, a.year, a.trim)
    b_match = _find_rv_model(models_db, b.manufacturer, b.model, b.year, b.trim)

    def safe_specs(match: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not match:
            return {}
        src = match.get("source_url")
        verified = not _is_placeholder(src)
        specs = match.get("specs") if isinstance(match.get("specs"), dict) else {}
        return {
            "status": "verified" if verified else "unverified",
            "source_url": src if verified else None,
            "last_verified": match.get("last_verified") if verified else None,
            "specs": specs
        }

    a_specs = safe_specs(a_match)
    b_specs = safe_specs(b_match)

    diffs = []
    same = []

    if a_specs.get("status") == "verified" and b_specs.get("status") == "verified":
        a_s = a_specs.get("specs", {})
        b_s = b_specs.get("specs", {})
        compare_fields = [
            ("sleeping_capacity", "Sleeping capacity"),
            ("black_tank_gal", "Black tank (gal)"),
            ("fresh_water_gal", "Fresh water (gal)"),
            ("gray_water_gal", "Gray water (gal)"),
            ("gvwr_lb", "GVWR (lb)"),
            ("gcwr_lb", "GCWR (lb)"),
            ("exterior_length_overall", "Exterior length overall"),
            ("exterior_width", "Exterior width"),
            ("exterior_height_with_ac", "Exterior height (with A/C)")
        ]
        for key, label in compare_fields:
            if key in a_s and key in b_s and a_s.get(key) is not None and b_s.get(key) is not None:
                if a_s.get(key) == b_s.get(key):
                    same.append({"field": key, "label": label, "value": a_s.get(key)})
                else:
                    diffs.append({"field": key, "label": label, "rv_a": a_s.get(key), "rv_b": b_s.get(key)})

    return {
        "tool": "rv_compare",
        "input": req.model_dump(),
        "output": {
            "rv_a_status": "found" if a_match else "not_found",
            "rv_b_status": "found" if b_match else "not_found",
            "specs": {"rv_a": a_specs, "rv_b": b_specs},
            "key_differences": {"differences": diffs, "same_on_both": same},
            "trust_disclosures": [
                "Specs are factual only when the dataset entry includes a verified source_url.",
                "If something is missing, we label it unknown rather than guessing."
            ]
        }
    }


# -----------------------------
# TOOL #3 — Cost & Depreciation
# -----------------------------
class CostEstimateProfile(BaseModel):
    miles_per_year: Optional[int] = Field(None, ge=0, le=60000)
    storage: Optional[Literal["garage", "covered", "outdoor_shade", "outdoor_sun"]] = None
    maintenance_records: Optional[Literal["unknown", "poor", "average", "excellent"]] = "unknown"


class CostDepreciationRequest(BaseModel):
    rv: RVIdentity
    purchase_price_usd: Optional[float] = Field(None, gt=0, le=5000000)
    rv_category: Optional[Literal["motorhome", "towable"]] = None
    profile: Optional[CostEstimateProfile] = None
    focus: Optional[Literal["depreciation_only", "ownership_only", "full"]] = "full"


@app.post("/tools/cost_depreciation_estimate")
def cost_depreciation_estimate(req: CostDepreciationRequest):
    dep_model = _load_json(DEPRECIATION_MODEL_FILE)
    models_db = _load_json(RV_MODELS_FILE)

    if not dep_model:
        return {"tool": "cost_depreciation_estimate", "input": req.model_dump(), "output": {"status": "error", "message": "Depreciation model not found. Create data/depreciation_model.json first."}}

    rv = req.rv
    profile = req.profile or CostEstimateProfile()

    inferred_category = None
    match = _find_rv_model(models_db, rv.manufacturer, rv.model, rv.year, rv.trim)
    if match and isinstance(match, dict):
        specs = match.get("specs") if isinstance(match.get("specs"), dict) else {}
        rv_type = specs.get("rv_type")
        if isinstance(rv_type, str):
            inferred_category = "motorhome" if "motorhome" in rv_type.lower() else None

    category = req.rv_category or inferred_category
    if category not in ("motorhome", "towable"):
        category = None

    curves = dep_model.get("default_curves", {}) if isinstance(dep_model, dict) else {}
    base = curves.get(category) if category and isinstance(curves, dict) else None

    if not base:
        return {
            "tool": "cost_depreciation_estimate",
            "input": req.model_dump(),
            "output": {
                "status": "unverified",
                "data_confidence": {"level": "stub", "meaning": "RV category not provided or could not be inferred; curve cannot be selected.", "can_be_used_as_fact": False},
                "what_you_can_do_next": ["Provide rv_category as 'motorhome' or 'towable'.", "Optionally provide purchase_price_usd to get dollar ranges."],
                "disclosures": [dep_model.get("disclosure", "Estimates only. Not financial advice.")]
            }
        }

    y1 = tuple(base.get("year_1_loss_pct_range", [0.0, 0.0]))
    y3 = tuple(base.get("year_3_total_loss_pct_range", [0.0, 0.0]))
    y5 = tuple(base.get("year_5_total_loss_pct_range", [0.0, 0.0]))

    adjustments = dep_model.get("adjustments", {}) if isinstance(dep_model, dict) else {}
    applied: List[Dict[str, Any]] = []

    high_miles = adjustments.get("high_miles_per_year")
    if profile.miles_per_year is not None and isinstance(high_miles, dict):
        trigger = high_miles.get("trigger_miles_per_year", 12000)
        add_loss = tuple(high_miles.get("add_loss_pct", [0.0, 0.0]))
        if profile.miles_per_year >= trigger:
            y1 = _add_ranges(y1, add_loss)
            y3 = _add_ranges(y3, add_loss)
            y5 = _add_ranges(y5, add_loss)
            applied.append({"rule": "high_miles_per_year", "add_loss_pct_range": list(add_loss), "reason": f"miles_per_year >= {trigger}"})

    outdoor_sun = adjustments.get("poor_storage_outdoor_sun")
    if profile.storage == "outdoor_sun" and isinstance(outdoor_sun, dict):
        add_loss = tuple(outdoor_sun.get("add_loss_pct", [0.0, 0.0]))
        y1 = _add_ranges(y1, add_loss)
        y3 = _add_ranges(y3, add_loss)
        y5 = _add_ranges(y5, add_loss)
        applied.append({"rule": "poor_storage_outdoor_sun", "add_loss_pct_range": list(add_loss), "reason": "storage == outdoor_sun"})

    excellent = adjustments.get("excellent_maintenance_records")
    if profile.maintenance_records == "excellent" and isinstance(excellent, dict):
        reduce = tuple(excellent.get("reduce_loss_pct", [0.0, 0.0]))
        y1 = (y1[0] - reduce[1], y1[1] - reduce[0])
        y3 = (y3[0] - reduce[1], y3[1] - reduce[0])
        y5 = (y5[0] - reduce[1], y5[1] - reduce[0])
        applied.append({"rule": "excellent_maintenance_records", "reduce_loss_pct_range": list(reduce), "reason": "maintenance_records == excellent"})

    y1 = _clamp_range(y1)
    y3 = _clamp_range(y3)
    y5 = _clamp_range(y5)

    pct = {
        "year_1_loss_pct_range": [round(y1[0], 3), round(y1[1], 3)],
        "year_3_total_loss_pct_range": [round(y3[0], 3), round(y3[1], 3)],
        "year_5_total_loss_pct_range": [round(y5[0], 3), round(y5[1], 3)]
    }

    dollars = None
    if req.purchase_price_usd is not None:
        p = req.purchase_price_usd
        y1_loss = _money_range_from_pct(p, y1)
        y3_rem = _remaining_value_range(p, y3)
        y5_rem = _remaining_value_range(p, y5)
        dollars = {
            "purchase_price_usd": p,
            "year_1_loss_usd_range": [_format_usd(y1_loss[0]), _format_usd(y1_loss[1])],
            "estimated_resale_value_year_3_usd_range": [_format_usd(y3_rem[0]), _format_usd(y3_rem[1])],
            "estimated_resale_value_year_5_usd_range": [_format_usd(y5_rem[0]), _format_usd(y5_rem[1])]
        }

    out: Dict[str, Any] = {
        "status": "ok",
        "rv": {"manufacturer": _norm_key(rv.manufacturer), "model": _norm_key(rv.model), "year": rv.year, "trim": _norm_key(rv.trim) if rv.trim else None, "category_used": category},
        "data_confidence": {"level": "partial", "meaning": "Estimates from a disclosed model. Not guarantees.", "can_be_used_as_fact": False},
        "model_meta": {"model_version": dep_model.get("version"), "last_updated": dep_model.get("last_updated")},
        "applied_adjustments": applied,
        "disclosures": [
            dep_model.get("disclosure", "Estimates only. Not financial advice."),
            "This tool does not determine a fair price, negotiate, or recommend dealers."
        ],
        "depreciation_estimate": {"status": "estimate", "percent_ranges": pct, "dollar_ranges": dollars}
    }

    return {"tool": "cost_depreciation_estimate", "input": req.model_dump(), "output": out}


# -----------------------------
# TOOL #4 — Deal Risk Scan (traffic-light flags)
# -----------------------------
class FeeLineItem(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    amount_usd: float = Field(..., ge=0, le=500000)
    category: Optional[Literal["tax", "title_registration", "doc_fee", "prep", "delivery", "warranty", "financing", "addon", "other"]] = "other"
    disclosed_as_optional: Optional[bool] = None


class FinancingTerms(BaseModel):
    apr_percent: Optional[float] = Field(None, ge=0, le=40)
    term_months: Optional[int] = Field(None, ge=0, le=360)
    down_payment_usd: Optional[float] = Field(None, ge=0, le=5000000)
    total_amount_financed_usd: Optional[float] = Field(None, ge=0, le=5000000)
    lender_name: Optional[str] = Field(None, max_length=120)


class TradeInInfo(BaseModel):
    has_trade_in: bool = False
    offered_trade_in_value_usd: Optional[float] = Field(None, ge=0, le=5000000)
    payoff_amount_usd: Optional[float] = Field(None, ge=0, le=5000000)


class DealRiskScanRequest(BaseModel):
    rv: Optional[RVIdentity] = None
    quoted_unit_price_usd: Optional[float] = Field(None, ge=0, le=5000000)
    fees: Optional[List[FeeLineItem]] = None
    financing: Optional[FinancingTerms] = None
    trade_in: Optional[TradeInInfo] = None
    buyer_priorities: Optional[List[Literal["lowest_total_cost", "lowest_monthly_payment", "lowest_upfront", "warranty_coverage", "simple_ownership", "resale_value"]]] = None


@app.post("/tools/deal_risk_scan")
def deal_risk_scan(req: DealRiskScanRequest):
    fees = req.fees or []
    financing = req.financing or FinancingTerms()
    trade = req.trade_in or TradeInInfo()
    priorities = req.buyer_priorities or []

    total_fees = sum(f.amount_usd for f in fees) if fees else 0.0
    quoted_price = req.quoted_unit_price_usd

    questions: List[str] = []
    if quoted_price is None or quoted_price == 0:
        questions.append("What is the quoted unit price (before taxes/fees)?")
    if not fees:
        questions.append("Can you list the taxes/fees/add-ons line by line (name + amount)?")

    flags: List[Dict[str, Any]] = []

    def add_flag(level: Literal["green", "yellow", "red"], title: str, why_it_matters: str, what_to_ask: List[str], evidence: Optional[Dict[str, Any]] = None):
        flags.append({
            "level": level,
            "title": title,
            "why_it_matters": why_it_matters,
            "what_to_ask": what_to_ask,
            "evidence": evidence or {}
        })

    if quoted_price and quoted_price > 0 and total_fees > 0:
        ratio = total_fees / quoted_price
        if ratio >= 0.12:
            add_flag("red", "Fees are a large share of the unit price",
                     "High fees can materially change the real price. Some fees are required; others may be negotiable or optional.",
                     ["Which fees are required by law vs dealer-charged?", "Which items are optional and removable?", "Provide an itemized out-the-door total."],
                     {"total_fees_usd": total_fees, "quoted_unit_price_usd": quoted_price, "fee_ratio": round(ratio, 3)})
        elif ratio >= 0.07:
            add_flag("yellow", "Fees are noticeable relative to the unit price",
                     "Clarity prevents surprise at signing.",
                     ["Please label each fee: required vs optional.", "If optional: what’s the price without it?"],
                     {"total_fees_usd": total_fees, "quoted_unit_price_usd": quoted_price, "fee_ratio": round(ratio, 3)})
        else:
            add_flag("green", "Fees appear proportionate (based on what was provided)",
                     "This doesn’t prove every fee is required; it suggests total fee load isn’t unusually high versus price.",
                     ["Still confirm which fees are required by law vs dealer-added."],
                     {"total_fees_usd": total_fees, "quoted_unit_price_usd": quoted_price, "fee_ratio": round(ratio, 3)})

    addon_lines = [f for f in fees if f.category in ("addon", "warranty") or ("warranty" in f.name.lower())]
    if addon_lines:
        unclear = [f for f in addon_lines if f.disclosed_as_optional is None]
        forced = [f for f in addon_lines if f.disclosed_as_optional is False]
        if forced:
            add_flag("red", "Add-ons appear included as non-optional",
                     "If add-ons aren’t optional, you may be paying for items you didn’t choose. Some add-ons can be valuable; the key is informed consent.",
                     ["Can you remove it and show the updated total?", "What exact coverage and exclusions?", "Is it refundable?"],
                     {"add_on_items": [{"name": f.name, "amount_usd": f.amount_usd, "category": f.category} for f in forced]})
        elif unclear:
            add_flag("yellow", "Some add-ons are not clearly marked optional",
                     "Unclear optionality increases confusion at signing.",
                     ["For each add-on: is it optional (yes/no)?", "What’s the out-the-door price without it?"],
                     {"add_on_items": [{"name": f.name, "amount_usd": f.amount_usd, "category": f.category} for f in unclear]})
        else:
            add_flag("green", "Add-ons appear clearly labeled as optional",
                     "Optional add-ons can be evaluated calmly when disclosed clearly.",
                     ["Ask for a baseline out-the-door number with all optional items removed."],
                     {"add_on_items": [{"name": f.name, "amount_usd": f.amount_usd, "category": f.category} for f in addon_lines]})

    if trade.has_trade_in and trade.offered_trade_in_value_usd is not None and trade.payoff_amount_usd is not None:
        neg_equity = trade.payoff_amount_usd - trade.offered_trade_in_value_usd
        if neg_equity > 0:
            add_flag("yellow", "Trade-in may have negative equity",
                     "If payoff exceeds trade value, the difference may be rolled into the new loan, increasing total cost.",
                     ["Will negative equity be added to the new loan? How much?", "Show trade value, payoff, and net separately on the deal sheet."],
                     {"offered_trade_in_value_usd": trade.offered_trade_in_value_usd, "payoff_amount_usd": trade.payoff_amount_usd, "estimated_negative_equity_usd": neg_equity})

    scripts = [
        {"purpose": "Ask for an itemized out-the-door total", "script": "Can you give me an itemized out-the-door total showing: unit price, taxes, required government fees, dealer fees, and optional add-ons—each labeled clearly?"},
        {"purpose": "Remove optional items to set a baseline", "script": "Please show me the out-the-door total with all optional add-ons removed, so I can decide on add-ons separately."},
        {"purpose": "Clarify financing", "script": "Can you confirm the APR, term length, total amount financed, and whether there is any prepayment penalty?"}
    ]

    checkpoint = {"suggested_next_move": "pause_and_clarify", "reason": "Key details are missing. Getting a clean itemized breakdown reduces surprise and makes comparisons fair."} if questions else {
        "suggested_next_move": "slow_down_and_verify" if any(f.get("level") in ("red", "yellow") for f in flags) else "proceed_if_comfortable",
        "reason": "Educational guidance only. Use the questions above to verify details before signing."
    }

    return {
        "tool": "deal_risk_scan",
        "input": req.model_dump(),
        "output": {
            "status": "ok",
            "data_confidence": {"level": "partial" if not questions else "stub", "meaning": "Flags clarity risks and cost drivers based on provided inputs; does not prove wrongdoing.", "can_be_used_as_fact": False},
            "summary": {"quoted_unit_price_usd": quoted_price, "total_fees_usd": total_fees if fees else None, "fee_line_count": len(fees), "has_trade_in": bool(trade.has_trade_in), "buyer_priorities": priorities},
            "clarifying_questions": questions,
            "flags": flags,
            "scripts_to_use_with_dealer": scripts,
            "checkpoint": checkpoint,
            "trust_disclosures": [
                "Educational use only; not legal or financial advice.",
                "This tool does not accuse any dealer of misconduct.",
                "Goal: informed consent—clear numbers, clear optionality, understandable terms."
            ]
        }
    }


# -----------------------------
# TOOL #5 — Maintenance & Repair Triage (SAFETY-FIRST)
# -----------------------------
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
    system: Literal[
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
        "engine_chassis"
    ]
    symptoms_text: str = Field(..., min_length=3, max_length=2000)
    red_flags: Optional[TriageRedFlags] = None
    context: Optional[MaintenanceTriageContext] = None


@app.post("/tools/maintenance_repair_triage")
def maintenance_repair_triage(req: MaintenanceRepairTriageRequest):
    rf = req.red_flags or TriageRedFlags()
    ctx = req.context or MaintenanceTriageContext()

    # ---------
    # Safety gate: STOP conditions
    # ---------
    stop_triggers = []
    if rf.propane_smell:
        stop_triggers.append("propane_smell")
    if rf.smoke_or_burning_smell:
        stop_triggers.append("smoke_or_burning_smell")
    if rf.carbon_monoxide_alarm:
        stop_triggers.append("carbon_monoxide_alarm")
    if rf.active_water_near_electrical:
        stop_triggers.append("active_water_near_electrical")
    if rf.brake_or_steering_issue:
        stop_triggers.append("brake_or_steering_issue")
    if rf.engine_overheating:
        stop_triggers.append("engine_overheating")
    if rf.sparking_or_arcing:
        stop_triggers.append("sparking_or_arcing")

    if stop_triggers:
        immediate_actions = [
            "Stop using the RV system involved right now.",
            "Move to fresh air if there is any smell of gas, smoke, or a CO alarm.",
            "If safe to do so, turn off the relevant supply (for example: propane at the tank, shore power/breakers, generator).",
            "Contact a qualified RV technician or emergency services if you believe there is immediate danger."
        ]

        return {
            "tool": "maintenance_repair_triage",
            "input": req.model_dump(),
            "output": {
                "status": "ok",
                "safety_level": "stop_now",
                "reason": "One or more high-risk safety red flags were selected. This tool will not provide DIY troubleshooting steps for these conditions.",
                "stop_triggers": stop_triggers,
                "immediate_actions": immediate_actions,
                "what_to_tell_the_pro": [
                    "Your RV category (motorhome/towable) and system involved",
                    "Exactly what you smelled/heard/observed and when it started",
                    "Whether shore power / generator / propane was on at the time",
                    "Any alarms or error codes"
                ],
                "trust_disclosures": [
                    "Safety-first guidance only; not professional repair advice.",
                    "If there is any immediate danger, prioritize safety and professional help."
                ]
            }
        }

    # ---------
    # Non-emergency: safe, educational triage
    # ---------
    safety_level: Literal["caution", "routine"] = "caution"

    # A few general safe checks by system (no dangerous instructions)
    steps: List[str] = []
    likely_causes: List[str] = []
    what_to_collect: List[str] = []
    when_to_stop_and_call_pro: List[str] = []

    # Common “safe-first” reminders
    base_reminders = [
        "If you notice gas smell, smoke/burning smell, sparking, or a CO alarm at any time: stop and seek professional help immediately.",
        "If you are unsure or uncomfortable, it’s okay to stop and call a qualified RV technician."
    ]

    if req.system == "water_pump_plumbing":
        safety_level = "routine"
        likely_causes = [
            "Fresh water tank is empty or valve is set to wrong source",
            "Pump has lost prime (air in the line)",
            "Inline screen/filter at the pump is clogged",
            "A valve is closed (winterization/bypass)",
            "Air leak on suction side (loose fitting)"
        ]
        steps = [
            "Confirm the fresh water tank level is not empty.",
            "Confirm the water source selector (city vs tank) is set correctly (if your RV has one).",
            "Open one cold faucet fully and listen: does the pump run continuously or cycle on/off?",
            "Check if any winterization/bypass valves are in the wrong position (common after storage).",
            "If your pump has an accessible clear strainer bowl, visually check if it looks full of water or full of air (do not force or overtighten parts).",
            "If you recently had freezing weather, consider that a frozen line can block flow—avoid forcing the pump and consider professional help."
        ]
        what_to_collect = [
            "Whether pump runs continuously or cycles",
            "Tank level, and whether city water works (if available)",
            "Any recent winterization or valve changes",
            "Any visible leaks or wet areas"
        ]
        when_to_stop_and_call_pro = [
            "You see active leaking and cannot identify the source",
            "Pump runs continuously with no water flow after basic checks",
            "You suspect frozen lines or burst plumbing"
        ]

    elif req.system in ("electrical_12v", "electrical_120v"):
        safety_level = "caution"
        likely_causes = [
            "Battery disconnect is off (12V)",
            "Tripped breaker or GFCI (120V)",
            "Low battery state of charge (12V)",
            "Blown fuse (12V circuit)",
            "Loose shore power connection (120V)"
        ]
        steps = [
            "If using shore power, confirm the cord is fully seated and the pedestal breaker is on.",
            "Check for a tripped GFCI outlet (often in bathroom/kitchen) and reset if tripped.",
            "Check your RV’s breaker panel for any tripped breakers and reset once (if it immediately trips again, stop).",
            "For 12V issues, confirm the battery disconnect switch is ON (if equipped).",
            "Check if other 12V items work (lights, fans). If none work, the issue may be upstream (battery/disconnect/main fuse)."
        ]
        what_to_collect = [
            "Which items are not working (everything vs one circuit)",
            "Whether you’re on shore power, generator, or battery",
            "Any recent changes (battery replacement, storage, upgrades)",
            "Any error indicators on inverter/charger (if present)"
        ]
        when_to_stop_and_call_pro = [
            "A breaker/GFCI trips repeatedly after one reset attempt",
            "You see heat, melting, or smell burning (stop immediately—use red flags next time)",
            "You are unsure how to access panels safely"
        ]

    elif req.system == "propane":
        safety_level = "caution"
        likely_causes = [
            "Propane is turned off at the tank",
            "Appliance needs to purge air from the line after refill",
            "Low propane level",
            "Appliance ignition fault"
        ]
        steps = [
            "Confirm propane cylinder/tank valve is open (do not force it).",
            "Confirm propane level is not empty.",
            "Try one propane appliance that is designed for simple startup (follow the appliance’s manual steps).",
            "If an appliance fails to ignite repeatedly, stop and avoid repeated attempts—follow the manufacturer guidance or call service."
        ]
        what_to_collect = [
            "Which appliance(s) fail (furnace, fridge, stove)",
            "Whether propane was recently refilled",
            "Any appliance error codes or indicator lights"
        ]
        when_to_stop_and_call_pro = [
            "You ever smell gas (use the red-flag propane_smell next time)",
            "Repeated ignition failures that persist after basic checks"
        ]

    elif req.system == "roof_leaks":
        safety_level = "caution"
        likely_causes = [
            "Sealant failure around roof penetrations (vents, skylights, antennas)",
            "Window/marker light seal failure",
            "Water tracking from a higher point than where it appears"
        ]
        steps = [
            "If safe, inspect interior for where water first appears (ceiling, wall corner, around a vent).",
            "Take photos of the wet area and any staining pattern.",
            "If safe to access, visually inspect roof penetrations for obvious cracks/gaps in sealant (do not attempt repairs in unsafe conditions).",
            "Dry the area as best you can and avoid running electrical items near wet zones."
        ]
        what_to_collect = [
            "When it happens (rain, washing, driving)",
            "Exact location of first moisture",
            "Photos of interior staining and any exterior suspect areas"
        ]
        when_to_stop_and_call_pro = [
            "Water is near electrical fixtures or the breaker panel (use red_flag active_water_near_electrical next time)",
            "You cannot safely access the roof",
            "Leak continues and you can’t identify the source"
        ]

    else:
        # Generic safe triage for other systems
        safety_level = "caution"
        likely_causes = ["Multiple possible causes depending on RV model and component design."]
        steps = [
            "Identify the exact component (brand/model if visible) and any error codes/indicator lights.",
            "Describe when the issue happens (always vs intermittent, after storage, only on shore power, etc.).",
            "Check your owner’s manual for any basic reset steps and safety notes for that component.",
            "Avoid repeated resets or cycling power if behavior seems abnormal."
        ]
        what_to_collect = [
            "Component brand/model label (photo if possible)",
            "Any error codes/messages",
            "What power source you were using (shore/generator/battery/propane)"
        ]
        when_to_stop_and_call_pro = [
            "If you are uncertain about safety or the system involves high voltage, propane, or engine/chassis concerns"
        ]

    return {
        "tool": "maintenance_repair_triage",
        "input": req.model_dump(),
        "output": {
            "status": "ok",
            "safety_level": safety_level,
            "summary": "Educational triage: likely causes, safe checks, and when to escalate. Not a guaranteed diagnosis.",
            "system": req.system,
            "likely_causes": likely_causes,
            "safe_checks": steps,
            "what_to_collect": what_to_collect,
            "when_to_stop_and_call_a_pro": when_to_stop_and_call_pro,
            "safety_reminders": base_reminders,
            "trust_disclosures": [
                "Educational information only; not professional repair advice.",
                "If conditions change (gas smell, smoke, CO alarm, sparking), stop and seek professional help."
            ]
        }
    }


# -----------------------------
# Safe error handling
# -----------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_error",
                "message": "Something went wrong. Please try again.",
            }
        },
    )
