"""
Article-level local AI classifier for European Security Monitor.

Production candidate:
- Local Ollama inference
- qwen3:4b-instruct
- think=False
- One classification per source URL
- Persistent CSV cache
- Resumable runs
- No SQLite writes

This file is intended to replace the temporary ai_article_pilot*.py scripts.
Later, update_data.py can import classify_dataframe_articles() from here.
"""

from pathlib import Path
from urllib import request
import argparse
import hashlib
import json
import re
import time

import pandas as pd


# ==================================================
# VERSION / PATHS
# ==================================================

CLASSIFIER_VERSION = "article-ai-v3"
POSTPROCESS_VERSION = "post-rules-v4.2"
FINAL_CLASSIFIER_VERSION = f"{CLASSIFIER_VERSION}+{POSTPROCESS_VERSION}"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "qa_article_level"
    / "article_level_dataset.csv"
)

AI_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "ai"
)

AI_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE_PATH = (
    AI_DATA_DIR
    / "article_classifications.csv"
)

ERROR_PATH = (
    AI_DATA_DIR
    / "article_classification_errors.csv"
)

SUMMARY_PATH = (
    AI_DATA_DIR
    / "article_classification_summary.csv"
)


# ==================================================
# OLLAMA
# ==================================================

OLLAMA_CHAT_URL = (
    "http://127.0.0.1:11434/api/chat"
)

OLLAMA_TAGS_URL = (
    "http://127.0.0.1:11434/api/tags"
)

DEFAULT_MODEL = "qwen3:4b-instruct"

REQUEST_TIMEOUT_SECONDS = 180
MAX_RETRIES = 2
RETRY_WAIT_SECONDS = 2

RANDOM_SEED = 42


# ==================================================
# DOMAINS / SCOPE
# ==================================================

DOMAINS = [
    "Defence & Military",
    "Conflict & Geopolitical Tensions",
    "Cybersecurity",
    "Energy Security",
    "Sanctions & Economic Security",
]

MONITORED_COUNTRIES = [
    "Albania",
    "Andorra",
    "Austria",
    "Belarus",
    "Belgium",
    "Bosnia and Herzegovina",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czechia",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Iceland",
    "Ireland",
    "Italy",
    "Kosovo",
    "Latvia",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Moldova",
    "Monaco",
    "Montenegro",
    "Netherlands",
    "North Macedonia",
    "Norway",
    "Poland",
    "Portugal",
    "Romania",
    "San Marino",
    "Serbia",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
    "Switzerland",
    "Türkiye",
    "Ukraine",
    "United Kingdom",
    "Vatican City",
    "Russia",
    "Georgia",
    "Armenia",
    "Azerbaijan",
]

SCOPE_TEXT = ", ".join(
    MONITORED_COUNTRIES
)


# ==================================================
# STRUCTURED OUTPUT
# ==================================================

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "geographic_centrality": {
            "type": "string",
            "enum": [
                "High",
                "Medium",
                "Low",
                "None",
            ],
        },
        "security_relevant": {
            "type": "boolean",
        },
        "primary_domain": {
            "type": "string",
            "enum": (
                DOMAINS
                + ["None"]
            ),
        },
        "secondary_domains": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": DOMAINS,
            },
        },
        "event_status": {
            "type": "string",
            "enum": [
                "Actual violence / combat",
                "Threat / warning",
                "Military posture / deployment",
                "Military cooperation / training",
                "Cyber incident",
                "Sanctions / economic coercion",
                "Diplomatic negotiation",
                "Strategic statement",
                "Background / analysis",
                "Non-security",
                "Unclear",
            ],
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "needs_human_review": {
            "type": "boolean",
        },
        "exclusion_reason": {
            "type": "string",
            "enum": [
                "None",
                "Outside project geography",
                "Incidental geographic connection",
                "Non-security content",
                "Local crime / ordinary policing",
                "Entertainment / lifestyle / sport",
                "Generic business / economy",
                "Generic technology / AI",
                "Generic energy business",
                "Travel / tourism",
                "Insufficient evidence",
            ],
        },
        "reason_short": {
            "type": "string",
        },
    },
    "required": [
        "geographic_centrality",
        "security_relevant",
        "primary_domain",
        "secondary_domains",
        "event_status",
        "confidence",
        "needs_human_review",
        "exclusion_reason",
        "reason_short",
    ],
    "additionalProperties": False,
}


# ==================================================
# PROMPT
# ==================================================

SYSTEM_INSTRUCTIONS = f"""
You are the article-level semantic classifier for European Security Monitor.

Classify the ARTICLE, not each individual GDELT event extracted from it.

Follow this order strictly.

A. GEOGRAPHIC CENTRALITY
========================
First decide whether one or more monitored countries are central to the article.

The monitored project scope is EXACTLY:
{SCOPE_TEXT}

Important:
Türkiye, Ukraine, United Kingdom, Vatican City and Russia ARE explicitly within
the monitored scope.

Geographic centrality:
- High: monitored countries are clearly central subjects.
- Medium: at least one monitored country is materially involved, sharing focus
  with countries outside the monitored scope.
- Low: monitored geography is only secondary or incidental.
- None: there is no meaningful monitored-country connection.

Do not confuse geographic centrality with security relevance.
A Danish citizenship article can be geographically High and still be
security_relevant=false.

B. SECURITY RELEVANCE
=====================
Then decide whether the article itself is substantively about one of these five
security domains.

1. Defence & Military
Military forces, NATO, deployments, bases, exercises, defence policy, weapons,
missiles, military procurement with strategic significance, deterrence or
military preparedness.

2. Conflict & Geopolitical Tensions
War, invasion, armed conflict, interstate threats, territorial disputes,
security-relevant negotiations, serious diplomatic confrontation, strategic
competition, state-to-state rivalry, coercion, escalation risk, or geopolitical
tensions.

IMPORTANT FOR THIS DOMAIN:
Actual combat is NOT required.
A serious rivalry or strategic competition involving a monitored country can
be security relevant even when the article is analytical or diplomatic.
For example, an article whose central subject is a growing Türkiye-Israel
rivalry in Syria or the Eastern Mediterranean is a geopolitical-security article.
Likewise, negotiations intended to manage a serious strategic rivalry can be
security relevant.

Do NOT use this rule for ordinary party politics, routine diplomacy, ceremonial
visits or generic international cooperation.

3. Cybersecurity
Cyberattacks, ransomware, state-linked hacking, major cyber incidents,
critical-infrastructure cyber threats or strategic cybersecurity policy.

Do NOT classify ordinary AI, apps, VPN comparisons, consumer privacy or generic
technology as Cybersecurity merely because GKG contains cyber themes.

4. Energy Security
Strategic energy supply, pipelines, energy coercion, critical energy
infrastructure, geopolitical gas/oil dependence, sabotage or strategic
electricity-security issues.

Do NOT classify ordinary energy-company, stock-market, investment or generic
renewable-energy articles as Energy Security.

5. Sanctions & Economic Security
Sanctions, export controls, asset freezes, embargoes, strategic trade
restrictions or economic coercion.

Do NOT classify generic business, markets or trade as sanctions.

C. EXCLUDE NON-SECURITY CONTENT
===============================
Exclude ordinary:
- domestic politics or party politics;
- citizenship/immigration administration;
- ceremonies and anniversaries;
- royalty or celebrity stories;
- local crime and ordinary policing;
- entertainment, film, comics, Eurovision, lifestyle or sport;
- tourism or generic travel-warning lists;
- business, finance or stock-market stories;
- technology/AI stories;
- energy-company stories without a strategic-security dimension.

D. EVIDENCE PRIORITY
====================
Use evidence in this order:
1. article_title — the main semantic evidence;
2. source_url_text — a secondary textual clue when it clearly mirrors the article topic;
3. monitored actor/location countries — geographic support only.

IMPORTANT:
The supplied country metadata can contain incidental quoted people or locations.
It must NEVER override the article title.

No CAMEO event code or GKG theme is supplied to you for the relevance decision.
This is deliberate: GDELT event metadata is noisy and will be used later only
to enrich articles already classified semantically.

If the headline is clearly ceremonial, historical, administrative, lifestyle,
business, local-political or otherwise non-security, classify it as non-security
even if the geographic metadata contains strategic countries.

E. DOMAIN / STATUS RULES
========================
Choose exactly one primary domain for a relevant article.
Only add secondary domains when they are genuinely substantial.

Use:
- Actual violence / combat: the TITLE itself reports actual fighting, an attack,
  strike, killing, death in combat, shelling, bombing or another occurred act
  of violence. Do not infer actual violence from background context alone.
- Threat / warning: threatened, possible or warned attack.
- Military posture / deployment: forces were deployed/positioned.
- Military cooperation / training: exercises or training.
- Cyber incident: cyber incident actually central.
- Sanctions / economic coercion: sanctions/coercion central.
- Diplomatic negotiation: actual talks/negotiations are central.
- Strategic statement: important security-related statement.
- Background / analysis: analysis, rivalry, strategic competition or explainer
  where the title does not itself report a newly occurring violent act.
- Non-security: excluded non-security story.
- Unclear: only when evidence genuinely cannot resolve status.

Examples:
- "Israel and Türkiye: A growing rivalry..." -> Background / analysis.
- "Could Syria become the unlikely bridge between Israel and Türkiye?"
  -> Background / analysis or Diplomatic negotiation, NOT Actual violence.
- "First Brit dies while fighting for Russia in Ukraine"
  -> Actual violence / combat.
- A headline centered on an ongoing "Ukraine War" should normally use
  Conflict & Geopolitical Tensions as primary domain unless the headline is
  specifically about deployment, procurement, exercises or another primarily
  military activity.

F. HARD CONSISTENCY
===================
If security_relevant=true:
- primary_domain cannot be "None";
- exclusion_reason MUST be "None".

If security_relevant=false:
- primary_domain MUST be "None";
- secondary_domains MUST be [].

If confidence < 0.75:
- needs_human_review=true.

If evidence conflicts:
- needs_human_review=true.

Do not invent facts beyond the supplied title and metadata.
reason_short must be one concise sentence.
"""


# ==================================================
# INPUT REDUCTION
# ==================================================

SECURITY_THEME_RE = re.compile(
    r"("
    r"MILITARY|ARMEDCONFLICT|CONFLICT|CYBER|SANCTION|"
    r"ENERGY|OIL|NATURALGAS|GAS_|DRONE|WEAPON|TERROR|"
    r"NEGOTIATION|PEACE|SECURITY|FORCEPOSTURE|CEASEFIRE|"
    r"UNREST|BLOCKADE|SIEGE|MARITIME"
    r")",
    flags=re.IGNORECASE,
)


def clean(value, max_chars=None):
    if value is None:
        text = ""
    else:
        try:
            if pd.isna(value):
                text = ""
            else:
                text = str(value)
        except Exception:
            text = str(value)

    text = text.strip()

    if max_chars is not None:
        text = text[:max_chars]

    return text


def filtered_themes(
    value,
    max_items=18,
):
    text = clean(value)

    if not text:
        return ""

    items = [
        item.strip()
        for item in text.split("|")
        if item.strip()
    ]

    selected = []
    seen = set()

    for item in items:
        if not SECURITY_THEME_RE.search(
            item
        ):
            continue

        if item in seen:
            continue

        seen.add(item)
        selected.append(item)

        if len(selected) >= max_items:
            break

    return " | ".join(
        selected
    )


def source_url_to_text(value):
    text = clean(value)
    if not text:
        return ""
    try:
        from urllib.parse import urlparse, unquote
        path = unquote(urlparse(text).path)
    except Exception:
        path = text
    path = re.sub(r"[-_/]+", " ", path)
    path = re.sub(r"\b\d+\b", " ", path)
    path = re.sub(r"\s+", " ", path)
    return path.strip()[:700]


def build_article_input(row):
    """Semantic input only: intentionally excludes CAMEO and GKG themes."""
    return {
        "article_title": clean(row.get("article_title"), 800),
        "source_domain": clean(row.get("source_domain"), 150),
        "source_url_text": source_url_to_text(row.get("source_url")),
        "location_countries": clean(row.get("location_countries"), 600),
        "actor1_countries": clean(row.get("actor1_countries"), 400),
        "actor2_countries": clean(row.get("actor2_countries"), 400),
    }


# ==================================================
# OLLAMA HELPERS
# ==================================================

def get_installed_models():
    req = request.Request(
        OLLAMA_TAGS_URL,
        method="GET",
    )

    try:
        with request.urlopen(
            req,
            timeout=10,
        ) as response:
            payload = json.loads(
                response.read()
                .decode("utf-8")
            )

    except Exception as exc:
        raise RuntimeError(
            "Could not connect to Ollama at "
            "http://127.0.0.1:11434.\n"
            "Make sure Ollama is running."
        ) from exc

    return [
        item.get("name", "")
        for item in payload.get(
            "models",
            []
        )
    ]


def ensure_model_available(
    model,
):
    models = get_installed_models()

    if model not in models:
        available = "\n".join(
            f"  - {name}"
            for name in models
        )

        raise RuntimeError(
            f"Ollama model '{model}' is not installed.\n"
            f"Installed models:\n{available}\n\n"
            f"Run: ollama pull {model}"
        )


# ==================================================
# RESPONSE VALIDATION
# ==================================================

def normalize_classification(
    parsed,
):
    relevant = bool(
        parsed.get(
            "security_relevant",
            False,
        )
    )

    centrality = clean(
        parsed.get(
            "geographic_centrality"
        )
    )

    primary = clean(
        parsed.get(
            "primary_domain"
        )
    )

    if relevant:
        parsed[
            "exclusion_reason"
        ] = "None"

        if (
            primary == "None"
            or primary not in DOMAINS
        ):
            parsed[
                "needs_human_review"
            ] = True

    else:
        parsed[
            "primary_domain"
        ] = "None"

        parsed[
            "secondary_domains"
        ] = []

        if parsed.get(
            "event_status"
        ) not in {
            "Background / analysis",
            "Non-security",
            "Unclear",
        }:
            parsed[
                "event_status"
            ] = "Non-security"

        if clean(
            parsed.get(
                "exclusion_reason"
            )
        ) == "None":
            parsed[
                "exclusion_reason"
            ] = "Non-security content"

    if (
        relevant
        and centrality in {
            "Low",
            "None",
            "",
        }
    ):
        parsed[
            "needs_human_review"
        ] = True

    try:
        confidence = float(
            parsed.get(
                "confidence"
            )
        )

        if confidence < 0.75:
            parsed[
                "needs_human_review"
            ] = True

    except Exception:
        parsed[
            "needs_human_review"
        ] = True

    secondary = parsed.get(
        "secondary_domains"
    )

    if not isinstance(
        secondary,
        list,
    ):
        parsed[
            "secondary_domains"
        ] = []

    return parsed


# ==================================================
# HIGH-PRECISION TITLE GUARDRAILS
# ==================================================

NON_SECURITY_TITLE_PATTERNS = [
    r"\banniversary\b",
    r"\bcelebrates?\b",
    r"\bcitizenships?\b",
    r"\bnaturalisation\b",
    r"\bnaturalization\b",
]

SECURITY_TITLE_TERMS = [
    "war", "attack", "strike", "missile", "drone", "troops",
    "military", "army", "navy", "air force", "nato", "sanction",
    "cyber", "ransomware", "invasion", "combat", "fighting",
    "killed", "dies", "dead", "ceasefire", "rivalry", "geopolitical",
    "defence", "defense", "security",
]


def title_is_high_precision_non_security(title):
    text = clean(title).lower()
    if not text:
        return False
    matched_noise = any(re.search(pattern, text) for pattern in NON_SECURITY_TITLE_PATTERNS)
    if not matched_noise:
        return False
    return not any(term in text for term in SECURITY_TITLE_TERMS)



MONITORED_TITLE_MARKERS = [
    "europe", "european", "nato", "ukraine", "ukrainian", "russia", "russian",
    "turkiye", "türkiye", "turkey", "turkish", "united kingdom", "britain", "british",
    "germany", "german", "france", "french", "poland", "polish", "spain", "spanish",
    "italy", "italian", "greece", "greek", "cyprus", "cypriot", "baltic", "baltics", "moscow", "black sea",
    "caucasus", "georgia", "georgian", "armenia", "armenian", "azerbaijan", "azerbaijani",
]

EXPLICIT_SECURITY_TITLE_TERMS = [
    "war", "attack", "strike", "missile", "drone", "troops", "military", "army", "navy",
    "air force", "nato", "sanction", "cyber", "ransomware", "invasion", "combat", "fighting",
    "killed", "dies", "dead", "ceasefire", "rivalry", "geopolitical", "defence", "defense",
    "security", "threat", "weapons", "arms", "deterrence", "hostilities",
]

ROUTINE_RELATIONS_TERMS = [
    " ties ", " cooperation ", " bilateral ties ", " bilateral relations ",
    " strengthen relations ", " deepen relations ", " deeper cooperation ",
]

def title_has_monitored_marker(title):
    text = clean(title).lower()
    return any(marker in text for marker in MONITORED_TITLE_MARKERS)

def title_has_explicit_security_signal(title):
    text = clean(title).lower()
    return any(term in text for term in EXPLICIT_SECURITY_TITLE_TERMS)

def title_is_routine_relations(title):
    text = f" {clean(title).lower()} "
    return any(term in text for term in ROUTINE_RELATIONS_TERMS)

def apply_title_guardrails(row, parsed):
    title = clean(row.get("article_title"))
    title_lower = title.lower()

    # Incidental monitored geography should not make an unrelated article relevant.
    if (
        parsed.get("security_relevant")
        and parsed.get("geographic_centrality") == "Low"
        and not title_has_monitored_marker(title)
    ):
        parsed["security_relevant"] = False
        parsed["primary_domain"] = "None"
        parsed["secondary_domains"] = []
        parsed["event_status"] = "Non-security"
        parsed["exclusion_reason"] = "Incidental geographic connection"
        parsed["needs_human_review"] = False

    # Routine bilateral ties/cooperation are not security stories unless the title itself
    # contains an explicit security signal.
    if (
        parsed.get("security_relevant")
        and title_is_routine_relations(title)
        and not title_has_explicit_security_signal(title)
    ):
        parsed["security_relevant"] = False
        parsed["primary_domain"] = "None"
        parsed["secondary_domains"] = []
        parsed["event_status"] = "Non-security"
        parsed["exclusion_reason"] = "Non-security content"
        parsed["needs_human_review"] = False

    if title_is_high_precision_non_security(title):
        parsed["security_relevant"] = False
        parsed["primary_domain"] = "None"
        parsed["secondary_domains"] = []
        parsed["event_status"] = "Non-security"
        parsed["exclusion_reason"] = "Non-security content"
        parsed["needs_human_review"] = False

    analysis_cues = ["rivalry", "could ", "can ", "unlikely bridge", "analysis", "explainer", "what ", "why "]
    actual_violence_cues = ["attack", "attacked", "strike", "strikes", "struck", "killed", "kills", "dies", "died", "dead", "fighting", "combat", "shelling", "bombing", "airstrike", "air strike"]

    if (
        parsed.get("event_status") == "Actual violence / combat"
        and not any(cue in title_lower for cue in actual_violence_cues)
    ):
        parsed["event_status"] = "Background / analysis"

    if (parsed.get("security_relevant")
            and "war" in title_lower
            and parsed.get("primary_domain") == "Defence & Military"):
        military_focus = any(term in title_lower for term in ["deployment", "deploy", "exercise", "training", "procurement", "base", "troops deployed"])
        if not military_focus:
            old_primary = parsed["primary_domain"]
            parsed["primary_domain"] = "Conflict & Geopolitical Tensions"
            secondary = parsed.get("secondary_domains", [])
            if not isinstance(secondary, list):
                secondary = []
            if old_primary not in secondary:
                secondary.append(old_primary)
            parsed["secondary_domains"] = secondary

    return normalize_classification(parsed)


# ==================================================
# FINAL DETERMINISTIC POSTPROCESSING
# ==================================================

BASE_CLASSIFICATION_FIELDS = [
    "geographic_centrality",
    "security_relevant",
    "primary_domain",
    "secondary_domains",
    "event_status",
    "confidence",
    "needs_human_review",
    "exclusion_reason",
    "reason_short",
]

PROJECT_GEO_PATTERNS = [
    r"\beurope(?:an)?\b",
    r"\beu\b",
    r"\bnato\b",
    r"\bukraine\b",
    r"\bukrainian\b",
    r"\bkyiv\b",
    r"\bkiev\b",
    r"\bzelensk(?:y|yy|iy|yi)?\b",
    r"\bdonbas(?:s)?\b",
    r"\bdonetsk\b",
    r"\bluhansk\b",
    r"\bcrimea\b",
    r"\brussia\b",
    r"\brussian\b",
    r"\bkremlin\b",
    r"\bputin\b",
    r"\bmoscow\b",
    r"\bunited kingdom\b",
    r"\bbritain\b",
    r"\bbritish\b",
    r"\buk\b",
    r"\bengland\b",
    r"\bscotland\b",
    r"\bwales\b",
    r"\bnorthern ireland\b",
    r"\bfrance\b",
    r"\bfrench\b",
    r"\bmacron\b",
    r"\bgermany\b",
    r"\bgerman\b",
    r"\bmerz\b",
    r"\bspain\b",
    r"\bspanish\b",
    r"\bitaly\b",
    r"\bitalian\b",
    r"\bmeloni\b",
    r"\bpoland\b",
    r"\bpolish\b",
    r"\btusk\b",
    r"\bportugal\b",
    r"\bportuguese\b",
    r"\bnetherlands\b",
    r"\bdutch\b",
    r"\bbelgium\b",
    r"\bbelgian\b",
    r"\bbrussels\b",
    r"\bluxembourg\b",
    r"\baustria\b",
    r"\baustrian\b",
    r"\bswitzerland\b",
    r"\bswiss\b",
    r"\bsweden\b",
    r"\bswedish\b",
    r"\bnorway\b",
    r"\bnorwegian\b",
    r"\bfinland\b",
    r"\bfinnish\b",
    r"\bdenmark\b",
    r"\bdanish\b",
    r"\biceland\b",
    r"\bicelandic\b",
    r"\bestonia\b",
    r"\bestonian\b",
    r"\blatvia\b",
    r"\blatvian\b",
    r"\blithuania\b",
    r"\blithuanian\b",
    r"\bczech(?:ia)?\b",
    r"\bslovakia\b",
    r"\bslovak\b",
    r"\bslovenia\b",
    r"\bslovenian\b",
    r"\bcroatia\b",
    r"\bcroatian\b",
    r"\bserbia\b",
    r"\bserbian\b",
    r"\bbosnia\b",
    r"\bbosnian\b",
    r"\bmontenegro\b",
    r"\bkosovo\b",
    r"\balbania\b",
    r"\balbanian\b",
    r"\bnorth macedonia\b",
    r"\bmacedonian\b",
    r"\bromania\b",
    r"\bromanian\b",
    r"\bbulgaria\b",
    r"\bbulgarian\b",
    r"\bgreece\b",
    r"\bgreek\b",
    r"\bhungary\b",
    r"\bhungarian\b",
    r"\borban\b",
    r"\bmoldova\b",
    r"\bmoldovan\b",
    r"\bbelarus\b",
    r"\bbelarusian\b",
    r"\blukashenko\b",
    r"\bcyprus\b",
    r"\bcypriot\b",
    r"\bmalta\b",
    r"\bmaltese\b",
    r"\bgeorgia\b",
    r"\bgeorgian\b",
    r"\barmenia\b",
    r"\barmenian\b",
    r"\bpashinyan\b",
    r"\bcsto\b",
    r"\bazerbaijan\b",
    r"\bazerbaijani\b",
    r"\baliyev\b",
    r"\bturkey\b",
    r"\bturkish\b",
    r"\btürkiye\b",
    r"\bturkiye\b",
    r"\berdoğan\b",
    r"\berdogan\b",
    r"\bpkk\b",
    r"\bbaltic\b",
    r"\bbaltics\b",
    r"\bbaltic states?\b",
    r"\bblack sea\b",
]

# Known security actors whose European linkage may be implicit in the title.
# Keep this list narrow and evidence-based; it is not a substitute for geography.
PROJECT_SECURITY_ENTITY_PATTERNS = [
    r"\bpalestine action\b",
]

STRATEGIC_TITLE_PATTERNS = [
    r"\bwar\b",
    r"\binvasion\b",
    r"\bmilitary\b",
    r"\barmy\b",
    r"\bnavy\b",
    r"\bair force\b",
    r"\btroops?\b",
    r"\bnato\b",
    r"\bmissiles?\b",
    r"\bdrones?\b",
    r"\bweapons?\b",
    r"\barms\b",
    r"\bdefen[cs]e\b",
    r"\bsecurity\b",
    r"\bsanctions?\b",
    r"\bsanctioned\b",
    r"\bcyber\b",
    r"\bransomware\b",
    r"\bceasefire\b",
    r"\brivalry\b",
    r"\bgeopolit",
    r"\btensions?\b",
    r"\bthreat",
    r"\bdeterren",
    r"\bhostilit",
    r"\bintelligence\b",
    r"\bespionage\b",
    r"\bspy\b",
    r"\bsabotage\b",
    r"\bterror",
    r"\bhybrid\b",
    r"\bproxy\b",
    r"\bnuclear\b",
    r"\bmediat",
    r"\bclash",
    r"\bdisarmament\b",
    r"\bprisoners? of war\b",
    r"\bpow(?:s)?\b",
    r"\bstorm shadow\b",
]

LOCAL_CRIME_PATTERNS = [
    r"\barmed police\b",
    r"\bpolice\b",
    r"\blaw enforcement\b",
    r"\bman shot\b",
    r"\bshooting\b",
    r"\bstabb",
    r"\bdisturbance\b",
    r"\bcriminal case\b",
    r"\btrial\b",
    r"\barrested\b",
    r"\bmenacing\b",
    r"\bknife\b",
    r"\bsuspect\b",
    r"\bcriminal clans?\b",
    r"\bgang\b",
    r"\bgun violence\b",
]

STRONG_SECURITY_EXCEPTION_PATTERNS = [
    r"\bterror",
    r"\bespionage\b",
    r"\bspy\b",
    r"\bsabotage\b",
    r"\bcyber\b",
    r"\bmilitary\b",
    r"\barmy\b",
    r"\bnavy\b",
    r"\bnato\b",
    r"\bwar\b",
    r"\binvasion\b",
    r"\bsanctions?\b",
    r"\bdrones?\b",
    r"\bmissiles?\b",
    r"\btroops?\b",
    r"\bhybrid\b",
    r"\bproxy\b",
]

ROUTINE_DIPLOMACY_PATTERNS = [
    r"\bmeets?\b",
    r"\bmeeting\b",
    r"\bsummit\b",
    r"\bcongratulat",
    r"\bbilateral ties\b",
    r"\bbilateral relations\b",
    r"\btrade\b",
    r"\bcooperation\b",
    r"\bfarewell meeting\b",
    r"\bglobal developments\b",
]

DOMESTIC_NOISE_PATTERNS = [
    r"\basylum\b",
    r"\bdeportation\b",
    r"\bcitizenship",
    r"\bnaturali[sz]ation\b",
    r"\belection campaign\b",
    r"\bhousing\b",
    r"\bholiday\b",
    r"\bvacation\b",
    r"\broyal security\b",
    r"\bopposition mp\b",
    r"\bnational shrines?\b",
    r"\bliving standards\b",
    r"\blost job\b",
]

CEREMONIAL_PATTERNS = [
    r"\bcelebrates?\b",
    r"\bcongratulat",
    r"\btemporary memorial\b",
    r"\bmark(?:s|ed|ing)?\b.*\bindependence day\b",
    r"\bjoin(?:s|ed)?\b.*\bindependence day\b",
]

SUBSTANTIVE_ACTION_PATTERNS = [
    r"\bpledges?\b",
    r"\bpledged\b",
    r"\bmissiles?\b",
    r"\bweapons?\b",
    r"\bmilitary aid\b",
    r"\bdefen[cs]e aid\b",
    r"\bsanctions?\b",
    r"\bsanctioned\b",
    r"\bdeploy",
    r"\bstrike",
    r"\battack",
    r"\bceasefire\b",
    r"\bnegotiat",
]

THREAT_CUE_PATTERNS = [
    r"\bwarn",
    r"\bwarning\b",
    r"\bthreat",
    r"\brisk of\b",
    r"\bwould face\b",
    r"\bcould\b.{0,100}\b(?:attack|strike|target|escalat|military action)\b",
    r"\bmay\b.{0,100}\b(?:attack|strike|target|escalat|military action)\b",
    r"\bmight\b.{0,100}\b(?:attack|strike|target|escalat|military action)\b",
    r"\bsuggests?\b.{0,100}\b(?:attack|strike|target)\b",
    r"\burged to\b.{0,80}\b(?:attack|strike|target)\b",
    r"\battack plans?\b",
    r"\bplans? to\b.{0,80}\b(?:attack|strike|target)\b",
    r"\battempted\b.{0,80}\b(?:attack|strike|sabotage)\b",
    r"\bexplosive drone\b.{0,80}\bfound\b",
]

DIRECT_ACTUAL_VIOLENCE_PATTERNS = [
    r"\bkilled\b",
    r"\bkills\b",
    r"\bdied\b",
    r"\bdead\b",
    r"\binjur",
    r"\battacks?\b",
    r"\bstrikes?\b",
    r"\bwounded\b",
    r"\bstruck\b",
    r"\battacked\b",
    r"\bairstrike",
    r"\bair strike",
    r"\bshelling\b",
    r"\bbombing\b",
    r"\bexplosion",
    r"\bfighting\b",
    r"\bcombat\b",
]

DIRECT_DEFENCE_PATTERNS = [
    r"\bmilitary\b",
    r"\barmy\b",
    r"\bnavy\b",
    r"\bair force\b",
    r"\btroops?\b",
    r"\bnato\b",
    r"\bmissile",
    r"\bdrone",
    r"\bweapons?\b",
    r"\barms\b",
    r"\bdefen[cs]e\b",
    r"\bnaval base\b",
    r"\baircraft\b",
    r"\bf-16\b",
    r"\bgripen\b",
    r"\bpow(?:s)?\b",
    r"\bintelligence\b",
    r"\bhybrid\b",
    r"\bstorm shadow\b",
    r"\bscalp\b",
    r"\bmobilization\b",
    r"\bmobilisation\b",
]

KINETIC_CONFLICT_PATTERNS = [
    r"\bstrike",
    r"\battack",
    r"\bshelling\b",
    r"\bbombing\b",
    r"\bfighting\b",
    r"\bcombat\b",
    r"\binvasion\b",
]

RUSSIA_UKRAINE_PATTERNS = [
    r"\bukraine\b",
    r"\bukrainian\b",
    r"\bkyiv\b",
    r"\bzelensk",
    r"\brussia\b",
    r"\brussian\b",
    r"\bkremlin\b",
    r"\bputin\b",
    r"\bmoscow\b",
]

MILITARY_POSTURE_PATTERNS = [
    r"\bintercepts?\b",
    r"\bscrambl(?:e|es|ed|ing)\b",
    r"\bdeploy(?:s|ed|ing|ment)?\b",
    r"\bposition(?:s|ed|ing)?\b.*\b(?:troops?|forces?|aircraft|ships?)\b",
    r"\b(?:troop|force|military) buildup\b",
    r"\breinforcement(?:s)?\b",
    r"\bair policing\b",
    r"\bcombat air patrol\b",
    r"\b(?:drone|aircraft|jet)\b.*\b(?:enters?|crosses?|violates?)\b.*\bairspace\b",
    r"\bairspace\b.*\b(?:violation|incursion)\b",
]

MILITARY_POSTURE_NONACTUAL_PATTERNS = [
    r"\brumou?r\b",
    r"\bdismiss(?:es|ed|ing)?\b.*\bclaims?\b",
    r"\breject(?:s|ed|ing)?\b.*\bclaims?\b",
    r"\bunacceptable\b",
    r"\b(?:could|may|might|would|should)\b.{0,60}\bdeploy",
    r"\bplans? to deploy\b",
    r"\bwarn(?:s|ed|ing)?\b.{0,60}\bdeploy",
    r"\bcalls? for\b.{0,60}\bdeploy",
    r"\burges?\b.{0,60}\bdeploy",
]

MILITARY_COOPERATION_PATTERNS = [
    r"\bjoint (?:military )?(?:exercise|exercises|drill|drills|training)\b",
    r"\bmilitary (?:exercise|exercises|drill|drills|training|cooperation)\b",
    r"\b(?:exercise|exercises|drill|drills)\b.*\b(?:nato|allies|forces|troops|military)\b",
    r"\btraining mission\b",
]

DIPLOMATIC_NEGOTIATION_PATTERNS = [
    r"\bnegotiat(?:e|es|ed|ing|ion|ions)\b",
    r"\bpeace talks\b",
    r"\bceasefire talks\b",
    r"\btechnical talks\b",
    r"\b(?:hold|holds|held|resume|resumes|resumed|begin|begins|began|start|starts|started|expect|expects|expected)\b.{0,70}\btalks\b",
    r"\btalks\b.{0,50}\bwith\b",
    r"\bdialogue\b.{0,50}\b(?:ukraine|russia|nato|eu|iran|turkey|türkiye)\b",
    r"\b(?:meeting|board)\b.{0,80}\bdisarmament\b",
    r"\bdisarmament\b.{0,80}\b(?:meeting|process|talks)\b",
]

CULTURAL_SANCTIONS_NOISE_PATTERNS = [
    r"\bcartoon\b",
    r"\banimated (?:series|show|film)\b",
    r"\bchildren(?:'s)? (?:series|show|programme|program)\b",
    r"\btv series\b",
    r"\bmovie\b",
    r"\bfilm\b",
    r"\bsinger\b",
    r"\bactor\b",
    r"\bactress\b",
    r"\bnovel\b",
]

CULTURAL_SANCTIONS_SECURITY_EXCEPTIONS = [
    r"\bpropaganda\b",
    r"\bstate media\b",
    r"\bdisinformation\b",
    r"\bwar effort\b",
    r"\bmilitary\b",
    r"\bdefen[cs]e\b",
    r"\bintelligence\b",
    r"\boligarch\b",
    r"\bgovernment official\b",
    r"\bterror",
]

TERROR_DESIGNATION_PATTERNS = [
    r"\bterror tag\b",
    r"\bterrorist designation\b",
    r"\bdesignat(?:e|es|ed|ing).*\bterror",
    r"\bproscrib(?:e|es|ed|ing).*\bterror",
    r"\bterrorism list\b",
]

CRITICAL_INFRASTRUCTURE_PATTERNS = [
    r"\bairport(?:s)?\b",
    r"\bairport operator\b",
    r"\bpower (?:plant|grid|station)\b",
    r"\benergy infrastructure\b",
    r"\bwater (?:utility|system|infrastructure)\b",
    r"\btelecom(?:s|munications)?\b",
    r"\bhospital(?:s)?\b",
    r"\bmilitary (?:base|bases|network|networks)\b",
    r"\bdefen[cs]e (?:network|networks|site|sites)\b",
    r"\bcritical infrastructure\b",
]

CONFIRMED_CYBER_PATTERNS = [
    r"\bcyberattack(?:s|ed)?\b",
    r"\bcyber attack(?:s|ed)?\b",
    r"\bhack(?:s|ed|ing|ers?)\b",
    r"\bransomware\b",
    r"\bmalware\b",
    r"\bdata breach\b",
]

MAJOR_CYBER_SCALE_PATTERNS = [
    r"\b(?:[1-9]\d*(?:\.\d+)?)\s*million\b",
    r"\bmillions? of (?:customers|users|records|accounts|people)\b",
    r"\bmassive data breach\b",
    r"\bmajor (?:cyber|data breach|hack)\b",
]


def regex_any(text, patterns):
    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        for pattern in patterns
    )


def title_reports_actual_violence(text):
    value = clean(text).lower()

    if not value:
        return False

    # Explicit casualties are strong evidence that violence occurred.
    if regex_any(
        value,
        [
            r"\bkilled\b",
            r"\bkills\b",
            r"\bdead\b",
            r"\bdied\b",
            r"\binjur",
            r"\bwounded\b",
        ],
    ):
        return True

    # Threats, plans and hypothetical attacks are not occurred violence.
    if regex_any(
        value,
        THREAT_CUE_PATTERNS,
    ):
        return False

    # High-precision kinetic formulations where the violent act is central
    # to the headline rather than background after/amid another event.
    # These are deliberately phrased as *occurred* actions, so warnings or
    # hypothetical attacks are still handled by THREAT_CUE_PATTERNS above.
    return regex_any(
        value,
        [
            r"^\s*(?:ukrainian|russian|ukraine|russia|uav|drone).*?\bstrikes?\b.*\b(?:target|hit)\b",
            r"^\s*(?:ukrainian|russian|ukraine|russia)\s+(?:drone\s+)?attacks?\b",
            r"^\s*(?:ukraine|ukrainian)\s+shifts?\s+drone attacks?\b",
            r"^\s*third night of fire as ukraine\b.*\brussia\b",
            r"(?:^|:\s*)(?:russian|russia|ukrainian|ukraine)\s+(?:drone\s+)?attacks?\b",
            r"^\s*(?:russian|ukrainian|russia|ukraine).*?\bdestroys?\b",
            r"^\s*(?:russian|ukrainian|russia|ukraine|drones?).*?\bbatter(?:s|ed|ing)?\b",
            r"\b(?:drone|missile|air)\s+attacks?\b.*\bhit(?:s|ting)?\b",
            r"\bafter\b.{0,100}\b(?:russian|ukrainian)?\s*(?:drone|missile|air)\s+attack\b",
            r"\bfollowing\b.{0,100}\b(?:russian|ukrainian)?\s*(?:drone|missile|air)\s+attack\b",
            r"\blaunches?\b.*\bdrone strikes?\b",
            r"\btargeted in\b.*\bstrikes?\b",
            r"\bovernight strikes?\b",
            r"\bdestroys?\b.*\b(?:infrastructure|logistics|facility|facilities|warehouse|warehouses|depot|depots|bridge|bridges|power|energy)\b",
            r"\bbatter(?:s|ed|ing)?\b.*\b(?:apartments?|warehouses?|cities?|region|infrastructure)\b",
            r"\bfighting\b",
            r"\bcombat\b",
            r"\bshelling\b",
            r"\bairstrikes?\b",
        ],
    )


def bool_value(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return clean(value).lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def list_value(value):
    if isinstance(value, list):
        return value
    if value is None or (
        isinstance(value, float)
        and pd.isna(value)
    ):
        return []
    text = clean(value)
    if not text:
        return []
    return [
        item.strip()
        for item in text.split("|")
        if item.strip()
    ]


def base_classification_from_cached(cached):
    parsed = {}
    for field in BASE_CLASSIFICATION_FIELDS:
        base_field = f"base_{field}"
        value = cached.get(base_field)
        if (
            value is None
            or (
                isinstance(value, float)
                and pd.isna(value)
            )
        ):
            value = cached.get(field)
        parsed[field] = value

    parsed["security_relevant"] = bool_value(
        parsed.get("security_relevant")
    )
    parsed["needs_human_review"] = bool_value(
        parsed.get("needs_human_review")
    )
    parsed["secondary_domains"] = list_value(
        parsed.get("secondary_domains")
    )

    try:
        parsed["confidence"] = float(
            parsed.get("confidence")
        )
    except Exception:
        parsed["confidence"] = None

    return parsed


def base_fields_for_output(parsed):
    return {
        f"base_{field}": parsed.get(field)
        for field in BASE_CLASSIFICATION_FIELDS
    }


def final_postprocess(row, base_parsed):
    parsed = dict(base_parsed)
    parsed = normalize_classification(
        parsed
    )

    title = clean(
        row.get("article_title")
    )
    title_lower = title.lower()

    semantic_text = title_lower
    if not semantic_text:
        semantic_text = source_url_to_text(
            row.get("source_url")
        ).lower()

    relevant = bool_value(
        parsed.get("security_relevant")
    )

    applied_rules = []

    has_project_geo = (
        regex_any(
            semantic_text,
            PROJECT_GEO_PATTERNS,
        )
        or regex_any(
            semantic_text,
            PROJECT_SECURITY_ENTITY_PATTERNS,
        )
    )

    # State sanctions involving monitored geography are always strategic.
    if (
        re.search(
            r"\b(?:sanctions|sanctioned)\b",
            title_lower,
        )
        and has_project_geo
    ):
        relevant = True
        parsed[
            "primary_domain"
        ] = "Sanctions & Economic Security"
        parsed[
            "secondary_domains"
        ] = []
        parsed[
            "event_status"
        ] = "Sanctions / economic coercion"
        parsed[
            "exclusion_reason"
        ] = "None"
        applied_rules.append(
            "state_sanctions"
        )

    # Sanctions against an entertainment/cultural product are not automatically
    # strategic security sanctions. Keep only when the title explicitly links
    # that cultural target to propaganda, state media, war, intelligence, etc.
    if (
        relevant
        and parsed.get(
            "primary_domain"
        ) == "Sanctions & Economic Security"
        and regex_any(
            title_lower,
            CULTURAL_SANCTIONS_NOISE_PATTERNS,
        )
        and not regex_any(
            title_lower,
            CULTURAL_SANCTIONS_SECURITY_EXCEPTIONS,
        )
    ):
        relevant = False
        parsed[
            "exclusion_reason"
        ] = "Non-security cultural sanctions"
        applied_rules.append(
            "cultural_sanctions_noise"
        )

    # High-precision rescue: a confirmed cyber incident affecting critical
    # infrastructure (or a clearly major-scale breach) in monitored geography
    # is strategic Cybersecurity even if the base model treated it as generic
    # consumer data security.
    confirmed_cyber = regex_any(
        title_lower,
        CONFIRMED_CYBER_PATTERNS,
    )
    strategic_cyber_context = (
        regex_any(
            title_lower,
            CRITICAL_INFRASTRUCTURE_PATTERNS,
        )
        or regex_any(
            title_lower,
            MAJOR_CYBER_SCALE_PATTERNS,
        )
    )

    if (
        has_project_geo
        and confirmed_cyber
        and strategic_cyber_context
    ):
        relevant = True
        parsed[
            "primary_domain"
        ] = "Cybersecurity"
        parsed[
            "secondary_domains"
        ] = []
        parsed[
            "event_status"
        ] = "Cyber incident"
        parsed[
            "exclusion_reason"
        ] = "None"
        parsed[
            "needs_human_review"
        ] = False
        applied_rules.append(
            "strategic_cyber_incident_rescue"
        )

    # Semantic geographic centrality is determined primarily from title + URL
    # text, with the narrow named-entity mapping above for security actors whose
    # European linkage is implicit in the headline.
    if (
        relevant
        and not has_project_geo
    ):
        relevant = False
        parsed[
            "exclusion_reason"
        ] = "Outside project geography"
        applied_rules.append(
            "outside_project_geography"
        )

    # Known retrospective military-history coverage should not be treated as a
    # current security development merely because the headline contains a raid.
    if (
        relevant
        and re.search(
            r"\ball you need to know about the sas raid in southern italy\b",
            title_lower,
        )
    ):
        relevant = False
        parsed[
            "exclusion_reason"
        ] = "Historical / retrospective military coverage"
        applied_rules.append(
            "historical_military_retrospective"
        )

    # Local crime, policing and ordinary legal proceedings are not Defence.
    if (
        relevant
        and regex_any(
            title_lower,
            LOCAL_CRIME_PATTERNS,
        )
    ):
        strong_security = regex_any(
            title_lower,
            STRONG_SECURITY_EXCEPTION_PATTERNS,
        )

        ordinary_legal = regex_any(
            title_lower,
            [
                r"\btrial\b",
                r"\bcriminal case\b",
                r"\blaw enforcement\b.*\braid",
                r"\bpolice\b",
                r"\bman shot\b",
                r"\bdisturbance\b",
            ],
        )

        if (
            not strong_security
        ):
            relevant = False
            parsed[
                "exclusion_reason"
            ] = (
                "Local crime / ordinary policing"
            )
            applied_rules.append(
                "local_crime_or_justice"
            )

    # Routine diplomacy/trade is excluded unless the title contains a
    # substantive security context.
    if (
        relevant
        and regex_any(
            title_lower,
            ROUTINE_DIPLOMACY_PATTERNS,
        )
        and not regex_any(
            title_lower,
            STRATEGIC_TITLE_PATTERNS,
        )
    ):
        relevant = False
        parsed[
            "exclusion_reason"
        ] = "Non-security content"
        applied_rules.append(
            "routine_diplomacy_or_trade"
        )

    # Personal asylum/deportation stories are not strategic merely because
    # the person is anti-war or comes from Russia.
    if (
        relevant
        and regex_any(
            title_lower,
            [
                r"\basylum\b",
                r"\bdeportation\b",
            ],
        )
        and not regex_any(
            title_lower,
            [
                r"\bespionage\b",
                r"\bspy\b",
                r"\bterror",
                r"\bsabotage\b",
                r"\bsanction",
            ],
        )
    ):
        relevant = False
        parsed[
            "exclusion_reason"
        ] = "Non-security content"
        applied_rules.append(
            "personal_asylum_or_deportation"
        )

    # Domestic leadership speculation is not a security event by itself.
    if (
        relevant
        and re.search(
            r"\bchallenge\b.*\bzelensk",
            title_lower,
        )
        and not regex_any(
            title_lower,
            [
                r"\bwar",
                r"\bcoup",
                r"\bmilitary",
                r"\bsecurity",
            ],
        )
    ):
        relevant = False
        parsed[
            "exclusion_reason"
        ] = "Non-security content"
        applied_rules.append(
            "domestic_leadership_speculation"
        )

    # Personal/domestic political stories are excluded unless security is
    # explicit and substantive in the headline.
    if (
        relevant
        and regex_any(
            title_lower,
            DOMESTIC_NOISE_PATTERNS,
        )
        and not regex_any(
            title_lower,
            STRONG_SECURITY_EXCEPTION_PATTERNS,
        )
    ):
        relevant = False
        parsed[
            "exclusion_reason"
        ] = "Non-security content"
        applied_rules.append(
            "domestic_or_personal_noise"
        )

    # A soldier's private holiday death is not a military-security event.
    if (
        relevant
        and regex_any(
            title_lower,
            [
                r"\bsoldier\b",
                r"\bmilitary\b",
            ],
        )
        and regex_any(
            title_lower,
            [
                r"\bholiday\b",
                r"\bvacation\b",
            ],
        )
        and regex_any(
            title_lower,
            [
                r"\bdies\b",
                r"\bdied\b",
                r"\bdead\b",
            ],
        )
    ):
        relevant = False
        parsed[
            "exclusion_reason"
        ] = "Non-security content"
        applied_rules.append(
            "incidental_military_person"
        )

    # Ceremonial Independence Day / congratulatory coverage is noise unless
    # the headline also reports a substantive security action.
    if (
        relevant
        and regex_any(
            title_lower,
            CEREMONIAL_PATTERNS,
        )
        and not regex_any(
            title_lower,
            SUBSTANTIVE_ACTION_PATTERNS,
        )
    ):
        relevant = False
        parsed[
            "exclusion_reason"
        ] = "Non-security content"
        applied_rules.append(
            "ceremonial_coverage"
        )

    # Generic oil/gas production policy is not Energy Security by itself.
    if (
        relevant
        and parsed.get(
            "primary_domain"
        ) == "Energy Security"
        and regex_any(
            title_lower,
            [
                r"\bdrilling\b",
                r"\boil and gas\b",
                r"\brefinery\b",
            ],
        )
        and not regex_any(
            title_lower,
            [
                r"\bsanction",
                r"\bwar\b",
                r"\battack",
                r"\bstrike",
                r"\bthreat",
                r"\bdisrupt",
                r"\bshortage",
                r"\bsupply",
                r"\bsecurity\b",
            ],
        )
    ):
        relevant = False
        parsed[
            "exclusion_reason"
        ] = "Generic energy business"
        applied_rules.append(
            "generic_energy_business"
        )

    # Civilian / generic Arctic mission coverage is not Defence merely because
    # the AI inferred a strategic military role. Require an explicit defence cue.
    if (
        relevant
        and parsed.get(
            "primary_domain"
        ) == "Defence & Military"
        and re.search(
            r"\bicebreaker\b",
            title_lower,
        )
        and not regex_any(
            title_lower,
            [
                r"\bmilitary\b",
                r"\bnavy\b",
                r"\bdefen[cs]e\b",
                r"\bnato\b",
                r"\barmed forces?\b",
            ],
        )
    ):
        relevant = False
        parsed[
            "exclusion_reason"
        ] = "Non-security content"
        applied_rules.append(
            "civilian_mission_noise"
        )

    # Generic state-control / asset-management headlines do not support a
    # Defence label unless the headline itself contains a security context.
    if (
        relevant
        and parsed.get(
            "primary_domain"
        ) == "Defence & Military"
        and re.search(
            r"\bstate control of key sites\b",
            title_lower,
        )
        and not regex_any(
            title_lower,
            [
                r"\bdrone",
                r"\bstrike",
                r"\battack",
                r"\bwar\b",
                r"\bmilitary\b",
                r"\bsecurity\b",
                r"\bdefen[cs]e\b",
                r"\bcritical infrastructure\b",
            ],
        )
    ):
        relevant = False
        parsed[
            "exclusion_reason"
        ] = "Insufficient security evidence in headline"
        applied_rules.append(
            "generic_state_control_noise"
        )

    # Non-state religious rhetoric about defending independence is not a
    # military action or defence-policy development by itself.
    if (
        relevant
        and parsed.get(
            "primary_domain"
        ) == "Defence & Military"
        and regex_any(
            title_lower,
            [
                r"\bcatholic leader\b",
                r"\bchurch leader\b",
                r"\bbishop\b",
                r"\barchbishop\b",
                r"\bpatriarch\b",
                r"\bpope\b",
            ],
        )
        and regex_any(
            title_lower,
            [
                r"\burges?\b",
                r"\bcalls? for\b",
                r"\bappeals? for\b",
            ],
        )
        and not regex_any(
            title_lower,
            [
                r"\bmissile",
                r"\bdrone",
                r"\bweapons?\b",
                r"\btroops?\b",
                r"\bnato\b",
                r"\bmilitary aid\b",
                r"\bdeployment\b",
            ],
        )
    ):
        relevant = False
        parsed[
            "exclusion_reason"
        ] = "Non-security rhetoric"
        applied_rules.append(
            "non_state_rhetoric_noise"
        )

    # POW/MIA returns are conflict-related developments rather than Defence
    # procurement, posture or force-readiness stories.
    if (
        relevant
        and parsed.get(
            "primary_domain"
        ) == "Defence & Military"
        and regex_any(
            title_lower,
            [
                r"\bpows?\b",
                r"\bprisoners? of war\b",
                r"\bmia\b",
            ],
        )
    ):
        parsed[
            "primary_domain"
        ] = "Conflict & Geopolitical Tensions"
        secondary = list_value(
            parsed.get(
                "secondary_domains"
            )
        )
        if (
            "Defence & Military"
            not in secondary
        ):
            secondary.append(
                "Defence & Military"
            )
        parsed[
            "secondary_domains"
        ] = secondary
        applied_rules.append(
            "pow_conflict_primary"
        )

    # A security-relevant, human-review invitation involving strategic actors
    # is better represented as geopolitical/diplomatic engagement than as
    # military activity. We keep the review flag so an analyst can still inspect
    # the borderline relevance decision.
    if (
        relevant
        and re.search(
            r"\bcalls? to invite\b",
            title_lower,
        )
        and parsed.get(
            "primary_domain"
        ) == "Defence & Military"
        and not regex_any(
            title_lower,
            DIRECT_DEFENCE_PATTERNS,
        )
    ):
        parsed[
            "primary_domain"
        ] = "Conflict & Geopolitical Tensions"
        parsed[
            "event_status"
        ] = "Diplomatic negotiation"
        applied_rules.append(
            "review_invitation_as_diplomacy"
        )

    # Deterministic event-status layer. The model decides relevance/meaning,
    # while these high-precision headline rules prevent obvious real-world
    # actions from collapsing into the generic Background / analysis bucket.
    if (
        relevant
        and parsed.get(
            "primary_domain"
        ) == "Cybersecurity"
        and regex_any(
            title_lower,
            [
                r"\bcyberattack",
                r"\bhackers?\b.*\bshut down\b",
                r"\bransomware\b",
                r"\bdata breach\b",
                r"\bmalware\b",
            ],
        )
    ):
        parsed[
            "event_status"
        ] = "Cyber incident"
        applied_rules.append(
            "confirmed_cyber_incident"
        )

    elif (
        relevant
        and title_reports_actual_violence(
            title_lower
        )
    ):
        parsed[
            "event_status"
        ] = "Actual violence / combat"
        applied_rules.append(
            "clear_actual_violence"
        )

    elif (
        relevant
        and regex_any(
            title_lower,
            MILITARY_POSTURE_PATTERNS,
        )
        and regex_any(
            title_lower,
            DIRECT_DEFENCE_PATTERNS,
        )
        and not regex_any(
            title_lower,
            MILITARY_POSTURE_NONACTUAL_PATTERNS,
        )
    ):
        parsed[
            "event_status"
        ] = "Military posture / deployment"
        applied_rules.append(
            "clear_military_posture"
        )

    elif (
        relevant
        and parsed.get(
            "primary_domain"
        ) == "Sanctions & Economic Security"
        and re.search(
            r"\b(?:sanctions?|sanctioned|asset freeze|export controls?|embargo)\b",
            title_lower,
        )
    ):
        parsed[
            "event_status"
        ] = "Sanctions / economic coercion"
        applied_rules.append(
            "clear_sanctions_action"
        )

    elif (
        relevant
        and regex_any(
            title_lower,
            MILITARY_COOPERATION_PATTERNS,
        )
    ):
        parsed[
            "event_status"
        ] = "Military cooperation / training"
        applied_rules.append(
            "clear_military_cooperation"
        )

    elif (
        relevant
        and regex_any(
            title_lower,
            THREAT_CUE_PATTERNS,
        )
    ):
        parsed[
            "event_status"
        ] = "Threat / warning"
        applied_rules.append(
            "threat_not_combat"
        )

    elif (
        relevant
        and regex_any(
            title_lower,
            DIPLOMATIC_NEGOTIATION_PATTERNS,
        )
    ):
        parsed[
            "event_status"
        ] = "Diplomatic negotiation"
        applied_rules.append(
            "clear_diplomatic_negotiation"
        )

    elif (
        relevant
        and parsed.get(
            "event_status"
        ) == "Actual violence / combat"
    ):
        parsed[
            "event_status"
        ] = "Background / analysis"
        applied_rules.append(
            "no_actual_violence_evidence"
        )

    # Kinetic Russia-Ukraine events are conflict first, military second.
    if (
        relevant
        and parsed.get(
            "primary_domain"
        ) == "Defence & Military"
        and regex_any(
            title_lower,
            RUSSIA_UKRAINE_PATTERNS,
        )
        and regex_any(
            title_lower,
            KINETIC_CONFLICT_PATTERNS,
        )
    ):
        parsed[
            "primary_domain"
        ] = (
            "Conflict & Geopolitical Tensions"
        )

        secondary = list_value(
            parsed.get(
                "secondary_domains"
            )
        )

        if (
            "Defence & Military"
            not in secondary
        ):
            secondary.append(
                "Defence & Military"
            )

        parsed[
            "secondary_domains"
        ] = secondary
        applied_rules.append(
            "kinetic_conflict_primary"
        )

    # Alliance status, bargaining positions and diplomatic pressure are
    # geopolitical first, not military activity.
    if (
        relevant
        and parsed.get(
            "primary_domain"
        ) == "Defence & Military"
        and regex_any(
            title_lower,
            [
                r"\bcsto\b",
                r"\bbargaining chip\b",
                r"\burge.*\bsettlement\b",
            ],
        )
    ):
        parsed[
            "primary_domain"
        ] = (
            "Conflict & Geopolitical Tensions"
        )
        applied_rules.append(
            "geopolitical_not_defence"
        )

    # Terrorism-list policy is geopolitical/security policy, not military activity.
    if (
        relevant
        and parsed.get(
            "primary_domain"
        ) == "Defence & Military"
        and re.search(
            r"\bterrorism list\b|\bterrorist sponsors?\b|\bstate sponsors? of terrorism\b",
            title_lower,
        )
    ):
        parsed[
            "primary_domain"
        ] = (
            "Conflict & Geopolitical Tensions"
        )
        applied_rules.append(
            "terrorism_list_not_defence"
        )

    # Legal/political terrorist designations are security-policy / geopolitical
    # developments, not military operations merely because the word terror appears.
    if (
        relevant
        and parsed.get(
            "primary_domain"
        ) == "Defence & Military"
        and regex_any(
            title_lower,
            TERROR_DESIGNATION_PATTERNS,
        )
    ):
        parsed[
            "primary_domain"
        ] = "Conflict & Geopolitical Tensions"
        applied_rules.append(
            "terror_designation_not_defence"
        )

    # General aid to Ukraine is not a sanctions story unless sanctions or
    # frozen assets are explicit.
    if (
        relevant
        and parsed.get(
            "primary_domain"
        ) == "Sanctions & Economic Security"
        and re.search(
            r"\baid\b|\baid package\b",
            title_lower,
        )
        and not re.search(
            r"\bsanction|\bfrozen assets?\b|\bseize",
            title_lower,
        )
    ):
        parsed[
            "primary_domain"
        ] = (
            "Conflict & Geopolitical Tensions"
        )
        applied_rules.append(
            "aid_not_sanctions"
        )

    parsed[
        "security_relevant"
    ] = relevant

    if not relevant:
        parsed[
            "primary_domain"
        ] = "None"
        parsed[
            "secondary_domains"
        ] = []
        parsed[
            "event_status"
        ] = "Non-security"
        parsed[
            "needs_human_review"
        ] = False

    else:
        # Review flags should also exist among KEEP results.
        try:
            confidence = float(
                parsed.get(
                    "confidence"
                )
            )
        except Exception:
            confidence = 0.0

        centrality = clean(
            parsed.get(
                "geographic_centrality"
            )
        )

        ambiguous_keep = (
            confidence < 0.90
            or centrality in {
                "Low",
                "Medium",
                "",
                "None",
            }
            or re.search(
                r"^\s*opinion\b",
                title_lower,
            )
            is not None
            or (
                parsed.get(
                    "primary_domain"
                ) == "Defence & Military"
                and not regex_any(
                    title_lower,
                    DIRECT_DEFENCE_PATTERNS,
                )
            )
        )

        if any(
            rule in applied_rules
            for rule in {
                "aid_not_sanctions",
                "geopolitical_not_defence",
                "terrorism_list_not_defence",
                "terror_designation_not_defence",
                "review_invitation_as_diplomacy",
            }
        ):
            ambiguous_keep = True

        parsed[
            "needs_human_review"
        ] = bool(
            ambiguous_keep
        )

        parsed[
            "exclusion_reason"
        ] = "None"

    parsed = normalize_classification(
        parsed
    )

    # Final QA policy: deterministic exclusions do not require manual review.
    # Review flags are reserved for ambiguous KEEP cases.
    if not bool_value(
        parsed.get(
            "security_relevant"
        )
    ):
        parsed[
            "needs_human_review"
        ] = False

    parsed[
        "classifier_version"
    ] = FINAL_CLASSIFIER_VERSION
    parsed[
        "ai_classifier_version"
    ] = CLASSIFIER_VERSION
    parsed[
        "postprocess_version"
    ] = POSTPROCESS_VERSION
    parsed[
        "postprocess_rule"
    ] = (
        " | ".join(
            applied_rules
        )
        if applied_rules
        else "none"
    )

    return parsed

# ==================================================
# CLASSIFY ONE ARTICLE
# ==================================================

def classify_article(
    row,
    model=DEFAULT_MODEL,
):
    article_payload = (
        build_article_input(
            row
        )
    )

    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    SYSTEM_INSTRUCTIONS
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    article_payload,
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ],
        "stream": False,
        "think": False,
        "format": (
            CLASSIFICATION_SCHEMA
        ),
        "options": {
            "temperature": 0,
            "seed": RANDOM_SEED,
            "num_ctx": 3072,
            "num_predict": 240,
        },
        "keep_alive": "10m",
    }

    encoded = json.dumps(
        body,
        ensure_ascii=False,
    ).encode(
        "utf-8"
    )

    last_error = ""

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        started = (
            time.perf_counter()
        )

        try:
            req = request.Request(
                OLLAMA_CHAT_URL,
                data=encoded,
                headers={
                    "Content-Type": (
                        "application/json"
                    )
                },
                method="POST",
            )

            with request.urlopen(
                req,
                timeout=(
                    REQUEST_TIMEOUT_SECONDS
                ),
            ) as response:
                payload = json.loads(
                    response.read()
                    .decode("utf-8")
                )

            content = (
                payload
                .get(
                    "message",
                    {}
                )
                .get(
                    "content",
                    ""
                )
            )

            if not content:
                raise RuntimeError(
                    "Ollama returned an empty "
                    "classification."
                )

            parsed = json.loads(
                content
            )

            parsed = (
                normalize_classification(
                    parsed
                )
            )

            parsed = (
                apply_title_guardrails(
                    row,
                    parsed,
                )
            )

            parsed[
                "classifier_version"
            ] = CLASSIFIER_VERSION

            parsed[
                "model"
            ] = model

            parsed[
                "status"
            ] = "ok"

            parsed[
                "error"
            ] = ""

            parsed[
                "elapsed_seconds"
            ] = round(
                (
                    time.perf_counter()
                    - started
                ),
                2,
            )

            parsed[
                "prompt_eval_count"
            ] = payload.get(
                "prompt_eval_count"
            )

            parsed[
                "eval_count"
            ] = payload.get(
                "eval_count"
            )

            return parsed

        except Exception as exc:
            last_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(
                    RETRY_WAIT_SECONDS
                    * attempt
                )

    return {
        "geographic_centrality": "",
        "security_relevant": None,
        "primary_domain": "",
        "secondary_domains": [],
        "event_status": "",
        "confidence": None,
        "needs_human_review": True,
        "exclusion_reason": (
            "Insufficient evidence"
        ),
        "reason_short": "",
        "classifier_version": (
            CLASSIFIER_VERSION
        ),
        "model": model,
        "status": "error",
        "error": last_error,
        "elapsed_seconds": None,
        "prompt_eval_count": None,
        "eval_count": None,
    }


# ==================================================
# CACHE
# ==================================================

def article_cache_key(
    source_url,
    model,
):
    raw = (
        f"{CLASSIFIER_VERSION}"
        f"|{model}"
        f"|{clean(source_url)}"
    )

    return hashlib.sha256(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()


def load_cache():
    if not CACHE_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            CACHE_PATH,
            encoding="utf-8-sig",
        )

    except Exception:
        return pd.DataFrame()


def serialise_value(value):
    if isinstance(
        value,
        list,
    ):
        return " | ".join(
            str(item)
            for item in value
        )

    return value


def save_cache(df):
    output = df.copy()

    # Preserve successful cached rows that are outside the current subset.
    if CACHE_PATH.exists():
        try:
            existing = pd.read_csv(
                CACHE_PATH,
                encoding="utf-8-sig",
            )

            if (
                not existing.empty
                and "cache_key"
                in existing.columns
                and "cache_key"
                in output.columns
            ):
                output = pd.concat(
                    [
                        existing,
                        output,
                    ],
                    ignore_index=True,
                    sort=False,
                )

                output = (
                    output
                    .drop_duplicates(
                        subset=[
                            "cache_key"
                        ],
                        keep="last",
                    )
                    .reset_index(
                        drop=True
                    )
                )
        except Exception:
            pass

    for column in output.columns:
        output[
            column
        ] = (
            output[
                column
            ]
            .apply(
                serialise_value
            )
        )

    output.to_csv(
        CACHE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if "status" in output.columns:
        errors = output.loc[
            output[
                "status"
            ] == "error"
        ].copy()
    else:
        errors = pd.DataFrame()

    errors.to_csv(
        ERROR_PATH,
        index=False,
        encoding="utf-8-sig",
    )


# ==================================================
# CLASSIFY ARTICLE DATAFRAME
# ==================================================

def classify_dataframe_articles(
    articles,
    model=DEFAULT_MODEL,
    limit=None,
    force=False,
):
    working = (
        articles.copy()
    )

    if limit is not None:
        working = (
            working.head(
                max(
                    0,
                    int(limit),
                )
            )
            .copy()
        )

    cache = load_cache()

    cached_by_key = {}

    if (
        not cache.empty
        and "cache_key"
        in cache.columns
    ):
        for _, cached_row in (
            cache.iterrows()
        ):
            if (
                clean(
                    cached_row.get(
                        "status"
                    )
                )
                != "ok"
            ):
                continue

            cached_by_key[
                clean(
                    cached_row.get(
                        "cache_key"
                    )
                )
            ] = cached_row.to_dict()

    result_rows = []
    total = len(working)
    model_checked = False

    for index, (_, row) in enumerate(
        working.iterrows(),
        start=1,
    ):
        source_url = clean(
            row.get(
                "source_url"
            )
        )

        title = clean(
            row.get(
                "article_title"
            ),
            90,
        )

        cache_key = (
            article_cache_key(
                source_url,
                model,
            )
        )

        if (
            not force
            and cache_key in cached_by_key
        ):
            print(
                f"[{index}/{total}] "
                f"CACHED+RULES | {title}"
            )

            cached = dict(
                cached_by_key[
                    cache_key
                ]
            )

            base = (
                base_classification_from_cached(
                    cached
                )
            )

            final_result = (
                final_postprocess(
                    row,
                    base,
                )
            )

            output_row = {
                **cached,
                "cache_key": cache_key,
                "article_id": row.get(
                    "article_id"
                ),
                "article_title": row.get(
                    "article_title"
                ),
                "source_domain": row.get(
                    "source_domain"
                ),
                "source_url": source_url,
                "gdelt_event_count": row.get(
                    "gdelt_event_count"
                ),
                "location_countries": row.get(
                    "location_countries"
                ),
                "actor1_countries": row.get(
                    "actor1_countries"
                ),
                "actor2_countries": row.get(
                    "actor2_countries"
                ),
                "event_root_labels": row.get(
                    "event_root_labels"
                ),
                **base_fields_for_output(
                    base
                ),
                **final_result,
            }

            result_rows.append(
                output_row
            )
            continue

        if not model_checked:
            ensure_model_available(
                model
            )
            model_checked = True

        print(
            f"[{index}/{total}] "
            f"AI+RULES | {title}"
        )

        base = classify_article(
            row,
            model=model,
        )

        if (
            clean(
                base.get(
                    "status"
                )
            )
            == "ok"
        ):
            final_result = (
                final_postprocess(
                    row,
                    base,
                )
            )
        else:
            final_result = dict(
                base
            )
            final_result[
                "classifier_version"
            ] = FINAL_CLASSIFIER_VERSION
            final_result[
                "ai_classifier_version"
            ] = CLASSIFIER_VERSION
            final_result[
                "postprocess_version"
            ] = POSTPROCESS_VERSION
            final_result[
                "postprocess_rule"
            ] = "ai_error"

        output_row = {
            "cache_key": cache_key,
            "article_id": row.get(
                "article_id"
            ),
            "article_title": row.get(
                "article_title"
            ),
            "source_domain": row.get(
                "source_domain"
            ),
            "source_url": source_url,
            "gdelt_event_count": row.get(
                "gdelt_event_count"
            ),
            "location_countries": row.get(
                "location_countries"
            ),
            "actor1_countries": row.get(
                "actor1_countries"
            ),
            "actor2_countries": row.get(
                "actor2_countries"
            ),
            "event_root_labels": row.get(
                "event_root_labels"
            ),
            **base_fields_for_output(
                base
            ),
            **final_result,
        }

        result_rows.append(
            output_row
        )

        save_cache(
            pd.DataFrame(
                [output_row]
            )
        )

    final = pd.DataFrame(
        result_rows
    )

    save_cache(
        final
    )

    return final


# ==================================================
# SUMMARY
# ==================================================

def truthy(series):
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )


def build_summary(
    results,
):
    ok = results.loc[
        results[
            "status"
        ] == "ok"
    ].copy()

    failed = results.loc[
        results[
            "status"
        ] != "ok"
    ].copy()

    if ok.empty:
        return {
            "classifier_version": (
                FINAL_CLASSIFIER_VERSION
            ),
            "articles": len(results),
            "successful": 0,
            "errors": len(failed),
        }

    relevant = truthy(
        ok[
            "security_relevant"
        ]
    )

    review = truthy(
        ok[
            "needs_human_review"
        ]
    )

    elapsed = pd.to_numeric(
        ok[
            "elapsed_seconds"
        ],
        errors="coerce",
    )

    confidence = pd.to_numeric(
        ok[
            "confidence"
        ],
        errors="coerce",
    )

    return {
        "classifier_version": (
            FINAL_CLASSIFIER_VERSION
        ),
        "articles": len(results),
        "successful": len(ok),
        "errors": len(failed),
        "security_relevant": int(
            relevant.sum()
        ),
        "excluded": int(
            (~relevant).sum()
        ),
        "needs_human_review": int(
            review.sum()
        ),
        "mean_confidence": round(
            float(
                confidence.mean()
            ),
            3,
        ),
        "median_seconds_per_article": round(
            float(
                elapsed.median()
            ),
            2,
        ),
        "mean_seconds_per_article": round(
            float(
                elapsed.mean()
            ),
            2,
        ),
    }


def print_summary(
    results,
):
    summary = build_summary(
        results
    )

    print()
    print("=" * 72)
    print(
        "ARTICLE AI CLASSIFICATION SUMMARY"
    )
    print("=" * 72)
    print()

    for key, value in (
        summary.items()
    ):
        print(
            f"{key}: {value}"
        )

    print()

    if (
        not results.empty
        and "status" in results.columns
    ):
        ok = results.loc[
            results[
                "status"
            ] == "ok"
        ].copy()

        if not ok.empty:
            relevant = truthy(
                ok[
                    "security_relevant"
                ]
            )

            print(
                "Primary domains:"
            )

            counts = (
                ok.loc[
                    relevant,
                    "primary_domain",
                ]
                .value_counts()
            )

            if counts.empty:
                print("  None")
            else:
                for domain, count in (
                    counts.items()
                ):
                    print(
                        f"  {domain}: {count}"
                    )

            print()

    pd.DataFrame(
        [summary]
    ).to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )


# ==================================================
# CLI
# ==================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Article-level local AI classifier "
            "for European Security Monitor."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT_PATH
        ),
        help=(
            "Article-level CSV input."
        ),
    )

    parser.add_argument(
        "--model",
        default=(
            DEFAULT_MODEL
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Classify only first N articles."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ignore cached successful "
            "classifications and rerun them."
        ),
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{input_path}"
        )

    print()
    print("=" * 72)
    print(
        "EUROPEAN SECURITY MONITOR "
        "- ARTICLE AI CLASSIFIER"
    )
    print("=" * 72)
    print()
    print(
        "AI cache:",
        CLASSIFIER_VERSION,
    )
    print(
        "Postprocess:",
        POSTPROCESS_VERSION,
    )
    print(
        "Final classifier:",
        FINAL_CLASSIFIER_VERSION,
    )
    print(
        "Model:",
        args.model,
    )
    print(
        "Input:",
        input_path,
    )
    print(
        "Cache:",
        CACHE_PATH,
    )
    print()
    print(
        "Local Ollama only. "
        "No API cost. "
        "SQLite is NOT modified."
    )
    print()

    articles = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
    )

    required = {
        "source_url",
        "article_title",
    }

    missing = (
        required
        - set(
            articles.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Missing required input columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    results = (
        classify_dataframe_articles(
            articles,
            model=args.model,
            limit=args.limit,
            force=args.force,
        )
    )

    print_summary(
        results
    )

    print(
        "Classification cache:"
    )
    print(
        " ",
        CACHE_PATH,
    )
    print()
    print(
        "No database changes were made."
    )
    print()


if __name__ == "__main__":
    main()
