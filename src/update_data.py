from pathlib import Path
from urllib.parse import unquote, urlparse
from datetime import datetime, timezone

import html
import json
import re
import shutil
import sqlite3

import numpy as np
import pandas as pd
import requests

try:
    from article_security_classifier import (
        classify_dataframe_articles,
        DEFAULT_MODEL,
        FINAL_CLASSIFIER_VERSION,
    )
except ImportError as exc:
    raise RuntimeError(
        "European Security Monitor requires the final "
        "src/article_security_classifier.py. "
        "Expected a classifier exposing "
        "classify_dataframe_articles, DEFAULT_MODEL and "
        "FINAL_CLASSIFIER_VERSION."
    ) from exc


EXPECTED_CLASSIFIER_VERSION = "article-ai-v3+post-rules-v4.2"

if FINAL_CLASSIFIER_VERSION != EXPECTED_CLASSIFIER_VERSION:
    raise RuntimeError(
        "Wrong article_security_classifier.py version. "
        f"Expected {EXPECTED_CLASSIFIER_VERSION}, "
        f"found {FINAL_CLASSIFIER_VERSION}."
    )


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DATABASE_PATH = DATA_DIR / "security_monitor.db"
PRE_V42_BACKUP_PATH = (
    DATA_DIR / "security_monitor_pre_v42_backup.db"
)

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# GDELT CONFIG
# ==================================================

LAST_UPDATE_URL = (
    "https://data.gdeltproject.org/gdeltv2/lastupdate.txt"
)


# ==================================================
# GEOGRAPHIC SCOPE
# ==================================================

EUROPEAN_COUNTRIES = [
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
    "Vatican City"
]


STRATEGIC_NEIGHBOURS = [
    "Russia",
    "Georgia",
    "Armenia",
    "Azerbaijan"
]


MONITORED_COUNTRIES = (
    EUROPEAN_COUNTRIES
    + STRATEGIC_NEIGHBOURS
)


MONITORED_ISO3 = {
    "ALB",
    "AND",
    "AUT",
    "BLR",
    "BEL",
    "BIH",
    "BGR",
    "HRV",
    "CYP",
    "CZE",
    "DNK",
    "EST",
    "FIN",
    "FRA",
    "DEU",
    "GRC",
    "HUN",
    "ISL",
    "IRL",
    "ITA",
    "LVA",
    "LIE",
    "LTU",
    "LUX",
    "MLT",
    "MDA",
    "MCO",
    "MNE",
    "NLD",
    "MKD",
    "NOR",
    "POL",
    "PRT",
    "ROU",
    "SMR",
    "SRB",
    "SVK",
    "SVN",
    "ESP",
    "SWE",
    "CHE",
    "TUR",
    "UKR",
    "GBR",
    "VAT",

    # Strategic neighbourhood
    "RUS",
    "GEO",
    "ARM",
    "AZE"
}


LOCATION_COUNTRY_ALIASES = {
    "Turkey": "Türkiye",
    "Czech Republic": "Czechia",
    "Russian Federation": "Russia",
    "Vatican": "Vatican City"
}


# ==================================================
# GDELT EVENT COLUMNS
# ==================================================

GDELT_EVENT_COLUMNS = [
    "GLOBALEVENTID",
    "SQLDATE",
    "MonthYear",
    "Year",
    "FractionDate",
    "Actor1Code",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor1KnownGroupCode",
    "Actor1EthnicCode",
    "Actor1Religion1Code",
    "Actor1Religion2Code",
    "Actor1Type1Code",
    "Actor1Type2Code",
    "Actor1Type3Code",
    "Actor2Code",
    "Actor2Name",
    "Actor2CountryCode",
    "Actor2KnownGroupCode",
    "Actor2EthnicCode",
    "Actor2Religion1Code",
    "Actor2Religion2Code",
    "Actor2Type1Code",
    "Actor2Type2Code",
    "Actor2Type3Code",
    "IsRootEvent",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "Actor1Geo_Type",
    "Actor1Geo_FullName",
    "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code",
    "Actor1Geo_ADM2Code",
    "Actor1Geo_Lat",
    "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type",
    "Actor2Geo_FullName",
    "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code",
    "Actor2Geo_ADM2Code",
    "Actor2Geo_Lat",
    "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type",
    "ActionGeo_FullName",
    "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code",
    "ActionGeo_ADM2Code",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED",
    "SOURCEURL"
]


# ==================================================
# GKG COLUMNS
# ==================================================

GKG_COLUMNS = [
    "GKGRECORDID",
    "V2DATE",
    "V2SOURCECOLLECTIONIDENTIFIER",
    "V2SOURCECOMMONNAME",
    "V2DOCUMENTIDENTIFIER",
    "V1COUNTS",
    "V2COUNTS",
    "V1THEMES",
    "V2ENHANCEDTHEMES",
    "V1LOCATIONS",
    "V2ENHANCEDLOCATIONS",
    "V1PERSONS",
    "V2ENHANCEDPERSONS",
    "V1ORGANIZATIONS",
    "V2ENHANCEDORGANIZATIONS",
    "V1TONE",
    "V2ENHANCEDDATES",
    "V2GCAM",
    "V2SHARINGIMAGE",
    "V2RELATEDIMAGES",
    "V2SOCIALIMAGEEMBEDS",
    "V2SOCIALVIDEOEMBEDS",
    "V2QUOTATIONS",
    "V2ALLNAMES",
    "V2AMOUNTS",
    "V2TRANSLATIONINFO",
    "V2EXTRASXML"
]


# ==================================================
# GET LATEST GDELT FILES
# ==================================================

def get_latest_gdelt_files():

    response = requests.get(
        LAST_UPDATE_URL,
        timeout=30
    )

    response.raise_for_status()

    lines = response.text.strip().splitlines()

    if len(lines) < 3:
        raise ValueError(
            "Unexpected response from GDELT lastupdate.txt"
        )

    export_url = (
        lines[0]
        .split()[-1]
        .replace("http://", "https://")
    )

    gkg_url = (
        lines[2]
        .split()[-1]
        .replace("http://", "https://")
    )

    return export_url, gkg_url


# ==================================================
# DOWNLOAD FILE
# ==================================================

def download_file(url):

    file_name = url.split("/")[-1]

    destination = RAW_DIR / file_name

    if destination.exists():

        print(
            f"Already exists: {file_name}"
        )

        return destination

    response = requests.get(
        url,
        timeout=60
    )

    response.raise_for_status()

    with open(
        destination,
        "wb"
    ) as file:

        file.write(
            response.content
        )

    print(
        f"Downloaded: {file_name}"
    )

    return destination


# ==================================================
# LOAD RAW EVENTS
# ==================================================

def load_events(file_path):

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=GDELT_EVENT_COLUMNS,
        compression="zip",
        low_memory=False
    )

    if len(df.columns) != 61:
        raise ValueError(
            f"Expected 61 GDELT Event columns, "
            f"but found {len(df.columns)}."
        )

    return df


# ==================================================
# LOAD RAW GKG
# ==================================================

def load_gkg(file_path):

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=GKG_COLUMNS,
        compression="zip",
        low_memory=False
    )

    if len(df.columns) != 27:
        raise ValueError(
            f"Expected 27 GKG columns, "
            f"but found {len(df.columns)}."
        )

    return df


# ==================================================
# PREPARE EVENTS
# ==================================================

def prepare_events(df):

    selected_columns = [
        "GLOBALEVENTID",
        "SQLDATE",

        "Actor1Name",
        "Actor1CountryCode",
        "Actor1Type1Code",
        "Actor1KnownGroupCode",

        "Actor2Name",
        "Actor2CountryCode",
        "Actor2Type1Code",
        "Actor2KnownGroupCode",

        "EventCode",
        "EventRootCode",
        "QuadClass",
        "GoldsteinScale",

        "NumMentions",
        "NumSources",
        "NumArticles",
        "AvgTone",

        "ActionGeo_FullName",
        "ActionGeo_CountryCode",
        "ActionGeo_Lat",
        "ActionGeo_Long",

        "SOURCEURL"
    ]

    events = df[
        selected_columns
    ].copy()

    events = events.rename(
        columns={
            "GLOBALEVENTID": "event_id",
            "SQLDATE": "event_date",

            "Actor1Name": "actor1",
            "Actor1CountryCode": "actor1_country",
            "Actor1Type1Code": "actor1_type",
            "Actor1KnownGroupCode": "actor1_group",

            "Actor2Name": "actor2",
            "Actor2CountryCode": "actor2_country",
            "Actor2Type1Code": "actor2_type",
            "Actor2KnownGroupCode": "actor2_group",

            "EventCode": "event_code",
            "EventRootCode": "event_root_code",
            "QuadClass": "quad_class",
            "GoldsteinScale": "goldstein_scale",

            "NumMentions": "num_mentions",
            "NumSources": "num_sources",
            "NumArticles": "num_articles",
            "AvgTone": "avg_tone",

            "ActionGeo_FullName": "location",
            "ActionGeo_CountryCode": "location_country",
            "ActionGeo_Lat": "latitude",
            "ActionGeo_Long": "longitude",

            "SOURCEURL": "source_url"
        }
    )

    events["event_date"] = pd.to_datetime(
        events["event_date"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    events = events[
        events["event_id"].notna()
    ].copy()

    events = events.drop_duplicates(
        subset="event_id"
    )

    return events


# ==================================================
# EXTRACT COUNTRY FROM GDELT LOCATION
# ==================================================

def extract_gdelt_location_country(location):

    if not isinstance(location, str):
        return []

    country_candidate = (
        location
        .split(",")[-1]
        .strip()
    )

    country_candidate = (
        LOCATION_COUNTRY_ALIASES.get(
            country_candidate,
            country_candidate
        )
    )

    if country_candidate in MONITORED_COUNTRIES:
        return [country_candidate]

    return []


# ==================================================
# FILTER EUROPE + STRATEGIC NEIGHBOURHOOD
# ==================================================

def filter_geographic_scope(events):

    events = events.copy()

    events["location_countries"] = (
        events["location"]
        .apply(
            extract_gdelt_location_country
        )
    )

    actor1_match = (
        events["actor1_country"]
        .isin(MONITORED_ISO3)
    )

    actor2_match = (
        events["actor2_country"]
        .isin(MONITORED_ISO3)
    )

    location_match = (
        events["location_countries"]
        .apply(
            lambda countries:
            len(countries) > 0
        )
    )

    europe = events[
        actor1_match
        | actor2_match
        | location_match
    ].copy()

    return europe

# ==================================================
# PREPARE GKG
# ==================================================

def prepare_gkg(df):

    selected_columns = [
        "GKGRECORDID",
        "V2DATE",
        "V2SOURCECOMMONNAME",
        "V2DOCUMENTIDENTIFIER",
        "V1THEMES",
        "V2ENHANCEDTHEMES",
        "V1LOCATIONS",
        "V1ORGANIZATIONS",
        "V1TONE"
    ]

    gkg = df[
        selected_columns
    ].copy()

    gkg = gkg.rename(
        columns={
            "GKGRECORDID": "gkg_record_id",
            "V2DATE": "gkg_date",
            "V2SOURCECOMMONNAME": "source_name",
            "V2DOCUMENTIDENTIFIER": "document_url",
            "V1THEMES": "themes",
            "V2ENHANCEDTHEMES": "enhanced_themes",
            "V1LOCATIONS": "gkg_locations",
            "V1ORGANIZATIONS": "organizations",
            "V1TONE": "gkg_tone"
        }
    )

    # Remove rows without a document URL
    gkg = gkg[
        gkg["document_url"].notna()
    ].copy()

    # One GKG record per article URL
    gkg = gkg.drop_duplicates(
        subset="document_url"
    )

    return gkg


# ==================================================
# ENRICH EVENTS WITH GKG
# ==================================================

def enrich_events_with_gkg(events, gkg):

    enriched = events.merge(
        gkg,
        left_on="source_url",
        right_on="document_url",
        how="left",
        validate="many_to_one"
    )

    return enriched

# ==================================================
# ARTICLE-LEVEL AI PRODUCTION HELPERS
# ==================================================

PAGE_TITLE_RE = re.compile(
    r"<PAGE_TITLE>(.*?)</PAGE_TITLE>",
    flags=re.IGNORECASE | re.DOTALL,
)


STATUS_SEVERITY_CAP = {
    "Actual violence / combat": 100.0,
    "Cyber incident": 90.0,
    "Military posture / deployment": 75.0,
    "Sanctions / economic coercion": 70.0,
    "Military cooperation / training": 60.0,
    "Threat / warning": 55.0,
    "Diplomatic negotiation": 40.0,
    "Strategic statement": 35.0,
    "Background / analysis": 30.0,
    "Unclear": 30.0,
}


def clean_text(value):

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "<na>",
    }:
        return ""

    return text


def to_bool(value):

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return (
        clean_text(value)
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
        }
    )


def list_value(value):

    if isinstance(value, list):
        return [
            clean_text(item)
            for item in value
            if clean_text(item)
        ]

    text = clean_text(value)

    if not text:
        return []

    return [
        item.strip()
        for item in text.split("|")
        if item.strip()
    ]


def join_unique(series):

    values = []
    seen = set()

    for value in series:

        candidates = (
            value
            if isinstance(value, list)
            else [value]
        )

        for candidate in candidates:

            text = clean_text(candidate)

            if (
                text
                and text not in seen
            ):
                seen.add(text)
                values.append(text)

    return " | ".join(values)


def first_nonempty(series):

    for value in series:
        text = clean_text(value)
        if text:
            return text

    return ""


def extract_domain(url):

    text = clean_text(url)

    if not text:
        return ""

    try:
        domain = (
            urlparse(text)
            .netloc
            .lower()
            .strip()
        )

        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    except Exception:
        return ""


def extract_page_title(extras_xml):

    if not isinstance(extras_xml, str):
        return ""

    match = PAGE_TITLE_RE.search(
        extras_xml
    )

    if not match:
        return ""

    title = html.unescape(
        match.group(1)
    )

    title = re.sub(
        r"<[^>]+>",
        " ",
        title,
    )

    title = re.sub(
        r"\s+",
        " ",
        title,
    )

    return title.strip()


def attach_page_titles(
    prepared_gkg,
    raw_gkg,
):

    title_source = raw_gkg[
        [
            "V2DOCUMENTIDENTIFIER",
            "V2EXTRASXML",
        ]
    ].copy()

    title_source[
        "article_title"
    ] = (
        title_source[
            "V2EXTRASXML"
        ]
        .apply(
            extract_page_title
        )
    )

    title_source = (
        title_source
        .dropna(
            subset=[
                "V2DOCUMENTIDENTIFIER"
            ]
        )
        .drop_duplicates(
            subset=[
                "V2DOCUMENTIDENTIFIER"
            ],
            keep="first",
        )
    )

    title_map = (
        title_source
        .set_index(
            "V2DOCUMENTIDENTIFIER"
        )[
            "article_title"
        ]
        .to_dict()
    )

    output = prepared_gkg.copy()

    output[
        "article_title"
    ] = (
        output[
            "document_url"
        ]
        .map(
            title_map
        )
        .fillna("")
    )

    return output


def add_basic_event_labels(df):

    output = df.copy()

    output[
        "event_root_label"
    ] = (
        output[
            "event_root_code"
        ]
        .map(
            CAMEO_ROOT_CODES
        )
    )

    output[
        "quad_class_label"
    ] = (
        output[
            "quad_class"
        ]
        .map(
            QUAD_CLASS_LABELS
        )
    )

    return output


def build_article_dataset(events):

    rows = []

    for source_url, group in (
        events.groupby(
            "source_url",
            dropna=False,
            sort=False,
        )
    ):

        url = clean_text(
            source_url
        )

        if not url:
            continue

        root_codes = (
            pd.to_numeric(
                group[
                    "event_root_code"
                ],
                errors="coerce",
            )
            .dropna()
            .astype(int)
        )

        unique_roots = list(
            dict.fromkeys(
                root_codes.tolist()
            )
        )

        rows.append(
            {
                "source_url": url,
                "source_domain": (
                    extract_domain(url)
                ),
                "article_title": (
                    first_nonempty(
                        group[
                            "article_title"
                        ]
                    )
                ),
                "gdelt_event_count": (
                    len(group)
                ),
                "location_countries": (
                    join_unique(
                        group[
                            "location_countries"
                        ]
                    )
                ),
                "actor1_countries": (
                    join_unique(
                        group[
                            "actor1_country"
                        ]
                    )
                ),
                "actor2_countries": (
                    join_unique(
                        group[
                            "actor2_country"
                        ]
                    )
                ),
                "event_root_labels": (
                    " | ".join(
                        CAMEO_ROOT_CODES.get(
                            code,
                            str(code),
                        )
                        for code
                        in unique_roots
                    )
                ),
            }
        )

    articles = pd.DataFrame(
        rows
    )

    if articles.empty:
        return articles

    articles = (
        articles
        .sort_values(
            [
                "gdelt_event_count",
                "article_title",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    articles.insert(
        0,
        "article_id",
        range(
            1,
            len(articles) + 1,
        ),
    )

    return articles


def validate_article_classifications(
    articles,
    classifications,
):

    if articles.empty:
        return

    if classifications.empty:
        raise RuntimeError(
            "Article classifier returned no results."
        )

    if "status" not in classifications.columns:
        raise RuntimeError(
            "Article classifier output is missing "
            "the 'status' column."
        )

    failed = (
        classifications[
            "status"
        ]
        .astype(str)
        .str.lower()
        .ne("ok")
    )

    if failed.any():
        raise RuntimeError(
            f"{int(failed.sum())} article "
            "classifications failed. "
            "The production database was NOT updated."
        )


def propagate_article_classification(
    events,
    classifications,
):

    if events.empty:
        return events.copy()

    if classifications.empty:
        output = events.copy()
        output[
            "security_relevant_ai"
        ] = False
        output[
            "needs_human_review_ai"
        ] = False
        output[
            "secondary_domains_ai"
        ] = [[] for _ in range(len(output))]
        output[
            "security_domains_ai"
        ] = [[] for _ in range(len(output))]
        return output

    keep_columns = [
        "source_url",
        "article_title",
        "security_relevant",
        "geographic_centrality",
        "primary_domain",
        "secondary_domains",
        "event_status",
        "confidence",
        "needs_human_review",
        "exclusion_reason",
        "reason_short",
        "classifier_version",
        "ai_classifier_version",
        "postprocess_version",
        "postprocess_rule",
        "status",
        "error",
    ]

    available = [
        column
        for column in keep_columns
        if column
        in classifications.columns
    ]

    article_map = (
        classifications[
            available
        ]
        .drop_duplicates(
            subset=[
                "source_url"
            ],
            keep="last",
        )
        .copy()
    )

    enriched = (
        events.merge(
            article_map,
            on="source_url",
            how="left",
            validate="many_to_one",
            suffixes=(
                "",
                "_ai",
            ),
        )
    )

    if "article_title_ai" in enriched.columns:

        enriched[
            "article_title"
        ] = (
            enriched[
                "article_title"
            ]
            .fillna("")
        )

        missing_title = (
            enriched[
                "article_title"
            ]
            .astype(str)
            .str.strip()
            .eq("")
        )

        enriched.loc[
            missing_title,
            "article_title",
        ] = (
            enriched.loc[
                missing_title,
                "article_title_ai",
            ]
        )

        enriched = (
            enriched.drop(
                columns=[
                    "article_title_ai"
                ]
            )
        )

    enriched[
        "security_relevant_ai"
    ] = (
        enriched[
            "security_relevant"
        ]
        .apply(
            to_bool
        )
    )

    enriched[
        "needs_human_review_ai"
    ] = (
        enriched[
            "needs_human_review"
        ]
        .apply(
            to_bool
        )
    )

    enriched[
        "secondary_domains_ai"
    ] = (
        enriched[
            "secondary_domains"
        ]
        .apply(
            list_value
        )
    )

    enriched[
        "security_domains_ai"
    ] = (
        enriched.apply(
            lambda row: (
                list(
                    dict.fromkeys(
                        (
                            [
                                clean_text(
                                    row[
                                        "primary_domain"
                                    ]
                                )
                            ]
                            if clean_text(
                                row[
                                    "primary_domain"
                                ]
                            )
                            not in {
                                "",
                                "None",
                            }
                            else []
                        )
                        + row[
                            "secondary_domains_ai"
                        ]
                    )
                )
            ),
            axis=1,
        )
    )

    return enriched


def apply_ai_severity_cap(df):

    # Backward-compatible wrapper used by the production pipeline.
    # The v4.2 Attention Score implementation applies both an AI-derived
    # severity floor and cap, plus a final status-specific attention cap.
    return calculate_attention_score(df)


def gdelt_batch_id_from_url(url):

    return (
        url
        .split("/")[-1]
        .split(".")[0]
    )


def find_complete_batches():

    export_files = {
        path.name.replace(
            ".export.CSV.zip",
            "",
        ): path
        for path in RAW_DIR.glob(
            "*.export.CSV.zip"
        )
    }

    gkg_files = {
        path.name.replace(
            ".gkg.csv.zip",
            "",
        ): path
        for path in RAW_DIR.glob(
            "*.gkg.csv.zip"
        )
    }

    batch_ids = sorted(
        set(export_files)
        & set(gkg_files)
    )

    return [
        {
            "batch_id": batch_id,
            "export_path": (
                export_files[
                    batch_id
                ]
            ),
            "gkg_path": (
                gkg_files[
                    batch_id
                ]
            ),
        }
        for batch_id in batch_ids
    ]


def process_batch_files(
    export_path,
    gkg_path,
    batch_id,
):

    events_raw = load_events(
        export_path
    )

    gkg_raw = load_gkg(
        gkg_path
    )

    events = prepare_events(
        events_raw
    )

    geographic = (
        filter_geographic_scope(
            events
        )
    )

    gkg = prepare_gkg(
        gkg_raw
    )

    gkg = attach_page_titles(
        gkg,
        gkg_raw,
    )

    enriched = (
        enrich_events_with_gkg(
            geographic,
            gkg,
        )
    )

    enriched = (
        add_basic_event_labels(
            enriched
        )
    )

    enriched[
        "gdelt_batch"
    ] = batch_id

    stats = {
        "events_processed": len(events),
        "geographic_events": len(geographic),
        "gkg_rows": len(gkg),
        "gkg_matches": int(
            enriched[
                "document_url"
            ]
            .notna()
            .sum()
        ),
    }

    return enriched, stats


# ==================================================
# CAMEO CLASSIFICATION
# ==================================================

CAMEO_ROOT_CODES = {
    1: "Make Public Statement",
    2: "Appeal",
    3: "Express Intent to Cooperate",
    4: "Consult",
    5: "Engage in Diplomatic Cooperation",
    6: "Engage in Material Cooperation",
    7: "Provide Aid",
    8: "Yield",
    9: "Investigate",
    10: "Demand",
    11: "Disapprove",
    12: "Reject",
    13: "Threaten",
    14: "Protest",
    15: "Exhibit Force Posture",
    16: "Reduce Relations",
    17: "Coerce",
    18: "Assault",
    19: "Fight",
    20: "Use Unconventional Mass Violence"
}


QUAD_CLASS_LABELS = {
    1: "Verbal Cooperation",
    2: "Material Cooperation",
    3: "Verbal Conflict",
    4: "Material Conflict"
}


STRATEGIC_ACTOR_TYPES = {
    "GOV",
    "MIL",
    "REB",
    "INS",
    "SEP",
    "SPY",
    "UAF",
    "IGO"
}


SECURITY_CONFLICT_ROOTS = {
    13,
    15,
    16,
    17,
    18,
    19,
    20
}

# ==================================================
# SECURITY THEMES
# ==================================================

DIRECT_DEFENCE_THEMES = {
    "MILITARY_COOPERATION",
    "TAX_WEAPONS_DRONE_STRIKE",
    "TAX_WEAPONS_ARTILLERY"
}


BROAD_DEFENCE_THEMES = {
    "MILITARY",
    "ARMEDCONFLICT",
    "WB_2470_PEACE_OPERATIONS_AND_CONFLICT_MANAGEMENT"
}


CYBER_THEMES = {
    "CYBER_ATTACK",
    "WB_670_ICT_SECURITY",
    "TAX_FNCACT_HACKER",
    "TAX_FNCACT_HACKERS"
}


SANCTIONS_THEMES = {
    "SANCTIONS"
}


DIRECT_CONFLICT_THEMES = {
    "WB_739_POLITICAL_VIOLENCE_AND_CIVIL_WAR",
    "WB_2462_POLITICAL_VIOLENCE_AND_WAR",
    "TERROR"
}


BROAD_CONFLICT_THEMES = {
    "ARMEDCONFLICT",
    "WB_2433_CONFLICT_AND_VIOLENCE",
    "WB_2432_FRAGILITY_CONFLICT_AND_VIOLENCE"
}


ENERGY_THEMES = {
    "WB_507_ENERGY_AND_EXTRACTIVES",
    "ENV_OIL",
    "ENV_NATURALGAS",
    "WB_539_OIL_AND_GAS_POLICY_STRATEGY_AND_INSTITUTIONS",
    "WB_2290_OIL_AND_GAS_EXPORT",
    "WB_544_MID_AND_DOWNSTREAM_OIL_AND_GAS",
    "WB_2273_UPSTREAM_OIL_AND_GAS",
    "WB_525_RENEWABLE_ENERGY",
    "WB_532_BIOFUELS_ENERGY",
    "WB_548_PPP_IN_OIL_AND_GAS"
}

# ==================================================
# GKG / TEXT HELPERS
# ==================================================

def split_gkg_themes(value):

    if pd.isna(value):
        return []

    return [
        theme.strip()
        for theme in str(value).split(";")
        if theme.strip()
    ]


def url_to_text(url):

    if not isinstance(url, str):
        return ""

    path = unquote(
        urlparse(url).path
    )

    text = re.sub(
        r"[-_/]+",
        " ",
        path
    )

    text = re.sub(
        r"\b\d+\b",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip().lower()

URL_SECURITY_KEYWORDS = {

    "Defence & Military": [
        "military",
        "armed forces",
        "air defence",
        "air defense",
        "missile",
        "missile strike",
        "drone strike",
        "drone strikes",
        "artillery",
        "troops",
        "nato",
        "defence ministry",
        "defense ministry"
    ],

    "Cybersecurity": [
        "cyberattack",
        "cyber attack",
        "cybersecurity",
        "ransomware",
        "malware",
        "hacking",
        "data breach"
    ],

    "Energy Security": [
        "energy security",
        "gas pipeline",
        "oil pipeline",
        "natural gas",
        "energy infrastructure",
        "power grid",
        "oil and gas"
    ],

    "Sanctions & Economic Security": [
        "sanctions",
        "economic sanctions",
        "export controls",
        "asset freeze",
        "embargo"
    ],

    "Conflict & Geopolitical Tensions": [
        "armed conflict",
        "military escalation",
        "invasion",
        "ceasefire",
        "hostilities",
        "airstrike",
        "air strike",
        "missile strike",
        "drone strike",
        "drone strikes",
        "shelling",
        "frontline",
        "front line"
    ]
}

def classify_url_security(text):

    if not isinstance(text, str):
        return []

    detected = []

    for domain, keywords in URL_SECURITY_KEYWORDS.items():

        for keyword in keywords:

            if keyword in text:
                detected.append(domain)
                break

    return detected


# ==================================================
# HIGH-PRECISION NON-SECURITY CONTENT FILTER
# ==================================================

NOISE_URL_SEGMENTS = {
    "Entertainment & Fiction": {
        "movie",
        "movies",
        "film",
        "films",
        "tv",
        "television",
        "anime",
        "comic",
        "comics",
        "entertainment",
        "celebrity",
        "celebrities",
        "music"
    },

    "Gaming": {
        "game",
        "games",
        "gaming",
        "video-games",
        "videogames",
        "esports",
        "e-sports"
    },

    "Sports": {
        "sport",
        "sports",
        "football",
        "soccer",
        "basketball",
        "baseball",
        "hockey",
        "tennis",
        "golf",
        "cricket",
        "nfl",
        "nba",
        "mlb",
        "nhl"
    },

    "Lifestyle": {
        "travel",
        "food",
        "recipes",
        "fashion",
        "lifestyle"
    }
}


def classify_noise_content(url):

    if not isinstance(url, str):
        return None

    path = unquote(
        urlparse(url).path
    ).lower()

    segments = {
        segment.strip()
        for segment in path.split("/")
        if segment.strip()
    }

    for category, blocked_segments in (
        NOISE_URL_SEGMENTS.items()
    ):

        if segments.intersection(
            blocked_segments
        ):
            return category

    return None

# ==================================================
# PREPARE SECURITY FEATURES
# ==================================================

def prepare_security_features(df):

    security = df.copy()

    # CAMEO labels
    security["event_root_label"] = (
        security["event_root_code"]
        .map(CAMEO_ROOT_CODES)
    )

    security["quad_class_label"] = (
        security["quad_class"]
        .map(QUAD_CLASS_LABELS)
    )

    # GKG themes
    security["theme_list"] = (
        security["themes"]
        .apply(split_gkg_themes)
    )

    # URL text
    security["url_text"] = (
        security["source_url"]
        .apply(url_to_text)
    )

    security["url_security_domains"] = (
        security["url_text"]
        .apply(classify_url_security)
    )

    # High-precision noise filter
    security["noise_category"] = (
        security["source_url"]
        .apply(classify_noise_content)
    )

    security["noise_excluded"] = (
        security["noise_category"]
        .notna()
    )

    # Strategic actors
    security["strategic_actor"] = (
        security["actor1_type"]
        .isin(STRATEGIC_ACTOR_TYPES)
        |
        security["actor2_type"]
        .isin(STRATEGIC_ACTOR_TYPES)
    )

    return security

# ==================================================
# COUNTRY CODE MAPPING
# ==================================================

COUNTRY_TO_CAMEO = {
    "Albania": "ALB",
    "Andorra": "AND",
    "Austria": "AUT",
    "Belarus": "BLR",
    "Belgium": "BEL",
    "Bosnia and Herzegovina": "BIH",
    "Bulgaria": "BGR",
    "Croatia": "HRV",
    "Cyprus": "CYP",
    "Czechia": "CZE",
    "Denmark": "DNK",
    "Estonia": "EST",
    "Finland": "FIN",
    "France": "FRA",
    "Germany": "DEU",
    "Greece": "GRC",
    "Hungary": "HUN",
    "Iceland": "ISL",
    "Ireland": "IRL",
    "Italy": "ITA",
    "Latvia": "LVA",
    "Liechtenstein": "LIE",
    "Lithuania": "LTU",
    "Luxembourg": "LUX",
    "Malta": "MLT",
    "Moldova": "MDA",
    "Monaco": "MCO",
    "Montenegro": "MNE",
    "Netherlands": "NLD",
    "North Macedonia": "MKD",
    "Norway": "NOR",
    "Poland": "POL",
    "Portugal": "PRT",
    "Romania": "ROU",
    "San Marino": "SMR",
    "Serbia": "SRB",
    "Slovakia": "SVK",
    "Slovenia": "SVN",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Türkiye": "TUR",
    "Ukraine": "UKR",
    "United Kingdom": "GBR",
    "Vatican City": "VAT",
    "Russia": "RUS",
    "Georgia": "GEO",
    "Armenia": "ARM",
    "Azerbaijan": "AZE"
}

# ==================================================
# STRATEGIC CONTEXT
# ==================================================

def get_location_cameo(location_countries):

    if not isinstance(location_countries, list):
        return None

    if len(location_countries) == 0:
        return None

    country = location_countries[0]

    return COUNTRY_TO_CAMEO.get(
        country
    )


def detect_cross_border_context(row):

    location_code = row["location_cameo"]

    if pd.isna(location_code):
        return False

    actor_codes = [
        row["actor1_country"],
        row["actor2_country"]
    ]

    actor_codes = [
        code
        for code in actor_codes
        if pd.notna(code)
    ]

    for actor_code in actor_codes:

        if (
            actor_code in MONITORED_ISO3
            and actor_code != location_code
        ):
            return True

    return False


def detect_black_sea_context(row):

    text = row["url_text"]

    if not isinstance(text, str):
        return False

    if "black sea" not in text:
        return False

    strategic_terms = [
        "ukraine",
        "ukrainian",
        "russia",
        "russian",
        "turkey",
        "turkish",
        "türkiye",
        "ankara"
    ]

    return any(
        term in text
        for term in strategic_terms
    )

def add_strategic_context(df):

    security = df.copy()

    security["location_cameo"] = (
        security["location_countries"]
        .apply(get_location_cameo)
    )

    security["cross_border_context"] = (
        security.apply(
            detect_cross_border_context,
            axis=1
        )
    )

    security["black_sea_context"] = (
        security.apply(
            detect_black_sea_context,
            axis=1
        )
    )

    security["strategic_context"] = (
        security["strategic_actor"]
        |
        security["cross_border_context"]
    )

    return security

# ==================================================
# SECURITY RELEVANCE V4
# ==================================================

def classify_security_domains_v4(row):

    # Exclude high-confidence non-security content
    # before using CAMEO or GKG security signals.
    if bool(row.get("noise_excluded", False)):
        return []

    themes = set(
        row["theme_list"]
        if isinstance(row["theme_list"], list)
        else []
    )

    url_domains = set(
        row["url_security_domains"]
        if isinstance(
            row["url_security_domains"],
            list
        )
        else []
    )

    strategic_actor = bool(
        row["strategic_actor"]
    )

    cross_border_context = bool(
        row["cross_border_context"]
    )

    black_sea_context = bool(
        row["black_sea_context"]
    )

    root_code = (
        row["event_root_code"]
    )

    domains = []


    # ----------------------------------------------
    # CYBERSECURITY
    # ----------------------------------------------

    if (
        themes.intersection(
            CYBER_THEMES
        )
        or "Cybersecurity"
        in url_domains
    ):
        domains.append(
            "Cybersecurity"
        )


    # ----------------------------------------------
    # SANCTIONS
    # ----------------------------------------------

    if (
        themes.intersection(
            SANCTIONS_THEMES
        )
        or
        "Sanctions & Economic Security"
        in url_domains
    ):
        domains.append(
            "Sanctions & Economic Security"
        )


    # ----------------------------------------------
    # DEFENCE & MILITARY
    # ----------------------------------------------

    direct_defence_theme = bool(
        themes.intersection(
            DIRECT_DEFENCE_THEMES
        )
    )

    broad_defence_theme = bool(
        themes.intersection(
            BROAD_DEFENCE_THEMES
        )
    )

    defence_url = (
        "Defence & Military"
        in url_domains
    )

    defence_context = (
        strategic_actor
        or black_sea_context
    )

    direct_defence_context = (
        defence_context
        or cross_border_context
    )

    if (
        defence_url
        or (
            direct_defence_theme
            and direct_defence_context
        )
        or (
            broad_defence_theme
            and defence_context
        )
    ):
        domains.append(
            "Defence & Military"
        )


    # ----------------------------------------------
    # CONFLICT & GEOPOLITICAL TENSIONS
    # ----------------------------------------------

    direct_conflict_theme = bool(
        themes.intersection(
            DIRECT_CONFLICT_THEMES
        )
    )

    broad_conflict_theme = bool(
        themes.intersection(
            BROAD_CONFLICT_THEMES
        )
    )

    conflict_url = (
        "Conflict & Geopolitical Tensions"
        in url_domains
    )

    conflict_event = (
        root_code
        in SECURITY_CONFLICT_ROOTS
    )

    direct_conflict_context = (
        strategic_actor
        or cross_border_context
        or black_sea_context
    )

    broad_conflict_context = (
        strategic_actor
        or black_sea_context
    )

    if (
        conflict_url
        or (
            conflict_event
            and direct_conflict_theme
            and direct_conflict_context
        )
        or (
            conflict_event
            and broad_conflict_theme
            and broad_conflict_context
        )
    ):
        domains.append(
            "Conflict & Geopolitical Tensions"
        )


    # ----------------------------------------------
    # ENERGY SECURITY
    # ----------------------------------------------

    energy_theme = bool(
        themes.intersection(
            ENERGY_THEMES
        )
    )

    energy_url = (
        "Energy Security"
        in url_domains
    )

    wider_security_context = bool(
        themes.intersection(
            SANCTIONS_THEMES
            | DIRECT_CONFLICT_THEMES
        )
    )

    energy_context = (
        strategic_actor
        or black_sea_context
        or wider_security_context
    )

    if (
        energy_url
        or (
            energy_theme
            and energy_context
        )
    ):
        domains.append(
            "Energy Security"
        )


    # Remove duplicates while preserving order
    return list(
        dict.fromkeys(domains)
    )


def apply_security_relevance_v4(df):

    security = df.copy()

    security["security_domains_v4"] = (
        security.apply(
            classify_security_domains_v4,
            axis=1
        )
    )

    security["security_relevant_v4"] = (
        security["security_domains_v4"]
        .apply(
            lambda domains:
            len(domains) > 0
        )
    )

    return security

# ==================================================
# ATTENTION SCORE
# ==================================================

CAMEO_SEVERITY = {
    1: 5,
    2: 5,
    3: 5,
    4: 5,
    5: 5,

    6: 10,
    7: 10,
    8: 15,

    9: 20,

    10: 35,
    11: 35,
    12: 40,
    13: 60,
    14: 45,
    15: 70,
    16: 65,
    17: 75,

    18: 90,
    19: 95,
    20: 100
}


def attention_band(score):

    if score >= 75:
        return "Critical"

    if score >= 55:
        return "High"

    if score >= 35:
        return "Medium"

    return "Low"


# The AI status now constrains CAMEO/Goldstein in BOTH directions.
# This prevents a diplomatic/background article from inheriting an extreme
# CAMEO score, but also prevents confirmed attacks from being artificially
# flattened when GDELT encodes the event weakly.
AI_EVENT_SEVERITY_RANGE = {
    "Actual violence / combat": (70.0, 100.0),
    "Cyber incident": (60.0, 90.0),
    "Military posture / deployment": (40.0, 75.0),
    "Sanctions / economic coercion": (35.0, 70.0),
    "Military cooperation / training": (25.0, 60.0),
    "Threat / warning": (30.0, 55.0),
    "Diplomatic negotiation": (15.0, 40.0),
    "Strategic statement": (10.0, 35.0),
    "Background / analysis": (5.0, 30.0),
    "Unclear": (5.0, 30.0)
}


# A second broad ceiling controls the final Attention Score so that
# non-kinetic developments cannot become Critical from media volume alone.
AI_ATTENTION_CAP = {
    "Actual violence / combat": 100.0,
    "Cyber incident": 94.99,
    "Military posture / deployment": 84.99,
    "Sanctions / economic coercion": 79.99,
    "Military cooperation / training": 69.99,
    "Threat / warning": 74.99,
    "Diplomatic negotiation": 69.99,
    "Strategic statement": 59.99,
    "Background / analysis": 54.99,
    "Unclear": 54.99
}


def calculate_attention_score(df):

    scored = df.copy()

    # ----------------------------------------------
    # CAMEO severity
    # ----------------------------------------------

    scored["cameo_severity"] = (
        scored["event_root_code"]
        .map(CAMEO_SEVERITY)
        .fillna(0)
    )

    # ----------------------------------------------
    # Goldstein conflict intensity
    # -10 -> 100
    #   0 -> 50
    # +10 -> 0
    # ----------------------------------------------

    scored["goldstein_conflict"] = (
        (
            10
            - pd.to_numeric(
                scored["goldstein_scale"],
                errors="coerce"
            )
        )
        / 20
        * 100
    ).clip(
        0,
        100
    ).fillna(50)

    # ----------------------------------------------
    # Raw combined event severity
    # ----------------------------------------------

    scored["event_severity_score_raw"] = (
        0.60
        * scored["cameo_severity"]
        +
        0.40
        * scored["goldstein_conflict"]
    )

    scored["event_severity_score"] = (
        scored["event_severity_score_raw"]
    )

    # Article-level AI determines the semantic severity band.
    # GDELT CAMEO/Goldstein then positions the event WITHIN that band.
    if "event_status" in scored.columns:

        severity_ranges = (
            scored["event_status"]
            .map(
                AI_EVENT_SEVERITY_RANGE
            )
        )

        scored["ai_severity_floor"] = (
            severity_ranges
            .apply(
                lambda value: value[0]
                if isinstance(value, tuple)
                else 5.0
            )
        )

        scored["ai_severity_cap"] = (
            severity_ranges
            .apply(
                lambda value: value[1]
                if isinstance(value, tuple)
                else 30.0
            )
        )

        scored["event_severity_score"] = (
            scored["event_severity_score_raw"]
            .clip(
                lower=scored["ai_severity_floor"],
                upper=scored["ai_severity_cap"]
            )
        )

    else:

        scored["ai_severity_floor"] = np.nan
        scored["ai_severity_cap"] = np.nan

    # ----------------------------------------------
    # Media attention
    # ----------------------------------------------

    mentions = pd.to_numeric(
        scored["num_mentions"],
        errors="coerce"
    ).fillna(0)

    mentions_reference = (
        mentions.quantile(0.95)
    )

    if (
        pd.isna(mentions_reference)
        or mentions_reference <= 0
    ):

        scored["media_attention_score"] = 0.0

    else:

        scored["media_attention_score"] = (
            np.log1p(
                mentions
            )
            /
            np.log1p(
                mentions_reference
            )
            * 100
        ).clip(
            0,
            100
        )

    # ----------------------------------------------
    # Negative tone
    # ----------------------------------------------

    scored["negative_tone_score"] = (
        (
            -pd.to_numeric(
                scored["avg_tone"],
                errors="coerce"
            ).fillna(0)
        )
        .clip(
            0,
            10
        )
        / 10
        * 100
    )

    # ----------------------------------------------
    # Recency
    # ----------------------------------------------

    scored["event_date"] = pd.to_datetime(
        scored["event_date"],
        errors="coerce"
    )

    reference_date = (
        scored["event_date"]
        .max()
    )

    if pd.isna(reference_date):

        scored["days_old"] = 0
        scored["recency_score"] = 0.0

    else:

        scored["days_old"] = (
            reference_date
            - scored["event_date"]
        ).dt.days

        scored["days_old"] = (
            scored["days_old"]
            .fillna(0)
            .clip(lower=0)
        )

        scored["recency_score"] = (
            np.exp(
                -scored["days_old"]
                / 14
            )
            * 100
        )

    # ----------------------------------------------
    # Final score
    # ----------------------------------------------

    scored["attention_score_raw"] = (
        0.55
        * scored["event_severity_score"]
        +
        0.20
        * scored["media_attention_score"]
        +
        0.15
        * scored["negative_tone_score"]
        +
        0.10
        * scored["recency_score"]
    ).round(2)

    scored["attention_score"] = (
        scored["attention_score_raw"]
    )

    if "event_status" in scored.columns:

        scored["ai_attention_cap"] = (
            scored["event_status"]
            .map(
                AI_ATTENTION_CAP
            )
            .fillna(69.99)
        )

        scored["attention_score"] = np.minimum(
            scored["attention_score"],
            scored["ai_attention_cap"]
        ).round(2)

    else:

        scored["ai_attention_cap"] = np.nan

    scored["attention_band"] = (
        scored["attention_score"]
        .apply(attention_band)
    )

    return scored


# ==================================================
# RECALCULATE ATTENTION ACROSS FULL HISTORY
# ==================================================

def recalculate_attention_history(df):

    history = df.copy()

    original_columns = list(
        history.columns
    )

    history[
        "event_date"
    ] = pd.to_datetime(
        history[
            "event_date"
        ],
        errors="coerce"
    )

    if (
        "event_status"
        in history.columns
    ):
        history = (
            apply_ai_severity_cap(
                history
            )
        )
    else:
        history = (
            calculate_attention_score(
                history
            )
        )

    history[
        "event_date"
    ] = (
        history[
            "event_date"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    history = history.reindex(
        columns=original_columns
    )

    return history
# ==================================================
# PREPARE SQLITE DATASET
# ==================================================

def prepare_sql_dataset(df):

    output = df.copy()

    if (
        "security_domains_ai"
        in output.columns
    ):
        output[
            "security_domains"
        ] = (
            output[
                "security_domains_ai"
            ]
            .apply(
                lambda values:
                " | ".join(values)
                if isinstance(
                    values,
                    list,
                )
                else clean_text(values)
            )
        )

    elif (
        "security_domains"
        not in output.columns
    ):
        output[
            "security_domains"
        ] = ""

    if (
        "secondary_domains_ai"
        in output.columns
    ):
        output[
            "secondary_domains"
        ] = (
            output[
                "secondary_domains_ai"
            ]
            .apply(
                lambda values:
                " | ".join(values)
                if isinstance(
                    values,
                    list,
                )
                else clean_text(values)
            )
        )

    elif (
        "secondary_domains"
        not in output.columns
    ):
        output[
            "secondary_domains"
        ] = ""

    if (
        "location_countries"
        in output.columns
    ):
        output[
            "location_countries_text"
        ] = (
            output[
                "location_countries"
            ]
            .apply(
                lambda values:
                " | ".join(values)
                if isinstance(
                    values,
                    list,
                )
                else clean_text(values)
            )
        )

    elif (
        "location_countries_text"
        not in output.columns
    ):
        output[
            "location_countries_text"
        ] = ""

    if (
        "needs_human_review_ai"
        in output.columns
    ):
        output[
            "needs_human_review"
        ] = (
            output[
                "needs_human_review_ai"
            ]
            .astype(bool)
        )

    final_columns = [
        "event_id",
        "event_date",

        "actor1",
        "actor1_country",
        "actor1_type",

        "actor2",
        "actor2_country",
        "actor2_type",

        "event_code",
        "event_root_code",
        "event_root_label",

        "quad_class",
        "quad_class_label",

        "goldstein_scale",
        "avg_tone",

        "num_mentions",
        "num_articles",

        "location",
        "location_countries_text",
        "latitude",
        "longitude",

        "security_domains",
        "primary_domain",
        "secondary_domains",

        "event_status",
        "geographic_centrality",
        "confidence",
        "needs_human_review",
        "exclusion_reason",
        "reason_short",

        "classifier_version",
        "ai_classifier_version",
        "postprocess_version",
        "postprocess_rule",

        "article_title",

        "attention_score",
        "attention_band",
        "event_severity_score",
        "event_severity_score_raw",
        "ai_severity_floor",
        "ai_severity_cap",
        "attention_score_raw",
        "ai_attention_cap",

        "source_name",
        "source_url",
        "gdelt_batch",
    ]

    for column in final_columns:
        if column not in output.columns:
            output[column] = np.nan

    output[
        "event_date"
    ] = (
        pd.to_datetime(
            output[
                "event_date"
            ],
            errors="coerce",
        )
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    return output[
        final_columns
    ].copy()


# ==================================================
# WRITE SQLITE - INCREMENTAL HISTORY
# ==================================================

def write_to_sqlite(df):

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    try:

        new_data = df.copy()

        original_columns = list(
            new_data.columns
        )

        table_check = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'security_events';
            """,
            conn
        )

        if table_check.empty:

            combined_data = (
                recalculate_attention_history(
                    new_data
                )
            )

            combined_data = (
                combined_data.reindex(
                    columns=original_columns
                )
            )

            combined_data.to_sql(
                "security_events",
                conn,
                if_exists="replace",
                index=False
            )

            return {
                "previous_rows": 0,
                "new_events": len(
                    combined_data
                ),
                "updated_events": 0,
                "total_rows": len(
                    combined_data
                )
            }

        existing_data = pd.read_sql_query(
            """
            SELECT *
            FROM security_events;
            """,
            conn
        )

        previous_rows = len(
            existing_data
        )

        existing_data = (
            existing_data
            .reindex(
                columns=original_columns
            )
        )

        existing_ids = set(
            existing_data[
                "event_id"
            ]
            .astype(str)
        )

        incoming_ids = set(
            new_data[
                "event_id"
            ]
            .astype(str)
        )

        new_event_ids = (
            incoming_ids
            - existing_ids
        )

        existing_event_ids = (
            incoming_ids
            & existing_ids
        )

        combined_data = pd.concat(
            [
                existing_data,
                new_data
            ],
            ignore_index=True,
            sort=False,
        )

        combined_data = (
            combined_data
            .drop_duplicates(
                subset="event_id",
                keep="last"
            )
            .reset_index(
                drop=True
            )
        )

        combined_data = (
            recalculate_attention_history(
                combined_data
            )
        )

        combined_data = (
            combined_data.reindex(
                columns=original_columns
            )
        )

        combined_data.to_sql(
            "security_events",
            conn,
            if_exists="replace",
            index=False
        )

        return {
            "previous_rows": previous_rows,
            "new_events": len(
                new_event_ids
            ),
            "updated_events": len(
                existing_event_ids
            ),
            "total_rows": len(
                combined_data
            )
        }

    finally:

        conn.close()


# ==================================================
# ONE-TIME DATABASE ATTENTION RECALCULATION
# ==================================================

def recalculate_existing_database_attention():

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    try:

        table_check = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'security_events';
            """,
            conn
        )

        if table_check.empty:

            print(
                "security_events table does not exist."
            )
            return

        data = pd.read_sql_query(
            """
            SELECT *
            FROM security_events;
            """,
            conn
        )

        if data.empty:

            print(
                "No security events found."
            )
            return

        data = (
            recalculate_attention_history(
                data
            )
        )

        data.to_sql(
            "security_events",
            conn,
            if_exists="replace",
            index=False
        )

        print(
            "Historical Attention Scores recalculated."
        )

        print(
            "Events recalculated:",
            len(data)
        )

        print(
            "Average Attention Score:",
            f"{data['attention_score'].mean():.2f}"
        )

        print(
            "Minimum Attention Score:",
            f"{data['attention_score'].min():.2f}"
        )

        print(
            "Maximum Attention Score:",
            f"{data['attention_score'].max():.2f}"
        )

        distribution = (
            data[
                "attention_band"
            ]
            .value_counts()
            .reindex(
                [
                    "Critical",
                    "High",
                    "Medium",
                    "Low"
                ],
                fill_value=0
            )
        )

        print()
        print(
            "Attention distribution:"
        )

        for level, count in (
            distribution.items()
        ):

            print(
                f"  {level}: {count}"
            )

    finally:

        conn.close()
# ==================================================
# UPDATE METADATA
# ==================================================

def write_update_metadata(
    export_url,
    total_events,
    geographic_events,
    relevant_events,
    database_stats,
    update_mode="incremental",
):

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    try:

        batch_id = (
            gdelt_batch_id_from_url(
                export_url
            )
        )

        update_time = (
            datetime.now(
                timezone.utc
            )
            .strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
        )

        metadata = pd.DataFrame(
            [
                {
                    "update_time": update_time,
                    "gdelt_batch": batch_id,
                    "classifier_version": (
                        FINAL_CLASSIFIER_VERSION
                    ),
                    "model": DEFAULT_MODEL,
                    "update_mode": update_mode,
                    "events_processed": (
                        total_events
                    ),
                    "geographic_events": (
                        geographic_events
                    ),
                    "security_relevant_events": (
                        relevant_events
                    ),
                    "new_events_added": (
                        database_stats[
                            "new_events"
                        ]
                    ),
                    "events_refreshed": (
                        database_stats[
                            "updated_events"
                        ]
                    ),
                    "database_total_events": (
                        database_stats[
                            "total_rows"
                        ]
                    )
                }
            ]
        )

        table_check = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'update_history';
            """,
            conn,
        )

        if table_check.empty:
            existing = pd.DataFrame()
        else:
            existing = pd.read_sql_query(
                """
                SELECT *
                FROM update_history;
                """,
                conn,
            )

        required_columns = set(
            metadata.columns
        )

        if (
            not existing.empty
            and not required_columns.issubset(
                set(
                    existing.columns
                )
            )
        ):
            existing = pd.DataFrame()

        combined = pd.concat(
            [
                existing,
                metadata,
            ],
            ignore_index=True,
            sort=False,
        )

        combined.to_sql(
            "update_history",
            conn,
            if_exists="replace",
            index=False
        )

    finally:

        conn.close()

# ==================================================
# CHECK PROCESSED BATCH
# ==================================================

def batch_already_processed(export_url):

    batch_id = (
        gdelt_batch_id_from_url(
            export_url
        )
    )

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    try:

        table_check = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'processed_batches';
            """,
            conn
        )

        if table_check.empty:
            return False

        result = pd.read_sql_query(
            """
            SELECT COUNT(*) AS matches
            FROM processed_batches
            WHERE gdelt_batch = ?
              AND classifier_version = ?;
            """,
            conn,
            params=(
                batch_id,
                FINAL_CLASSIFIER_VERSION,
            )
        )

        return (
            int(
                result.iloc[0][
                    "matches"
                ]
            )
            > 0
        )

    finally:

        conn.close()

        
# ==================================================
# AI DATABASE / MIGRATION HELPERS
# ==================================================

def is_ai_database_ready():

    if not DATABASE_PATH.exists():
        return False

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    try:

        table_check = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'security_events';
            """,
            conn,
        )

        if table_check.empty:
            return False

        columns = pd.read_sql_query(
            """
            PRAGMA table_info(
                security_events
            );
            """,
            conn,
        )[
            "name"
        ].tolist()

        required = {
            "classifier_version",
            "article_title",
            "primary_domain",
            "event_status",
            "ai_severity_floor",
            "ai_severity_cap",
            "attention_score_raw",
            "ai_attention_cap",
        }

        if not required.issubset(
            set(columns)
        ):
            return False

        counts = pd.read_sql_query(
            """
            SELECT
                COUNT(*) AS total_rows,
                SUM(
                    CASE
                        WHEN classifier_version = ?
                        THEN 1
                        ELSE 0
                    END
                ) AS current_rows
            FROM security_events;
            """,
            conn,
            params=(
                FINAL_CLASSIFIER_VERSION,
            ),
        )

        total_rows = int(
            counts.iloc[0][
                "total_rows"
            ]
            or 0
        )

        current_rows = int(
            counts.iloc[0][
                "current_rows"
            ]
            or 0
        )

        return (
            total_rows == 0
            or total_rows == current_rows
        )

    finally:

        conn.close()


def backup_pre_v42_database():

    if (
        DATABASE_PATH.exists()
        and not PRE_V42_BACKUP_PATH.exists()
    ):
        shutil.copy2(
            DATABASE_PATH,
            PRE_V42_BACKUP_PATH,
        )

        print(
            "Pre-v4.2 production database backup created:"
        )
        print(
            PRE_V42_BACKUP_PATH
        )
        print()


def mark_batches_processed(
    batch_ids,
    update_mode,
):

    if not batch_ids:
        return

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    try:

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_batches (
                gdelt_batch TEXT NOT NULL,
                classifier_version TEXT NOT NULL,
                model TEXT,
                processed_at_utc TEXT,
                update_mode TEXT,
                PRIMARY KEY (
                    gdelt_batch,
                    classifier_version
                )
            );
            """
        )

        processed_at = (
            datetime.now(
                timezone.utc
            )
            .strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
        )

        for batch_id in batch_ids:

            conn.execute(
                """
                INSERT OR REPLACE INTO processed_batches (
                    gdelt_batch,
                    classifier_version,
                    model,
                    processed_at_utc,
                    update_mode
                )
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    batch_id,
                    FINAL_CLASSIFIER_VERSION,
                    DEFAULT_MODEL,
                    processed_at,
                    update_mode,
                ),
            )

        conn.commit()

    finally:

        conn.close()


def serialize_sql_object(value):

    if isinstance(
        value,
        (
            list,
            dict,
            tuple,
            set,
        ),
    ):
        return json.dumps(
            list(value)
            if isinstance(
                value,
                set,
            )
            else value,
            ensure_ascii=False,
        )

    return value


def write_article_classifications(
    classifications,
    replace=False,
):

    if classifications.empty:
        return

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    try:

        output = (
            classifications.copy()
        )

        if not replace:

            table_check = (
                pd.read_sql_query(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                      AND name = 'article_classifications';
                    """,
                    conn,
                )
            )

            if not table_check.empty:

                existing = pd.read_sql_query(
                    """
                    SELECT *
                    FROM article_classifications;
                    """,
                    conn,
                )

                output = pd.concat(
                    [
                        existing,
                        output,
                    ],
                    ignore_index=True,
                    sort=False,
                )

        dedup_column = (
            "cache_key"
            if "cache_key"
            in output.columns
            else "source_url"
        )

        if dedup_column in output.columns:

            output = (
                output
                .drop_duplicates(
                    subset=[
                        dedup_column
                    ],
                    keep="last",
                )
                .reset_index(
                    drop=True
                )
            )

        for column in output.columns:

            if (
                output[
                    column
                ].dtype
                == "object"
            ):
                output[
                    column
                ] = (
                    output[
                        column
                    ]
                    .apply(
                        serialize_sql_object
                    )
                )

        output.to_sql(
            "article_classifications",
            conn,
            if_exists="replace",
            index=False,
        )

    finally:

        conn.close()


def classify_event_frame(events):

    articles = (
        build_article_dataset(
            events
        )
    )

    print(
        "Unique articles:",
        len(articles)
    )

    if articles.empty:

        empty = events.copy()
        empty[
            "security_relevant_ai"
        ] = False

        return (
            empty.iloc[0:0].copy(),
            articles,
            pd.DataFrame(),
        )

    articles_with_title = int(
        articles[
            "article_title"
        ]
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    print(
        "Articles with PAGE_TITLE:",
        articles_with_title
    )

    print()
    print(
        "Applying local article AI classifier..."
    )
    print(
        "Classifier:",
        FINAL_CLASSIFIER_VERSION
    )
    print(
        "Model:",
        DEFAULT_MODEL
    )
    print()

    classifications = (
        classify_dataframe_articles(
            articles,
            model=DEFAULT_MODEL,
            force=False,
        )
    )

    validate_article_classifications(
        articles,
        classifications,
    )

    classified_events = (
        propagate_article_classification(
            events,
            classifications,
        )
    )

    relevant_events = (
        classified_events.loc[
            classified_events[
                "security_relevant_ai"
            ]
        ]
        .copy()
    )

    print()
    print(
        "Relevant articles:",
        int(
            classifications[
                "security_relevant"
            ]
            .apply(
                to_bool
            )
            .sum()
        )
    )

    print(
        "Relevant GDELT event rows:",
        len(relevant_events)
    )

    print(
        "Unique relevant event IDs:",
        relevant_events[
            "event_id"
        ].nunique()
    )

    return (
        relevant_events,
        articles,
        classifications,
    )


def write_full_ai_database(
    sql_events,
    classifications,
):

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    try:

        sql_events.to_sql(
            "security_events",
            conn,
            if_exists="replace",
            index=False,
        )

    finally:

        conn.close()

    write_article_classifications(
        classifications,
        replace=True,
    )


def rebuild_production_history():

    print()
    print("=" * 72)
    print(
        "ONE-TIME V4.2 PRODUCTION DATABASE REBUILD"
    )
    print("=" * 72)
    print()

    batches = (
        find_complete_batches()
    )

    if not batches:
        raise RuntimeError(
            "No complete local GDELT "
            "Events + GKG batches found "
            "in data/raw."
        )

    backup_pre_v42_database()

    event_frames = []
    total_events = 0
    geographic_events = 0

    print(
        "Complete local batches:",
        len(batches)
    )
    print()

    for index, batch in enumerate(
        batches,
        start=1,
    ):

        print(
            f"[{index}/{len(batches)}] "
            f"Preparing batch "
            f"{batch['batch_id']}..."
        )

        batch_events, stats = (
            process_batch_files(
                batch[
                    "export_path"
                ],
                batch[
                    "gkg_path"
                ],
                batch[
                    "batch_id"
                ],
            )
        )

        event_frames.append(
            batch_events
        )

        total_events += (
            stats[
                "events_processed"
            ]
        )

        geographic_events += (
            stats[
                "geographic_events"
            ]
        )

        print(
            "  Geographic event rows:",
            len(batch_events)
        )

    all_events = pd.concat(
        event_frames,
        ignore_index=True,
        sort=False,
    )

    all_events = (
        all_events
        .drop_duplicates(
            subset=[
                "event_id"
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        "Geographic events after "
        "event-id dedup:",
        len(all_events)
    )

    (
        relevant_events,
        articles,
        classifications,
    ) = classify_event_frame(
        all_events
    )

    relevant_events = (
        apply_ai_severity_cap(
            relevant_events
        )
    )

    sql_events = (
        prepare_sql_dataset(
            relevant_events
        )
    )

    write_full_ai_database(
        sql_events,
        classifications,
    )

    mark_batches_processed(
        [
            batch[
                "batch_id"
            ]
            for batch in batches
        ],
        update_mode="full_rebuild",
    )

    latest_batch = (
        batches[-1][
            "batch_id"
        ]
    )

    latest_export_url = (
        "https://data.gdeltproject.org/"
        "gdeltv2/"
        f"{latest_batch}.export.CSV.zip"
    )

    database_stats = {
        "previous_rows": 0,
        "new_events": len(
            sql_events
        ),
        "updated_events": 0,
        "total_rows": len(
            sql_events
        ),
    }

    write_update_metadata(
        export_url=latest_export_url,
        total_events=total_events,
        geographic_events=(
            geographic_events
        ),
        relevant_events=len(
            sql_events
        ),
        database_stats=(
            database_stats
        ),
        update_mode="full_rebuild",
    )

    print()
    print(
        "AI production database rebuilt:"
    )
    print(
        DATABASE_PATH
    )

    print(
        "Security events:",
        len(sql_events)
    )

    print(
        "Article classifications:",
        len(classifications)
    )

    print()

    return database_stats

# ==================================================
# MAIN PIPELINE
# ==================================================

def main():

    print()
    print(
        "EUROPEAN SECURITY MONITOR - PRODUCTION AI UPDATE"
    )

    print("=" * 60)

    print(
        "Project root:",
        PROJECT_ROOT
    )

    print(
        "Raw data:",
        RAW_DIR
    )

    print(
        "Database:",
        DATABASE_PATH
    )

    print(
        "Classifier:",
        FINAL_CLASSIFIER_VERSION
    )

    print(
        "Ollama model:",
        DEFAULT_MODEL
    )

    print()

    # ----------------------------------------------
    # 1. FIND LATEST RELEASE
    # ----------------------------------------------

    print(
        "1. Checking latest GDELT release..."
    )

    export_url, gkg_url = (
        get_latest_gdelt_files()
    )

    export_batch = (
        gdelt_batch_id_from_url(
            export_url
        )
    )

    gkg_batch = (
        gdelt_batch_id_from_url(
            gkg_url
        )
    )

    if export_batch != gkg_batch:

        raise RuntimeError(
            "GDELT Events and GKG latest "
            "releases do not share the "
            "same batch ID."
        )

    print(
        "Latest batch:",
        export_batch
    )

    print(
        "Latest Events:",
        export_url
    )

    print(
        "Latest GKG:",
        gkg_url
    )

    print()

    # ----------------------------------------------
    # 2. DOWNLOAD LATEST FILES
    # ----------------------------------------------

    print(
        "2. Downloading latest GDELT files..."
    )

    export_path = download_file(
        export_url
    )

    gkg_path = download_file(
        gkg_url
    )

    print()

    # ----------------------------------------------
    # 3. ONE-TIME MIGRATION TO ARTICLE AI
    # ----------------------------------------------

    if not is_ai_database_ready():

        print(
            "3. Legacy / missing AI production "
            "database detected."
        )

        print(
            "Rebuilding production history "
            "with the final article-level "
            "AI classifier."
        )

        print()

        rebuild_production_history()

        print("=" * 60)

        print(
            "AI migration and current update "
            "completed successfully."
        )

        print()

        print(
            "Streamlit database ready:",
            DATABASE_PATH
        )

        return

    print(
        "3. AI production database: OK"
    )

    print()

    # ----------------------------------------------
    # 4. SKIP CURRENT CLASSIFIER/BATCH IF DONE
    # ----------------------------------------------

    if batch_already_processed(
        export_url
    ):

        print(
            f"GDELT batch {export_batch} "
            "has already been processed "
            f"with {FINAL_CLASSIFIER_VERSION}."
        )

        print(
            "No database update required."
        )

        return

    # ----------------------------------------------
    # 5. PREPARE LATEST BATCH
    # ----------------------------------------------

    print(
        "4. Preparing latest batch..."
    )

    df_events, batch_stats = (
        process_batch_files(
            export_path,
            gkg_path,
            export_batch,
        )
    )

    print(
        "Raw / prepared Events:",
        batch_stats[
            "events_processed"
        ]
    )

    print(
        "European / strategic events:",
        batch_stats[
            "geographic_events"
        ]
    )

    print(
        "Events with GKG match:",
        batch_stats[
            "gkg_matches"
        ]
    )

    print()

    # ----------------------------------------------
    # 6. ARTICLE-LEVEL AI CLASSIFICATION
    # ----------------------------------------------

    print(
        "5. Classifying unique articles..."
    )

    (
        df_relevant,
        articles,
        classifications,
    ) = classify_event_frame(
        df_events
    )

    # ----------------------------------------------
    # 7. ATTENTION SCORE
    # ----------------------------------------------

    print()
    print(
        "6. Calculating AI-aware Attention Score..."
    )

    df_relevant = (
        apply_ai_severity_cap(
            df_relevant
        )
    )

    if df_relevant.empty:

        print(
            "No security-relevant events "
            "in this batch."
        )

    else:

        print(
            "Average Attention Score:",
            f"{df_relevant['attention_score'].mean():.2f}"
        )

        print(
            "Minimum Attention Score:",
            f"{df_relevant['attention_score'].min():.2f}"
        )

        print(
            "Maximum Attention Score:",
            f"{df_relevant['attention_score'].max():.2f}"
        )

        print()

        attention_distribution = (
            df_relevant[
                "attention_band"
            ]
            .value_counts()
            .reindex(
                [
                    "Critical",
                    "High",
                    "Medium",
                    "Low"
                ],
                fill_value=0
            )
        )

        print(
            "Attention distribution:"
        )

        for level, count in (
            attention_distribution.items()
        ):

            print(
                f"  {level}: {count}"
            )

    print()

    # ----------------------------------------------
    # 8. SQLITE OUTPUT
    # ----------------------------------------------

    print(
        "7. Updating SQLite database..."
    )

    df_sql = prepare_sql_dataset(
        df_relevant
    )

    database_stats = write_to_sqlite(
        df_sql
    )

    write_article_classifications(
        classifications,
        replace=False,
    )

    mark_batches_processed(
        [
            export_batch
        ],
        update_mode="incremental",
    )

    write_update_metadata(
        export_url=export_url,
        total_events=(
            batch_stats[
                "events_processed"
            ]
        ),
        geographic_events=(
            batch_stats[
                "geographic_events"
            ]
        ),
        relevant_events=len(
            df_relevant
        ),
        database_stats=(
            database_stats
        ),
        update_mode="incremental",
    )

    print(
        "Previous database rows:",
        database_stats[
            "previous_rows"
        ]
    )

    print(
        "New event IDs:",
        database_stats[
            "new_events"
        ]
    )

    print(
        "Existing event IDs refreshed:",
        database_stats[
            "updated_events"
        ]
    )

    print(
        "Total database rows:",
        database_stats[
            "total_rows"
        ]
    )

    print()

    # ----------------------------------------------
    # DOMAIN DISTRIBUTION
    # ----------------------------------------------

    if not classifications.empty:

        relevant_articles = (
            classifications.loc[
                classifications[
                    "security_relevant"
                ]
                .apply(
                    to_bool
                )
            ]
        )

        domain_distribution = (
            relevant_articles[
                "primary_domain"
            ]
            .value_counts()
        )

        print(
            "Primary security domains "
            "(article level):"
        )

        if domain_distribution.empty:

            print(
                "  None"
            )

        else:

            for domain, count in (
                domain_distribution.items()
            ):

                print(
                    f"  {domain}: {count}"
                )

        print()

    # ----------------------------------------------
    # FINAL STATUS
    # ----------------------------------------------

    print("=" * 60)

    print(
        "Full AI update pipeline "
        "completed successfully."
    )

    print()

    print(
        "Events file:",
        export_path
    )

    print(
        "GKG file:",
        gkg_path
    )

    print()

    print(
        "Streamlit database ready:",
        DATABASE_PATH
    )

    print()

    print(
        "Current database size:",
        database_stats[
            "total_rows"
        ],
        "security events"
    )


# ==================================================
# RUN SCRIPT
# ==================================================

if __name__ == "__main__":
    main()
