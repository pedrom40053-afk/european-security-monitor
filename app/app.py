import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="European Security Monitor",
    layout="wide"
)


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "security_monitor.db"
)


# ==================================================
# DATABASE FUNCTIONS
# ==================================================

def load_data():

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    try:

        df = pd.read_sql_query(
            """
            SELECT *
            FROM security_events;
            """,
            conn
        )

    finally:

        conn.close()

    return df


def load_update_metadata():

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    try:

        metadata = pd.read_sql_query(
            """
            SELECT
                update_time,
                gdelt_batch,
                events_processed,
                geographic_events,
                security_relevant_events,
                new_events_added,
                events_refreshed,
                database_total_events
            FROM update_history
            ORDER BY rowid DESC
            LIMIT 1;
            """,
            conn
        )

    except Exception:

        metadata = pd.DataFrame()

    finally:

        conn.close()

    return metadata


# ==================================================
# HELPERS
# ==================================================

def split_pipe_values(series):

    values = []

    for value in (
        series
        .replace("", pd.NA)
        .dropna()
    ):

        for item in str(value).split("|"):

            item = item.strip()

            if item:
                values.append(item)

    return sorted(
        set(values)
    )


def contains_pipe_value(value, target):

    if pd.isna(value):
        return False

    values = [
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    ]

    return target in values


def apply_filters(
    df,
    date_range,
    selected_country,
    selected_domain,
    selected_attention
):

    filtered_df = df.copy()

    # ----------------------------------------------
    # DATE
    # ----------------------------------------------

    if (
        isinstance(date_range, tuple)
        and len(date_range) == 2
    ):

        start_date = pd.Timestamp(
            date_range[0]
        )

        # Exclusive upper limit makes this robust
        # even if event_date later contains times.
        end_date = (
            pd.Timestamp(date_range[1])
            + pd.Timedelta(days=1)
        )

        filtered_df = filtered_df[
            (
                filtered_df["event_date"]
                >= start_date
            )
            &
            (
                filtered_df["event_date"]
                < end_date
            )
        ]


    # ----------------------------------------------
    # COUNTRY
    # ----------------------------------------------

    if selected_country != "All":

        country_mask = (
            filtered_df[
                "location_countries_text"
            ]
            .apply(
                lambda value:
                contains_pipe_value(
                    value,
                    selected_country
                )
            )
        )

        filtered_df = filtered_df[
            country_mask
        ]


    # ----------------------------------------------
    # SECURITY DOMAIN
    # ----------------------------------------------

    if selected_domain != "All":

        domain_mask = (
            filtered_df[
                "security_domains"
            ]
            .apply(
                lambda value:
                contains_pipe_value(
                    value,
                    selected_domain
                )
            )
        )

        filtered_df = filtered_df[
            domain_mask
        ]


    # ----------------------------------------------
    # ATTENTION LEVEL
    # ----------------------------------------------

    if selected_attention != "All":

        filtered_df = filtered_df[
            filtered_df["attention_band"]
            == selected_attention
        ]


    return filtered_df


# ==================================================
# INITIAL DATA
# Used only to build the sidebar controls.
# The live dashboard reloads SQLite every 60 seconds.
# ==================================================

df_initial = load_data()

if df_initial.empty:

    st.error(
        "The security_events table is empty."
    )

    st.stop()


df_initial["event_date"] = pd.to_datetime(
    df_initial["event_date"],
    errors="coerce"
)


# ==================================================
# HEADER
# ==================================================

st.title(
    "European Security Monitor"
)

st.write(
    "Interactive monitoring of European security "
    "and geopolitical events using GDELT data."
)


# ==================================================
# SIDEBAR FILTERS
# ==================================================

st.sidebar.header(
    "Filters"
)


# --------------------------------------------------
# DATE FILTER
# --------------------------------------------------

valid_dates = (
    df_initial["event_date"]
    .dropna()
)

min_date = (
    valid_dates
    .min()
    .date()
)

max_date = (
    valid_dates
    .max()
    .date()
)

date_range = st.sidebar.date_input(
    "Date range",
    value=(
        min_date,
        max_date
    ),
    min_value=min_date,
    max_value=max_date
)


# --------------------------------------------------
# COUNTRY FILTER
# --------------------------------------------------

countries = split_pipe_values(
    df_initial[
        "location_countries_text"
    ]
)

selected_country = (
    st.sidebar.selectbox(
        "Country",
        ["All"] + countries
    )
)


# --------------------------------------------------
# SECURITY DOMAIN FILTER
# --------------------------------------------------

domains = split_pipe_values(
    df_initial[
        "security_domains"
    ]
)

selected_domain = (
    st.sidebar.selectbox(
        "Security Domain",
        ["All"] + domains
    )
)


# --------------------------------------------------
# ATTENTION LEVEL FILTER
# --------------------------------------------------

attention_levels = [
    "Low",
    "Medium",
    "High",
    "Critical"
]

selected_attention = (
    st.sidebar.selectbox(
        "Attention Level",
        ["All"] + attention_levels
    )
)


# ==================================================
# LIVE DASHBOARD
# SQLite is reloaded every 60 seconds.
# ==================================================

@st.fragment(
    run_every="60s"
)
def live_dashboard(
    date_range,
    selected_country,
    selected_domain,
    selected_attention
):

    # ----------------------------------------------
    # RELOAD CURRENT DATABASE
    # ----------------------------------------------

    df = load_data()

    df["event_date"] = pd.to_datetime(
        df["event_date"],
        errors="coerce"
    )

    latest_update = (
        load_update_metadata()
    )


    # ----------------------------------------------
    # APPLY FILTERS
    # ----------------------------------------------

    filtered_df = apply_filters(
        df=df,
        date_range=date_range,
        selected_country=selected_country,
        selected_domain=selected_domain,
        selected_attention=selected_attention
    )


    # ==================================================
    # UPDATE STATUS
    # ==================================================

    if not latest_update.empty:

        update = latest_update.iloc[0]

        st.caption(
            f"Last updated: "
            f"{update['update_time']}  ·  "
            f"GDELT batch: "
            f"{update['gdelt_batch']}  ·  "
            f"Historical database: "
            f"{int(update['database_total_events'])} events"
        )

        with st.expander(
            "Latest update details"
        ):

            (
                col_a,
                col_b,
                col_c,
                col_d
            ) = st.columns(4)

            col_a.metric(
                "Events Processed",
                int(
                    update[
                        "events_processed"
                    ]
                )
            )

            col_b.metric(
                "Geographic Events",
                int(
                    update[
                        "geographic_events"
                    ]
                )
            )

            col_c.metric(
                "Relevant Events",
                int(
                    update[
                        "security_relevant_events"
                    ]
                )
            )

            col_d.metric(
                "New Events Added",
                int(
                    update[
                        "new_events_added"
                    ]
                )
            )


    # ==================================================
    # KPIs
    # ==================================================

    total_events = len(
        filtered_df
    )

    critical_events = (
        filtered_df[
            "attention_band"
        ]
        .eq("Critical")
        .sum()
    )

    average_attention = (
        filtered_df[
            "attention_score"
        ]
        .mean()
    )

    filtered_countries = (
        split_pipe_values(
            filtered_df[
                "location_countries_text"
            ]
        )
    )

    countries_covered = len(
        filtered_countries
    )


    (
        col1,
        col2,
        col3,
        col4
    ) = st.columns(4)


    col1.metric(
        "Security Events",
        total_events
    )

    col2.metric(
        "Critical Events",
        critical_events
    )

    col3.metric(
        "Avg Attention Score",
        (
            f"{average_attention:.1f}"
            if pd.notna(
                average_attention
            )
            else "—"
        )
    )

    col4.metric(
        "Countries Covered",
        countries_covered
    )


    # ==================================================
    # EMPTY RESULT CHECK
    # ==================================================

    if filtered_df.empty:

        st.warning(
            "No events match the selected filters."
        )

        return


    # ==================================================
    # INTERACTIVE MAP
    # ==================================================

    st.subheader(
        "Security Events Map"
    )

    map_df = filtered_df[
        filtered_df[
            "latitude"
        ].notna()
        &
        filtered_df[
            "longitude"
        ].notna()
        &
        filtered_df[
            "location_countries_text"
        ].notna()
        &
        (
            filtered_df[
                "location_countries_text"
            ] != ""
        )
    ].copy()


    if not map_df.empty:

        fig_map = px.scatter_map(
            map_df,
            lat="latitude",
            lon="longitude",
            size="attention_score",
            hover_name="location",
            hover_data={
                "actor1": True,
                "actor2": True,
                "event_root_label": True,
                "security_domains": True,
                "attention_score": True,
                "attention_band": True,
                "latitude": False,
                "longitude": False
            },
            zoom=3,
            center={
                "lat": 54,
                "lon": 15
            },
            size_max=24
        )

        fig_map.update_layout(
            map_style="open-street-map",
            margin={
                "l": 0,
                "r": 0,
                "t": 0,
                "b": 0
            }
        )

        st.plotly_chart(
            fig_map,
            use_container_width=True
        )

    else:

        st.info(
            "No mapped events match "
            "the selected filters."
        )


    # ==================================================
    # EVENT ANALYSIS
    # ==================================================

    st.subheader(
        "Event Analysis"
    )

    (
        chart_col1,
        chart_col2
    ) = st.columns(2)


    # --------------------------------------------------
    # ATTENTION LEVEL DISTRIBUTION
    # --------------------------------------------------

    attention_order = [
        "Low",
        "Medium",
        "High",
        "Critical"
    ]

    attention_chart = (
        filtered_df[
            "attention_band"
        ]
        .value_counts()
        .reindex(
            attention_order,
            fill_value=0
        )
        .rename_axis(
            "Attention Level"
        )
        .reset_index(
            name="Events"
        )
    )


    fig_attention = px.bar(
        attention_chart,
        x="Attention Level",
        y="Events",
        title=(
            "Events by Attention Level"
        )
    )

    fig_attention.update_layout(
        margin={
            "l": 20,
            "r": 20,
            "t": 50,
            "b": 20
        }
    )

    chart_col1.plotly_chart(
        fig_attention,
        use_container_width=True
    )


    # --------------------------------------------------
    # SECURITY DOMAIN DISTRIBUTION
    # --------------------------------------------------

    domain_rows = []

    for value in (
        filtered_df[
            "security_domains"
        ]
        .replace("", pd.NA)
        .dropna()
    ):

        for domain in (
            str(value).split("|")
        ):

            domain = domain.strip()

            if domain:
                domain_rows.append(
                    domain
                )


    if domain_rows:

        domain_chart = (
            pd.Series(
                domain_rows,
                name="Security Domain"
            )
            .value_counts()
            .rename_axis(
                "Security Domain"
            )
            .reset_index(
                name="Events"
            )
        )

        fig_domains = px.bar(
            domain_chart,
            x="Events",
            y="Security Domain",
            orientation="h",
            title=(
                "Security Domains Associated "
                "with Filtered Events"
            )
        )

        fig_domains.update_layout(
            margin={
                "l": 20,
                "r": 20,
                "t": 50,
                "b": 20
            }
        )

        chart_col2.plotly_chart(
            fig_domains,
            use_container_width=True
        )

    else:

        chart_col2.info(
            "No security-domain data "
            "for the current selection."
        )


    # ==================================================
    # HIGHEST ATTENTION DEVELOPMENTS
    # ==================================================

    st.subheader(
        "Highest Attention Developments"
    )

    developments = (
        filtered_df
        .sort_values(
            "attention_score",
            ascending=False
        )
        .drop_duplicates(
            subset="source_url"
        )
        .head(10)
        .copy()
    )


    for _, row in developments.iterrows():

        with st.container(
            border=True
        ):

            (
                col_info,
                col_score
            ) = st.columns(
                [4, 1]
            )


            # --------------------------------------
            # MAIN INFORMATION
            # --------------------------------------

            with col_info:

                location = (
                    row["location"]
                    if pd.notna(
                        row["location"]
                    )
                    else "Unknown location"
                )

                st.markdown(
                    f"### {location}"
                )

                event_label = (
                    row[
                        "event_root_label"
                    ]
                    if pd.notna(
                        row[
                            "event_root_label"
                        ]
                    )
                    else "Unknown"
                )

                st.write(
                    f"**Event:** "
                    f"{event_label}"
                )

                domains_text = (
                    row[
                        "security_domains"
                    ]
                    if pd.notna(
                        row[
                            "security_domains"
                        ]
                    )
                    else "Unclassified"
                )

                st.write(
                    f"**Domain:** "
                    f"{domains_text}"
                )


                actor1 = (
                    row["actor1"]
                    if pd.notna(
                        row["actor1"]
                    )
                    else "Unknown"
                )

                actor2 = (
                    row["actor2"]
                    if pd.notna(
                        row["actor2"]
                    )
                    else "Unknown"
                )

                st.write(
                    f"**Actors:** "
                    f"{actor1} → {actor2}"
                )


                if pd.notna(
                    row["event_date"]
                ):

                    event_date = (
                        row["event_date"]
                        .date()
                    )

                    st.write(
                        f"**Date:** "
                        f"{event_date}"
                    )


            # --------------------------------------
            # ATTENTION SCORE
            # --------------------------------------

            with col_score:

                score = (
                    row[
                        "attention_score"
                    ]
                )

                st.metric(
                    "Attention Score",
                    (
                        f"{score:.1f}"
                        if pd.notna(score)
                        else "—"
                    )
                )

                if pd.notna(
                    row["attention_band"]
                ):

                    st.write(
                        f"**"
                        f"{row['attention_band']}"
                        f"**"
                    )


            # --------------------------------------
            # SOURCE LINK
            # --------------------------------------

            if (
                pd.notna(
                    row["source_url"]
                )
                and str(
                    row["source_url"]
                ).strip()
            ):

                st.link_button(
                    "Open source article",
                    row["source_url"]
                )


    # ==================================================
    # LIVE STATUS
    # ==================================================

    st.caption(
        "Dashboard data refreshes automatically "
        "every 60 seconds."
    )


# ==================================================
# RUN LIVE DASHBOARD
# ==================================================

live_dashboard(
    date_range,
    selected_country,
    selected_domain,
    selected_attention
)