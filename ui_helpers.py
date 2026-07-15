import hashlib
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.emotion_templates import EMOTION_RESPONSES

EMOTION_COLORS = {
    "Bored": "#95a5a6",
    "Confident": "#2ecc71",
    "Confused": "#3498db",
    "Curious": "#9b59b6",
    "Frustrated": "#e74c3c",
}


def get_mixed_emotions(scores, threshold=0.15):
    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    primary = sorted_scores[0]
    mixed = [primary]

    for emotion, score in sorted_scores[1:]:
        if score >= threshold:
            mixed.append((emotion, score))

    return mixed


def get_template_response(emotion):
    template = EMOTION_RESPONSES[emotion]
    return (
        f"{template['emoji']} {template['response']}\n\n"
        f"**Suggested Action:** {template['action']}"
    )


def history_to_dataframe(history):
    if not history:
        return pd.DataFrame()

    records = []
    for item in history:
        records.append(
            {
                "timestamp": item["timestamp"],
                "field": item["field"],
                "problem": item["problem"],
                "emotion": item["emotion"],
                "confidence": item["confidence"],
                "model": item["model"],
            }
        )

    return pd.DataFrame(records)


def _history_cache_key(history):
    payload = [
        {
            "timestamp": item["timestamp"].isoformat(),
            "field": item["field"],
            "emotion": item["emotion"],
            "confidence": item["confidence"],
            "model": item["model"],
        }
        for item in history
    ]
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@st.cache_data(show_spinner=False)
def build_emotion_distribution_chart(history_key, history_json):
    del history_key
    history = json.loads(history_json)
    if not history:
        return None

    df = pd.DataFrame(history)
    counts = df["emotion"].value_counts().reset_index()
    counts.columns = ["emotion", "count"]

    fig = px.pie(
        counts,
        names="emotion",
        values="count",
        title="Emotion Distribution Across Sessions",
        color="emotion",
        color_discrete_map=EMOTION_COLORS,
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        showlegend=True,
        margin=dict(l=20, r=20, t=50, b=20),
        legend_title_text="Emotion",
    )
    return fig


@st.cache_data(show_spinner=False)
def build_confidence_timeline_chart(history_key, history_json):
    del history_key
    history = json.loads(history_json)
    if not history:
        return None

    df = pd.DataFrame(history)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["session"] = range(1, len(df) + 1)

    fig = px.line(
        df,
        x="session",
        y="confidence",
        color="emotion",
        markers=True,
        title="Emotional Journey — Confidence Over Time",
        labels={
            "session": "Session",
            "confidence": "Confidence",
            "emotion": "Emotion",
        },
        color_discrete_map=EMOTION_COLORS,
    )
    fig.update_layout(
        yaxis_tickformat=".0%",
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
    )
    return fig


@st.cache_data(show_spinner=False)
def build_field_emotion_chart(history_key, history_json):
    del history_key
    history = json.loads(history_json)
    if not history:
        return None

    df = pd.DataFrame(history)
    grouped = (
        df.groupby(["field", "emotion", "model"])
        .size()
        .reset_index(name="count")
    )

    models = sorted(df["model"].unique())
    if len(models) > 1:
        fig = px.bar(
            grouped,
            x="field",
            y="count",
            color="emotion",
            facet_col="model",
            barmode="group",
            title="Emotion Distribution by Study Field",
            labels={"count": "Interactions", "field": "Field"},
            color_discrete_map=EMOTION_COLORS,
        )
    else:
        field_counts = (
            df.groupby(["field", "emotion"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            field_counts,
            x="field",
            y="count",
            color="emotion",
            barmode="group",
            title="Emotion Distribution by Study Field",
            labels={"count": "Interactions", "field": "Field"},
            color_discrete_map=EMOTION_COLORS,
        )

    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_tickangle=-35,
    )
    return fig


@st.cache_data(show_spinner=False)
def build_summary_metrics(history_key, history_json):
    del history_key
    history = json.loads(history_json)
    if not history:
        return {}

    df = pd.DataFrame(history)
    top_emotion = (
        df["emotion"].value_counts().idxmax()
        if not df.empty
        else "N/A"
    )
    top_field = (
        df["field"].value_counts().idxmax()
        if not df.empty
        else "N/A"
    )

    models = df["model"].unique()
    agreement_rate = None
    if len(models) > 1:
        pivot = df.pivot_table(
            index=["timestamp", "field", "problem"],
            columns="model",
            values="emotion",
            aggfunc="first",
        )
        if pivot.shape[1] >= 2:
            model_cols = list(pivot.columns[:2])
            matches = pivot[model_cols[0]] == pivot[model_cols[1]]
            agreement_rate = matches.mean()

    return {
        "total_predictions": len(df),
        "unique_sessions": df[["timestamp", "field", "problem"]].drop_duplicates().shape[0],
        "avg_confidence": df["confidence"].mean(),
        "top_emotion": top_emotion,
        "top_field": top_field,
        "fields_explored": df["field"].nunique(),
        "agreement_rate": agreement_rate,
    }


def serialize_history_for_cache(history):
    return json.dumps(
        [
            {
                "timestamp": item["timestamp"].isoformat(),
                "field": item["field"],
                "problem": item["problem"],
                "emotion": item["emotion"],
                "confidence": float(item["confidence"]),
                "model": item["model"],
            }
            for item in history
        ]
    )


def render_model_comparison(
    emotion,
    confidence,
    bilstm_result,
    bert_result,
    mixed_emotions,
    bert_mixed,
    show_scores,
):
    st.subheader("📊 Model Predictions Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧠 BiLSTM Student Adaptive")

        if len(mixed_emotions) > 1:
            mixed_text = " + ".join(
                [
                    f"{EMOTION_RESPONSES[e[0]]['emoji']} {e[0]}"
                    for e in mixed_emotions
                ]
            )
            st.metric(
                "Mixed Emotions",
                mixed_text,
                f"Primary: {mixed_emotions[0][1]:.1%}",
            )
        else:
            emoji = EMOTION_RESPONSES[emotion]["emoji"]
            st.metric(
                "Emotion",
                f"{emoji} {emotion}",
                f"{confidence:.1%}",
            )

        if show_scores:
            for emotion_name, score in sorted(
                bilstm_result["scores"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                st.progress(float(score), text=f"{emotion_name}: {score:.1%}")

    with col2:
        st.markdown("### 🤖 BERT Transformer")

        if len(bert_mixed) > 1:
            mixed_text = " + ".join(
                [
                    f"{EMOTION_RESPONSES[e[0]]['emoji']} {e[0]}"
                    for e in bert_mixed
                ]
            )
            st.metric(
                "Mixed Emotions",
                mixed_text,
                f"Primary: {bert_mixed[0][1]:.1%}",
            )
        else:
            emoji = EMOTION_RESPONSES[bert_result["emotion"]]["emoji"]
            st.metric(
                "Emotion",
                f"{emoji} {bert_result['emotion']}",
                f"{bert_result['confidence']:.1%}",
            )

        if show_scores:
            for emotion_name, score in sorted(
                bert_result["scores"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                st.progress(float(score), text=f"{emotion_name}: {score:.1%}")


def render_analytics_dashboard(history):
    st.divider()
    st.subheader("📈 Analytics Dashboard")

    history_json = serialize_history_for_cache(history)
    cache_key = _history_cache_key(history)

    tab_emotions, tab_fields, tab_summary = st.tabs(
        ["Emotions", "Fields", "Summary"]
    )

    with tab_emotions:
        col_pie, col_line = st.columns(2)

        with col_pie:
            pie_fig = build_emotion_distribution_chart(cache_key, history_json)
            if pie_fig:
                st.plotly_chart(pie_fig, use_container_width=True)

        with col_line:
            line_fig = build_confidence_timeline_chart(cache_key, history_json)
            if line_fig:
                st.plotly_chart(line_fig, use_container_width=True)

    with tab_fields:
        bar_fig = build_field_emotion_chart(cache_key, history_json)
        if bar_fig:
            st.plotly_chart(bar_fig, use_container_width=True)

    with tab_summary:
        metrics = build_summary_metrics(cache_key, history_json)
        if metrics:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Predictions", metrics["total_predictions"])
            c2.metric("Unique Sessions", metrics["unique_sessions"])
            c3.metric("Avg Confidence", f"{metrics['avg_confidence']:.1%}")
            c4.metric("Fields Explored", metrics["fields_explored"])

            c5, c6, c7 = st.columns(3)
            c5.metric("Most Common Emotion", metrics["top_emotion"])
            c6.metric("Top Field", metrics["top_field"])
            if metrics["agreement_rate"] is not None:
                c7.metric(
                    "Model Agreement",
                    f"{metrics['agreement_rate']:.1%}",
                )
            else:
                c7.metric("Model Agreement", "N/A")
