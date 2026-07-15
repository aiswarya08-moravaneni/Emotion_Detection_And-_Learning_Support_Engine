# ============================================================
# AI LEARNING ASSISTANT
# Streamlit Application
# ============================================================

import os

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

from src.emotion_predictor import EmotionPredictor
from src.bert_classifier import BERTEmotionClassifier
from src.mixed_emotion_detector import MixedEmotionDetector
from src.history_manager import add_to_history
from src.csv_manager import CSVManager
from src.ui_helpers import (
    get_template_response,
    render_analytics_dashboard,
    render_model_comparison,
)

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
else:
    gemini_model = None

csv_manager = CSVManager()

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🎓",
    layout="wide",
)

# ============================================================
# LOAD MODELS (CACHE)
# ============================================================


@st.cache_resource
def load_models():
    bilstm = EmotionPredictor()
    bert = BERTEmotionClassifier()
    detector = MixedEmotionDetector()
    return bilstm, bert, detector


# ============================================================
# SESSION STATE
# ============================================================


def initialize_session():
    if "emotion_history" not in st.session_state:
        st.session_state.emotion_history = []


# ============================================================
# GEMINI RESPONSE
# ============================================================


def get_gemini_response(field, problem, emotion, confidence):
    if gemini_model is None:
        return None

    prompt = f"""
You are an AI Learning Assistant.

Student Field:
{field}

Detected Emotion:
{emotion}

Confidence:
{confidence:.2%}

Problem:
{problem}

Please respond with:

1. Acknowledge the student's emotion.

2. Give one field-specific learning tip.

3. Suggest one encouraging next step.

Keep the answer friendly and concise.
"""

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return None


def resolve_ai_response(
    field,
    problem,
    emotion,
    confidence,
    use_ai,
    use_csv_prediction,
):
    if use_csv_prediction:
        csv_response = csv_manager.get_example_response(problem, emotion)
        if csv_response:
            return csv_response, "CSV Example"

    if use_ai:
        ai_response = get_gemini_response(field, problem, emotion, confidence)
        if ai_response:
            return ai_response, "Gemini AI"

    return get_template_response(emotion), "Template"


# ============================================================
# MAIN APPLICATION
# ============================================================


def main():
    initialize_session()

    try:
        bilstm_model, bert_model, mixed_detector = load_models()
        models_ready = True
    except Exception as exc:
        models_ready = False
        st.error(f"Failed to load models: {exc}")

    st.title("🎓 AI Learning Assistant")
    st.caption(
        "Emotion-Aware Personalized Learning Support using BiLSTM, BERT and Gemini AI"
    )
    st.divider()

    # =======================================================
    # SIDEBAR
    # =======================================================

    with st.sidebar:
        st.header("📊 Dashboard")

        if models_ready:
            st.success("Models Loaded")
        else:
            st.error("Models Unavailable")

        st.metric(
            "Total Interactions",
            len(st.session_state.emotion_history),
        )

        csv_count = csv_manager.count_examples()
        st.metric("CSV Examples", csv_count)

        st.divider()
        st.subheader("⚙️ Settings")

        use_ai = st.checkbox("Use AI Response (Gemini)", value=True)
        save_csv = st.checkbox("Save Interaction to CSV", value=True)
        show_scores = st.checkbox("Show Emotion Scores", value=True)
        use_csv_prediction = st.checkbox("Use CSV-Based Prediction", value=False)

        if use_csv_prediction:
            if csv_count > 0:
                st.info(
                    f"CSV prediction enabled. {csv_count} example(s) available "
                    "for emotion-matched responses."
                )
            else:
                st.warning(
                    "CSV prediction enabled, but no examples found yet. "
                    "Responses will fall back to AI or templates."
                )

        st.divider()

        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.emotion_history = []
            st.success("History cleared.")
            st.rerun()

        st.divider()
        st.subheader("🕒 Recent Interactions")

        if not st.session_state.emotion_history:
            st.info("No interactions yet.")
        else:
            for item in reversed(st.session_state.emotion_history[-3:]):
                st.write(f"**{item['field']}**")
                st.write(f"{item['emotion']} ({item['confidence']:.1%})")
                st.caption(item["timestamp"].strftime("%H:%M"))
                st.divider()

    if not models_ready:
        st.stop()

    # =======================================================
    # USER INPUT
    # =======================================================

    field = st.selectbox(
        "📚 Select your field of study",
        [
            "Computer Science",
            "Mathematics",
            "Physics",
            "Chemistry",
            "Biology",
            "Engineering",
            "Business",
            "Literature",
            "History",
            "Psychology",
            "Other",
        ],
    )

    problem = st.text_area(
        f"✍ Describe your {field} problem or challenge",
        placeholder=f"Example: I'm struggling with algorithms in {field}.",
        height=150,
    )

    analyze_button = st.button(
        "🔍 Get AI Learning Help",
        use_container_width=True,
    )

    # =======================================================
    # ANALYSIS
    # =======================================================

    if analyze_button:
        if not problem.strip():
            st.warning("Please enter your learning problem before analyzing.")
        else:
            try:
                with st.spinner("Analyzing your learning state..."):
                    bilstm_result = bilstm_model.predict(problem)
                    emotion = bilstm_result["emotion"]
                    confidence = bilstm_result["confidence"]
                    scores = bilstm_result["scores"]

                    bert_result = bert_model.predict(problem)
                    mixed_emotions = mixed_detector.detect(scores)
                    bert_mixed = mixed_detector.detect(bert_result["scores"])

                    ai_response, response_source = resolve_ai_response(
                        field,
                        problem,
                        emotion,
                        confidence,
                        use_ai,
                        use_csv_prediction,
                    )

                    if save_csv:
                        csv_manager.save_interaction(
                            field,
                            problem,
                            emotion,
                            confidence,
                            ai_response,
                        )

                    add_to_history(
                        field,
                        problem,
                        emotion,
                        confidence,
                        ai_response,
                        scores,
                        "BiLSTM",
                    )
                    add_to_history(
                        field,
                        problem,
                        bert_result["emotion"],
                        bert_result["confidence"],
                        ai_response,
                        bert_result["scores"],
                        "BERT",
                    )

                st.success("Analysis complete.")

                render_model_comparison(
                    emotion,
                    confidence,
                    bilstm_result,
                    bert_result,
                    mixed_emotions,
                    bert_mixed,
                    show_scores,
                )

                st.divider()
                st.subheader("🤖 AI Learning Support")
                st.caption(f"Response source: {response_source}")
                st.write(ai_response)

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    # =======================================================
    # ANALYTICS DASHBOARD
    # =======================================================

    if st.session_state.emotion_history:
        render_analytics_dashboard(st.session_state.emotion_history)

    # =======================================================
    # SESSION SUMMARY
    # =======================================================

    st.divider()
    st.subheader("📜 Current Session History")

    if not st.session_state.emotion_history:
        st.info("No interactions available.")
    else:
        for index, interaction in enumerate(
            reversed(st.session_state.emotion_history[-5:]),
            start=1,
        ):
            with st.expander(
                f"{index}. {interaction['field']} - "
                f"{interaction['emotion']} "
                f"({interaction['confidence']:.2%})"
            ):
                st.write(f"**Problem:** {interaction['problem']}")
                st.write(f"**Model:** {interaction['model']}")
                st.write(f"**Emotion:** {interaction['emotion']}")
                st.write(f"**Confidence:** {interaction['confidence']:.2%}")
                st.write("**AI Response:**")
                st.write(interaction["ai_response"])
                st.caption(f"Timestamp: {interaction['timestamp']}")


if __name__ == "__main__":
    main()
