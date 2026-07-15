from datetime import datetime
import streamlit as st

def add_to_history(
    field,
    problem,
    emotion,
    confidence,
    ai_response,
    scores,
    model
):

    st.session_state.emotion_history.append({

        "timestamp": datetime.now(),

        "field": field,

        "problem": problem,

        "emotion": emotion,

        "confidence": confidence,

        "ai_response": ai_response,

        "all_scores": scores,

        "model": model

    })