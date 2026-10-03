import sys
from pathlib import Path

import streamlit as st

# src qovlugunu import yoluna elave edirik
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from predict import Predictor  # noqa: E402

MAX_CHARS = 2000


@st.cache_resource
def get_predictor():
    """Model yalniz bir defe yuklenir."""
    return Predictor()


st.set_page_config(page_title="Sentiment Analizi")
st.title("Sentiment Analizi")
st.write(
    "Ingilis dilinde bir film reyi yazin. "
    "Model onun musbet ve ya menfi oldugunu mueyyen edecek."
)

text = st.text_area("Rey", height=150, max_chars=MAX_CHARS)

if st.button("Yoxla"):
    if not text.strip():
        st.warning("Evvelce metn yazin.")
    else:
        data = get_predictor().predict(text)
        total = data["total_words"]
        reliable = total > 0 and data["known_words"] / total >= 0.5

        if data["label"] == "musbet":
            st.success("Netice: MUSBET")
        else:
            st.error("Netice: MENFI")

        st.progress(float(data["confidence"]))
        st.caption(
            f"Modelin daxili qiymeti: {data['confidence']:.0%} "
            "(bu, neticenin dogru oldugunu demek deyil)"
        )
        st.caption(f"Taninan sozler: {data['known_words']}/{data['total_words']}")

        if not reliable:
            st.warning(
                "Diqqet: sozlerin yarisindan coxu modele tanis deyil "
                "(model yalniz ingilis dilinde oyredilib). Netice etibarsizdir."
            )

st.divider()
st.caption(
    "Model PyTorch ile sifirdan yazilib ve IMDB film reyleri ile oyredilib. "
    "Test deqiqliyi: 85.5%. Mehdudiyyetler: yalniz ingilis dili; inkar "
    "(not bad) ve kinayede yanila biler."
)
