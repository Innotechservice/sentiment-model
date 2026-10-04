import sys
from pathlib import Path

import streamlit as st

# src qovlugunu import yoluna elave edirik
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from predict import Predictor  # noqa: E402

MAX_CHARS = 2000

# Her dil ucun: gosterilen ad, izah metni, deqiqlik
LANG_INFO = {
    "en": {
        "name": "English",
        "intro": "Ingilis dilinde bir film reyi yazin. "
        "Model onun musbet ve ya menfi oldugunu mueyyen edecek.",
        "accuracy": "85.5%",
        "data": "IMDB film reyleri",
        "language": "ingilis",
    },
    "az": {
        "name": "Azərbaycan",
        "intro": "Azerbaycan dilinde bir rey yazin. "
        "Model onun musbet ve ya menfi oldugunu mueyyen edecek.",
        "accuracy": "82.7%",
        "data": "Azerbaycan dilindeki reyler",
        "language": "azerbaycan",
    },
}


@st.cache_resource
def get_predictor(lang):
    """Her dilin modeli yalniz bir defe yuklenir."""
    return Predictor(lang)


st.set_page_config(page_title="Sentiment Analizi")
st.title("Sentiment Analizi")

lang = st.radio(
    "Dil",
    options=list(LANG_INFO),
    format_func=lambda code: LANG_INFO[code]["name"],
    horizontal=True,
)
info = LANG_INFO[lang]

st.write(info["intro"])

text = st.text_area("Rey", height=150, max_chars=MAX_CHARS)

if st.button("Yoxla"):
    if not text.strip():
        st.warning("Evvelce metn yazin.")
    else:
        data = get_predictor(lang).predict(text)
        total = data["total_words"]
        reliable = total > 0 and data["known_words"] / total >= 0.5

        if data["label"] == "musbet":
            st.success("Netice: MUSBET")
        else:
            st.error("Netice: MENFI")

        st.progress(float(data["confidence"]))
        st.caption(
            f"Modelin daxili qiymeti: {data['confidence']:.0%} "
            "(bu, neticenin dogru oldugu demek deyil)"
        )
        st.caption(f"Taninan sozler: {data['known_words']}/{data['total_words']}")

        if not reliable:
            st.warning(
                "Diqqet: sozlerin yarisindan coxu modele tanis deyil "
                f"(secilen model yalniz {info['language']} dilinde oyredilib). "
                "Netice etibarsizdir."
            )

st.divider()
st.caption(
    f"Model PyTorch ile sifirdan yazilib ve {info['data']} ile oyredilib. "
    f"Test deqiqliyi: {info['accuracy']}. Mehdudiyyetler: yalniz secilen dil; "
    "inkar (not bad) ve kinayede yanila biler."
)