import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"
MAX_CHARS = 2000

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
        try:
            response = requests.post(
                f"{API_URL}/predict", json={"text": text}, timeout=10
            )
        except requests.exceptions.RequestException:
            st.error("API-ye qosulmaq mumkun olmadi. API serverinin isledyini yoxlayin.")
        else:
            if response.status_code != 200:
                st.error(f"API xeta qaytardi (kod {response.status_code}).")
            else:
                data = response.json()
                if data["label"] == "musbet":
                    st.success("Netice: MUSBET")
                else:
                    st.error("Netice: MENFI")

                st.progress(float(data["confidence"]))
                st.caption(
                    f"Modelin daxili qiymeti: {data['confidence']:.0%} "
                    "(bu, neticenin dogru oldugunu demek deyil)"
                )
                st.caption(
                    f"Taninan sozler: {data['known_words']}/{data['total_words']}"
                )

                if not data["reliable"]:
                    st.warning(
                        "Diqqet: sozlerin yarisindan coxu modele tanis deyil "
                        "(model yalniz ingilis dilinde oyredilib). Netice etibarsizdir."
                    )
