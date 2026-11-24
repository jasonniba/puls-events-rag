
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

API_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8001"
).rstrip("/")

API_KEY = os.getenv("RAG_API_KEY", "")
DEMO_MODE = os.getenv(
    "STREAMLIT_DEMO_MODE", "false"
).lower() == "true"


st.set_page_config(
    page_title="Puls-Events",
    page_icon="🎭",
    layout="centered"
)

st.title("🎭 Puls-Events")
st.subheader("Assistant culturel intelligent")

st.write(
    "Découvrez des événements culturels à Paris "
    "en posant votre question."
)

st.caption("FAISS • LangChain • Mistral API")

if DEMO_MODE:
    st.warning(
        "Mode démonstration : réponses simulées. "
        "Aucun appel à Mistral ni à FastAPI."
    )

st.divider()

st.write("**Exemples de questions**")
st.write("🎵 Quels concerts sont prévus à Paris ?")
st.write("🎨 Je cherche une exposition artistique.")
st.write("👨‍👩‍👧 Quelles activités faire avec des enfants ?")

question = st.chat_input(
    "Quel événement recherchez-vous ?"
)

if question:

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        if DEMO_MODE:

            st.info(
                "Ceci est une réponse simulée "
                "pour vérifier l'interface."
            )

            st.write(
                "Voici comment les recommandations "
                "apparaîtront lorsque Mistral API "
                "sera disponible."
            )

            sources = [
                {
                    "titre": "Événement de démonstration A",
                    "lieu": "Lieu fictif – Paris",
                    "date_fin": "Date de démonstration",
                    "url": ""
                },
                {
                    "titre": "Événement de démonstration B",
                    "lieu": "Lieu fictif – Paris",
                    "date_fin": "Date de démonstration",
                    "url": ""
                }
            ]

        else:

            if not API_KEY:
                st.error(
                    "RAG_API_KEY manquante dans .env"
                )
                st.stop()

            with st.spinner(
                "Recherche des événements..."
            ):

                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"question": question},
                        headers={"X-API-Key": API_KEY},
                        timeout=120
                    )

                    response.raise_for_status()
                    data = response.json()

                except requests.exceptions.RequestException as exc:
                    st.error(
                        "Impossible d'obtenir la réponse "
                        "de FastAPI."
                    )
                    st.caption(str(exc))
                    st.stop()

            st.markdown(data.get("reponse", ""))
            sources = data.get("sources", [])

        if sources:

            with st.expander(
                "📚 Consulter les événements sources"
            ):

                for source in sources:

                    st.markdown(
                        f"**{source.get('titre', '')}**"
                    )

                    st.write(
                        "📍 Lieu :",
                        source.get("lieu", "")
                    )

                    st.write(
                        "📅 Fin :",
                        source.get("date_fin", "")
                    )

                    url = source.get("url", "")

                    if url and url.startswith(
                        ("https://", "http://")
                    ):
                        st.link_button(
                            "Voir l'événement",
                            url
                        )

                    st.divider()
