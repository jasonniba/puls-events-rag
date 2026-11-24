
from pathlib import Path

import faiss
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "data" / "processed" / "openagenda_clean.csv"
INDEX_PATH = BASE_DIR / "data" / "faiss_index" / "index.faiss"


def test_index_existe():
    assert INDEX_PATH.exists()


def test_dimensions_index():
    index = faiss.read_index(str(INDEX_PATH))

    assert index.ntotal > 0
    assert index.d == 384


def test_evenements_csv():
    df = pd.read_csv(CSV_PATH)

    assert len(df) > 0
    assert df["Identifiant"].is_unique
    assert df["texte_rag"].notna().all()
