from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import uuid4

import chromadb
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


REQUIRED_COLUMNS = {"Existing Category", "Updated"}


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_workbook(uploaded_bytes: bytes) -> dict[str, pd.DataFrame]:
    return pd.read_excel(BytesIO(uploaded_bytes), sheet_name=None)


def normalize_header(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def resolve_required_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    normalized_to_original = {normalize_header(col): str(col) for col in df.columns}
    existing_col = normalized_to_original.get(normalize_header("Existing Category"))
    updated_col = normalized_to_original.get(normalize_header("Updated"))
    if existing_col and updated_col:
        return existing_col, updated_col
    return None


@st.cache_resource
def get_hf_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def build_collection_with_existing_category(
    existing_texts: list[str],
    model: SentenceTransformer,
) -> chromadb.api.models.Collection.Collection:
    client = chromadb.Client()
    collection_name = f"existing_category_{uuid4().hex}"
    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    embeddings = model.encode(existing_texts, normalize_embeddings=True).tolist()
    ids = [f"existing_{idx}" for idx in range(len(existing_texts))]
    metadatas = [{"row_index": idx} for idx in range(len(existing_texts))]

    collection.add(ids=ids, documents=existing_texts, embeddings=embeddings, metadatas=metadatas)
    return collection


def find_best_match_for_updated(
    updated_texts: list[str],
    collection: chromadb.api.models.Collection.Collection,
    model: SentenceTransformer,
) -> tuple[list[str | None], list[float | None]]:
    best_matches: list[str | None] = []
    best_scores: list[float | None] = []

    progress = st.progress(0, text="Matching Updated rows against Existing Category...")

    for idx, text in enumerate(updated_texts):
        if not text:
            best_matches.append(None)
            best_scores.append(None)
        else:
            query_embedding = model.encode(text, normalize_embeddings=True).tolist()
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                include=["documents", "distances"],
            )

            has_match = bool(result.get("documents") and result["documents"][0])
            if not has_match:
                best_matches.append(None)
                best_scores.append(None)
            else:
                matched_text = result["documents"][0][0]
                distance = float(result["distances"][0][0])
                similarity = max(0.0, 1.0 - distance)

                best_matches.append(str(matched_text))
                best_scores.append(round(similarity, 4))

        progress.progress((idx + 1) / max(1, len(updated_texts)))

    progress.empty()
    return best_matches, best_scores


def write_result_columns(
    df: pd.DataFrame,
    best_matches: list[str | None],
    best_scores: list[float | None],
) -> pd.DataFrame:
    output = df.copy()
    output["Best Match (Existing Category)"] = pd.Series(best_matches, index=output.index, dtype="string")
    output["Best Match Score (HF+Chroma)"] = pd.Series(best_scores, index=output.index, dtype="Float64")
    return output


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, sheet_df in sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output.getvalue()


def run_page() -> None:
    st.set_page_config(page_title="Chroma Best Match", layout="wide")
    st.title("Chroma Best-Match Finder")
    st.caption(
        "Upload an Excel file. This page embeds 'Existing Category' and finds the best matching entry for each 'Updated' row."
    )

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], key="chroma_match_uploader")
    if uploaded_file is None:
        st.info("Upload an Excel file to begin.")
        return

    uploaded_bytes = uploaded_file.getvalue()
    all_sheets = load_workbook(uploaded_bytes)

    sheet_names = list(all_sheets.keys())
    selected_sheet = st.selectbox(
        "Select sheet to preview",
        options=sheet_names,
        index=0,
        key="chroma_match_preview_sheet",
    )
    st.subheader(f"Preview: {selected_sheet}")
    st.dataframe(all_sheets[selected_sheet], use_container_width=True, height=440)

    resolved_columns = resolve_required_columns(all_sheets[selected_sheet])
    if resolved_columns is None:
        st.error(
            "Selected sheet must contain 'Existing Category' and 'Updated' columns "
            "(header matching ignores case and extra spaces)."
        )
        return

    existing_col, updated_col = resolved_columns
    st.info(f"Using columns -> Existing: '{existing_col}' | Updated: '{updated_col}'")

    if st.button("Run Best-Match Search + Prepare Download", type="primary"):
        with st.spinner("Embedding Existing Category and querying best matches..."):
            work_df = all_sheets[selected_sheet].copy()
            existing = work_df[existing_col].map(normalize_text)
            updated = work_df[updated_col].map(normalize_text)

            model = get_hf_model()
            collection = build_collection_with_existing_category(existing.tolist(), model)
            best_matches, best_scores = find_best_match_for_updated(updated.tolist(), collection, model)

            all_sheets[selected_sheet] = write_result_columns(work_df, best_matches, best_scores)
            updated_excel = to_excel_bytes(all_sheets)

        st.success(
            "Done. Added 'Best Match (Existing Category)' and 'Best Match Score (HF+Chroma)' columns "
            f"to sheet '{selected_sheet}'."
        )
        st.dataframe(all_sheets[selected_sheet], use_container_width=True, height=440)

        updated_name = f"{Path(uploaded_file.name).stem}_best_match.xlsx"
        st.download_button(
            "Download Updated Excel",
            data=updated_excel,
            file_name=updated_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    run_page()
