
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def compute_rowwise_cosine(existing: pd.Series, updated: pd.Series) -> list[float | None]:
    corpus = pd.concat([existing, updated], ignore_index=True).tolist()
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    vectorizer.fit(corpus)

    existing_matrix = vectorizer.transform(existing.tolist())
    updated_matrix = vectorizer.transform(updated.tolist())

    scores: list[float | None] = []
    for idx in range(len(existing)):
        if not existing.iloc[idx] or not updated.iloc[idx]:
            scores.append(None)
            continue
        score = float(cosine_similarity(existing_matrix[idx], updated_matrix[idx])[0][0])
        scores.append(round(score, 4))
    return scores


@st.cache_resource
def get_hf_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def compute_rowwise_cosine_hf(
    existing: pd.Series,
    updated: pd.Series,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[float | None]:
    scores: list[float | None] = [None] * len(existing)
    valid_rows = [idx for idx in range(len(existing)) if existing.iloc[idx] and updated.iloc[idx]]

    if not valid_rows:
        return scores

    model = get_hf_model(model_name)
    existing_texts = [existing.iloc[idx] for idx in valid_rows]
    updated_texts = [updated.iloc[idx] for idx in valid_rows]

    existing_emb = model.encode(existing_texts, normalize_embeddings=True)
    updated_emb = model.encode(updated_texts, normalize_embeddings=True)
    pair_scores = (existing_emb * updated_emb).sum(axis=1).tolist()

    for idx, score in zip(valid_rows, pair_scores):
        scores[idx] = round(float(score), 4)

    return scores


def write_score_columns(
    df: pd.DataFrame,
    tfidf_scores: list[float | None],
    hf_scores: list[float | None],
) -> pd.DataFrame:
    tfidf_col_name = "Cosine Score (TF-IDF)"
    hf_col_name = "Cosine Score (HF)"
    tfidf_series = pd.Series(tfidf_scores, index=df.index, dtype="Float64")
    hf_series = pd.Series(hf_scores, index=df.index, dtype="Float64")

    if df.shape[1] >= 4:
        original_fourth_col = df.columns[3]
        df = df.drop(columns=[original_fourth_col])
        df.insert(3, tfidf_col_name, tfidf_series)
    else:
        df.insert(3, tfidf_col_name, tfidf_series)

    if df.shape[1] >= 5:
        original_fifth_col = df.columns[4]
        df = df.drop(columns=[original_fifth_col])
        df.insert(4, hf_col_name, hf_series)
    else:
        df.insert(4, hf_col_name, hf_series)

    return df


def load_workbook(uploaded_bytes: bytes) -> dict[str, pd.DataFrame]:
    return pd.read_excel(BytesIO(uploaded_bytes), sheet_name=None)


def find_scoring_sheet(sheets: dict[str, pd.DataFrame]) -> str | None:
    required = {"Existing Category", "Updated"}
    for name, df in sheets.items():
        if required.issubset(set(df.columns)):
            return name
    return None


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, sheet_df in sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output.getvalue()


def run_app() -> None:
    st.set_page_config(page_title="Excel Similarity Tool", layout="wide")
    st.title("Excel Upload + Preview + Download")
    st.caption("Upload an Excel file, view a sheet, compute cosine score, and download the updated workbook.")

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    if uploaded_file is None:
        st.info("Upload an Excel file to begin.")
        return

    uploaded_bytes = uploaded_file.getvalue()
    all_sheets = load_workbook(uploaded_bytes)
    scoring_sheet = find_scoring_sheet(all_sheets)

    sheet_names = list(all_sheets.keys())
    default_index = sheet_names.index(scoring_sheet) if scoring_sheet in sheet_names else 0
    selected_sheet = st.selectbox("Select sheet to display", options=sheet_names, index=default_index)
    preview_df = all_sheets[selected_sheet]

    st.subheader(f"Preview: {selected_sheet}")
    st.dataframe(preview_df, use_container_width=True, height=460)

    if scoring_sheet is None:
        st.error("No sheet found with both 'Existing Category' and 'Updated' columns.")
    else:
        st.info(f"Scoring will use sheet: {scoring_sheet}")

    col1, col2 = st.columns(2)
    with col1:
        original_name = f"{Path(uploaded_file.name).stem}_original.xlsx"
        st.download_button(
            "Download Original Excel",
            data=uploaded_bytes,
            file_name=original_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with col2:
        if scoring_sheet is None:
            return

        if st.button("Compute Cosine Score + Prepare Download", type="primary"):
            work_df = all_sheets[scoring_sheet].copy()
            existing = work_df["Existing Category"].map(normalize_text)
            updated = work_df["Updated"].map(normalize_text)
            tfidf_scores = compute_rowwise_cosine(existing=existing, updated=updated)
            hf_scores = compute_rowwise_cosine_hf(existing=existing, updated=updated)

            all_sheets[scoring_sheet] = write_score_columns(work_df, tfidf_scores=tfidf_scores, hf_scores=hf_scores)
            updated_excel = to_excel_bytes(all_sheets)

            st.success(
                f"Cosine scores added to '{scoring_sheet}': column 4 (TF-IDF) and column 5 (HuggingFace)."
            )
            st.dataframe(all_sheets[scoring_sheet], use_container_width=True, height=460)

            updated_name = f"{Path(uploaded_file.name).stem}_with_scores.xlsx"
            st.download_button(
                "Download Updated Excel",
                data=updated_excel,
                file_name=updated_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


if __name__ == "__main__":
    run_app()
