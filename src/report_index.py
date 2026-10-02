"""
Builds a retrieval index over real IU/OpenI radiology reports.
"""

import argparse
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from paths import REPORT_INDEX_PATH


def load_openi_reports(reports_csv, projections_csv):
    reports = pd.read_csv(reports_csv)
    projections = pd.read_csv(projections_csv)

    merged = projections.merge(reports, on="uid", how="left")
    merged["text"] = (
        merged["findings"].fillna("") + " " + merged["impression"].fillna("")
    ).str.strip()
    merged["labels"] = merged["Problems"].fillna("normal")

    merged = merged[merged["text"].str.len() > 0].reset_index(drop=True)
    return merged


def build_index(reports_csv, projections_csv, out_path=REPORT_INDEX_PATH):
    df = load_openi_reports(reports_csv, projections_csv)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["text"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {"vectorizer": vectorizer, "tfidf_matrix": tfidf_matrix, "reports": df},
            f,
        )
    print(f"Built index over {len(df)} reports -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="data/openi/indiana_reports.csv")
    parser.add_argument("--projections", default="data/openi/indiana_projections.csv")
    parser.add_argument("--out", default=REPORT_INDEX_PATH)
    args = parser.parse_args()
    build_index(args.reports, args.projections, args.out)