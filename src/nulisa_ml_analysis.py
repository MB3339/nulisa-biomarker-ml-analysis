"""
NULISA Biomarker Machine Learning Analysis
Author: Meet P Bhatt

Description:
This script demonstrates a machine learning workflow for analyzing
high-dimensional biomarker data generated from the NULISA platform.
The analysis includes data preprocessing, dimensionality reduction,
clustering, and supervised classification.

This repository is intended to showcase a reproducible computational
workflow for biomedical data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data(file_path: str) -> pd.DataFrame:
    """Load biomarker dataset from Excel file."""
    df = pd.read_excel(file_path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset and retain numeric biomarker features."""
    df = df.dropna(axis=1, thresh=len(df) * 0.7)
    df = df.fillna(df.median(numeric_only=True))
    features = df.select_dtypes(include=[np.number])
    print(f"Numeric feature matrix shape: {features.shape}")
    return features


def correlation_plot(features: pd.DataFrame) -> None:
    """Generate correlation heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(features.corr(), cmap="coolwarm")
    plt.title("Biomarker Correlation Matrix")
    plt.tight_layout()
    plt.show()


def run_pca(features: pd.DataFrame) -> np.ndarray:
    """Perform PCA for visualization."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(components[:, 0], components[:, 1], alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection of Biomarker Data")
    plt.tight_layout()
    plt.show()

    return components


def run_clustering(features: pd.DataFrame, n_clusters: int = 3) -> np.ndarray:
    """Apply KMeans clustering."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(scaled)

    print("Cluster distribution:")
    print(pd.Series(clusters).value_counts().sort_index())

    return clusters


def run_supervised_model(features: pd.DataFrame, labels: pd.Series) -> None:
    """Train and evaluate a Random Forest classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Model Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


def main() -> None:
    file_path = "NULISA_dataset.xlsx"  # Replace with your local file path if needed

    df = load_data(file_path)
    features = preprocess_data(df)

    correlation_plot(features)
    run_pca(features)
    run_clustering(features)

    if "label" in df.columns:
        run_supervised_model(features, df["label"])
    else:
        print("No 'label' column found. Supervised step skipped.")


if __name__ == "__main__":
    main()
