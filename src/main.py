# main.py
# Car Price Prediction and Feature Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# -----------------------------
# 1. Load Dataset
# -----------------------------
def load_data(path):
    df = pd.read_csv(Car Features and MSRP.csv)
    print("Dataset loaded successfully")
    print(df.head())
    return df


# -----------------------------
# 2. Data Cleaning
# -----------------------------
def clean_data(df):
    # Drop duplicate rows
    df = df.drop_duplicates()

    # Handle missing values
    df = df.dropna()

    # Rename columns (optional but clean)
    df.columns = df.columns.str.replace(" ", "_")

    return df


# -----------------------------
# 3. Exploratory Data Analysis
# -----------------------------
def perform_eda(df):
    print("\n--- Dataset Info ---")
    print(df.info())

    print("\n--- Statistical Summary ---")
    print(df.describe())

    # Price distribution
    plt.figure()
    sns.histplot(df["MSRP"], bins=50, kde=True)
    plt.title("Distribution of Car Prices (MSRP)")
    plt.show()

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=np.number)
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()


# -----------------------------
# 4. Feature Encoding
# -----------------------------
def encode_features(df):
    encoder = LabelEncoder()
    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    return df


# -----------------------------
# 5. Model Building
# -----------------------------
def build_model(df):
    X = df.drop("MSRP", axis=1)
    y = df["MSRP"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)

    return model, X_test, y_test


# -----------------------------
# 6. Model Evaluation
# -----------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\n--- Model Performance ---")
    print("R2 Score:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


# -----------------------------
# 7. Main Execution
# -----------------------------
def main():
    data_path = "data/Car Features and MSRP.csv"  

    df = load_data(data_path)
    df = clean_data(df)
    perform_eda(df)
    df = encode_features(df)

    model, X_test, y_test = build_model(df)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
