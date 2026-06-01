import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.ensemble import (
    AdaBoostRegressor
)

from sklearn.tree import (
    DecisionTreeRegressor
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="AdaBoost Regression",
    layout="wide"
)

st.title(
    "Housing Price Prediction using AdaBoost Regression"
)

os.makedirs(
    "models",
    exist_ok=True
)

# =====================================================
# Load Dataset
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/housing.csv"
    )

    df.dropna(inplace=True)

    return df


df = load_data()

st.subheader("Dataset Preview")

st.dataframe(df.head())

# =====================================================
# Feature Selection
# =====================================================

target_column = "median_house_value"

X = df.drop(
    target_column,
    axis=1
)

y = df[target_column]

X = pd.get_dummies(X)

X = X.fillna(X.mean())

# =====================================================
# Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# Scaling
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

pickle.dump(
    scaler,
    open(
        "models/scaler.pkl",
        "wb"
    )
)

# =====================================================
# AdaBoost Regressor
# =====================================================

base_estimator = DecisionTreeRegressor(
    max_depth=4
)

ada = AdaBoostRegressor(
    estimator=base_estimator,
    random_state=42
)

param_grid = {
    "n_estimators": [
        50,
        100,
        200
    ],
    "learning_rate": [
        0.01,
        0.1,
        1.0
    ]
}

grid_search = GridSearchCV(
    ada,
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

with st.spinner(
    "Training AdaBoost..."
):

    grid_search.fit(
        X_train_scaled,
        y_train
    )

best_model = grid_search.best_estimator_

pickle.dump(
    best_model,
    open(
        "models/adaboost_regressor.pkl",
        "wb"
    )
)

# =====================================================
# Prediction
# =====================================================

y_pred = best_model.predict(
    X_test_scaled
)

# =====================================================
# Metrics
# =====================================================

mse = mean_squared_error(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

st.subheader(
    "Model Evaluation"
)

st.write(
    "Best Parameters:",
    grid_search.best_params_
)

st.write(
    "MSE:",
    round(mse, 2)
)

st.write(
    "MAE:",
    round(mae, 2)
)

st.write(
    "R2 Score:",
    round(r2, 4)
)

# =====================================================
# Plot
# =====================================================

fig, ax = plt.subplots(
    figsize=(6,5)
)

ax.scatter(
    y_test,
    y_pred
)

ax.set_xlabel(
    "Actual Values"
)

ax.set_ylabel(
    "Predicted Values"
)

ax.set_title(
    "Actual vs Predicted"
)

st.pyplot(fig)

# =====================================================
# Prediction Section
# =====================================================

st.subheader(
    "Predict House Value"
)

input_data = []

for col in X.columns:

    value = st.number_input(
        f"Enter {col}",
        value=float(
            X[col].mean()
        )
    )

    input_data.append(value)

if st.button(
    "Predict"
):

    scaler = pickle.load(
        open(
            "models/scaler.pkl",
            "rb"
        )
    )

    model = pickle.load(
        open(
            "models/adaboost_regressor.pkl",
            "rb"
        )
    )

    input_array = np.array(
        input_data
    ).reshape(1,-1)

    input_scaled = scaler.transform(
        input_array
    )

    prediction = model.predict(
        input_scaled
    )

    st.success(
        f"Predicted House Value: ${round(prediction[0],2)}"
    )