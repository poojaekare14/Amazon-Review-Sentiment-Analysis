import streamlit as st
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    page_icon="🛒",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    tfidf = joblib.load("vectorizer (1).joblib")
    scaler = joblib.load("scaler (1).joblib")
    pca = joblib.load("pca (1).joblib")
    model = joblib.load("random_forest_model.pkl.gz")

    return tfidf, scaler, pca, model


tfidf, scaler, pca, model = load_models()

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🛒 Amazon Review Sentiment Analysis")

st.write(
    "Enter an Amazon product review below to predict its sentiment."
)

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

review = st.text_area(
    "Enter your review:",
    placeholder="Example: The product is excellent and works perfectly!",
    height=150
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔍 Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        try:
            # TF-IDF
            X_tfidf = tfidf.transform([review])

            # Convert sparse matrix to dense
            X_dense = X_tfidf.toarray()

            # Scaling
            X_scaled = scaler.transform(X_dense)

            # PCA
            X_pca = pca.transform(X_scaled)

            # Prediction
            prediction = model.predict(X_pca)[0]

            # Display prediction only
            st.success(f"Predicted Sentiment: {prediction}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")


