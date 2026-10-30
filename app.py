
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

MODEL_ID = "SHAILESHGOKU/imdb-distilbert-sentiment"
MAX_LENGTH = 256

# --------------------------------------------------
# LOAD MODEL AND TOKENIZER
# --------------------------------------------------

@st.cache_resource
def load_model():

    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


# --------------------------------------------------
# MODEL INITIALIZATION
# --------------------------------------------------

try:
    tokenizer, model, device = load_model()

except Exception as e:
    st.error("Unable to load the sentiment analysis model.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# SENTIMENT PREDICTION
# --------------------------------------------------

def predict_sentiment(review):

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model(**inputs)

        probabilities = F.softmax(
            outputs.logits,
            dim=1
        )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    negative_probability = probabilities[0][0].item() * 100
    positive_probability = probabilities[0][1].item() * 100

    sentiment = (
        "Positive" if predicted_class == 1 else "Negative"
    )

    confidence = probabilities[0][predicted_class].item() * 100

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "positive_probability": positive_probability,
        "negative_probability": negative_probability
    }


# --------------------------------------------------
# USER INTERFACE
# --------------------------------------------------

st.title("🎬 IMDb Sentiment Analyzer")

st.markdown(
    """
    ### Understand the emotion behind movie reviews

    Powered by a fine-tuned **DistilBERT** deep learning model.

    Enter an IMDb-style movie review to predict whether
    the sentiment is positive or negative.
    """
)

st.divider()

review = st.text_area(
    "Write your movie review",
    placeholder=(
        "Example: This movie was fantastic! "
        "The storyline, acting and direction were brilliant..."
    ),
    height=200
)

analyze_button = st.button(
    "Analyze Sentiment",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# DISPLAY PREDICTION
# --------------------------------------------------

if analyze_button:

    if not review.strip():

        st.warning("Please enter a movie review.")

    else:

        with st.spinner("Analyzing your review..."):

            result = predict_sentiment(review)

        st.divider()

        st.subheader("Sentiment Analysis Result")

        if result["sentiment"] == "Positive":

            st.success("😊 Positive Sentiment")

        else:

            st.error("😞 Negative Sentiment")

        st.metric(
            label="Prediction Confidence",
            value=f'{result["confidence"]:.2f}%'
        )

        st.subheader("Class Probabilities")

        st.write(
            f'Positive: {result["positive_probability"]:.2f}%'
        )

        st.progress(
            result["positive_probability"] / 100
        )

        st.write(
            f'Negative: {result["negative_probability"]:.2f}%'
        )

        st.progress(
            result["negative_probability"] / 100
        )

        st.caption(
            "Confidence represents the model's softmax probability. "
            "It is not a guarantee that the prediction is correct."
        )


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

with st.expander("About this project"):

    st.markdown(
        """
        **Model:** DistilBERT Base Uncased

        **Task:** Binary Sentiment Classification

        **Dataset:** IMDb 50K Movie Reviews

        **Training Framework:** PyTorch and Hugging Face Transformers

        **Data Engineering:** Databricks Medallion Architecture

        **Test Accuracy:** 91.69%

        **Test F1-score:** 91.79%

        **Inference:** CPU
        """
    )

st.divider()

st.caption(
    "Built with Python | PyTorch | Hugging Face | DistilBERT | Streamlit"
)
