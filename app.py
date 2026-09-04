import streamlit as st
import joblib
import pandas as pd


# PAGE CONFIGURATION


st.set_page_config(
    page_title="RAG Hallucination Detector",
    page_icon="🧠",
    layout="wide"
)



# LOAD MODEL


@st.cache_resource
def load_model():
    model = joblib.load("rag_hallucination_detector.pkl")
    features = joblib.load("rag_hallucination_features.pkl")
    return model, features


try:
    model, features = load_model()
except Exception as e:
    st.error("❌ Model files could not be loaded.")
    st.code(str(e))
    st.stop()



# HEADER

st.title("🧠 RAG Hallucination Detector")

st.markdown(
    """
    ### Trained Model-powered hallucination risk detection

    Enter the retrieval and response characteristics below to estimate
    whether an AI-generated response is likely to contain hallucinations.
    """
)

st.divider()


# SIDEBAR


with st.sidebar:

    st.header("⚙️ Model Information")

    st.write("**Algorithm:** Random Forest")
    st.write("**Features:** 6")
    st.write("**Training Samples:** 24,000")
    st.write("**Test Samples:** 6,000")

    st.divider()

    st.write("### 📊 Test Performance")

    st.metric("Accuracy", "96.22%")
    st.metric("Precision", "96.81%")
    st.metric("Recall", "97.24%")
    st.metric("F1 Score", "97.03%")
    st.metric("ROC-AUC", "99.22%")



# INPUT SECTION

st.subheader("🔢 Input Features")

col1, col2, col3 = st.columns(3)

with col1:

    temperature = st.slider(
        "🌡️ Temperature Setting",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        help="Generation temperature used by the language model."
    )

    vector_db_similarity = st.slider(
        "🔎 Vector DB Similarity Score",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.01,
        help="Similarity between the query and retrieved documents."
    )


with col2:

    prompt_tokens = st.number_input(
        "📝 Prompt Tokens",
        min_value=1,
        max_value=5000,
        value=100,
        step=1
    )

    response_tokens = st.number_input(
        "💬 Response Tokens",
        min_value=1,
        max_value=5000,
        value=18,
        step=1
    )


with col3:

    sentence_complexity = st.slider(
        "📚 Sentence Complexity Index",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.01
    )

    subjectivity = st.slider(
        "🎭 Subjectivity Score",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01
    )


st.divider()


# PREDICTION


if st.button(
    "🚀 Detect Hallucination",
    use_container_width=True,
    type="primary"
):

    input_data = {
        "Temperature_Setting": temperature,
        "Vector_DB_Similarity_Score": vector_db_similarity,
        "Prompt_Tokens": prompt_tokens,
        "Response_Tokens": response_tokens,
        "Sentence_Complexity_Index": sentence_complexity,
        "Subjectivity_Score": subjectivity
    }

    input_df = pd.DataFrame([input_data])

    # Ensure exact training feature order
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    hallucination_probability = float(probabilities[1])
    no_hallucination_probability = float(probabilities[0])

    confidence = max(
        hallucination_probability,
        no_hallucination_probability
    )



    st.subheader("📊 Detection Result")

    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        if prediction == 1:
            st.error("🔴 HALLUCINATION")
        else:
            st.success("🟢 NO HALLUCINATION")


    with result_col2:

        st.metric(
            "Hallucination Probability",
            f"{hallucination_probability * 100:.2f}%"
        )


    with result_col3:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


    st.divider()


   

    st.subheader("📈 Prediction Probabilities")

    probability_df = pd.DataFrame(
        {
            "Probability": [
                no_hallucination_probability,
                hallucination_probability
            ]
        },
        index=[
            "No Hallucination",
            "Hallucination"
        ]
    )

    st.bar_chart(probability_df)




    st.subheader("🔍 Detailed Analysis")

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:

        st.write("### Input Values")

        st.dataframe(
            input_df.T.rename(columns={0: "Value"}),
            use_container_width=True
        )


    with detail_col2:

        st.write("### Model Output")

        result_data = {
            "Prediction": [
                "Hallucination" if prediction == 1
                else "No Hallucination"
            ],
            "Hallucination Probability": [
                f"{hallucination_probability * 100:.2f}%"
            ],
            "No Hallucination Probability": [
                f"{no_hallucination_probability * 100:.2f}%"
            ],
            "Confidence": [
                f"{confidence * 100:.2f}%"
            ]
        }

        st.dataframe(
            pd.DataFrame(result_data),
            use_container_width=True
        )



st.divider()

st.caption(
    "RAG Hallucination Detector made by Devendra. "
    
)