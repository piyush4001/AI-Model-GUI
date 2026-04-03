import os
import warnings

import streamlit as st

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def load_legacy_joblib(path):
    import joblib

    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except Exception:
        InconsistentVersionWarning = None

    with warnings.catch_warnings():
        if InconsistentVersionWarning is not None:
            warnings.simplefilter("ignore", InconsistentVersionWarning)
        return joblib.load(path)


def get_tensorflow():
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
        absl.logging.set_stderrthreshold(absl.logging.ERROR)
    except Exception:
        pass

    return tf

# =========================
# 🎨 CUSTOM DARK UI
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #ffffff;
}
section[data-testid="stSidebar"] {
    background-color: #111827;
}
h1, h2, h3 {
    color: #00F5D4 !important;
}
.stButton > button {
    background: linear-gradient(90deg, #00F5D4, #0ea5e9);
    color: black;
    border-radius: 10px;
    padding: 0.5em 1em;
    font-weight: bold;
    border: none;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #0ea5e9, #00F5D4);
}
.stTextInput input, .stTextArea textarea {
    background-color: #1f2937;
    color: white;
    border-radius: 8px;
}
.stNumberInput input {
    background-color: #1f2937;
    color: white;
}
.stFileUploader {
    background-color: #1f2937;
    border-radius: 10px;
    padding: 10px;
}
hr {
    border: 1px solid #374151;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🧠 AI Model Hub")

option = st.sidebar.radio(
    "Select Model",
    [
        "📱 Mobile Price",
        "🎬 Movie Review",
        "🩺 Pneumonia",
        "🎧 Emotion",
        "🎥 Action"
    ]
)

# =========================
# HEADER
# =========================
st.markdown("""
# 🧠 AI Model Hub
### 🚀 Multi-Modal Intelligence System
""")

st.markdown("---")

# =========================
# 📱 MOBILE MODEL
# =========================
# =========================
# 📱 MOBILE MODEL (FIXED)
# =========================
# =========================
# 📱 MOBILE MODEL (RAM FIXED)
# =========================
if option == "📱 Mobile Price":

    import numpy as np
    import pandas as pd

    st.header("📱 Mobile Price Prediction")

    pipeline = load_legacy_joblib("models/numeric/mobile_pipeline.pkl")
    model = pipeline["model"]
    scaler = pipeline["scaler"]
    features = pipeline["features"]

    # =========================
    important_features = [
        "ram", "battery_power", "px_height",
        "px_width", "int_memory", "mobile_wt",
        "clock_speed", "n_cores"
    ]

    binary_features = [
        "blue", "dual_sim", "four_g",
        "three_g", "touch_screen", "wifi"
    ]

    # =========================
    # 🎯 DEFAULT VALUES (RAM in GB now)
    # =========================
    default_values = {
        "ram": 2,   # ✅ GB (NOT MB)
        "battery_power": 1500,
        "px_height": 1000,
        "px_width": 1500,
        "int_memory": 64,
        "mobile_wt": 180,
        "clock_speed": 2,
        "n_cores": 4,

        "blue": 1,
        "dual_sim": 1,
        "four_g": 1,
        "three_g": 1,
        "touch_screen": 1,
        "wifi": 1
    }

    # =========================
    if st.button("⚡ Fill Sample Data", key="sample_btn"):
        for key, value in default_values.items():
            st.session_state[f"val_{key}"] = value

    st.markdown("### ⚡ Quick Input")

    user_data = {}
    col1, col2 = st.columns(2)

    # =========================
    # 🔢 IMPORTANT INPUTS
    # =========================
    for i, feature in enumerate(important_features):
        if feature in features:
            with col1 if i % 2 == 0 else col2:

                # 🔥 SPECIAL CASE: RAM (GB → MB)
                if feature == "ram":
                    ram_gb = st.number_input(
                        "RAM (GB)",
                        value=st.session_state.get("val_ram", 4),
                        key="val_ram"
                    )
                    user_data["ram"] = ram_gb * 1024   # ✅ CONVERT HERE

                else:
                    user_data[feature] = st.number_input(
                        feature,
                        value=st.session_state.get(
                            f"val_{feature}",
                            default_values.get(feature, 1)
                        ),
                        key=f"val_{feature}"
                    )

    st.markdown("---")

    # =========================
    # ⚙️ ADVANCED SETTINGS
    # =========================
    with st.expander("⚙️ Advanced Settings (Optional)"):

        for feature in features:
            if feature not in important_features:

                if feature in binary_features:
                    val = st.selectbox(
                        feature,
                        ["Yes", "No"],
                        index=0,
                        key=f"adv_{feature}"
                    )
                    user_data[feature] = 1 if val == "Yes" else 0

                else:
                    user_data[feature] = st.number_input(
                        feature,
                        value=default_values.get(feature, 1),
                        key=f"adv_{feature}"
                    )

    # =========================
    # 🚀 PREDICTION
    # =========================
    if st.button("🚀 Predict Price", key="mobile_btn"):

        df = pd.DataFrame([user_data])

        for f in features:
            if f not in df:
                df[f] = default_values.get(f, 0)

        df = df[features]

        st.write("### 🔍 Input Features")
        st.dataframe(df)

        # ⚠️ IMPORTANT: choose ONE depending on your pipeline

        # 👉 If model is pipeline:
        pred = model.predict(df)[0]

        # 👉 If model is NOT pipeline, use this instead:
        # arr = scaler.transform(df)
        # pred = model.predict(arr)[0]

        labels = {
            0: "Low Cost 📉",
            1: "Medium Cost 💰",
            2: "High Cost 💎",
            3: "Very High Cost 🚀"
        }

        st.success(f"### {labels[pred]}")
# =========================
# 🎬 TEXT MODEL
# =========================
elif option == "🎬 Movie Review":

    st.header("🎬 Sentiment Analysis of Movie Reviews")

    pipeline = load_legacy_joblib("models/text/sentiment_pipeline.pkl")

    model = pipeline["model"]
    vectorizer = pipeline["vectorizer"]
    encoder = pipeline["encoder"]

    review = st.text_area("Enter review", height=150, key="text_input")

    if st.button("🔍 Predict", key="text_btn"):
        if review.strip() == "":
            st.warning("Enter text")
        else:
            vec = vectorizer.transform([review])
            pred = model.predict(vec)
            sentiment = encoder.inverse_transform(pred)[0]

            if sentiment == "positive":
                st.success("😊 Positive")
            else:
                st.error("😡 Negative")

# =========================
# 🩺 IMAGE MODEL
# =========================
elif option == "🩺 Pneumonia":

    tf = get_tensorflow()
    from PIL import Image
    import numpy as np

    st.header("🩺 Pneumonia Detection From X-ray")

    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model("models/image/pneumonia_model.keras")

    model = load_model()

    file = st.file_uploader("Upload X-ray", type=["jpg","png"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, width="stretch")

        if st.button("🔍 Predict", key="img_btn"):
            img = img.resize((224,224))
            img = np.array(img)/255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)[0][0]

            if pred > 0.5:
                st.error("❌ Pneumonia")
            else:
                st.success("✅ Normal")

# =========================
# 🎧 AUDIO MODEL
# =========================
elif option == "🎧 Emotion":

    tf = get_tensorflow()
    import librosa
    import numpy as np

    st.header("🎧 Emotion Detection through audio")

    @st.cache_resource
    def load_audio():
        return tf.keras.models.load_model("models/audio/audio_model.keras")

    model = load_audio()
    le = load_legacy_joblib("models/audio/audio_label_encoder.pkl")

    file = st.file_uploader("Upload WAV", type=["wav"])

    def extract(f):
        audio, sr = librosa.load(f, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        if mfcc.shape[1] < 128:
            mfcc = np.pad(mfcc, ((0,0),(0,128-mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :128]

        return mfcc

    if file:
        st.audio(file)

        if st.button("🔍 Predict", key="audio_btn"):
            mfcc = extract(file)
            mfcc = mfcc[..., np.newaxis]
            mfcc = np.expand_dims(mfcc, axis=0)

            pred = model.predict(mfcc)[0]
            idx = np.argmax(pred)

            emotion = le.inverse_transform([idx])[0]
            st.success(f"{emotion} ({pred[idx]*100:.2f}%)")

# =========================
# 🎥 VIDEO MODEL
# =========================
elif option == "🎥 Action":

    tf = get_tensorflow()
    import cv2
    import numpy as np
    import tempfile

    st.header("🎥 Action Recognition for sports")

    @st.cache_resource
    def load_video():
        return tf.keras.models.load_model("models/video/action_model.keras")

    model = load_video()
    classes = np.load("models/video/action_classes.npy")

    file = st.file_uploader("Upload Video", type=["mp4","avi"])

    def extract(video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip = max(total//10,1)

        for i in range(10):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i*skip)
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame,(64,64))
            frame = frame/255.0
            frames.append(frame)

        cap.release()
        return np.array(frames)

    if file:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(file.read())

        st.video(file)

        if st.button("🔍 Predict", key="video_btn"):
            frames = extract(temp.name)

            if len(frames) == 10:
                frames = np.expand_dims(frames, axis=0)
                pred = model.predict(frames)[0]

                idx = np.argmax(pred)
                st.success(f"{classes[idx]} ({pred[idx]*100:.2f}%)")
            else:
                st.error("Video too short")
