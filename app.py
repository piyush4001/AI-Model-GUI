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
       # pred = model.predict(df)[0]

        # 👉 If model is NOT pipeline, use this instead:
        arr = scaler.transform(df)
        pred = model.predict(arr)[0]

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

# import os

# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# import warnings

# import streamlit as st

# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# # ─── Page config ──────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="AI Model Hub",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

# # ─── Global CSS ───────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

# /* ── Reset & base ── */
# *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

# html, body, .stApp {
#     background: #080b12 !important;
#     color: #e2e8f0;
#     font-family: 'Syne', sans-serif;
# }

# /* hide default streamlit chrome */
# #MainMenu, footer, header { visibility: hidden; }
# .block-container { padding: 2rem 2rem 4rem !important; max-width: 1200px !important; }

# /* ── Scrollbar ── */
# ::-webkit-scrollbar { width: 6px; }
# ::-webkit-scrollbar-track { background: #0d1117; }
# ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

# /* ── Typography ── */
# h1, h2, h3, h4 { font-family: 'Syne', sans-serif; letter-spacing: -0.02em; }

# /* ── Animated gradient background ── */
# .bg-grid {
#     position: fixed; top: 0; left: 0; width: 100%; height: 100%;
#     background-image:
#         linear-gradient(rgba(0,245,212,0.03) 1px, transparent 1px),
#         linear-gradient(90deg, rgba(0,245,212,0.03) 1px, transparent 1px);
#     background-size: 60px 60px;
#     pointer-events: none; z-index: 0;
# }

# /* ── Orbs ── */
# .orb {
#     position: fixed; border-radius: 50%; filter: blur(100px); opacity: 0.15;
#     pointer-events: none; z-index: 0; animation: drift 12s ease-in-out infinite alternate;
# }
# .orb-1 { width: 500px; height: 500px; background: #00F5D4; top: -150px; right: -100px; animation-delay: 0s; }
# .orb-2 { width: 400px; height: 400px; background: #0ea5e9; bottom: -100px; left: -80px; animation-delay: -5s; }
# .orb-3 { width: 300px; height: 300px; background: #8b5cf6; top: 40%; left: 40%; animation-delay: -3s; }

# @keyframes drift {
#     from { transform: translate(0, 0) scale(1); }
#     to   { transform: translate(30px, 20px) scale(1.05); }
# }

# /* ── Hero ── */
# .hero {
#     position: relative; z-index: 1;
#     text-align: center; padding: 4rem 2rem 2rem;
# }
# .hero-badge {
#     display: inline-block;
#     border: 1px solid rgba(0,245,212,0.4);
#     color: #00F5D4;
#     font-family: 'Space Mono', monospace;
#     font-size: 0.72rem;
#     letter-spacing: 0.18em;
#     text-transform: uppercase;
#     padding: 0.35rem 1.1rem;
#     border-radius: 100px;
#     background: rgba(0,245,212,0.07);
#     margin-bottom: 1.5rem;
# }
# .hero-title {
#     font-size: clamp(2.8rem, 6vw, 5rem);
#     font-weight: 800;
#     line-height: 1.05;
#     background: linear-gradient(135deg, #ffffff 30%, #00F5D4 70%, #0ea5e9 100%);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
#     background-clip: text;
#     margin-bottom: 1rem;
# }
# .hero-sub {
#     color: #64748b;
#     font-size: 1.05rem;
#     max-width: 520px;
#     margin: 0 auto 3rem;
#     line-height: 1.7;
#     font-family: 'Space Mono', monospace;
# }

# /* ── Cards grid ── */
# .cards-grid {
#     display: grid;
#     grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
#     gap: 1.25rem;
#     position: relative; z-index: 1;
#     margin-bottom: 3rem;
# }

# .model-card {
#     position: relative;
#     border: 1px solid rgba(255,255,255,0.06);
#     border-radius: 20px;
#     padding: 2rem 1.5rem 1.6rem;
#     background: rgba(255,255,255,0.025);
#     backdrop-filter: blur(12px);
#     cursor: pointer;
#     transition: transform 0.28s cubic-bezier(.34,1.56,.64,1), border-color 0.25s, box-shadow 0.25s;
#     overflow: hidden;
#     text-align: center;
# }
# .model-card::before {
#     content: '';
#     position: absolute; inset: 0;
#     background: var(--card-glow);
#     opacity: 0;
#     transition: opacity 0.3s;
#     border-radius: 20px;
# }
# .model-card:hover { transform: translateY(-6px) scale(1.02); }
# .model-card:hover::before { opacity: 1; }
# .model-card:hover { border-color: var(--card-accent); box-shadow: 0 0 28px -6px var(--card-accent); }

# .card-icon {
#     font-size: 2.8rem;
#     display: block;
#     margin-bottom: 1rem;
#     filter: drop-shadow(0 0 12px var(--card-accent));
# }
# .card-title {
#     font-size: 1.05rem;
#     font-weight: 700;
#     color: #f1f5f9;
#     margin-bottom: 0.4rem;
# }
# .card-desc {
#     font-size: 0.78rem;
#     color: #64748b;
#     line-height: 1.55;
#     font-family: 'Space Mono', monospace;
# }
# .card-tag {
#     display: inline-block;
#     margin-top: 1rem;
#     font-size: 0.68rem;
#     font-family: 'Space Mono', monospace;
#     letter-spacing: 0.1em;
#     text-transform: uppercase;
#     color: var(--card-accent);
#     border: 1px solid var(--card-accent);
#     padding: 0.2rem 0.7rem;
#     border-radius: 100px;
#     opacity: 0.8;
# }

# /* ── Back button ── */
# .back-btn-wrap { position: relative; z-index: 1; margin-bottom: 0.5rem; }

# /* ── Page header ── */
# .page-header {
#     position: relative; z-index: 1;
#     display: flex; align-items: center; gap: 1.2rem;
#     margin-bottom: 2rem;
#     padding: 1.8rem 2rem;
#     border: 1px solid rgba(255,255,255,0.06);
#     border-radius: 20px;
#     background: rgba(255,255,255,0.025);
#     backdrop-filter: blur(12px);
# }
# .page-header-icon { font-size: 2.8rem; filter: drop-shadow(0 0 14px var(--page-accent)); }
# .page-header-title { font-size: 1.8rem; font-weight: 800; color: var(--page-accent); }
# .page-header-sub { font-size: 0.82rem; color: #64748b; font-family: 'Space Mono', monospace; margin-top: 0.2rem; }

# /* ── Panels ── */
# .panel {
#     position: relative; z-index: 1;
#     border: 1px solid rgba(255,255,255,0.06);
#     border-radius: 16px;
#     padding: 1.8rem;
#     background: rgba(255,255,255,0.022);
#     backdrop-filter: blur(12px);
#     margin-bottom: 1.25rem;
# }
# .panel-title {
#     font-size: 0.72rem;
#     font-family: 'Space Mono', monospace;
#     letter-spacing: 0.15em;
#     text-transform: uppercase;
#     color: var(--page-accent);
#     margin-bottom: 1.2rem;
#     display: flex; align-items: center; gap: 0.5rem;
# }
# .panel-title::before {
#     content: '';
#     display: inline-block; width: 6px; height: 6px;
#     background: var(--page-accent); border-radius: 50%;
# }

# /* ── Input overrides ── */
# .stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox select {
#     background: rgba(255,255,255,0.04) !important;
#     border: 1px solid rgba(255,255,255,0.09) !important;
#     border-radius: 10px !important;
#     color: #f1f5f9 !important;
#     font-family: 'Syne', sans-serif !important;
#     transition: border-color 0.2s;
# }
# .stTextInput input:focus, .stTextArea textarea:focus {
#     border-color: var(--page-accent, #00F5D4) !important;
#     box-shadow: 0 0 0 2px rgba(0,245,212,0.12) !important;
# }
# label, .stTextInput label, .stNumberInput label, .stTextArea label, .stSelectbox label {
#     color: #94a3b8 !important;
#     font-size: 0.82rem !important;
#     font-family: 'Space Mono', monospace !important;
#     letter-spacing: 0.05em !important;
# }

# /* ── Buttons ── */
# .stButton > button {
#     background: linear-gradient(135deg, var(--page-accent, #00F5D4), #0ea5e9) !important;
#     color: #080b12 !important;
#     border: none !important;
#     border-radius: 12px !important;
#     padding: 0.65rem 1.6rem !important;
#     font-weight: 700 !important;
#     font-family: 'Syne', sans-serif !important;
#     font-size: 0.9rem !important;
#     letter-spacing: 0.03em !important;
#     transition: opacity 0.2s, transform 0.15s !important;
#     box-shadow: 0 4px 20px rgba(0,245,212,0.25) !important;
# }
# .stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

# /* ── File uploader ── */
# .stFileUploader {
#     border: 2px dashed rgba(255,255,255,0.1) !important;
#     border-radius: 14px !important;
#     padding: 1rem !important;
#     background: rgba(255,255,255,0.02) !important;
#     transition: border-color 0.2s;
# }
# .stFileUploader:hover { border-color: var(--page-accent, #00F5D4) !important; }

# /* ── Success / error / warning alerts ── */
# .stSuccess, .element-container .stSuccess {
#     background: rgba(0,245,212,0.08) !important;
#     border: 1px solid rgba(0,245,212,0.3) !important;
#     border-radius: 12px !important;
#     color: #00F5D4 !important;
# }
# .stError {
#     background: rgba(239,68,68,0.08) !important;
#     border: 1px solid rgba(239,68,68,0.3) !important;
#     border-radius: 12px !important;
# }
# .stWarning {
#     background: rgba(245,158,11,0.08) !important;
#     border: 1px solid rgba(245,158,11,0.3) !important;
#     border-radius: 12px !important;
# }

# /* ── Progress / spinner ── */
# .stSpinner > div { border-top-color: #00F5D4 !important; }

# /* ── Dataframe ── */
# .stDataFrame { border-radius: 12px !important; overflow: hidden; }

# /* ── Audio player ── */
# .stAudio { border-radius: 12px !important; }

# /* ── Expander ── */
# .streamlit-expanderHeader {
#     background: rgba(255,255,255,0.03) !important;
#     border-radius: 10px !important;
#     color: #94a3b8 !important;
#     font-family: 'Space Mono', monospace !important;
#     font-size: 0.82rem !important;
# }

# /* ── Divider ── */
# hr { border-color: rgba(255,255,255,0.06) !important; }

# /* ── Metric ── */
# .stMetric label { color: #64748b !important; font-family: 'Space Mono', monospace !important; font-size: 0.75rem !important; }
# .stMetric .stMetricValue { color: #f1f5f9 !important; font-family: 'Syne', sans-serif !important; }

# /* ── Result card ── */
# .result-card {
#     position: relative; z-index: 1;
#     border-radius: 16px;
#     padding: 1.6rem 2rem;
#     text-align: center;
#     margin-top: 1rem;
# }
# .result-card.positive {
#     background: rgba(0,245,212,0.07);
#     border: 1px solid rgba(0,245,212,0.3);
# }
# .result-card.negative {
#     background: rgba(239,68,68,0.07);
#     border: 1px solid rgba(239,68,68,0.3);
# }
# .result-card.warning {
#     background: rgba(245,158,11,0.07);
#     border: 1px solid rgba(245,158,11,0.3);
# }
# .result-emoji { font-size: 3rem; display: block; margin-bottom: 0.5rem; }
# .result-label { font-size: 1.5rem; font-weight: 800; }
# .result-sub { font-size: 0.82rem; color: #64748b; font-family: 'Space Mono', monospace; margin-top: 0.3rem; }

# /* ── Sidebar hidden ── */
# section[data-testid="stSidebar"] { display: none !important; }

# /* ── Step badges ── */
# .step-badge {
#     display: inline-flex; align-items: center; justify-content: center;
#     width: 28px; height: 28px; border-radius: 50%;
#     background: var(--page-accent, #00F5D4);
#     color: #080b12; font-weight: 700; font-size: 0.8rem;
#     margin-right: 0.6rem; flex-shrink: 0;
# }
# .step-row { display: flex; align-items: center; margin-bottom: 0.8rem; color: #94a3b8; font-family: 'Space Mono', monospace; font-size: 0.8rem; }
# </style>
# """, unsafe_allow_html=True)

# # Background decorations
# st.markdown("""
# <div class="bg-grid"></div>
# <div class="orb orb-1"></div>
# <div class="orb orb-2"></div>
# <div class="orb orb-3"></div>
# """, unsafe_allow_html=True)

# # ─── Helpers ──────────────────────────────────────────────────────────────────
# def load_legacy_joblib(path):
#     import joblib
#     try:
#         from sklearn.exceptions import InconsistentVersionWarning
#         w = InconsistentVersionWarning
#     except Exception:
#         w = None
#     with warnings.catch_warnings():
#         if w is not None:
#             warnings.simplefilter("ignore", w)
#         return joblib.load(path)

# def get_tensorflow():
#     import tensorflow as tf
#     tf.get_logger().setLevel("ERROR")
#     try:
#         import absl.logging
#         absl.logging.set_verbosity(absl.logging.ERROR)
#         absl.logging.set_stderrthreshold(absl.logging.ERROR)
#     except Exception:
#         pass
#     return tf

# def back_button():
#     st.markdown('<div class="back-btn-wrap">', unsafe_allow_html=True)
#     if st.button("← Back to Hub"):
#         st.session_state.page = "home"
#         st.rerun()
#     st.markdown('</div>', unsafe_allow_html=True)

# # ─── Session state ─────────────────────────────────────────────────────────────
# if "page" not in st.session_state:
#     st.session_state.page = "home"

# # ─── Card click JS helper ──────────────────────────────────────────────────────
# # We use a Streamlit button hidden behind a custom card HTML via columns trick.

# CARDS = [
#     {
#         "key": "mobile",
#         "icon": "📱",
#         "title": "Mobile Price",
#         "desc": "Predict smartphone price tier from hardware specs",
#         "tag": "Numeric · ML",
#         "accent": "#00F5D4",
#         "glow": "linear-gradient(135deg, rgba(0,245,212,0.06), transparent)",
#     },
#     {
#         "key": "movie",
#         "icon": "🎬",
#         "title": "Movie Review",
#         "desc": "Sentiment analysis on cinema reviews",
#         "tag": "Text · NLP",
#         "accent": "#f59e0b",
#         "glow": "linear-gradient(135deg, rgba(245,158,11,0.06), transparent)",
#     },
#     {
#         "key": "pneumonia",
#         "icon": "🩺",
#         "title": "Pneumonia",
#         "desc": "Detect pneumonia from chest X-ray images",
#         "tag": "Image · CNN",
#         "accent": "#ef4444",
#         "glow": "linear-gradient(135deg, rgba(239,68,68,0.06), transparent)",
#     },
#     {
#         "key": "emotion",
#         "icon": "🎧",
#         "title": "Emotion",
#         "desc": "Recognise emotion from raw audio recordings",
#         "tag": "Audio · LSTM",
#         "accent": "#8b5cf6",
#         "glow": "linear-gradient(135deg, rgba(139,92,246,0.06), transparent)",
#     },
#     {
#         "key": "action",
#         "icon": "🎥",
#         "title": "Action",
#         "desc": "Sports action recognition from video clips",
#         "tag": "Video · Deep Learning",
#         "accent": "#0ea5e9",
#         "glow": "linear-gradient(135deg, rgba(14,165,233,0.06), transparent)",
#     },
# ]

# # ══════════════════════════════════════════════════════════════════════════════
# # LANDING PAGE
# # ══════════════════════════════════════════════════════════════════════════════
# if st.session_state.page == "home":
#     st.markdown("""
#     <div class="hero">
#         <div class="hero-badge">Multi-Modal Intelligence System</div>
#         <h1 class="hero-title">AI Model Hub</h1>
#         <p class="hero-sub">Five production-ready models. One unified interface.<br>
#         Pick a modality and run inference instantly.</p>
#     </div>
#     """, unsafe_allow_html=True)

#     cols = st.columns(5)
#     for i, card in enumerate(CARDS):
#         with cols[i]:
#             st.markdown(f"""
#             <div class="model-card"
#                  style="--card-accent:{card['accent']}; --card-glow:{card['glow']};">
#                 <span class="card-icon">{card['icon']}</span>
#                 <div class="card-title">{card['title']}</div>
#                 <div class="card-desc">{card['desc']}</div>
#                 <div class="card-tag">{card['tag']}</div>
#             </div>
#             """, unsafe_allow_html=True)
#             if st.button(f"Open {card['title']}", key=f"nav_{card['key']}",
#                          use_container_width=True):
#                 st.session_state.page = card["key"]
#                 st.rerun()

#     # Stats strip
#     st.markdown("<br>", unsafe_allow_html=True)
#     c1, c2, c3, c4 = st.columns(4)
#     for col, label, value in [
#         (c1, "Models Deployed", "5"),
#         (c2, "Data Modalities", "5"),
#         (c3, "Inference", "Real-time"),
#         (c4, "Stack", "TF · SK · CV"),
#     ]:
#         with col:
#             st.markdown(f"""
#             <div style="text-align:center; padding:1.2rem; border:1px solid rgba(255,255,255,0.05);
#                         border-radius:14px; background:rgba(255,255,255,0.02);">
#                 <div style="font-size:1.6rem; font-weight:800; color:#f1f5f9;">{value}</div>
#                 <div style="font-size:0.72rem; color:#475569; font-family:'Space Mono',monospace;
#                             letter-spacing:0.1em; text-transform:uppercase; margin-top:0.2rem;">{label}</div>
#             </div>
#             """, unsafe_allow_html=True)


# # ══════════════════════════════════════════════════════════════════════════════
# # 📱 MOBILE PRICE PAGE
# # ══════════════════════════════════════════════════════════════════════════════
# elif st.session_state.page == "mobile":
#     import numpy as np
#     import pandas as pd

#     st.markdown('<style>:root { --page-accent: #00F5D4; }</style>', unsafe_allow_html=True)
#     back_button()

#     st.markdown("""
#     <div class="page-header">
#         <span class="page-header-icon">📱</span>
#         <div>
#             <div class="page-header-title">Mobile Price Prediction</div>
#             <div class="page-header-sub">Classify smartphones into price tiers using hardware specs</div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

#     pipeline = load_legacy_joblib("models/numeric/mobile_pipeline.pkl")
#     model    = pipeline["model"]
#     scaler   = pipeline["scaler"]
#     features = pipeline["features"]

#     important_features = ["ram","battery_power","px_height","px_width",
#                           "int_memory","mobile_wt","clock_speed","n_cores"]
#     binary_features    = ["blue","dual_sim","four_g","three_g","touch_screen","wifi"]

#     default_values = {
#         "ram":2, "battery_power":1500, "px_height":1000, "px_width":1500,
#         "int_memory":64, "mobile_wt":180, "clock_speed":2, "n_cores":4,
#         "blue":1,"dual_sim":1,"four_g":1,"three_g":1,"touch_screen":1,"wifi":1
#     }

#     # Quick fill
#     col_fill, col_sp = st.columns([1, 3])
#     with col_fill:
#         if st.button("⚡ Fill Sample Data"):
#             for k, v in default_values.items():
#                 st.session_state[f"val_{k}"] = v

#     # Core specs panel
#     st.markdown('<div class="panel"><div class="panel-title">Core Specifications</div>', unsafe_allow_html=True)
#     user_data = {}
#     c1, c2, c3, c4 = st.columns(4)
#     spec_cols = [c1, c2, c3, c4]
#     for idx, feature in enumerate(important_features):
#         if feature in features:
#             with spec_cols[idx % 4]:
#                 if feature == "ram":
#                     v = st.number_input("RAM (GB)", min_value=1, max_value=64,
#                                         value=st.session_state.get("val_ram", 4),
#                                         key="val_ram")
#                     user_data["ram"] = v * 1024
#                 else:
#                     user_data[feature] = st.number_input(
#                         feature.replace("_", " ").title(),
#                         value=int(st.session_state.get(f"val_{feature}",
#                                   default_values.get(feature, 1))),
#                         key=f"val_{feature}")
#     st.markdown('</div>', unsafe_allow_html=True)

#     # Advanced panel
#     with st.expander("⚙️  Advanced / Connectivity Features"):
#         st.markdown('<div style="padding-top:0.5rem;">', unsafe_allow_html=True)
#         adv_cols = st.columns(6)
#         adv_idx = 0
#         for feature in features:
#             if feature not in important_features:
#                 with adv_cols[adv_idx % 6]:
#                     if feature in binary_features:
#                         val = st.selectbox(feature.replace("_"," ").title(),
#                                            ["Yes","No"], index=0, key=f"adv_{feature}")
#                         user_data[feature] = 1 if val == "Yes" else 0
#                     else:
#                         user_data[feature] = st.number_input(
#                             feature.replace("_"," ").title(),
#                             value=int(default_values.get(feature, 1)),
#                             key=f"adv_{feature}")
#                 adv_idx += 1
#         st.markdown('</div>', unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)
#     if st.button("🚀  Predict Price Range", use_container_width=False):
#         df = pd.DataFrame([user_data])
#         for f in features:
#             if f not in df:
#                 df[f] = default_values.get(f, 0)
#         df = df[features]

#         pred = model.predict(df)[0]

#         tiers = {
#             0: ("Low Cost",      "📉", "#64748b", "Entry-level device — budget-friendly",      "positive"),
#             1: ("Medium Cost",   "💰", "#f59e0b", "Mid-range device — balanced performance",   "positive"),
#             2: ("High Cost",     "💎", "#00F5D4", "Premium device — flagship experience",      "positive"),
#             3: ("Very High Cost","🚀", "#8b5cf6", "Ultra-premium — top-tier performance",       "positive"),
#         }
#         label, emoji, color, sub, cls = tiers[pred]

#         st.markdown(f"""
#         <div class="result-card {cls}">
#             <span class="result-emoji">{emoji}</span>
#             <div class="result-label" style="color:{color};">{label}</div>
#             <div class="result-sub">{sub}</div>
#         </div>
#         """, unsafe_allow_html=True)

#         with st.expander("🔍  View input features"):
#             st.dataframe(df, use_container_width=True)


# # ══════════════════════════════════════════════════════════════════════════════
# # 🎬 MOVIE REVIEW PAGE
# # ══════════════════════════════════════════════════════════════════════════════
# elif st.session_state.page == "movie":
#     st.markdown('<style>:root { --page-accent: #f59e0b; }</style>', unsafe_allow_html=True)
#     back_button()

#     st.markdown("""
#     <div class="page-header">
#         <span class="page-header-icon">🎬</span>
#         <div>
#             <div class="page-header-title">Movie Review Sentiment</div>
#             <div class="page-header-sub">NLP model trained on IMDB-style reviews</div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

#     pipeline  = load_legacy_joblib("models/text/sentiment_pipeline.pkl")
#     model     = pipeline["model"]
#     vectorizer = pipeline["vectorizer"]
#     encoder   = pipeline["encoder"]

#     st.markdown('<div class="panel"><div class="panel-title">Input Review</div>', unsafe_allow_html=True)

#     review = st.text_area(
#         "Paste or type a movie review below",
#         height=180,
#         placeholder="e.g. 'An absolutely breathtaking film — the cinematography alone is worth watching for...'",
#         key="text_input",
#     )

#     # Character count indicator
#     char_count = len(review)
#     st.markdown(f'<div style="text-align:right; color:#475569; font-size:0.72rem; font-family:monospace; margin-top:-0.5rem;">{char_count} characters</div>', unsafe_allow_html=True)
#     st.markdown('</div>', unsafe_allow_html=True)

#     # Sample buttons
#     st.markdown("<br>", unsafe_allow_html=True)
#     sc1, sc2, sc3 = st.columns(3)
#     samples = [
#         ("😊 Positive Sample", "An absolute masterpiece. The director weaves a story that is both emotionally gripping and visually stunning. Every frame is a work of art."),
#         ("😡 Negative Sample", "Terrible waste of time. The plot makes no sense, the acting is wooden and the special effects look like they were done on a budget of $10."),
#         ("🤔 Neutral Sample", "The film had its moments but ultimately fell flat. Some scenes were genuinely exciting while others dragged on far too long."),
#     ]
#     for col, (btn_lbl, txt) in zip([sc1, sc2, sc3], samples):
#         with col:
#             if st.button(btn_lbl, use_container_width=True):
#                 st.session_state["text_input"] = txt
#                 st.rerun()

#     st.markdown("<br>", unsafe_allow_html=True)
#     if st.button("🔍  Analyse Sentiment", use_container_width=False):
#         if review.strip() == "":
#             st.warning("Please enter a review first.")
#         else:
#             vec = vectorizer.transform([review])
#             pred = model.predict(vec)
#             sentiment = encoder.inverse_transform(pred)[0]

#             if sentiment == "positive":
#                 st.markdown("""
#                 <div class="result-card positive">
#                     <span class="result-emoji">😊</span>
#                     <div class="result-label" style="color:#00F5D4;">Positive Sentiment</div>
#                     <div class="result-sub">The review expresses a favourable opinion</div>
#                 </div>""", unsafe_allow_html=True)
#             else:
#                 st.markdown("""
#                 <div class="result-card negative">
#                     <span class="result-emoji">😡</span>
#                     <div class="result-label" style="color:#ef4444;">Negative Sentiment</div>
#                     <div class="result-sub">The review expresses an unfavourable opinion</div>
#                 </div>""", unsafe_allow_html=True)


# # ══════════════════════════════════════════════════════════════════════════════
# # 🩺 PNEUMONIA PAGE
# # ══════════════════════════════════════════════════════════════════════════════
# elif st.session_state.page == "pneumonia":
#     tf = get_tensorflow()
#     from PIL import Image
#     import numpy as np

#     st.markdown('<style>:root { --page-accent: #ef4444; }</style>', unsafe_allow_html=True)
#     back_button()

#     st.markdown("""
#     <div class="page-header">
#         <span class="page-header-icon">🩺</span>
#         <div>
#             <div class="page-header-title">Pneumonia Detection</div>
#             <div class="page-header-sub">CNN analysis of chest X-ray images</div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

#     @st.cache_resource
#     def load_pneumonia():
#         return tf.keras.models.load_model("models/image/pneumonia_model.keras")
#     model = load_pneumonia()

#     # How-to guide
#     st.markdown("""
#     <div class="panel">
#         <div class="panel-title">How it works</div>
#         <div class="step-row"><span class="step-badge">1</span>Upload a chest X-ray image (JPG or PNG)</div>
#         <div class="step-row"><span class="step-badge">2</span>Image is resized to 224×224 and normalised</div>
#         <div class="step-row"><span class="step-badge">3</span>CNN model outputs probability of pneumonia</div>
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown('<div class="panel"><div class="panel-title">Upload X-Ray</div>', unsafe_allow_html=True)
#     file = st.file_uploader("Choose a chest X-ray image", type=["jpg","jpeg","png"], label_visibility="collapsed")
#     st.markdown('</div>', unsafe_allow_html=True)

#     if file:
#         img = Image.open(file).convert("RGB")
#         c_img, c_res = st.columns([1, 1])
#         with c_img:
#             st.markdown('<div class="panel"><div class="panel-title">Uploaded Image</div>', unsafe_allow_html=True)
#             st.image(img, use_column_width=True)
#             st.markdown('</div>', unsafe_allow_html=True)

#         with c_res:
#             st.markdown('<div class="panel"><div class="panel-title">Analysis</div>', unsafe_allow_html=True)
#             st.markdown(f"""
#             <div style="color:#64748b; font-family:'Space Mono',monospace; font-size:0.78rem; margin-bottom:1rem;">
#                 Size: {img.size[0]} × {img.size[1]} px<br>
#                 Mode: {img.mode}
#             </div>
#             """, unsafe_allow_html=True)

#             if st.button("🔬  Run Analysis", use_container_width=True):
#                 with st.spinner("Analysing…"):
#                     arr = img.resize((224,224))
#                     arr = np.array(arr)/255.0
#                     arr = np.expand_dims(arr, axis=0)
#                     pred = model.predict(arr)[0][0]

#                 conf = pred if pred > 0.5 else 1 - pred
#                 if pred > 0.5:
#                     st.markdown(f"""
#                     <div class="result-card negative">
#                         <span class="result-emoji">🔴</span>
#                         <div class="result-label" style="color:#ef4444;">Pneumonia Detected</div>
#                         <div class="result-sub">Confidence: {conf*100:.1f}% — Please consult a physician</div>
#                     </div>""", unsafe_allow_html=True)
#                 else:
#                     st.markdown(f"""
#                     <div class="result-card positive">
#                         <span class="result-emoji">🟢</span>
#                         <div class="result-label" style="color:#00F5D4;">Normal — No Pneumonia</div>
#                         <div class="result-sub">Confidence: {conf*100:.1f}%</div>
#                     </div>""", unsafe_allow_html=True)

#             st.markdown('<div style="margin-top:1rem; padding:0.8rem; background:rgba(245,158,11,0.07); border:1px solid rgba(245,158,11,0.3); border-radius:10px; font-size:0.72rem; color:#92400e; font-family:monospace;">⚠️ For research / educational purposes only. Not a substitute for professional medical diagnosis.</div>', unsafe_allow_html=True)
#             st.markdown('</div>', unsafe_allow_html=True)


# # ══════════════════════════════════════════════════════════════════════════════
# # 🎧 EMOTION PAGE
# # ══════════════════════════════════════════════════════════════════════════════
# elif st.session_state.page == "emotion":
#     tf = get_tensorflow()
#     import librosa
#     import numpy as np

#     st.markdown('<style>:root { --page-accent: #8b5cf6; }</style>', unsafe_allow_html=True)
#     back_button()

#     st.markdown("""
#     <div class="page-header">
#         <span class="page-header-icon">🎧</span>
#         <div>
#             <div class="page-header-title">Emotion Detection</div>
#             <div class="page-header-sub">LSTM model on MFCC audio features</div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

#     @st.cache_resource
#     def load_audio_model():
#         return tf.keras.models.load_model("models/audio/audio_model.keras")
#     model = load_audio_model()
#     le    = load_legacy_joblib("models/audio/audio_label_encoder.pkl")

#     # Info panel
#     st.markdown("""
#     <div class="panel">
#         <div class="panel-title">How it works</div>
#         <div class="step-row"><span class="step-badge">1</span>Upload a WAV audio file</div>
#         <div class="step-row"><span class="step-badge">2</span>40 MFCC features are extracted (first 3s)</div>
#         <div class="step-row"><span class="step-badge">3</span>LSTM network classifies the emotion</div>
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown('<div class="panel"><div class="panel-title">Upload Audio</div>', unsafe_allow_html=True)
#     file = st.file_uploader("Drop a WAV file here", type=["wav"], label_visibility="collapsed")
#     st.markdown('</div>', unsafe_allow_html=True)

#     def extract_mfcc(f):
#         audio, sr = librosa.load(f, duration=3, offset=0.5)
#         mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
#         if mfcc.shape[1] < 128:
#             mfcc = np.pad(mfcc, ((0,0),(0,128-mfcc.shape[1])))
#         else:
#             mfcc = mfcc[:, :128]
#         return mfcc

#     EMOTION_EMOJIS = {
#         "happy":"😄","sad":"😢","angry":"😠","fear":"😱","disgust":"🤢",
#         "surprise":"😲","neutral":"😐","calm":"😌","ps":"😊","bored":"😴"
#     }

#     if file:
#         c_a, c_r = st.columns([1,1])
#         with c_a:
#             st.markdown('<div class="panel"><div class="panel-title">Playback</div>', unsafe_allow_html=True)
#             st.audio(file)
#             st.markdown('</div>', unsafe_allow_html=True)

#         with c_r:
#             st.markdown('<div class="panel"><div class="panel-title">Result</div>', unsafe_allow_html=True)
#             if st.button("🎙️  Detect Emotion", use_container_width=True):
#                 with st.spinner("Extracting features…"):
#                     mfcc = extract_mfcc(file)
#                     mfcc = mfcc[..., np.newaxis]
#                     mfcc = np.expand_dims(mfcc, axis=0)
#                     pred = model.predict(mfcc)[0]

#                 idx     = np.argmax(pred)
#                 emotion = le.inverse_transform([idx])[0]
#                 conf    = pred[idx]*100
#                 emoji   = EMOTION_EMOJIS.get(emotion.lower(), "🎵")

#                 st.markdown(f"""
#                 <div class="result-card positive" style="background:rgba(139,92,246,0.08); border-color:rgba(139,92,246,0.3);">
#                     <span class="result-emoji">{emoji}</span>
#                     <div class="result-label" style="color:#8b5cf6;">{emotion.title()}</div>
#                     <div class="result-sub">Confidence: {conf:.1f}%</div>
#                 </div>""", unsafe_allow_html=True)

#                 # Top-3 bar
#                 st.markdown("<br>", unsafe_allow_html=True)
#                 top3 = np.argsort(pred)[::-1][:3]
#                 for i in top3:
#                     lbl = le.inverse_transform([i])[0]
#                     pct = pred[i]*100
#                     st.markdown(f"""
#                     <div style="margin-bottom:0.6rem;">
#                         <div style="display:flex; justify-content:space-between; font-size:0.78rem;
#                                     font-family:'Space Mono',monospace; color:#94a3b8; margin-bottom:3px;">
#                             <span>{lbl.title()}</span><span>{pct:.1f}%</span>
#                         </div>
#                         <div style="height:6px; background:rgba(255,255,255,0.06); border-radius:3px;">
#                             <div style="width:{pct:.1f}%; height:100%; background:linear-gradient(90deg,#8b5cf6,#0ea5e9);
#                                         border-radius:3px; transition:width 0.6s;"></div>
#                         </div>
#                     </div>""", unsafe_allow_html=True)

#             st.markdown('</div>', unsafe_allow_html=True)


# # ══════════════════════════════════════════════════════════════════════════════
# # 🎥 ACTION PAGE
# # ══════════════════════════════════════════════════════════════════════════════
# elif st.session_state.page == "action":
#     tf = get_tensorflow()
#     import cv2
#     import numpy as np
#     import tempfile

#     st.markdown('<style>:root { --page-accent: #0ea5e9; }</style>', unsafe_allow_html=True)
#     back_button()

#     st.markdown("""
#     <div class="page-header">
#         <span class="page-header-icon">🎥</span>
#         <div>
#             <div class="page-header-title">Sports Action Recognition</div>
#             <div class="page-header-sub">Deep learning on video frame sequences</div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

#     @st.cache_resource
#     def load_video_model():
#         return tf.keras.models.load_model("models/video/action_model.keras")
#     model   = load_video_model()
#     classes = np.load("models/video/action_classes.npy")

#     st.markdown("""
#     <div class="panel">
#         <div class="panel-title">How it works</div>
#         <div class="step-row"><span class="step-badge">1</span>Upload an MP4 or AVI sports video</div>
#         <div class="step-row"><span class="step-badge">2</span>10 frames are sampled evenly across the clip</div>
#         <div class="step-row"><span class="step-badge">3</span>Frames are resized to 64×64 and fed to the model</div>
#     </div>
#     """, unsafe_allow_html=True)

#     def extract_frames(video_path):
#         frames = []
#         cap = cv2.VideoCapture(video_path)
#         total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         skip  = max(total//10, 1)
#         for i in range(10):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, i*skip)
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.resize(frame, (64,64)) / 255.0
#             frames.append(frame)
#         cap.release()
#         return np.array(frames)

#     st.markdown('<div class="panel"><div class="panel-title">Upload Video</div>', unsafe_allow_html=True)
#     file = st.file_uploader("Upload MP4 or AVI", type=["mp4","avi"], label_visibility="collapsed")
#     st.markdown('</div>', unsafe_allow_html=True)

#     if file:
#         c_v, c_r = st.columns([1,1])
#         with c_v:
#             st.markdown('<div class="panel"><div class="panel-title">Preview</div>', unsafe_allow_html=True)
#             st.video(file)
#             st.markdown('</div>', unsafe_allow_html=True)

#         with c_r:
#             st.markdown('<div class="panel"><div class="panel-title">Result</div>', unsafe_allow_html=True)
#             if st.button("🏃  Classify Action", use_container_width=True):
#                 temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#                 temp.write(file.read())
#                 temp.flush()

#                 with st.spinner("Sampling frames…"):
#                     frames = extract_frames(temp.name)

#                 if len(frames) == 10:
#                     frames_input = np.expand_dims(frames, axis=0)
#                     pred = model.predict(frames_input)[0]
#                     idx  = np.argmax(pred)
#                     conf = pred[idx]*100
#                     action_name = classes[idx]

#                     st.markdown(f"""
#                     <div class="result-card positive" style="background:rgba(14,165,233,0.08); border-color:rgba(14,165,233,0.3);">
#                         <span class="result-emoji">🏆</span>
#                         <div class="result-label" style="color:#0ea5e9;">{action_name}</div>
#                         <div class="result-sub">Confidence: {conf:.1f}%</div>
#                     </div>""", unsafe_allow_html=True)

#                     # Top classes
#                     st.markdown("<br>", unsafe_allow_html=True)
#                     top3 = np.argsort(pred)[::-1][:3]
#                     for i in top3:
#                         lbl = classes[i]
#                         pct = pred[i]*100
#                         st.markdown(f"""
#                         <div style="margin-bottom:0.6rem;">
#                             <div style="display:flex; justify-content:space-between; font-size:0.78rem;
#                                         font-family:'Space Mono',monospace; color:#94a3b8; margin-bottom:3px;">
#                                 <span>{lbl}</span><span>{pct:.1f}%</span>
#                             </div>
#                             <div style="height:6px; background:rgba(255,255,255,0.06); border-radius:3px;">
#                                 <div style="width:{pct:.1f}%; height:100%; background:linear-gradient(90deg,#0ea5e9,#8b5cf6);
#                                             border-radius:3px;"></div>
#                             </div>
#                         </div>""", unsafe_allow_html=True)
#                 else:
#                     st.error("Video too short — need at least 10 extractable frames.")
#             st.markdown('</div>', unsafe_allow_html=True)