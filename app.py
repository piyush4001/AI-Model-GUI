# import streamlit as st
# import joblib
# import numpy as np
# # Page config
# st.set_page_config(page_title="AI Model Hub", layout="wide")

# # Title
# st.title("🧠 AI Model Hub")
# st.markdown("### Select a Model")

# # Menu
# option = st.selectbox(
#     "Choose Application",
#     [
#         "Mobile Price Prediction",
#         "Movie Review Classification",
#         "Pneumonia Detection",
#         "Emotion Detection",
#         "Action Recognition"
#     ]
# )

# # =========================
# # 🎬 TEXT MODEL (placeholder)
# # =========================
# if option == "Movie Review Classification":

#     st.title("🎬 Movie Review Sentiment Analysis")

#     # Load pipeline
#     pipeline = joblib.load("models/text/sentiment_pipeline.pkl")

#     model = pipeline["model"]
#     vectorizer = pipeline["vectorizer"]
#     encoder = pipeline["encoder"]

#     st.markdown("### Enter your movie review below:")

#     # Input
#     review = st.text_area("Write your review here...", key="text_input")

#     col1, col2 = st.columns(2)

#     with col1:
#         predict_btn = st.button("🔍 Predict", key="text_predict")

#     with col2:
#         clear_btn = st.button("🧹 Clear", key="text_clear")

#     if predict_btn:
#         if review.strip() == "":
#             st.warning("⚠️ Please enter a review")
#         else:
#             # Transform
#             review_vec = vectorizer.transform([review])

#             # Predict
#             pred = model.predict(review_vec)
#             sentiment = encoder.inverse_transform(pred)[0]

#             # Confidence
#             try:
#                 proba = model.predict_proba(review_vec)
#                 confidence = max(proba[0]) * 100
#             except:
#                 confidence = None

#             st.markdown("---")

#             # Output
#             if sentiment == "positive":
#                 st.success("😊 Positive Review")
#             else:
#                 st.error("😡 Negative Review")

#             if confidence:
#                 st.info(f"Confidence: {confidence:.2f}%")

#     if clear_btn:
#         st.rerun()

# # =========================
# # 📱 MOBILE MODEL
# # =========================
# elif option == "Mobile Price Prediction":

#     st.title("📱 Mobile Price Prediction")

#     # Load pipeline (only once)
#     pipeline = joblib.load("models/numeric/mobile_pipeline.pkl")

#     model = pipeline["model"]
#     scaler = pipeline["scaler"]
#     features = pipeline["features"]

#     st.markdown("### Enter Mobile Specifications")

#     user_input = []

#     # Dynamic inputs with unique keys
#     for i, feature in enumerate(features):
#         value = st.number_input(
#             f"{feature}",
#             value=0,
#             key=f"input_{i}"
#         )
#         user_input.append(value)

#     # Predict button with key
#     if st.button("Predict Price Range", key="mobile_predict"):

#         input_array = np.array(user_input).reshape(1, -1)

#         # Scale input
#         input_scaled = scaler.transform(input_array)

#         # Predict
#         prediction = model.predict(input_scaled)[0]

#         # Labels
#         labels = {
#             0: "Low Cost 📉",
#             1: "Medium Cost 💰",
#             2: "High Cost 💎",
#             3: "Very High Cost 🚀"
#         }

#         st.success(f"Predicted Price Range: {labels[prediction]}")

# # =========================
# # 🩺 IMAGE MODEL
# # =========================
# elif option == "Pneumonia Detection":

#     import tensorflow as tf
#     from PIL import Image
#     import numpy as np

#     st.title("🩺 Pneumonia Detection from X-ray")

#     # Load model (cache to avoid reloading every time)
#     @st.cache_resource
#     def load_model():
#         return tf.keras.models.load_model("models/image/pneumonia_model.keras")

#     model = load_model()

#     st.markdown("### Upload Chest X-ray Image")

#     uploaded_file = st.file_uploader(
#         "Choose an image...",
#         type=["jpg", "png", "jpeg"],
#         key="image_upload"
#     )

#     if uploaded_file is not None:

#         image = Image.open(uploaded_file).convert("RGB")
#         st.image(image, caption="Uploaded Image", use_container_width=True)

#         # Preprocess
#         img = image.resize((224, 224))
#         img = np.array(img) / 255.0
#         img = np.expand_dims(img, axis=0)

#         if st.button("🔍 Predict", key="image_predict"):

#             prediction = model.predict(img)[0][0]

#             st.markdown("---")

#             # Output
#             if prediction > 0.5:
#                 st.error(f"❌ Pneumonia Detected ({prediction*100:.2f}% confidence)")
#             else:
#                 st.success(f"✅ Normal ({(1-prediction)*100:.2f}% confidence)")

# # =========================
# # 🎧 AUDIO MODEL
# # =========================
# elif option == "Emotion Detection":

#     import tensorflow as tf
#     import librosa
#     import numpy as np
#     import soundfile as sf

#     st.title("🎧 Speech Emotion Detection")

#     # Load model (cached)
#     @st.cache_resource
#     def load_audio_model():
#         return tf.keras.models.load_model("models/audio/audio_model.keras")

#     model = load_audio_model()

#     # Emotion labels (IMPORTANT: same order as training)
#     emotion_labels = [
#         "neutral", "calm", "happy", "sad",
#         "angry", "fearful", "disgust", "surprised"
#     ]

#     st.markdown("### Upload Audio File (.wav)")

#     uploaded_audio = st.file_uploader(
#         "Choose audio file",
#         type=["wav"],
#         key="audio_upload"
#     )

#     def extract_features(file):
#         audio, sr = librosa.load(file, duration=3, offset=0.5)

#         mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

#         max_len = 128
#         if mfcc.shape[1] < max_len:
#             pad_width = max_len - mfcc.shape[1]
#             mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
#         else:
#             mfcc = mfcc[:, :max_len]

#         return mfcc

#     if uploaded_audio is not None:

#         st.audio(uploaded_audio)

#         if st.button("🔍 Predict Emotion", key="audio_predict"):

#             mfcc = extract_features(uploaded_audio)

#             # reshape
#             mfcc = mfcc[..., np.newaxis]
#             mfcc = np.expand_dims(mfcc, axis=0)

#             prediction = model.predict(mfcc)[0]

#             # predicted_class = np.argmax(prediction)
#             le = joblib.load("models/audio/audio_label_encoder.pkl")
#             predicted_class = np.argmax(prediction)
#             emotion = le.inverse_transform([predicted_class])[0]
#             confidence = prediction[predicted_class] * 100

#             st.markdown("---")

#             st.success(
#                 f"🎯 Emotion: {emotion_labels[predicted_class]} "
#                 f"({confidence:.2f}%)"
#             )

# # =========================
# # 🎥 VIDEO MODEL
# # =========================
# elif option == "Action Recognition":

#     import tensorflow as tf
#     import cv2
#     import numpy as np
#     import tempfile

#     st.title("🎥 Human Action Recognition")

#     # Load model
#     @st.cache_resource
#     def load_video_model():
#         return tf.keras.models.load_model("models/video/action_model.keras")

#     model = load_video_model()

#     # Load class names
#     classes = np.load("models/video/action_classes.npy")

#     IMG_SIZE = 64
#     SEQUENCE_LENGTH = 10

#     st.markdown("### Upload a video file")

#     uploaded_video = st.file_uploader(
#         "Choose video",
#         type=["mp4", "avi", "mov"],
#         key="video_upload"
#     )

#     def extract_frames(video_path):
#         frames = []
#         cap = cv2.VideoCapture(video_path)

#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         skip = max(total_frames // SEQUENCE_LENGTH, 1)

#         for i in range(SEQUENCE_LENGTH):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
#             success, frame = cap.read()

#             if not success:
#                 break

#             frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
#             frame = frame / 255.0
#             frames.append(frame)

#         cap.release()
#         return np.array(frames)

#     if uploaded_video is not None:

#         # Save temp video
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_video.read())

#         st.video(uploaded_video)

#         if st.button("🔍 Predict Action", key="video_predict"):

#             frames = extract_frames(tfile.name)

#             if len(frames) != SEQUENCE_LENGTH:
#                 st.error("❌ Could not extract enough frames")
#             else:
#                 frames = np.expand_dims(frames, axis=0)

#                 prediction = model.predict(frames)[0]

#                 predicted_class = np.argmax(prediction)
#                 confidence = prediction[predicted_class] * 100

#                 st.markdown("---")

#                 st.success(
#                     f"🎯 Action: {classes[predicted_class]} "
#                     f"({confidence:.2f}%)"
#                 )

import streamlit as st

st.set_page_config(
    page_title="AI Model Hub",
    page_icon="🧠",
    layout="wide"
)

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
if option == "📱 Mobile Price":

    import joblib
    import numpy as np

    st.header("📱 Mobile Price Prediction")

    pipeline = joblib.load("models/numeric/mobile_pipeline.pkl")
    model = pipeline["model"]
    scaler = pipeline["scaler"]
    features = pipeline["features"]

    col1, col2 = st.columns(2)
    user_input = []

    for i, feature in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            val = st.number_input(feature, value=0, key=f"mobile_{i}")
            user_input.append(val)

    if st.button("🚀 Predict Price", key="mobile_btn"):
        arr = np.array(user_input).reshape(1, -1)
        arr = scaler.transform(arr)
        pred = model.predict(arr)[0]

        labels = {
            0: "Low Cost 📉",
            1: "Medium Cost 💰",
            2: "High Cost 💎",
            3: "Very High Cost 🚀"
        }

        st.success(labels[pred])

# =========================
# 🎬 TEXT MODEL
# =========================
elif option == "🎬 Movie Review":

    import joblib

    st.header("🎬 Sentiment Analysis")

    pipeline = joblib.load("models/text/sentiment_pipeline.pkl")

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

    import tensorflow as tf
    from PIL import Image
    import numpy as np

    st.header("🩺 Pneumonia Detection")

    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model("models/image/pneumonia_model.keras")

    model = load_model()

    file = st.file_uploader("Upload X-ray", type=["jpg","png"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, use_container_width=True)

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

    import tensorflow as tf
    import librosa
    import numpy as np
    import joblib

    st.header("🎧 Emotion Detection")

    @st.cache_resource
    def load_audio():
        return tf.keras.models.load_model("models/audio/audio_model.keras")

    model = load_audio()
    le = joblib.load("models/audio/audio_label_encoder.pkl")

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

    import tensorflow as tf
    import cv2
    import numpy as np
    import tempfile

    st.header("🎥 Action Recognition")

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