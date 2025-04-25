import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import pipeline
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SimpleRNN as OriginalSimpleRNN, Embedding
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2
import tempfile
import os
from deepface import DeepFace
from ultralytics import YOLO
from tensorflow.keras import layers, models

@st.cache_data
def load_word_index():
    return imdb.get_word_index()

@st.cache_resource
def load_sentiment_model():
    class CustomSimpleRNN(OriginalSimpleRNN):
        def __init__(self, *args, **kwargs):
            kwargs.pop('time_major', None)
            super().__init__(*args, **kwargs)
    return load_model('simple_rnn_imdb.h5', custom_objects={'SimpleRNN': CustomSimpleRNN})

@st.cache_resource
def load_text_generator():
    return pipeline("text-generation", model="gpt2")

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

# New: Small MLP model for feature fusion
@st.cache_resource
def load_fusion_model():
    model = models.Sequential([
        layers.Input(shape=(7,)),  # 1 text score + 6 emotion scores
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output: probability of positive sentiment
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# Preprocessing

def preprocess_text(text, word_index, vocab_size=10000):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    encoded_review = [min(idx, vocab_size - 1) for idx in encoded_review]
    return sequence.pad_sequences([encoded_review], maxlen=500)

def predict_sentiment(text, model, word_index):
    preprocessed = preprocess_text(text, word_index)
    prediction = model.predict(preprocessed)
    sentiment = "Positive" if prediction[0][0] > 0.7 else "Negative"
    return sentiment, float(prediction[0][0])

def generate_explanation(text, generator):
    prompt = f"Review: {text}\nSentiment:"
    output = generator(prompt, max_length=100)[0]['generated_text']
    return output

def embedding_demo(sentence, voc_size=10000, sent_length=8, dim=10):
    one_hot_repr = [min(idx, voc_size - 1) for idx in one_hot(sentence, voc_size)]
    padded_seq = pad_sequences([one_hot_repr], maxlen=sent_length, padding='pre')
    model = tf.keras.Sequential()
    model.add(Embedding(voc_size, dim, input_length=sent_length))
    model.compile('adam', 'mse')
    embeddings = model.predict(padded_seq)
    return padded_seq, embeddings

# New: Extract full emotion scores from DeepFace analysis

def extract_emotion_scores(face_result):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    scores = [face_result['emotion'].get(e, 0) for e in emotions]
    return scores

# New: Perform feature-level fusion with text and emotion features

def feature_fusion(text_score, emotion_scores):
    features = np.array([text_score] + emotion_scores).reshape(1, -1)
    fusion_model = load_fusion_model()
    fused_output = fusion_model.predict(features)[0][0]
    if fused_output > 0.6:
        return "Strongly Positive"
    elif fused_output < 0.4:
        return "Strongly Negative"
    else:
        return "Mixed Sentiment"

# Modified: Full DeepFace output for emotions

def analyze_image(image_file, return_full_face=False):
    suffix = os.path.splitext(image_file.name)[-1]
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(image_file.read())
    tfile.flush()
    img_path = tfile.name
    tfile.close()

    yolo_model = load_yolo_model()
    yolo_results = yolo_model(img_path)[0]
    yolo_labels = [yolo_model.names[int(cls)] for cls in yolo_results.boxes.cls]

    try:
        face_result = DeepFace.analyze(img_path, actions=['emotion'], enforce_detection=False)[0]
        emotion = face_result['dominant_emotion']
    except:
        face_result = None
        emotion = "Face not detected"

    if return_full_face:
        return yolo_labels, face_result
    else:
        return yolo_labels, emotion

def analyze_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(frame_count // 10, 1)

    yolo_model = load_yolo_model()
    object_counts = {}
    emotions = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            tmp_path = os.path.join(tempfile.gettempdir(), f"frame_{frame_idx}.jpg")
            cv2.imwrite(tmp_path, frame)
            yolo_results = yolo_model(tmp_path)[0]
            for cls in yolo_results.boxes.cls:
                label = yolo_model.names[int(cls)]
                object_counts[label] = object_counts.get(label, 0) + 1
            try:
                face_result = DeepFace.analyze(tmp_path, actions=['emotion'], enforce_detection=False)[0]
                emotions.append(face_result['dominant_emotion'])
            except:
                pass
        frame_idx += 1

    cap.release()
    dominant_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    dominant_emotion = max(set(emotions), key=emotions.count) if emotions else "None"
    return dominant_objects, dominant_emotion

# Streamlit App

st.sidebar.title("Navigation")
choice = st.sidebar.selectbox("Choose a feature", [
    "Sentiment Analysis", "RAG-Based Explanation", "Word Embedding Demo", "Image Sentiment Analysis", "Video Sentiment Analysis"])

if choice == "Sentiment Analysis":
    st.title("Text Sentiment Analysis with Simple RNN")
    text_input = st.text_area("Enter Text:")
    if st.button("Analyze"):
        model = load_sentiment_model()
        word_index = load_word_index()
        sentiment, score = predict_sentiment(text_input, model, word_index)
        st.write(f"**Sentiment**: {sentiment}")
        st.write(f"**Score**: {score:.4f}")

elif choice == "RAG-Based Explanation":
    st.title("Sentiment Explanation with GPT-2")
    text_input = st.text_area("Enter Review for Explanation:")
    if st.button("Generate Explanation"):
        generator = load_text_generator()
        explanation = generate_explanation(text_input, generator)
        st.write("**Explanation**:")
        st.write(explanation)

elif choice == "Word Embedding Demo":
    st.title("Embedding Visualization")
    sentence = st.text_input("Enter a sentence:", "the glass of milk")
    if st.button("Generate Embedding"):
        padded, embedding = embedding_demo(sentence)
        st.write("**Padded Sequence**:", padded)
        st.write("**Embedding Vector**:", embedding[0])

elif choice == "Image Sentiment Analysis":
    st.title("Image-Based Sentiment Analysis with Feature Fusion")
    uploaded_img = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
    text_input = st.text_input("Enter accompanying text for fusion:")
    if uploaded_img and st.button("Analyze Image"):
        model = load_sentiment_model()
        word_index = load_word_index()
        sentiment, text_score = predict_sentiment(text_input, model, word_index)
        labels, face_result = analyze_image(uploaded_img, return_full_face=True)
        if face_result:
            emotion_scores = extract_emotion_scores(face_result)
            fusion = feature_fusion(text_score, emotion_scores)
        else:
            fusion = "Face not detected"
        st.image(uploaded_img, caption='Uploaded Image', use_column_width=True)
        st.write("**Detected Objects**:", labels)
        st.write("**Fusion Sentiment (Feature-level)**:", fusion)

elif choice == "Video Sentiment Analysis":
    st.title("Video-Based Sentiment Analysis")
    uploaded_video = st.file_uploader("Upload a Video", type=['mp4'])
    text_input = st.text_input("Enter accompanying text for fusion:")
    if uploaded_video and st.button("Analyze Video"):
        model = load_sentiment_model()
        word_index = load_word_index()
        sentiment, _ = predict_sentiment(text_input, model, word_index)
        objects, emotion = analyze_video(uploaded_video)
        fusion = fuse_sentiment(sentiment, emotion)
        st.write("**Top Detected Objects**:", objects)
        st.write("**Dominant Emotion**:", emotion)
        st.write("**Fused Sentiment**:", fusion)
