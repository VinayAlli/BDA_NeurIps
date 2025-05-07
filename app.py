import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import pipeline, AutoTokenizer, AutoModel
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
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

@st.cache_resource
def load_text_encoder():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
    return tokenizer, model

# ----------- VC-CSA Components -----------

class VideoEncoder(nn.Module):
    def __init__(self):
        super(VideoEncoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=7, padding=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x.mean(dim=2)

class FusionClassifier(nn.Module):
    def __init__(self, input_dim=768+256):
        super(FusionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2_opinion = nn.Linear(128, 3)
        self.fc2_emotion = nn.Linear(128, 8)

    def forward(self, fused):
        x = torch.relu(self.fc1(fused))
        opinion = self.fc2_opinion(x)
        emotion = self.fc2_emotion(x)
        return opinion, emotion

class VCCSAModel(nn.Module):
    def __init__(self):
        super(VCCSAModel, self).__init__()
        self.video_encoder = VideoEncoder()
        self.fusion = FusionClassifier()

    def forward(self, video_feats, text_feats):
        video_feats = video_feats.permute(0, 1, 4, 2, 3)
        video_feats = video_feats.mean(dim=[3, 4])
        video_feats = video_feats.permute(0, 2, 1)
        x_video = self.video_encoder(video_feats)
        combined = torch.cat((x_video, text_feats), dim=1)
        opinion, emotion = self.fusion(combined)
        return opinion, emotion

# ----------- Utility Functions -----------

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

def analyze_image(image_file):
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
        emotion = "Face not detected"

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

def extract_frames_from_video(video_file, max_frames=16):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(total // max_frames, 1)
    count = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % sample_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (64, 64))
            frames.append(frame)
        count += 1

    cap.release()
    frames = np.stack(frames, axis=0)
    frames = frames / 255.0
    return torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)

def extract_text_features(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

def fuse_sentiment(text_sentiment, image_emotion):
    if text_sentiment == "Positive" and image_emotion in ["happy", "surprise"]:
        return "Strongly Positive"
    elif text_sentiment == "Negative" and image_emotion in ["angry", "sad"]:
        return "Strongly Negative"
    else:
        return "Mixed Sentiment"

opinion_labels = ["Positive", "Negative", "Neutral"]
emotion_labels = ["Fear", "Disgust", "Anger", "Sadness", "Joy", "Trust", "Anticipation", "Surprise"]

# ---- Streamlit UI ----

st.sidebar.title("Navigation")
choice = st.sidebar.selectbox("Choose a feature", [
    "Sentiment Analysis", "RAG-Based Explanation", "Word Embedding Demo", "Image Sentiment Analysis", "Video Sentiment Analysis", "VC-CSA Sentiment Analysis"])

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
    st.title("Image-Based Sentiment Analysis")
    uploaded_img = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
    text_input = st.text_input("Enter accompanying text for fusion:")
    if uploaded_img and st.button("Analyze Image"):
        model = load_sentiment_model()
        word_index = load_word_index()
        sentiment, _ = predict_sentiment(text_input, model, word_index)
        labels, emotion = analyze_image(uploaded_img)
        fusion = fuse_sentiment(sentiment, emotion)
        st.image(uploaded_img, caption='Uploaded Image', use_column_width=True)
        st.write("**Detected Objects**:", labels)
        st.write("**Facial Emotion**:", emotion)
        st.write("**Fused Sentiment**:", fusion)

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

elif choice == "VC-CSA Sentiment Analysis":
    st.title("VC-CSA Based Induced Sentiment Analysis")
    video_file = st.file_uploader("Upload a Video File (.mp4)", type=['mp4'])
    comment_text = st.text_input("Enter Viewer Comment:")
    if video_file and comment_text and st.button("Analyze Sentiment (VC-CSA)"):
        with st.spinner("Processing..."):
            tokenizer, text_model = load_text_encoder()
            vccsa_model = VCCSAModel().to(device)
            vccsa_model.eval()
            frames = extract_frames_from_video(video_file)
            text_feats = extract_text_features(comment_text, tokenizer, text_model)
            opinion_logits, emotion_logits = vccsa_model(frames, text_feats)
            opinion_pred = opinion_logits.argmax(dim=1).item()
            emotion_pred = emotion_logits.argmax(dim=1).item()
            st.success(f"**Predicted Opinion:** {opinion_labels[opinion_pred]}")
            st.success(f"**Predicted Emotion:** {emotion_labels[emotion_pred]}")
