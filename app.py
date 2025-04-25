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
from tensorflow.keras import layers, models
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Set device
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

@st.cache_resource
def load_fusion_model():
    model = models.Sequential([
        layers.Input(shape=(7,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

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

class ConsensusTransformer(nn.Module):
    def __init__(self, hidden_dim=256):
        super(ConsensusTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        return self.transformer(x)

class GoldenGrounding(nn.Module):
    def __init__(self, hidden_dim=256):
        super(GoldenGrounding, self).__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.relu(out)
        return out.mean(dim=1)

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
        self.text_encoder_dim = 768
        self.consensus = ConsensusTransformer()
        self.golden = GoldenGrounding()
        self.fusion = FusionClassifier()

    def forward(self, video_feats, text_feats):
        x_video = self.video_encoder(video_feats.permute(0, 2, 1))
        x_text = text_feats
        combined = torch.cat((x_video, x_text), dim=1)
        opinion, emotion = self.fusion(combined)
        return opinion, emotion

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

opinion_labels = ["Positive", "Negative", "Neutral"]
emotion_labels = ["Fear", "Disgust", "Anger", "Sadness", "Joy", "Trust", "Anticipation", "Surprise"]

# Streamlit App
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
