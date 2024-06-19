import streamlit as st
import pyaudio
import wave
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from io import BytesIO
import tensorflow as tf
import pywt
import warnings
import time
from transformers import pipeline
import speech_recognition as sr

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Emotion labels
emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}

# Initialize PyAudio parameters
RATE = 24414
CHUNK = 512
RECORD_SECONDS = 7.1
FORMAT = pyaudio.paInt32
CHANNELS = 1

# Load emotion detection model
emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Load saved LSTM model for audio emotion prediction
saved_model_path = 'gru_model.json'
saved_weights_path = 'gru_model_weights.weights.h5'

with open(saved_model_path, 'r') as json_file:
    json_savedModel = json_file.read()

model_audio = tf.keras.models.model_from_json(json_savedModel)
model_audio.load_weights(saved_weights_path)

# Functions for feature extraction
def extract_mfcc(y, sr):
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

def extract_wavelet_features(y):
    coeffs = pywt.wavedec(y, 'db4', level=5)
    cA = coeffs[0]
    cD = coeffs[1:]
    features = [np.mean(cA), np.std(cA)]
    for detail in cD:
        features.append(np.mean(detail))
        features.append(np.std(detail))
    return features

def extract_features(wav_file_name):
    y, sr = librosa.load(wav_file_name)
    mfcc_features = extract_mfcc(y, sr)
    wavelet_features = extract_wavelet_features(y)
    combined_features = np.concatenate((mfcc_features, wavelet_features))
    return combined_features

def predict_emotion_audio(wav_filepath):
    test_point = extract_features(wav_filepath)
    test_point = np.reshape(test_point, newshape=(1, 52, 1))
    predictions = model_audio.predict(test_point)
    predicted_emotion = emotions[np.argmax(predictions[0]) + 1]
    return predicted_emotion

def recognize_and_predict_text(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data)
            st.write("You said:", text)
            emotion_result = emotion_pipe(text)[0]["label"]
            return text, emotion_result
        except sr.UnknownValueError:
            st.write("Sorry, I could not understand the audio.")
            return None, None
def linechart(emotion_live):
    time_intervals = [i*7 for i in range(len(emotion_live))]
    emotion_values = {'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3, 'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7}
    
    # Convert emotions to numerical values
    emotion_numerical = [emotion_values[emotion] for emotion in emotion_live]
    
    # Generate the line chart
    plt.figure(figsize=(10, 6))
    plt.plot(time_intervals, emotion_numerical, marker='o')

    # Customize the plot
    plt.yticks(list(emotion_values.values()), list(emotion_values.keys()))
    plt.xlabel('Time (s)')
    plt.ylabel('Emotion')
    plt.title('Emotion Variation Over Time')
    plt.grid(True)

    # Display the plot using Streamlit
    st.pyplot(plt)



# Audio recording function
def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    st.write("Recording...")
    frames = []
    for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    st.write("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    output_file = "audio.wav"
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return output_file

# Initialize Streamlit interface
def show():
    st.title("Magnus")
    st.write("Experience the power of emotional acoustics!")
    
    if "stop_session" not in st.session_state:
        st.session_state.stop_session = False

    if "emotion_counts" not in st.session_state:
        st.session_state.emotion_counts = {emotion: 0 for emotion in emotions.values()}

    if "emotion_live" not in st.session_state:
        st.session_state.emotion_live = []
        
    def main():
        start_recording = st.button("Start Session")
        stop_recording = st.button("Stop Session")
        
        if start_recording:
            st.session_state.stop_session = False
            st.session_state.emotion_counts = {emotion: 0 for emotion in emotions.values()}
            st.session_state.emotion_live = []
            while not st.session_state.stop_session:
                audio_file = record_audio()
                
                # Load audio data for visualization and playback
                y, sr = librosa.load(audio_file, sr=None)
                plt.figure(figsize=(10, 4))
                librosa.display.waveshow(y, sr=sr, color='b')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.title('Audio Waveform')
                buf = BytesIO()
                plt.savefig(buf, format="png")
                st.image(buf)
                
                # Predict emotion from audio
                predicted_emotion_audio = predict_emotion_audio(audio_file)
                #st.write("Predicted Emotion:", predicted_emotion_audio)
                st.subheader(f"Emotion: {predicted_emotion_audio} ")
                st.session_state.emotion_counts[predicted_emotion_audio] += 1
                st.session_state.emotion_live.append(predicted_emotion_audio)
                
        
        if stop_recording:
            st.session_state.stop_session = True
            col1, col2 = st.columns(2)
            with col1:
                linechart(st.session_state.emotion_live)
            with col2:
                fig, ax = plt.subplots()
                filtered_emotion_counts = {emotion: count for emotion, count in st.session_state.emotion_counts.items() if count > 0}
                ax.pie(filtered_emotion_counts.values(), labels=filtered_emotion_counts.keys(), autopct='%1.1f%%')
                ax.axis('equal')
                plt.title('Emotion Distribution')
                st.pyplot(fig)

    
    main()

show()
