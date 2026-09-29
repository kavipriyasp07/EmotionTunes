# 🎵 EmotionTunes

### Facial Emotion-Based Music Recommendation System

EmotionTunes is an AI-powered music recommendation system that detects a user's facial emotion in real time and recommends music based on their mood, preferred language, and singer.

## ✨ Features

- 🎥 Real-time webcam emotion detection
- 🧠 AI-based emotion classification
- 📍 MediaPipe facial & hand landmark detection
- 🎭 Final emotion detection from multiple predictions
- 🌐 Language preference
- 🎤 Singer preference
- 🎵 Mood-based YouTube music search
- 💻 Streamlit web interface

## 🏗️ How It Works

```text
Webcam
   ↓
MediaPipe Holistic
   ↓
Facial & Hand Landmarks
   ↓
1020 Feature Vector
   ↓
Keras Emotion Model
   ↓
Final Emotion
   ↓
Language + Singer
   ↓
YouTube Music Search
🛠️ Tech Stack
Python
TensorFlow / Keras
MediaPipe
OpenCV
Streamlit
Streamlit-WebRTC
NumPy
📂 Project Structure
EmotionTunes/
├── music.py
├── model.keras
├── model.h5
├── labels.npy
├── requirements.txt
├── run.bat
├── README.md
└── .gitignore
🚀 Run Locally
1. Clone
git clone https://github.com/kavipriyasp07/EmotionTunes.git
cd EmotionTunes
2. Create environment
python -m venv .venv
.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Run
streamlit run music.py

Open the Streamlit URL shown in the terminal.

🎯 Emotion Pipeline

The system processes:

Face + Hand Landmarks → 1020 Features → Emotion Model → Final Emotion → Music Search

The current model contains seven output labels:

disgust, happy, mad, netural, rock, sad, surprise

Note: netural is the existing label stored in the trained model.

🔮 Future Improvements
Improve emotion recognition accuracy
Better happy/sad/neutral classification
Verified mood-based song database
Spotify / YouTube Music integration
Personalized playlists
Online deployment
👩‍💻 Author

**Kavipriya SP
Artificial Intelligence and Data Science**

GitHub

⭐ EmotionTunes — Detect the emotion. Discover the music.


Then save `README.md` and run:

```powershell
git add README.md
git commit -m "Update professional README"
git push
