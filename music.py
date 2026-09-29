import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase
)

import cv2
import numpy as np
import mediapipe as mp

from keras.models import load_model

import webbrowser

from collections import Counter

import os


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.keras"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "labels.npy"
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model(
        MODEL_PATH,
        compile=False
    )

except Exception as e:

    st.error("❌ Could not load model.keras")
    st.code(str(e))
    st.stop()


# ============================================================
# LOAD LABELS
# ============================================================

try:

    label = np.load(
        LABEL_PATH,
        allow_pickle=True
    ).tolist()

    label = [
        x.decode("utf-8")
        if isinstance(x, bytes)
        else str(x)
        for x in label
    ]

except Exception as e:

    st.error("❌ Could not load labels.npy")
    st.code(str(e))
    st.stop()


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# ============================================================
# EMOTION PROCESSOR
# ============================================================

class EmotionProcessor(VideoTransformerBase):

    def __init__(self):

        self.holis = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Store all detected emotions
        self.emotion_history = []

        # Current emotion
        self.current_emotion = None

        # Current confidence
        self.current_confidence = 0.0


    # ========================================================
    # PROCESS CAMERA FRAME
    # ========================================================

    def transform(self, frame):

        # ----------------------------------------------------
        # GET CAMERA FRAME
        # ----------------------------------------------------

        frm = frame.to_ndarray(
            format="bgr24"
        )

        # Mirror camera
        frm = cv2.flip(
            frm,
            1
        )


        # ----------------------------------------------------
        # MEDIAPIPE
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frm,
            cv2.COLOR_BGR2RGB
        )

        results = self.holis.process(
            rgb
        )


        # Feature vector
        lst = []


        # ====================================================
        # FACE
        # ====================================================

        if results.face_landmarks:

            base = (
                results
                .face_landmarks
                .landmark[1]
            )

            for lm in (
                results
                .face_landmarks
                .landmark
            ):

                lst.extend([
                    lm.x - base.x,
                    lm.y - base.y
                ])

        else:

            cv2.putText(
                frm,
                "FACE NOT DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

            return frm


        # ====================================================
        # LEFT HAND
        # ====================================================

        if results.left_hand_landmarks:

            base = (
                results
                .left_hand_landmarks
                .landmark[8]
            )

            for lm in (
                results
                .left_hand_landmarks
                .landmark
            ):

                lst.extend([
                    lm.x - base.x,
                    lm.y - base.y
                ])

        else:

            # 21 landmarks × 2
            lst.extend(
                [0.0] * 42
            )


        # ====================================================
        # RIGHT HAND
        # ====================================================

        if results.right_hand_landmarks:

            base = (
                results
                .right_hand_landmarks
                .landmark[8]
            )

            for lm in (
                results
                .right_hand_landmarks
                .landmark
            ):

                lst.extend([
                    lm.x - base.x,
                    lm.y - base.y
                ])

        else:

            # 21 landmarks × 2
            lst.extend(
                [0.0] * 42
            )


        # ====================================================
        # CHECK FEATURE SIZE
        # ====================================================

        expected_size = model.input_shape[1]

        if len(lst) != expected_size:

            cv2.putText(
                frm,
                f"Feature error: {len(lst)} / {expected_size}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            print(
                "FEATURE SIZE ERROR:",
                len(lst),
                "EXPECTED:",
                expected_size
            )

            return frm


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        try:

            data = np.array(
                lst,
                dtype=np.float32
            ).reshape(
                1,
                -1
            )


            probabilities = model.predict(
                data,
                verbose=0
            )[0]


            # Highest probability
            pred_index = int(
                np.argmax(probabilities)
            )


            # Check label index
            if pred_index >= len(label):

                raise ValueError(
                    "Model output does not match labels.npy"
                )


            # Emotion
            detected_emotion = label[
                pred_index
            ]


            # Confidence
            confidence = float(
                probabilities[
                    pred_index
                ]
            )


            # Save current result
            self.current_emotion = (
                detected_emotion
            )

            self.current_confidence = (
                confidence
            )


            # =================================================
            # PRINT LIVE RESULT TO TERMINAL
            # =================================================

            print(
                "LIVE:",
                detected_emotion,
                "| CONFIDENCE:",
                round(confidence, 3),
                "| FEATURES:",
                len(lst)
            )


            # =================================================
            # STORE CONFIDENT PREDICTIONS
            # =================================================

            if confidence >= 0.45:

                self.emotion_history.append(
                    detected_emotion
                )


            # =================================================
            # DISPLAY ON CAMERA
            # =================================================

            cv2.putText(
                frm,
                f"Emotion: {detected_emotion}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )


            cv2.putText(
                frm,
                f"Confidence: {confidence * 100:.0f}%",
                (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


        except Exception as e:

            print(
                "PREDICTION ERROR:",
                e
            )

            cv2.putText(
                frm,
                "Prediction error",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


        # ====================================================
        # DRAW FACE
        # ====================================================

        if results.face_landmarks:

            mp_drawing.draw_landmarks(
                frm,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS
            )


        # ====================================================
        # DRAW LEFT HAND
        # ====================================================

        if results.left_hand_landmarks:

            mp_drawing.draw_landmarks(
                frm,
                results.left_hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


        # ====================================================
        # DRAW RIGHT HAND
        # ====================================================

        if results.right_hand_landmarks:

            mp_drawing.draw_landmarks(
                frm,
                results.right_hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


        return frm


# ============================================================
# STREAMLIT UI
# ============================================================

st.title(
    "🎵 Facial Emotion-Based Music Recommender"
)

st.write(
    "Detect your facial emotion and "
    "get music recommendations."
)


# ============================================================
# LANGUAGE
# ============================================================

lang = st.text_input(
    "🌐 Language",
    placeholder="Example: Tamil"
)


# ============================================================
# SINGER
# ============================================================

singer = st.text_input(
    "🎤 Singer",
    placeholder="Example: Anirudh"
)


# ============================================================
# CAMERA
# ============================================================

ctx = None


if lang and singer:

    st.info(
        "📷 Start the camera and keep your face "
        "visible for 5–10 seconds."
    )


    try:

        ctx = webrtc_streamer(

            key="emotion-detection",

            video_processor_factory=EmotionProcessor,

            media_stream_constraints={
                "video": True,
                "audio": False
            },

            async_processing=True
        )


    except Exception as e:

        st.error(
            "❌ Camera could not start."
        )

        st.code(
            str(e)
        )


# ============================================================
# RECOMMEND SONGS
# ============================================================
# ============================================================
# RECOMMEND SONGS
# ============================================================

if lang and singer:

    st.write("")

    if st.button("🎧 Recommend me songs"):

        processor = None

        try:
            if ctx is not None:
                processor = ctx.video_processor
        except Exception:
            processor = None

        if processor is None:
            st.warning("⚠️ Please start the camera first.")

        elif len(processor.emotion_history) == 0:
            st.warning("⚠️ No confident emotion detected.")
            st.write(
                "Keep your face visible for 5–10 seconds and try again."
            )

        else:

            # Take only the most recent 30 predictions
            recent_emotions = processor.emotion_history[-30:]

            emotion_counts = Counter(recent_emotions)

            # Ignore neutral when a real emotion was detected
            non_neutral = [
                e for e in recent_emotions
                if e.lower() not in ["netural", "neutral"]
            ]

            if len(non_neutral) >= 5:

                final_emotion = Counter(
                    non_neutral
                ).most_common(1)[0][0]

            else:

                final_emotion = Counter(
                    recent_emotions
                ).most_common(1)[0][0]

            # Fix model's spelling
            if final_emotion.lower() == "netural":
                final_emotion = "neutral"

            # Count final emotion
            if final_emotion == "neutral":
                final_count = emotion_counts.get("netural", 0)
                final_count += emotion_counts.get("neutral", 0)
            else:
                final_count = emotion_counts.get(
                    final_emotion,
                    0
                )

            total = len(recent_emotions)

            percentage = (
                final_count / total
            ) * 100

            # -----------------------------------------
            # FINAL RESULT
            # -----------------------------------------

            st.success(
                f"🎭 Final Emotion: "
                f"{final_emotion.upper()}"
            )

            st.info(
                f"{final_emotion} was detected in "
                f"{percentage:.0f}% of the recent predictions."
            )

            # -----------------------------------------
            # DEBUG INFORMATION
            # -----------------------------------------

            st.write("### 📊 Recent Emotion Detection")

            display_counts = Counter()

            for emotion, count in emotion_counts.items():

                clean_emotion = (
                    "neutral"
                    if emotion.lower() == "netural"
                    else emotion
                )

                display_counts[clean_emotion] += count

            st.write(dict(display_counts))

            # -----------------------------------------
            # SONG SEARCH
            # -----------------------------------------

            query = (
                f"{lang} "
                f"{final_emotion} "
                f"songs "
                f"{singer}"
            )

            st.success(
                f"🎵 Searching: {query}"
            )

            search_url = (
                "https://www.youtube.com/results?"
                "search_query="
                + query.replace(" ", "+")
            )

            webbrowser.open(search_url)