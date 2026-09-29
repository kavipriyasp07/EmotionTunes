# EmotionTunes

Facial-expression based music recommendation system.

## Run on Windows

Use **Python 3.11**.

```bat
cd C:\Users\admin\Downloads\emotionmusic-ready
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run music.py
```

## How it works

1. Enter Language and Singer.
2. Click START on the webcam.
3. Keep your face visible for 5–10 seconds.
4. The model collects confident emotion predictions.
5. Click STOP.
6. The session's most frequent emotion becomes the final emotion.
7. The system maps that emotion to a music category and opens a YouTube search.

## Model loading fix

The supplied model was saved with Keras 3.9.2. This project does **not** call `load_model()` on the supplied `.keras`/`.h5` model. Instead, it rebuilds the exact Dense network used by the original training script and loads the numeric weights from `model.h5` with `h5py`. This avoids the Keras name-scope `pop` loading error.

Keep these files in the same directory as `music.py`:

- `model.h5`
- `model.keras` (kept as the original model artifact; not required by the runtime loader)
- `labels.npy`
- `music.py`

`model.h5` is used by the runtime loader.
