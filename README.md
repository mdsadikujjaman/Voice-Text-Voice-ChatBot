# 🎤 Voice Txt Voice Chatbot


A conversational voice-enabled chatbot that combines:
- Vosk (offline ASR for speech-to-text)
- Microsoft DialoGPT-medium (via Hugging Face for dialogue)
- gTTS / pyttsx3 (for text-to-speech replies)


# 🚀 Features

- Speech-to-Text (ASR): Offline transcription with Vosk.
- Dialogue Generation (LLM): Conversational responses with DialoGPT-medium.
- Text-to-Speech (TTS): Speaks replies using gTTS or pyttsx3.
- Configurable: Choose TTS engine, model, mic device.
- Offline ASR: Works without internet for speech recognition.

# 📂 Project Structure


- │── main.py
- │── requirements.txt
- │── README.md
- │── vosk-model-small-en-us-0.15/

# ⚙️ Installation

1. Clone the project
   Download all files.

2. Install dependencies
   python -m pip install -r requirements.txt

   Or manually:
   pip install torch transformers vosk sounddevice pyttsx3 gTTS pygame

3. Download the Vosk Model
   - Download vosk-model-small-en-us-0.15 from:
     https://alphacephei.com/vosk/models
   - Extract folder into project root.

# 📌 Visualizations

## 1st Step
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/74bb97f7-e294-40af-8e2f-adcb426cea71" />

## 2nd Step
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/57fe4be8-0a4d-4dc6-a188-d7024dce5cf2" />

## 3rd Step
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/8b787a39-0a36-4bc2-a8c7-18b5aeab7798" />

## 4th Step
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/3920e32f-1341-4ca1-9457-ddf04e359476" />

# 🛠 Troubleshooting

- No speech detected → run sd.query_devices() and update mic device index in main.py.
- Bot replies are slow → use a smaller model like microsoft/DialoGPT-small.
- No sound output → switch to "gtts" in main.py.
- Hugging Face error → ensure internet connection is available.

# 📌 Future Improvements

- Add multilingual support with different Vosk models.
- Enhance conversation with memory and tutoring-style explanations.
- Build a simple GUI or web interface.

# 👨‍💻 Author

- Developed by Md. Sadikujjaman
- Powered by Vosk ASR + Hugging Face DialoGPT

- ## [GitHub](https://www.linkedin.com/in/sadikujjaman/)
- ## [Kaggle](https://www.kaggle.com/mdsadikujjamanshihab)
==========================
