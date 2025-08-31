import os
import queue
import json
import warnings
import sounddevice as sd
import vosk
import pyttsx3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Suppress Hugging Face warnings
warnings.filterwarnings("ignore", message=".*Xet Storage.*")
warnings.filterwarnings("ignore", message=".*hf_xet.*")


# ==========================
# ASR Class (Vosk)
# ==========================
class ASR:
    def __init__(self, model_path="vosk-model-small-en-us-0.15", device=None):
        if not os.path.exists(model_path):
            raise RuntimeError(f"[ASR] Vosk model not found at {model_path}")
        print(f"[ASR] Loading Vosk model from {model_path}...")
        self.model = vosk.Model(model_path)
        self.q = queue.Queue()
        self.device = device

    def _callback(self, indata, frames, time, status):
        if status:
            print("[ASR] Mic status:", status)
        self.q.put(bytes(indata))

    def listen_once(self):
        samplerate = 16000
        try:
            with sd.RawInputStream(device=self.device, samplerate=samplerate,
                                   blocksize=8000, dtype="int16", channels=1,
                                   callback=self._callback):
                rec = vosk.KaldiRecognizer(self.model, samplerate)
                print("🎤 Speak now… (Press Ctrl+C to stop listening)")

                # Clear the queue first
                while not self.q.empty():
                    self.q.get()

                # Listen for a reasonable amount of time or until we get a result
                silence_count = 0
                max_silence = 50  # Adjust based on your needs

                while silence_count < max_silence:
                    try:
                        data = self.q.get(timeout=0.1)
                        if rec.AcceptWaveform(data):
                            result = json.loads(rec.Result())
                            text = result.get("text", "").strip()
                            if text:
                                print(f"[ASR] Heard: {text}")
                                return text
                        else:
                            partial = json.loads(rec.PartialResult()).get("partial", "")
                            if partial:
                                print(f"\r[ASR] Listening... {partial}", end="")
                                silence_count = 0  # Reset silence counter if we hear something
                            else:
                                silence_count += 1
                    except queue.Empty:
                        silence_count += 1
                        continue

                print("\n[ASR] Listening timeout - no speech detected")
                return ""

        except KeyboardInterrupt:
            print("\n[ASR] Listening interrupted by user")
            return ""
        except Exception as e:
            print(f"[ASR] Error during listening: {e}")
            return ""


# ==========================
# TTS Class (pyttsx3)
# ==========================
class TTS:
    def __init__(self, config):
        try:
            self.engine = pyttsx3.init()
            rate_wpm = config.get("audio", {}).get("rate_wpm", 180)
            self.engine.setProperty("rate", rate_wpm)

            voice_style = config.get("audio", {}).get("voice_style", "formal")
            voices = self.engine.getProperty('voices')
            if voice_style == "formal" and voices and len(voices) > 0:
                self.engine.setProperty('voice', voices[0].id)

            # Test TTS engine
            print("[TTS] Text-to-speech engine initialized")
        except Exception as e:
            print(f"[TTS] Warning: Could not initialize TTS engine: {e}")
            self.engine = None

    def speak(self, text):
        if self.engine is None:
            print(f"[TTS] Would speak: {text}")
            return

        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"[TTS] Error speaking: {e}")
            print(f"[TTS] Text was: {text}")


# ==========================
# LLM Class (DialoGPT)
# ==========================
class LLM_HF:
    def __init__(self, config):
        self.model_name = config.get("huggingface", {}).get("model", "microsoft/DialoGPT-medium")
        print(f"[LLM] Loading local model: {self.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.chat_history_ids = None
            print("[LLM] Model loaded successfully")
        except Exception as e:
            print(f"[LLM] Error loading model: {e}")
            raise

    def chat(self, text):
        try:
            # Encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = self.tokenizer.encode(text + self.tokenizer.eos_token, return_tensors='pt')

            # Append the new user input tokens to the chat history
            bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids],
                                      dim=-1) if self.chat_history_ids is not None else new_user_input_ids

            # Generate a response while limiting the total chat history to 1000 tokens
            self.chat_history_ids = self.model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

            # Pretty print last output tokens from bot
            reply = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                                          skip_special_tokens=True)

            return reply.strip()
        except Exception as e:
            print(f"[LLM] Error generating response: {e}")
            return "Sorry, I encountered an error while generating a response."


# ==========================
# Logger
# ==========================
class Logger:
    def __init__(self, path="session_log.jsonl"):
        self.path = path
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def log(self, role, text):
        try:
            import datetime
            entry = {"role": role, "text": text, "timestamp": datetime.datetime.now().isoformat()}
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[Logger] Error logging: {e}")


# ==========================
# Main
# ==========================
def create_default_config():
    """Create a default config.json if it doesn't exist"""
    default_config = {
        "audio": {
            "rate_wpm": 180,
            "voice_style": "formal"
        },
        "huggingface": {
            "model": "microsoft/DialoGPT-medium"
        },
        "logging": {
            "path": "session_log.jsonl",
            "save_transcript": True
        }
    }

    with open("config.json", "w") as f:
        json.dump(default_config, f, indent=2)
    print("[Config] Created default config.json")
    return default_config


if __name__ == "__main__":
    print("=== Voice Chatbot Startup ===")

    # Check dependencies first
    try:
        import sounddevice as sd

        print("✓ SoundDevice available")
    except ImportError:
        print("✗ SoundDevice not found - run: pip install sounddevice")
        exit(1)

    try:
        import vosk

        print("✓ Vosk available")
    except ImportError:
        print("✗ Vosk not found - run: pip install vosk")
        exit(1)

    try:
        import pyttsx3

        print("✓ pyttsx3 available")
    except ImportError:
        print("✗ pyttsx3 not found - run: pip install pyttsx3")
        exit(1)

    try:
        import torch

        print("✓ PyTorch available")
    except ImportError:
        print("✗ PyTorch not found - run: pip install torch")
        exit(1)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("✓ Transformers available")
    except ImportError:
        print("✗ Transformers not found - run: pip install transformers")
        exit(1)

    try:
        # Load config
        if not os.path.exists("config.json"):
            print("[Config] config.json not found, creating default...")
            config = create_default_config()
        else:
            with open("config.json") as f:
                config = json.load(f)

        # Initialize components
        print("\nInitializing components...")

        asr = ASR(model_path="vosk-model-small-en-us-0.15", device=None)
        tts = TTS(config)
        llm = LLM_HF(config)
        logger = Logger(path=config.get("logging", {}).get("path", "session_log.jsonl"))

        print("\n=== Voice Chatbot Ready ===")
        print("Type your question or press Enter to use the mic. Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("Goodbye!")
                    break

                if user_input == "":
                    user_input = asr.listen_once()

                if not user_input:
                    continue

                # Log user input
                if config.get("logging", {}).get("save_transcript", True):
                    logger.log("user", user_input)

                # Generate bot reply
                print("[Bot] Thinking...")
                bot_reply = llm.chat(user_input)
                print("Bot:", bot_reply)

                # Log bot reply
                if config.get("logging", {}).get("save_transcript", True):
                    logger.log("bot", bot_reply)

                # Speak reply
                tts.speak(bot_reply)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue

    except Exception as e:
        print(f"Fatal error during initialization: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Vosk model is downloaded and in the correct path")
        print("2. Check your microphone permissions")
        print("3. Install missing dependencies:")
        print("   pip install sounddevice vosk pyttsx3 torch transformers huggingface_hub[hf_xet]")