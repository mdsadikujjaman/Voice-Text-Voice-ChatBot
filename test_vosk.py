import os, queue, json
import sounddevice as sd
import vosk

# ✅ Path to your model folder (adjust if needed)
model_path = "vosk-model-small-en-us-0.15"

if not os.path.exists(model_path):
    raise RuntimeError("Model folder not found! Check path.")

print("Loading Vosk model...")
model = vosk.Model(model_path)

# Audio settings
samplerate = 16000
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

# ✅ Show all audio devices to pick the right mic
print("\nAvailable audio devices:")
print(sd.query_devices())

# 👉 Change 'device=None' to the correct mic index (from sd.query_devices())
with sd.RawInputStream(device=None, samplerate=samplerate, blocksize=8000,
                       dtype="int16", channels=1, callback=callback):

    rec = vosk.KaldiRecognizer(model, samplerate)
    print("\n🎤 Speak now (Ctrl+C to stop)...\n")

    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "").strip()
            if text:
                print("✅ You said:", text)
        else:
            partial = json.loads(rec.PartialResult()).get("partial", "")
            if partial:
                print("...hearing:", partial)
