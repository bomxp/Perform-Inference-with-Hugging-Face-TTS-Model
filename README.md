# Hugging Face TTS Demo with Sentence Pausing

This project demonstrates how to generate human-like speech using the [facebook/fastspeech2-en-ljspeech](https://huggingface.co/facebook/fastspeech2-en-ljspeech) model from Hugging Face and Fairseq. It includes a feature to insert **2-second pauses between sentences** for more natural audio playback.

---

## ✅ Features

- Uses Hugging Face TTS model (`fastspeech2-en-ljspeech`) with HiFiGAN vocoder
- Converts multiple sentences into speech
- Inserts 2 seconds of silence between sentences
- Saves final result as a `.wav` file
- Plays audio on macOS automatically

---

## 🧠 Requirements

- Python 3.8 or above
- Libraries:
  - `fairseq`
  - `torchaudio`
  - `soundfile`
  - `nltk`
  - `torch`
  - `numpy`

Install via pip:

```bash
pip install fairseq torchaudio soundfile nltk torch numpy
```

---

## 🚀 How to Run

1. Clone the repo or copy the script.
2. Make sure you're on **macOS** if using `os.system("open tts_output.wav")` to play audio.
3. Run the script with Python:

```bash
python main.py
```

The script will:
- Generate speech for each sentence
- Insert 2 seconds of silence between them
- Concatenate and export all to `tts_output.wav`
- Play the audio file

---

## 🔊 Example Sentences Used

```text
1. Hello! How are you doing today?
2. This is a demonstration of a text-to-speech model using Hugging Face.
3. The weather is nice and sunny. Let’s go outside!
4. Artificial Intelligence is transforming the future of communication.
5. Welcome to the AI-powered voice synthesis system!
```

---

## 📝 Notes

- If this is your first time running the script, uncomment the `nltk.download(...)` line to download required POS tagger.
- The model supports **English input only**.
- For Windows/Linux, replace `os.system("open ...")` with appropriate command to play `.wav`.

---

## 📄 License

For educational/demo purposes only. Model is hosted and licensed under Hugging Face and Facebook AI terms.