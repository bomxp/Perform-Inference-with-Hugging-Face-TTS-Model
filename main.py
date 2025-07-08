import os
import nltk
import torch
import soundfile as sf
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

# Download POS tagger required by g2p_en (only once)
# nltk.download('averaged_perceptron_tagger_eng')

# Step 1: Load the pre-trained TTS model from Hugging Face
models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False}
)
model = models[0]
model.eval()

# Step 2: Prepare TTS generator
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(models, cfg)

# Step 3: Define your input text (split by line for 2s pause between sentences)
sentences = [
    "Hello! How are you doing today?",
    "This is a demonstration of a text-to-speech model using Hugging Face.",
    "The weather is nice and sunny. Let’s go outside!",
    "Artificial Intelligence is transforming the future of communication.",
    "Welcome to the AI-powered voice synthesis system!"
]

# Step 4: Convert each sentence to audio and add 2-second pauses
all_waveforms = []
for sentence in sentences:
    sample = TTSHubInterface.get_model_input(task, sentence)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    all_waveforms.append(torch.tensor(wav))

    # Append 2 seconds of silence (zeros)
    silence = torch.zeros(int(rate * 2.0))
    all_waveforms.append(silence)

# Concatenate all waveforms
final_waveform = torch.cat(all_waveforms).numpy()

# Step 5: Save final output to WAV file
sf.write("tts_output.wav", final_waveform, rate)

# Step 6: Play the audio file on macOS
os.system("open tts_output.wav")
