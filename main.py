import os
import nltk
import soundfile as sf
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

# Tải POS tagger cần thiết cho g2p_en (1 lần)
nltk.download('averaged_perceptron_tagger_eng')

# Step 1: Load model from Hugging Face Hub
models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False}
)
model = models[0]
model.eval()

# Step 2: Prepare generator
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(models, cfg)

# Step 3: Input your custom text
text = "Welcome to Hugging Face text to speech demo. This is a test using the VITS model."

# Step 4: Convert text to waveform
sample = TTSHubInterface.get_model_input(task, text)
waveform, sample_rate = TTSHubInterface.get_prediction(task, model, generator, sample)

# Step 5: Play audio directly
# ipd.display(ipd.Audio(waveform, rate=sample_rate))

# : Save as WAV file
sf.write("tts_output.wav", waveform, sample_rate)

"""
    Generate speech from text using espnet VITS model and play on macOS.
    """
# sample = TTSHubInterface.get_model_input(task, text)
# wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
# sf.write(filename, wav, rate)
os.system(f"open tts_output.wav")  # Phát âm thanh trên macOS
