# OpenAI Whisper-Base Fine-Tuned Model for Speech-to-Text

This repository hosts a fine-tuned version of the OpenAI Whisper-Base model optimized for speech-to-text tasks using the [Mozilla Common Voice 13.0](https://commonvoice.mozilla.org/) dataset. The model is designed to efficiently transcribe speech into text while maintaining high accuracy.

## Model Details
- **Model Architecture**: OpenAI Whisper-Base  
- **Task**: Audio-to-Text  
- **Dataset**: [Mozilla Common Voice 11.0](https://commonvoice.mozilla.org/)  
- **Fine-tuning Framework**: Hugging Face Transformers  

## 🚀 Usage

### Installation
```bash
pip install transformers torch
```

### Loading the Model
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "AventIQ-AI/whisper-audio-to-text"
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
processor = WhisperProcessor.from_pretrained(model_name)
```

### Speech-to-Text Inference
```python
import torchaudio

# Load and process audio file
def load_audio(file_path, target_sampling_rate=16000):
    # Load audio file
    waveform, sample_rate = torchaudio.load(file_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != target_sampling_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sampling_rate)(waveform)

    return waveform.squeeze(0).numpy()

input_audio_path = "/kaggle/input/test-data-2/Friday 4h04m pm.m4a"  # Change this to your audio file
audio_array = load_audio(input_audio_path)

input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features
input_features = input_features.to(device)

forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

with torch.no_grad():
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

# Decode output
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(f"Transcribed Text: {transcription}")
```

## 📊 Evaluation Results
After fine-tuning the Whisper-Base model for speech-to-text, we evaluated the model's performance on the validation set from the Common Voice 11.0 dataset. The following results were obtained:

| Metric      | Score  | Meaning |
|------------|--------|------------------------------------------------|
| **WER**    | 9.2%   | Word Error Rate: Measures transcription accuracy |
| **CER**    | 5.5%   | Character Error Rate: Measures character-level accuracy |

## Fine-Tuning Details

### Dataset
The Mozilla Common Voice 11.0 dataset, containing diverse multilingual speech samples, was used for fine-tuning the model.

### Training
- **Number of epochs**: 6 
- **Batch size**: 16  
- **Evaluation strategy**: epochs
- **Learning Rate**: 5e-6

## 📂 Repository Structure
```bash
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safetensors/   # Quantized Model
├── README.md            # Model documentation
```

## ⚠️ Limitations
- The model may struggle with highly noisy or overlapping speech.
- Performance may vary across different accents and dialects.

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.

