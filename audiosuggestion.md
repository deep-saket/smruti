# Audio Processing Suggestions

> TODO checklist — ordered plan. We'll implement these one-by-one; tell me which to start with.

## TODO (implement in order)
1. Upgrade STT (speech-to-text)
   - Goal: improve accuracy and latency on edge devices; support streaming inference and quantized models.
   - Mark as: [ ]

2. Upgrade TTS (text-to-speech)
   - Goal: faster, higher-quality synthesis with low latency; consider FastSpeech2 / VITS optimizations and caching.
   - Mark as: [ ]

3. Upgrade audio processing
   - Goal: robust input pre-processing (resampling, AGC, VAD, pre-emphasis) and streaming-friendly buffers.
   - Mark as: [ ]

4. Speech enhancement
   - Goal: better denoising / dereverberation pipeline (lightweight models or OpenVINO/TensorRT optimized models).
   - Mark as: [ ]

5. Pipeline optimization
   - Goal: end-to-end latency reduction (quantization, batching, hardware accel, async I/O).
   - Mark as: [ ]

6. Auto-termination of audio (endpointing)
   - Goal: replace fixed-duration recording with robust voice endpointing (VAD + energy + hysteresis + max-duration guard).
   - Mark as: [ ]

---

## Identified Classes for Audio Processing

### 1. AudioToAudioComponent
- **File**: `common/AudioToAudioComponent.py`
- **Description**: Base class for all audio-to-audio processing components.
- **Subclasses**:
  - **SpeechBrainInfer** (`models/SpeechBrainInfer.py`):
    - Uses SpeechBrain's SpectralMaskEnhancement models for speech denoising.
    - Supports models like MetricGAN+, SepFormer, and Conv-TasNet.
    - Input: Mono, 16kHz, float32 audio.
    - Output: Denoised audio.
  - **DeepFilterNet3Infer** (`models/DeepFilterNet3Infer.py`):
    - Uses DeepFilterNet 3 for speech enhancement, optimized with OpenVINO.
    - Input: 48kHz mono audio.
    - Output: Denoised audio.

### 2. InferenceAudioComponent
- **File**: `common/InferenceAudioComponent.py`
- **Description**: Base class for all audio inference components.
- **Subclasses**:
  - **KyutaiSTTInfer** (`models/KyutaiSTTInfer.py`):
    - Implements Speech-to-Text (STT) using the Moshi model.
    - Uses a streaming approach for real-time transcription.
  - **OpenAIWhisperTinyInfer** (`models/OpenAIWhisperTinyInfer.py`):
    - Implements STT using OpenAI's Whisper Tiny model.
    - Uses Hugging Face Transformers for model loading and inference.

### 3. InferenceTTSComponent
- **File**: `common/InferenceTTSComponent.py`
- **Description**: Base class for all Text-to-Speech (TTS) inference components.
- **Subclasses**:
  - **VITSTTSInfer** (`models/VITSTTSInfer.py`):
    - Implements TTS using the VITS model.
    - Supports multi-speaker synthesis and GPU acceleration.

### 4. ModelManager
- **File**: `models/ModelManager.py`
- **Description**: Dynamically loads and manages model instances.
  - Reads model class names from `settings["agent"]["models"]`.
  - Instantiates models with configurations from `settings["models"]`.

## Best Practices for Edge Devices

### Speech-to-Text (STT)
- **Recommended Models**:
  - Whisper Tiny
  - Conformer
- **Optimization Techniques**:
  - Quantization: Reduces model size and improves inference speed.
  - Pruning: Removes redundant parameters to optimize performance.
- **Implementation Tips**:
  - Use streaming inference for real-time transcription.
  - Preprocess audio to remove noise using lightweight libraries like WebRTC.

### Text-to-Speech (TTS)
- **Recommended Models**:
  - FastSpeech2
  - VITS
- **Optimization Techniques**:
  - Use pre-trained phoneme embeddings to reduce computational overhead.
  - Quantize models for deployment on edge devices.
- **Implementation Tips**:
  - Implement caching for frequently used phrases to reduce latency.

### Audio Processing
- **Techniques**:
  - Use real-time noise suppression libraries like WebRTC.
  - Employ lightweight neural networks for audio denoising.
- **Tools**:
  - PyTorch or TensorFlow Lite for deploying optimized models.

## Observations and Suggestions for Improvement

### Speech Enhancement
- **Current Approach**:
  - SpeechBrain and DeepFilterNet3 are used for denoising.
- **Suggestions**:
  - Use quantized versions of these models for edge devices.
  - Consider lightweight alternatives like RNNoise for real-time noise suppression.

### Speech-to-Text (STT)
- **Current Approach**:
  - Whisper Tiny and Moshi models are used.
- **Suggestions**:
  - Optimize Whisper Tiny using ONNX or TensorFlow Lite for faster inference.
  - Evaluate Conformer models for better accuracy on edge devices.

### Text-to-Speech (TTS)
- **Current Approach**:
  - VITS is used for TTS.
- **Suggestions**:
  - Quantize the VITS model to reduce memory usage.
  - Use FastSpeech2 for faster synthesis on edge devices.

### Pipeline Optimization
- Implement streaming inference for both STT and TTS to reduce latency.
- Use OpenVINO or TensorRT for hardware-accelerated inference.

## Recommendations for Incorporation

1. **Model Selection**:
   - Use quantized versions of Whisper Tiny for STT and FastSpeech2 for TTS.
   - Evaluate the performance of these models on edge hardware.

2. **Pipeline Optimization**:
   - Implement streaming inference for real-time STT and TTS.
   - Optimize audio preprocessing to minimize latency.

3. **Testing and Benchmarking**:
   - Test models on target edge devices to ensure acceptable latency and accuracy.
   - Benchmark performance with and without optimizations (e.g., quantization).

4. **Integration**:
   - Integrate the optimized models into the `InferenceAudioComponent` and `InferenceTTSComponent` classes.
   - Ensure compatibility with the existing `AudioToAudioComponent` pipeline.

---

This document provides a detailed analysis of the current audio processing pipeline and recommendations for improvement. Let me know how you'd like to proceed or if further details are needed.
