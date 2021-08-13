#!/usr/bin/env python3

import pathlib
import time
import wave

# NeMo's "core" package
import nemo
# NeMo's ASR collection - this collections contains complete ASR models and
# building blocks (modules) for ASR
import nemo.collections.asr as nemo_asr

# Import post-processing models
from nemo.collections.nlp.models import PunctuationCapitalizationModel

# This line will download pre-trained QuartzNet15x5 model from NVIDIA's NGC cloud and instantiate it for you
#quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
# Load quartznet from a locally downloaded .nemo file
#quartznet = nemo_asr.models.EncDecCTCModel.restore_from("models/QuartzNet15x5NR-En.nemo")
# Use a conformer CTC model
quartznet = nemo.collections.asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_large")

# Download and load the pre-trained BERT-based model
punct_model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_distilbert")

audio_files = list(pathlib.Path('.').glob('*.wav'))

print("Audio files: {}".format(audio_files))

for wav in audio_files:
    wav_name = str(wav)

    with wave.open(wav_name, 'r') as wave_data:
        frames = wave_data.getnframes()
        rate = wave_data.getframerate()
        wav_duration = frames / float(rate)

    t_start = time.time()
    raw_transcript = quartznet.transcribe(paths2audio_files=[wav_name])
    punct_transcript = punct_model.add_punctuation_capitalization(raw_transcript)
    t_end = time.time()
    t_duration = t_end - t_start

    speedup = wav_duration / t_duration
    print("---")
    print(f"{wav_name}: duration {wav_duration:.3f} sec, {t_duration:.3f} sec to transcribe ({speedup:.3f}x)")
    print(f"Audio was recognized as: {raw_transcript}")
    print(f"After punctuation: {punct_transcript}")
    print("---")
