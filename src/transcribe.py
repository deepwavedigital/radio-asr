#!/usr/bin/env python3

import pathlib
import time
import wave

# NeMo's "core" package
import nemo
# NeMo's ASR collection - this collections contains complete ASR models and
# building blocks (modules) for ASR
import nemo.collections.asr as nemo_asr

# This line will download pre-trained QuartzNet15x5 model from NVIDIA's NGC cloud and instantiate it for you
quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

audio_files = list(pathlib.Path('.').glob('*.wav'))

print("Audio files: {}".format(audio_files))

fname = '/tmp/test.wav'

for wav in audio_files:
    wav_name = str(wav)

    with wave.open(wav_name, 'r') as wave_data:
        frames = wave_data.getnframes()
        rate = wave_data.getframerate()
        wav_duration = frames / float(rate)

    t_start = time.time()
    transcription = quartznet.transcribe(paths2audio_files=[wav_name])
    t_end = time.time()
    t_duration = t_end - t_start

    speedup = wav_duration / t_duration
    print("---")
    print(f"{wav_name}: duration {wav_duration:.3f} sec, {t_duration:.3f} sec to transcribe ({speedup:.3f}x)")
    print(f"Audio was recognized as: {transcription}")
    print("---")
