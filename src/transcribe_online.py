#!/usr/bin/env python3

import copy
import pathlib
import time
import wave

import numpy as np
import omegaconf
import torch

# NeMo's "core" package
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType

# NeMo's ASR collection - this collections contains complete ASR models and
# building blocks (modules) for ASR
import nemo.collections.asr as nemo_asr

# Import post-processing models
from nemo.collections.nlp.models import PunctuationCapitalizationModel


class AudioDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        """Data layer to pass audio signal.
           Workflow: call ADL.set_signal(signal), then run inference
        """
        super().__init__()
        self._sample_rate = sample_rate
        self.signal_tensor = None
        self.shape_tensor = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.signal_tensor is None:
            raise StopIteration
        result = (self.signal_tensor, self.shape_tensor)
        self.signal_tensor = None
        self.shape_tensor = None
        return result

    def set_signal(self, signal):
        self.signal_tensor = torch.as_tensor(signal, dtype=torch.float32)
        self.shape_tensor = torch.as_tensor(signal.size, dtype=torch.int64)

    def __len__(self):
        return 1


class SpeechInference():
    """Wraps creation and configuration of model objects for inference.

       1) Call reset() method to reset internal history of decoded letters
       2) Call transcribe(frame) repeatedly to do ASR on frames of a
          contiguous signal
    """

    def __init__(self,
                 sample_rate: float = 16000.0,
                 frame_len: float = 2.0,
                 frame_overlap: float = 2.5,
                 offset: int = 4) -> None:
        """
        Args:
          sample_rate: sample rate of input signal, Hz
          frame_len: frame's duration, sec
          frame_overlap: duration of overlaps before/after current frame, sec
          offset: number of symbols to drop for smooth streaming
        """
        self.sample_rate = sample_rate
        self.frame_len = frame_len
        self.frame_overlap = frame_overlap
        self.offset = offset

        # a) This line will download pre-trained QuartzNet15x5 model from
        #    NVIDIA's NGC cloud and instantiate it for you
        self.asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
             model_name="QuartzNet15x5Base-En")
        # b) Load quartznet from a locally downloaded .nemo file
        #self.asr_model = nemo_asr.models.EncDecCTCModel.restore_from(
        #    "models/QuartzNet15x5NR-En.nemo")
        # c) Use a conformer CTC model (download from NGC cloud)
        #self.asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        #    model_name="stt_en_conformer_ctc_large")
        ## WARNING: the conformer CTC model needs a different _streaming_config()
        ##          function, not yet implemented. When testing offline it seemed
        ##          to produce better translations in noisy data, so we want to look
        ##          at this some point soon.

        # Download and load the pre-trained BERT-based model
        self.punct_model = PunctuationCapitalizationModel.from_pretrained(
            "punctuation_en_distilbert")

        self.data_layer = AudioDataLayer(sample_rate=self.sample_rate)
        self.data_loader = torch.utils.data.DataLoader(
            self.data_layer,
            batch_size=1,
            collate_fn=self.data_layer.collate_fn)

        self._inference_config()
        self._streaming_config()

    def _inference_config(self):
        # Preserve a copy of the full config
        cfg = copy.deepcopy(self.asr_model._cfg)
        print(omegaconf.OmegaConf.to_yaml(cfg))

        # Make config overwrite-able
        omegaconf.OmegaConf.set_struct(cfg.preprocessor, False)

        # some changes for streaming scenario
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0

        # Disable config overwriting
        omegaconf.OmegaConf.set_struct(cfg.preprocessor, True)

        self.asr_model.preprocessor = \
            self.asr_model.from_config_dict(cfg.preprocessor)

        # Set model to inference mode
        self.asr_model.eval()
        self.asr_model = self.asr_model.to(self.asr_model.device)

    def _streaming_config(self):
        cfg = self.asr_model._cfg
        self.vocab = list(cfg.decoder.vocabulary)
        self.vocab.append('_')

        timestep_duration = cfg.preprocessor['window_stride']
        for block in cfg.encoder['jasper']:
            timestep_duration *= block['stride'][0] ** block['repeat']

        self.n_frame_len = int(self.frame_len * self.sample_rate)
        self.n_frame_overlap = int(self.frame_overlap * self.sample_rate)
        self.n_timesteps_overlap = int(self.frame_overlap / timestep_duration) - 2
        self.buffer = np.zeros(shape=2*self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
        self.reset()

    def reset(self):
        """Reset frame_history and streaming decoder's state"""
        self.buffer[:] = 0.0
        self.prev_char = ''

    @torch.no_grad()
    def transcribe(self, frame=None):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)],
                           'constant')
        return self._greedy_merge(self._decode(frame, self.offset))

    def _decode(self, frame, offset=0):
        assert len(frame) == self.n_frame_len
        self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
        self.buffer[-self.n_frame_len:] = frame
        logits = self._inference(self.buffer).cpu().numpy()[0]
        decoded = self._greedy_decoder(
            logits[self.n_timesteps_overlap:-self.n_timesteps_overlap],
            self.vocab
        )
        return decoded[:len(decoded)-offset]

    @staticmethod
    def _greedy_decoder(logits, vocab):
        s = ''
        for i in range(logits.shape[0]):
            s += vocab[np.argmax(logits[i])]
        return s

    def _greedy_merge(self, s):
        s_merged = ''
        for i in range(len(s)):
            if s[i] != self.prev_char:
                self.prev_char = s[i]
                if self.prev_char != '_':
                    s_merged += self.prev_char
        return s_merged

    # inference method for audio samples (single instance)
    def _inference(self, input_samples):
        self.data_layer.set_signal(input_samples)
        batch = next(iter(self.data_loader))
        signal, signal_len = batch
        signal = signal.to(self.asr_model.device)
        signal_len = signal_len.to(self.asr_model.device)
        log_probs, encoded_len, predictions = self.asr_model.forward(
            input_signal=signal, input_signal_length=signal_len)
        return log_probs


speech = SpeechInference(sample_rate=16000.0)
# Get calculated number of samples to pass to each transcribe() call
num_samples = speech.n_frame_len

audio_files = list(pathlib.Path('.').glob('*.wav'))

print("Audio files: {}".format(audio_files))

for wav in audio_files:
    wav_name = str(wav)

    with wave.open(wav_name, 'r') as wave_data:
        frames = wave_data.getnframes()
        rate = wave_data.getframerate()
        wav_duration = frames / float(rate)

        audio_buf = wave_data.readframes(frames)
        audio = np.frombuffer(audio_buf, dtype=np.int16)
        # Normalize to float32, range (-1.0, +1.0)
        audio = audio.astype(np.float32) * (1.0 / 2**15)
        print(audio.shape)

        # Clip to number of full ASR model input frames and reshape
        num_model_frames = int(len(audio) / num_samples)
        audio = audio[:num_samples*num_model_frames].reshape(
            num_model_frames, num_samples)
        print(num_samples)
        print(audio.shape)

    empty_counter = 0
    raw_transcript = ""
    speech.reset()
    t_start = time.time()

    for frame in audio:
        text = speech.transcribe(frame)
        if len(text):
            raw_transcript += text
            empty_counter = speech.offset
        elif empty_counter > 0:
            empty_counter -= 1
            if empty_counter == 0:
                raw_transcript += " "
    # Pad with zero frames to get the last bit of transcription
    for _ in range(speech.offset):
        text = speech.transcribe(None)
        if len(text):
            raw_transcript += text
            empty_counter = speech.offset
        elif empty_counter > 0:
            empty_counter -= 1
            if empty_counter == 0:
                raw_transcript += " "
    t_end = time.time()
    t_duration = t_end - t_start

    speedup = wav_duration / t_duration
    print("---")
    print(f"{wav_name}: duration {wav_duration:.3f} sec, {t_duration:.3f} sec to transcribe ({speedup:.3f}x)")
    print(f"Audio was recognized as: {raw_transcript}")
    print("---")
