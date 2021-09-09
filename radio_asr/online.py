#!/usr/bin/env python3

import copy
import math
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
from nemo.collections.nlp.models.language_modeling import TransformerLMModel

import logging
logging.getLogger('nemo_logger').setLevel(logging.ERROR)


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
                 frame_len: float = 8.0,
                 frame_overlap: float = 0.1,
                 offset: int = 0,
                 rescore_beams: bool = False) -> None:
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
        self.rescore_beams = rescore_beams
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

        self.vocab = list(self.asr_model.decoder.vocabulary)
        self.vocab.append('_')

        # Set model to inference mode
        self.asr_model.eval()
        self.asr_model = self.asr_model.to(self.asr_model.device)

        # Create the beam search decoder
        self.beam_width = 32
        self.beam_search_lm = nemo_asr.modules.BeamSearchDecoderWithLM(
            vocab=list(cfg.decoder.vocabulary),
            beam_width=self.beam_width,
            alpha=1.0,
            beta=0.0,
            lm_path=None,
            num_cpus=2,
            input_tensor=False)

        if self.rescore_beams:
            self.asr_lm_model = TransformerLMModel.restore_from(
                "../models/asrlm_en_transformer_large_ls.nemo")
        else:
            self.asr_lm_model = None

    def _streaming_config(self):
        cfg = self.asr_model._cfg

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
            offset = 0
        elif len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)],
                           'constant')
            offset = 0
        else:
            offset = self.offset
        result = self._decode(frame, offset)
        return result

    def _decode(self, frame, offset=0):
        assert len(frame) == self.n_frame_len
        self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
        self.buffer[-self.n_frame_len:] = frame
        logits = self._inference(self.buffer)
        #logits = logits[:, self.n_timesteps_overlap:-self.n_timesteps_overlap, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        beams = self.beam_search_lm.forward(
            log_probs=probs, log_probs_length=None,)[0]

        if self.rescore_beams:
            scores = self._rescore(beams, max_length=64, rescore_alpha=0.4, rescore_beta=0.4)
            best_idx = np.argmax(scores[:len(beams)].cpu().numpy())
            print(f"scores: {scores}, BEST: {best_idx}")
            decoded = beams[best_idx][1]
        else:
            # Just use the top beam from the acoustic model
            decoded = beams[0][1]

        # In high-noise environments with no speaker, the model outputs sequences
        # of one-character words, all vowels. If that's all we have, we have silence.
        chars = set('aeiou _')
        if all((c in chars) for c in decoded):
            return ""
        else:
            return decoded  #[:len(decoded)-offset]

    def _rescore(self, beams, max_length, rescore_alpha, rescore_beta):
        # Create input to rescorer
        tokenizer = self.asr_lm_model.tokenizer
        max_length = 64
        c_scores = torch.empty(self.beam_width).cuda()
        c_lengths = torch.empty(self.beam_width).cuda()
        text_ids = torch.full((self.beam_width, max_length), tokenizer.pad_id)
        for i, beam in enumerate(beams):
            c_scores[i] = beam[0]
            text = beam[1]
            if len(text) == 0:
                continue
            tokens = [tokenizer.bos_id] + tokenizer.text_to_ids(text) + [tokenizer.eos_id]
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            text_ids[i, :len(tokens)] = torch.as_tensor(tokens)
            c_lengths[i] = len(tokens)
        input_mask = (text_ids != tokenizer.pad_id)

        text_ids = text_ids.cuda()
        input_mask = input_mask.cuda()

        # Rescore
        rescore_alpha = 0.4
        rescore_beta = 0.4
        log_probs = self.asr_lm_model.forward(text_ids[:, :-1], input_mask[:, :-1])
        target_log_probs = log_probs.gather(2, text_ids[:, 1:].unsqueeze(2)).squeeze(2)
        neural_lm_score = torch.sum(target_log_probs * input_mask[:, 1:], dim=-1)
        scores = c_scores + rescore_alpha * neural_lm_score + rescore_beta * c_lengths
        return scores

    @staticmethod
    def _greedy_decoder(logits, vocab):
        s = ''
        for i in range(logits.shape[0]):
            s += vocab[np.argmax(logits[i])]
        return s

    def _greedy_merge(self, s):
        s_merged = ''
        for elem in self.prev_char + s:
            if elem == self.prev_char:
                continue
            if self.prev_char in (',', '.', '?', '!') and elem != ' ':
                s_merged += ' '
            self.prev_char = elem
            s_merged += self.prev_char
        if self.prev_char in (',', '.', '?', '!'):
            s_merged += ' '
            self.prev_char = ' '
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
