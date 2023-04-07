import os.path
import re
import subprocess
import tempfile
from typing import List, Tuple, Union

import librosa
import nltk
import numpy as np
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


TARGET_SAMPLE_RATE: int = 16_000
MODEL_NAME = 'cointegrated/rut5-base-paraphraser'


def paraphrase_base(text: str, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration,
                    beams: int = 30, grams: int = 4, variants: int = 10) -> List[str]:
    re_for_digit = re.compile(r'\d+')
    x = tokenizer([text for _ in range(variants)],
                  return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 1.5 + 10)
    out = model.generate(**x, encoder_no_repeat_ngram_size=grams, early_stopping=True,
                         num_beams=beams, max_length=max_size, no_repeat_ngram_size=4,
                         num_return_sequences=variants)
    paraphrases = [tokenizer.decode(out[variant_idx], skip_special_tokens=True)
                   for variant_idx in range(variants)]
    normalized_paraphrases = {
        ' '.join(list(
            filter(
                lambda it2: len(it2) > 0,
                map(
                    lambda it1: it1.lower().strip(),
                    nltk.wordpunct_tokenize(text)
                )
            )
        ))
    }
    filtered_paraphrases = []
    for v in paraphrases:
        k = ' '.join(list(
            filter(
                lambda it2: len(it2) > 0,
                map(
                    lambda it1: it1.lower().strip(),
                    nltk.wordpunct_tokenize(v)
                )
            )
        ))
        if (k not in normalized_paraphrases) and (re_for_digit.search(k) is None):
            normalized_paraphrases.add(k)
            filtered_paraphrases.append(v)
    return filtered_paraphrases


def generate_sound(text: str, voice: str) -> np.ndarray:
    sound_fname = ''
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.wav', delete=False) as fp:
            sound_fname = fp.name
        cmd = f'echo "{text}" | RHVoice-client -s {voice}+CLB > "{sound_fname}"'
        retcode = subprocess.run(cmd, shell=True, encoding='utf-8').returncode
        err_msg = f'The text "{text}" cannot be synthesized into ' \
                  f'the file "{sound_fname}" with RHVoice\'s voice "{voice}"!'
        if retcode != 0:
            raise ValueError(err_msg)
        if not os.path.isfile(sound_fname):
            raise ValueError(err_msg)
        new_sound, rate = librosa.load(sound_fname)
        if len(new_sound.shape) > 1:
            new_sound = librosa.to_mono(new_sound)
        if rate != TARGET_SAMPLE_RATE:
            new_sound = librosa.resample(new_sound, orig_sr=rate, target_sr=TARGET_SAMPLE_RATE,
                                         res_type='kaiser_best', scale=True)
        new_sound = np.round(new_sound * 32767.0)
        err_msg += ' The synthetic sound is wrong!'
        if np.max(np.abs(new_sound)) > 32768.0:
            raise ValueError(err_msg)
        if np.min(new_sound) >= 0:
            raise ValueError(err_msg)
        if new_sound.shape[0] <= (TARGET_SAMPLE_RATE // 4):
            err_msg += f' It is too short! Expected greater than {TARGET_SAMPLE_RATE // 4}, got {new_sound.shape[0]}.'
            raise ValueError(err_msg)
        new_sound[new_sound > 32767] = 32767
        new_sound = new_sound.astype(np.int16)
    finally:
        if os.path.isfile(sound_fname):
            os.remove(sound_fname)
    return new_sound


def generate_sounds_using_different_voices(text: str) -> List[Union[np.ndarray, str]]:
    voices = ['Anna', 'Elena', 'Irina', 'Aleksandr', 'Artemiy']
    sounds = []
    for cur_voice in voices:
        try:
            new_sound = generate_sound(text, cur_voice)
        except BaseException as ex:
            new_sound = str(ex)
        sounds.append(new_sound)
    return sounds


def create_sounds(texts: List[str], variants_of_paraphrasing: int = 0) -> List[Tuple[np.ndarray, str]]:
    if variants_of_paraphrasing > 0:
        paraphraser = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        if torch.cuda.is_available():
            print('CUDA is available! It\'s great!')
            paraphraser.cuda()
        paraphraser.eval()
    else:
        paraphraser = None
        tokenizer = None

    all_sounds_and_transcriptions = []
    for sound_idx, cur_text in enumerate(tqdm(texts)):
        synthetic_sounds = generate_sounds_using_different_voices(cur_text)
        for new_sound in synthetic_sounds:
            all_sounds_and_transcriptions.append((new_sound, cur_text))

        if variants_of_paraphrasing > 0:
            paraphrases = paraphrase_base(text=cur_text, tokenizer=tokenizer, model=paraphraser,
                                          variants=variants_of_paraphrasing * 2 + 1)
            if len(paraphrases) > variants_of_paraphrasing:
                paraphrases = paraphrases[:variants_of_paraphrasing]
            for cur_paraphrase in paraphrases:
                synthetic_sounds = generate_sounds_using_different_voices(cur_paraphrase)
                for new_sound in synthetic_sounds:
                    all_sounds_and_transcriptions.append((new_sound, cur_paraphrase))

    return all_sounds_and_transcriptions
