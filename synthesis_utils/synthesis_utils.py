import gc
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

from normalization_utils.normalization_utils import apply_transliteration


TARGET_SAMPLE_RATE: int = 16_000
MODEL_NAME = 'cointegrated/rut5-base-paraphraser'


CONSONANTS = {
    'б': 'бэ',
    'в': 'вэ',
    'г': 'гэ',
    'д': 'дэ',
    'ж': 'жэ',
    'з': 'зэ',
    'й': 'йы',
    'к': 'ка',
    'л': 'эл',
    'м': 'эм',
    'н': 'эн',
    'п': 'пэ',
    'р': 'эр',
    'с': 'эс',
    'т': 'тэ',
    'ф': 'эф',
    'х': 'хэ',
    'ц': 'цэ',
    'ч': 'чэ',
    'ш': 'ша',
    'щ:': 'ща'
}


PRONOUNCING_EXCLUSIONS = {
    'нгу': 'энгэу',
    'мгу': 'эмгэу',
    'мфти': 'эмфэтэи',
    'доннту': 'донэнтэу',
    'мгту': 'эмгэтэу',
    'спбгу': 'эспэбэгэу',
    'ивт': 'ивэтэ',
    'ивц': 'ивэцэ',
    'ммф': 'эмэмэф',
    'ивммг': 'ивээмэмгэ',
    'ивмимг': 'ивээмиэмгэ',
    'лабадт': 'лабадэтэ'
}


def voice_abbreviation(old_text: str) -> str:
    old_text_ = old_text.strip().lower()
    if len(old_text_) == 0:
        return ''
    if set(old_text_) in {'ш', 'щ'}:
        return old_text_[0]
    re_for_vowels = re.compile(f'[ауоыиэяюёе]+')
    if re_for_vowels.search(old_text_) is not None:
        return old_text
    new_text = CONSONANTS[old_text_[0]]
    for c in old_text_[1:]:
        new_text += CONSONANTS[c]
    return new_text


def generate_pronouncing_exclusion(s: str) -> str:
    s_ = s.strip().lower()
    if s_ in PRONOUNCING_EXCLUSIONS:
        return PRONOUNCING_EXCLUSIONS[s_]
    return s


def paraphrase_base(text: str, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration,
                    beams: int = 30, grams: int = 4, variants: int = 10) -> List[str]:
    re_for_digit = re.compile(r'\d+')
    lat_numbers = {'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV',
                   'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX', 'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI',
                   'XXVII', 'XXVIII', 'XXIX', 'XXX'}
    x = tokenizer([text for _ in range(variants)],
                  return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 1.5 + 10)
    out = model.generate(**x, encoder_no_repeat_ngram_size=grams, early_stopping=True,
                         num_beams=beams, max_length=max_size, no_repeat_ngram_size=4,
                         num_return_sequences=variants).to('cpu')
    paraphrases = [tokenizer.decode(out[variant_idx], skip_special_tokens=True)
                   for variant_idx in range(variants)]
    del out, x
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
        if (k not in normalized_paraphrases) and (re_for_digit.search(k) is None) and \
                (len(set(k.upper().split()) & lat_numbers) == 0):
            normalized_paraphrases.add(k)
            filtered_paraphrases.append(v)
    return filtered_paraphrases


def generate_sound(text: str, voice: str) -> np.ndarray:
    sound_fname = ''
    re_for_word = re.compile(r'\w+')
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.wav', delete=False) as fp:
            sound_fname = fp.name
        text_for_synthesis = text.replace('"', '').replace("'", '')
        transliterated_text_for_synthesis = ''
        search_res = re_for_word.search(text_for_synthesis)
        if search_res is None:
            raise ValueError(f'The text {text} is wrong, and it cannot be synthesized.')
        start_pos = search_res.start()
        end_pos = search_res.end()
        if start_pos > 0:
            transliterated_text_for_synthesis = text_for_synthesis[0:start_pos]
        transliterated_text_for_synthesis += generate_pronouncing_exclusion(
            voice_abbreviation(
                apply_transliteration(text_for_synthesis[start_pos:end_pos])
            )
        )
        text_for_synthesis = text_for_synthesis[end_pos:]
        search_res = re_for_word.search(text_for_synthesis)
        while search_res is not None:
            start_pos = search_res.start()
            end_pos = search_res.end()
            if start_pos < 0:
                break
            if start_pos > 0:
                transliterated_text_for_synthesis += text_for_synthesis[0:start_pos]
            transliterated_text_for_synthesis += generate_pronouncing_exclusion(
                voice_abbreviation(
                    apply_transliteration(text_for_synthesis[start_pos:end_pos])
                )
            )
            text_for_synthesis = text_for_synthesis[end_pos:]
            search_res = re_for_word.search(text_for_synthesis)
        transliterated_text_for_synthesis += text_for_synthesis
        cmd = f'echo "{transliterated_text_for_synthesis}" | RHVoice-client -s {voice}+CLB > "{sound_fname}"'
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

        gc.collect()

    return all_sounds_and_transcriptions
