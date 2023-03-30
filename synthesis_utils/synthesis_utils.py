from typing import List, Tuple
import warnings

import librosa
import nltk
import numpy as np
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


TARGET_SAMPLE_RATE: int = 16_000
MODEL_NAME = 'cointegrated/rut5-base-paraphraser'


def paraphrase_base(text: str, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration,
                    beams: int = 30, grams: int = 4, variants: int = 10) -> List[str]:
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
        if k not in normalized_paraphrases:
            normalized_paraphrases.add(k)
            filtered_paraphrases.append(v)
    return filtered_paraphrases


def create_sounds(texts: List[str], variants_of_paraphrasing: int = 0) -> List[Tuple[np.ndarray, str]]:
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        "facebook/tts_transformer-ru-cv7_css10",
        arg_overrides={"vocoder": "hifigan", "fp16": False}
    )
    speech_model = models[0]
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(models, cfg)

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
        with warnings.catch_warnings():
            sample = TTSHubInterface.get_model_input(task, cur_text)
            wav, rate = TTSHubInterface.get_prediction(task, speech_model, generator, sample)
        new_sound = wav.numpy()
        if len(new_sound.shape) > 1:
            new_sound = librosa.to_mono(new_sound)
        if rate != TARGET_SAMPLE_RATE:
            new_sound = librosa.resample(new_sound, orig_sr=rate, target_sr=TARGET_SAMPLE_RATE,
                                         res_type='kaiser_best', scale=True)
        new_sound = np.round(new_sound * 32767.0)
        err_msg = f'The synthetic sound {sound_idx} for the text "{cur_text}" is wrong!'
        if np.max(np.abs(new_sound)) > 32768.0:
            raise ValueError(err_msg)
        if np.min(new_sound) >= 0:
            raise ValueError(err_msg)
        if new_sound.shape[0] <= (TARGET_SAMPLE_RATE // 4):
            err_msg += f' It is too short! Expected greater than {TARGET_SAMPLE_RATE // 4}, got {new_sound.shape[0]}.'
            raise ValueError(err_msg)
        new_sound[new_sound > 32767] = 32767
        new_sound = new_sound.astype(np.int16)
        all_sounds_and_transcriptions.append((new_sound, cur_text))
        del sample, wav, new_sound

        if variants_of_paraphrasing > 0:
            paraphrases = paraphrase_base(text=cur_text, tokenizer=tokenizer, model=paraphraser,
                                          variants=variants_of_paraphrasing + 2)
            if len(paraphrases) > variants_of_paraphrasing:
                paraphrases = paraphrases[:variants_of_paraphrasing]
            for cur_paraphrase in paraphrases:
                with warnings.catch_warnings():
                    sample = TTSHubInterface.get_model_input(task, cur_paraphrase)
                    wav, rate = TTSHubInterface.get_prediction(task, speech_model, generator, sample)
                new_sound = wav.numpy()
                if len(new_sound.shape) > 1:
                    new_sound = librosa.to_mono(new_sound)
                if rate != TARGET_SAMPLE_RATE:
                    new_sound = librosa.resample(new_sound, orig_sr=rate, target_sr=TARGET_SAMPLE_RATE,
                                                 res_type='kaiser_best', scale=True)
                new_sound = np.round(new_sound * 32767.0)
                new_sound[new_sound > 32767] = 32767
                new_sound = new_sound.astype(np.int16)
                all_sounds_and_transcriptions.append((new_sound, cur_paraphrase))

    return all_sounds_and_transcriptions
