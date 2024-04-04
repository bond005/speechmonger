import codecs
import re
from typing import List
import wave

from nltk import wordpunct_tokenize
import numpy as np

from synthesis_utils.synthesis_utils import TARGET_SAMPLE_RATE


def load_text_corpus(fname: str) -> List[str]:
    line_idx = 1
    re_for_russian = re.compile(f'^[абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ]+$')
    texts = []
    with codecs.open(fname, mode='r', encoding='utf-8') as fp:
        curline = fp.readline()
        while len(curline) > 0:
            prepline = curline.strip()
            if len(prepline) > 0:
                err_msg = f'The file "{fname}": line {line_idx} is wrong!'
                words = list(filter(
                    lambda it2: it2.isalpha(),
                    map(lambda it1: it1.strip(), wordpunct_tokenize(prepline))
                ))
                if len(words) == 0:
                    err_msg += ' There are no letters in this line!'
                    raise ValueError(err_msg)
                if any(map(lambda it: re_for_russian.search(it) is not None, words)):
                    texts.append(prepline)
            curline = fp.readline()
            line_idx += 1
    if len(texts) == 0:
        raise ValueError(f'The text file "{fname}" is empty!')
    return texts


def save_sound(fname: str, sound: np.ndarray):
    with wave.open(fname, 'wb') as fp:
        fp.setframerate(TARGET_SAMPLE_RATE)
        fp.setsampwidth(2)
        fp.setnchannels(1)
        fp.writeframes(sound.tobytes())
