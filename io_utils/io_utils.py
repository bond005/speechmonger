import codecs
import csv
import os
import re
import shutil
from typing import List, Tuple
import wave

from nltk import wordpunct_tokenize
import numpy as np

from synthesis_utils.synthesis_utils import TARGET_SAMPLE_RATE


def load_text_corpus(fname: str) -> List[str]:
    line_idx = 1
    re_for_digit = re.compile(r'\d+')
    re_for_russian = re.compile(f'^[абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ]+$')
    texts = []
    with codecs.open(fname, mode='r', encoding='utf-8') as fp:
        curline = fp.readline()
        while len(curline) > 0:
            prepline = curline.strip()
            if len(prepline) > 0:
                err_msg = f'The file "{fname}": line {line_idx} is wrong!'
                if re_for_digit.search(prepline) is not None:
                    err_msg += ' This line contains some digits!'
                    raise ValueError(err_msg)
                words = list(filter(
                    lambda it2: it2.isalpha(),
                    map(lambda it1: it1.strip(), wordpunct_tokenize(prepline))
                ))
                if len(words) == 0:
                    err_msg += ' There are no letters in this line!'
                if not all(map(lambda it: re_for_russian.search(it) is not None, words)):
                    err_msg += ' There are some non-Russian words in this line!'
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


def save_speech_corpus(speech_corpus_dir: str, speech_corpus_data: List[Tuple[np.ndarray, str, str]]):
    if not os.path.isdir(speech_corpus_dir):
        raise ValueError(f'The directory "{speech_corpus_dir}" does not exist!')
    metadata_fname = os.path.join(speech_corpus_dir, 'metadata.csv')
    data_dir = os.path.join(speech_corpus_dir, 'data')
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)
    with codecs.open(metadata_fname, mode='w', encoding='utf-8') as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['file_name', 'transcription', 'normalized'])
        for counter, (speech, text, norm) in enumerate(speech_corpus_data):
            speech_path = 'data/sound{0:>04}.wav'.format(counter + 1)
            data_writer.writerow([speech_path, text, norm])
            save_sound(fname=os.path.join(speech_corpus_dir, os.path.normpath(speech_path)), sound=speech)
