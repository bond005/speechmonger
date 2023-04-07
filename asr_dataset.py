from argparse import ArgumentParser
import logging
import os
import sys

from io_utils.io_utils import load_text_corpus, save_speech_corpus
from synthesis_utils.synthesis_utils import create_sounds


asr_dataset_logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='dataset_name', type=str, required=True,
                        help='The target directory with dataset which will be built.')
    parser.add_argument('-l', '--list', dest='text_list', type=str, required=True,
                        help='The file with text list for speech synthesis.')
    parser.add_argument('-p', '--paraphrases', dest='paraphrases', type=int, required=False, default=0,
                        help='The paraphrases number.')
    args = parser.parse_args()

    dataset_path = os.path.normpath(args.dataset_name)
    dataset_parent_dir = os.path.dirname(dataset_path)
    dataset_name = os.path.basename(dataset_path)
    if len(dataset_name) == 0:
        err_msg = f'The dataset path "{dataset_path}" is wrong!'
        asr_dataset_logger.error(err_msg)
        raise IOError(err_msg)
    if len(dataset_parent_dir) > 0:
        if not os.path.isdir(dataset_parent_dir):
            err_msg = f'The directory "{dataset_path}" does not exist!'
            asr_dataset_logger.error(err_msg)
            raise IOError(err_msg)
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)

    text_corpus_fname = os.path.normpath(args.text_list)
    if not os.path.isfile(text_corpus_fname):
        err_msg = f'The file "{text_corpus_fname}" does not exist!'
        asr_dataset_logger.error(err_msg)
        raise IOError(err_msg)
    try:
        text_corpus = load_text_corpus(text_corpus_fname)
    except BaseException as ex:
        err_msg = str(ex)
        asr_dataset_logger.error(err_msg)
        raise
    info_msg = f'The text corpus is loaded from the "{text_corpus_fname}". There are {len(text_corpus)} texts.'
    asr_dataset_logger.info(info_msg)

    number_of_paraphrases = args.paraphrases
    if number_of_paraphrases < 0:
        err_msg = f'The paraphrases number is wrong! Expected a non-negative integer, got {number_of_paraphrases}.'
        asr_dataset_logger.error(err_msg)
        raise IOError(err_msg)

    try:
        speech_corpus = create_sounds(texts=text_corpus, variants_of_paraphrasing=number_of_paraphrases)
    except BaseException as ex:
        err_msg = str(ex)
        asr_dataset_logger.error(err_msg)
        raise
    info_msg = f'There are {len(speech_corpus)} samples in the created speech corpus.'
    asr_dataset_logger.info(info_msg)
    filtered_speech_corpus = []
    n_errors = 0
    for sound, annotation in speech_corpus:
        if isinstance(sound, str):
            n_errors += 1
            asr_dataset_logger.warning(sound)
        else:
            filtered_speech_corpus.append((sound, annotation))
    del speech_corpus
    info_msg = f'There are {len(filtered_speech_corpus)} samples after filtering.'
    if n_errors > 0:
        info_msg += f' There are {n_errors} errors.'
    asr_dataset_logger.info(info_msg)

    try:
        save_speech_corpus(dataset_path, filtered_speech_corpus)
    except BaseException as ex:
        err_msg = str(ex)
        asr_dataset_logger.error(err_msg)
        raise


if __name__ == '__main__':
    asr_dataset_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    asr_dataset_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('asr_dataset.log')
    file_handler.setFormatter(formatter)
    asr_dataset_logger.addHandler(file_handler)
    main()
