import random
import re

from nltk import wordpunct_tokenize
from transliterate import translit


SPECIAL_WORDS = {
    'microsoft': ['микрософт', 'майкрософт'],
    'science': ['сайнс'],
    'scopus': ['скопус'],
    'google': ['гугл'],
    'yandex': ['яндекс'],
    'huawei': ['хуавей'],
    'openai': ['опенаи', 'опенэйай'],
    'python': ['питон', 'пайтон'],
    'pytorch': ['пайторч'],
    'ainl': ['аинл'],
    'nips': ['нипс'],
    'neurips': ['нейрипс'],
    'icml': ['айсиэмэль'],
    'iclr': ['айсиэлэр'],
    'specom': ['спеком', 'спиком'],
    'deadline': ['дедлайн'],
    'energy': ['энерджи'],
    'vocal': ['вокал'],
    'vocals': ['вокалс', 'вокалз'],
    'kubespray': ['куберспрей'],
    'helion': ['гелион', 'хелион'],
    'chatgpt': ['чатгэпэтэ', 'чатгпт', 'чатжэпэтэ', 'чатжпт'],
    'gpt': ['гэпэтэ', 'гпт', 'жэпэтэ', 'жпт'],
    'bert': ['берт', 'бёрт'],
    'bart': ['барт'],
    'github': ['гитхаб'],
    'gitlab': ['гитлаб'],
    'gitflic': ['гитфлик'],
    'cuda': ['куда'],
    'openml': ['опенмл', 'опенэмэль'],
    'opencl': ['опенсл', 'опенсиэль'],
    'wiki': ['вики'],
    'wikipedia': ['викиедия'],
    'female': ['фемэйл', 'фимэйл'],
    'male': ['мэйл'],
    'mail': ['мэйл'],
    'email': ['емэйл', 'имэйл'],
    'http': ['хттп', 'эйчтитипи', 'аштитипи', 'аштэтэпэ'],
    'https': ['хттпс', 'эйчтитипиэс', 'аштитипиэс', 'аштэтэпээс'],
    'ip': ['айпи'],
    'it': ['айти'],
    'tcp': ['тисипи'],
    'cisco': ['циско', 'циска'],
    'java': ['джава'],
    'linux': ['линукс'],
    'apple': ['эппл'],
    'android': ['андроид'],
    'macos': ['макось'],
    'mac': ['мак'],
    'iphone': ['айфон'],
    'nvidia': ['энвидиа', 'энвидия'],
    'intel': ['интел'],
    'expasoft': ['экспасофт'],
    'promsoft': ['промсофт'],
    'sber': ['сбер'],
    'sberbank': ['сбербанк'],
    'ai': ['аи', 'эйай'],
    'redmi': ['редми'],
    'xiaomi': ['сяоми'],
    'priora': ['приора'],
    'sleep': ['слип'],
    'shutdown': ['шатдаун'],
    'standup': ['стэндап'],
    'up': ['ап'],
    'down': ['даун'],
    'break': ['брейк'],
    'LabADT': ['лабадт'],
    'main': ['мэйн'],
    'develop': ['дэвэлоп', 'девелоп'],
    'chat': ['чат'],
    'devops': ['девопс'],
    'creta': ['крета'],
    'changan': ['чанган'],
    'vesta': ['веста'],
    'lada': ['лада'],
    'haval': ['хавейл', 'хавал'],
    'toyota': ['тойота'],
    'aurus': ['аурус'],
    'chery': ['чери'],
    'cherry': ['чери'],
    'tiggo': ['тиго'],
    'bmw': ['бээмвэ'],
    'fiat': ['фиат'],
    'ford': ['форд'],
    'renault': ['рено'],
    'peugeot': ['пежо'],
    'lama': ['лама'],
    'llama': ['лама'],
    'patriot': ['патриот'],
    'uaz': ['уаз'],
    'vaz': ['ваз'],
    'samsung': ['самсунг'],
    'nokia': ['нокия']
}


def apply_transliteration(source_word: str) -> str:
    if len(source_word.strip()) == 0:
        return ''
    re_for_russian = re.compile(r'^[абвгдеёжзийклмнопрстуфхцчшщъыьэюя]+$')
    re_for_english = re.compile(r'^[abcdefghijklmnopqrstuvwxyz]+$')
    if re_for_russian.search(source_word.lower()) is not None:
        return source_word
    if re_for_english.search(source_word.lower()) is None:
        if source_word.isalnum():
            raise ValueError(f'The word {source_word} is inadmissible!')
        subwords = list(filter(lambda it: it.isalnum(), wordpunct_tokenize(source_word)))
        if len(subwords) != 0:
            raise ValueError(f'The word {source_word} is inadmissible!')
        return source_word
    lowered_source_word = source_word.lower()
    if lowered_source_word in SPECIAL_WORDS:
        res = random.choice(SPECIAL_WORDS[lowered_source_word])
    else:
        res = translit(lowered_source_word, 'ru')
    if source_word.isupper():
        res = res.upper()
    elif source_word[0].isupper():
        res = res[0].upper() + res[1:]
    return res


def normalize_text(s: str) -> str:
    re_for_english = re.compile(r'[abcdefghijklmnopqrstuvwxyz]+')
    re_for_digit = re.compile(r'\d+')
    words = list(filter(
        lambda it2: it2.isalnum(),
        map(lambda it1: it1.strip().lower(), wordpunct_tokenize(s))
    ))
    if len(words) == 0:
        return ''
    normalized_words = []
    for cur in words:
        if re_for_digit.search(cur) is not None:
            err_msg = f'Text "{s}" is wrong, because it contains some digits!'
            raise ValueError(err_msg)
        search_res = re_for_english.search(cur)
        if search_res is None:
            normalized_words.append(cur)
        else:
            start_pos = search_res.start()
            end_pos = search_res.end()
            if start_pos < 0:
                normalized_words.append(cur)
            else:
                if start_pos > 0:
                    new_word = cur[0:start_pos]
                else:
                    new_word = ''
                new_word += apply_transliteration(cur[start_pos:end_pos])
                old_word = cur[end_pos:]
                search_res = re_for_english.search(old_word)
                while search_res is not None:
                    start_pos = search_res.start()
                    end_pos = search_res.end()
                    if start_pos < 0:
                        break
                    if start_pos > 0:
                        new_word += old_word[0:start_pos]
                    new_word += apply_transliteration(old_word[start_pos:end_pos])
                    old_word = old_word[end_pos:]
                    search_res = re_for_english.search(old_word)
                if len(old_word) > 0:
                    new_word += old_word
                normalized_words.append(new_word)
    return ' '.join(normalized_words).replace('ё', 'е')
