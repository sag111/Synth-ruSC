# author: naumov_al <sanya-naumov@mail.ru>
# coding: utf-8


"""
Скрипт `stt_prepare-vosk.py` — предназначен для получения STT результатов с 
использованием модели Vosk-model-ru-0.42
(Vosk, https://alphacephei.com/vosk/models).
"""

import pickle
import argparse as ap
from pathlib import Path
from collections import defaultdict


def save_obj(obj, path: Path or str = 'object.pkl'):
    '''Function for save object to pickle file
    '''
    
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def transcription_normalizer(string: str):
    '''Function to normalize text transcriptions from STT models
    '''
    
    for symbol in ['.','!','?',',']:
        string = string.replace(symbol, '')
    string = string.replace('ё', 'е')
    for i, j in zip([str(k) for k in range(10)], 
                    ['ноль', 'один', 'два', 'три', 'четыре', 'пять', 
                     'шесть', 'семь', 'восемь', 'девять']):
        string = string.replace(i, j)
    return string.strip().lower()


en_ru_label_enc = {'yes': 'да', 'no': 'нет', 'up': 'вверх', 'down': 'вниз', 'left': 'налево', 
                   'right': 'направо', 'on': 'включи', 'off': 'выключи', 'stop': 'стоп', 
                   'go': 'иди', 'forward': 'вперед', 'backward': 'назад', 
                   'follow': 'следуй', 'visual': 'наблюдай', 'learn': 'изучай', 
                   
                   'zero': 'ноль', 'one': 'один', 'two': 'два', 'three': 'три', 'four': 'четыре', 
                   'five': 'пять', 'six': 'шесть', 'seven': 'семь', 'eight': 'восемь', 'nine': 'девять',
                   
                   'create': 'создай', 'cry': 'зарыдай', 'jumpup': 'вскочи', 'above': 'сверх', 
                   'dissonance': 'разлад', 'harm': 'вред', 'dies': 'гибнет', 'nails': 'гвозди', 
                   'rustier': 'ржавее', 'exclude': 'исключи', 'slogan': 'девиз', 'trouble': 'беда', 
                   'newer': 'новее', 'knock': 'стучи', 'blow': 'сдуй',
                  }


if __name__ == '__main__':
    args_parser = ap.ArgumentParser()
    args_parser.add_argument(
        '-inp',
        default='./results/res_stt-new/vosk/',
        help='inp',
    )
    args_parser.add_argument(
        '-out',
        default='./results/',
        help='out',
    )
    args_parser.add_argument(
        '--exp_name',
        default='new',
        help='exp_name',
    )
    args = args_parser.parse_args()
    
    path_to_vosk_res = Path(args.inp)
    path_to_save = Path(args.out) / f'res_stt-{args.exp_name}'
    
    res = defaultdict(dict)
    for path_data_part in path_to_vosk_res.iterdir():
        for class_pathway in path_data_part.iterdir():
            if class_pathway.name.lower() not in en_ru_label_enc.keys():
                print(f'Unknown eng word: "{class_pathway.name.lower()}"')
                continue
            
            for fp in class_pathway.iterdir():
                name = f'{class_pathway.name.lower()}_{fp.stem}'
                
                with open(fp, "r") as f:
                    transcription = f.read()
                
                res[path_data_part.name][name] = {
                    'true': en_ru_label_enc[class_pathway.name.lower()], 
                    'pred': transcription_normalizer(transcription)
                }
    
    save_obj(res, path_to_save/'res_stt-vosk_model_ru_042.pkl')
