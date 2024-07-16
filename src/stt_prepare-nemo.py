# author: naumov_al <sanya-naumov@mail.ru>
# coding: utf-8


"""
Скрипт `stt_prepare-nemo.py` — предназначен для получения STT результатов с 
использованием модели NVIDIA Conformer-CTC Large 
(NeMo, https://github.com/NVIDIA/NeMo).
"""

import os
import pickle
import datetime
import argparse as ap
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import nemo.collections.asr as nemo_asr
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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


def batch_gen(df: list, bs: int = 10):
    '''Function for generation batch with size 'bs'
    '''
    
    for i in range(0, len(df), bs):
        yield df[i:i+bs]


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
        default='./data/raw/rusc/',
        help='inp',
    )
    args_parser.add_argument(
        '-out',
        default='./results/',
        help='out',
    )
    args_parser.add_argument(
        '-bs',
        default=10,
        help='bs',
    )
    args_parser.add_argument(
        '--exp_name',
        default='new',
        help='exp_name',
    )
    args = args_parser.parse_args()

    bs = int(args.bs)
    path_to_data = Path(args.inp)
    path_to_save = Path(args.out) / f'res_stt-{args.exp_name}'

    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name="stt_ru_conformer_ctc_large")
    
    res = defaultdict(dict)
    for part_pathway in path_to_data.iterdir():
        y_true, y_pred = [], []
        audio_names, audio_pathways = [], []
        
        for class_pathway in part_pathway.iterdir():
            if class_pathway.name.lower() not in en_ru_label_enc.keys():
                print(f'Unknown eng word: "{class_pathway.name.lower()}"')
                continue
            
            for fp in class_pathway.iterdir():
                audio_pathways.append(str(fp))
                name = f'{class_pathway.name.lower()}_{fp.stem}'
                audio_names.append(name)
                
                res[part_pathway.name][name] = {
                    'true': en_ru_label_enc[class_pathway.name.lower()], 
                    'pred': None
                }
        
        start_time = datetime.datetime.now()
        print(f'Start transcribe {len(audio_pathways)} audio')
        
        transcriptions = []
        for batch in tqdm(
            batch_gen(df=audio_pathways, bs=bs), 
            total=len(audio_pathways)//bs + \
                  (1 if len(audio_pathways)%bs!=0 else 0)
        ):
            batch_transcript = asr_model.transcribe(
                paths2audio_files=batch, batch_size=bs, verbose=False)
            transcriptions.extend(batch_transcript)
        print(f'Get transcriptions with: {str(datetime.datetime.now()-start_time)}')
        
        for ind, name in enumerate(audio_names):
            res[part_pathway.name][name]['pred'] = transcription_normalizer(transcriptions[ind])
    
    save_obj(res, path_to_save/'res_stt-stt_ru_conformer_ctc_large.pkl')
