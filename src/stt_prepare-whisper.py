# author: naumov_al <sanya-naumov@mail.ru>
# coding: utf-8


"""
Скрипт `stt_prepare-whisper.py` — предназначен для получения STT результатов с 
использованием модели Whisper v3
(Whisper, https://huggingface.co/openai/whisper-large-v3).
"""

import os
import pickle
import datetime
import argparse as ap
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import torch
import librosa

from transformers import WhisperProcessor, WhisperForConditionalGeneration

device = 'cpu'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#torch_dtype = torch.float16


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


def get_whisper_transcriptions(audio_list: list, sr: int):
    # get input feats
    inputs = processor(audio_list, sampling_rate=sr, return_tensors="pt")
    #inputs = inputs.to("cuda", torch_dtype)
    
    # generate token ids
    predicted_ids = model.generate(
        inputs.input_features, forced_decoder_ids=forced_decoder_ids)
    
    # decode token ids to text
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)
    

en_ru_label_enc = {'yes': 'да', 'no': 'нет', 'up': 'вверх', 'down': 'вниз', 'left': 'налево', 
                   'right': 'направо', 'on': 'включи', 'off': 'выключи', 'stop': 'стоп', 
                   'go': 'иди', 'forward': 'вперед', 'backward': 'назад', 
                   'follow': 'следуй', 'visual': 'наблюдай', 'learn': 'изучай', 
                   
                   'zero': 'ноль', 'one': 'один', 'two': 'два', 'three': 'три', 'four': 'четыре', 
                   'five': 'пять', 'six': 'шесть', 'seven': 'семь', 'eight': 'восемь', 'nine': 'девять',
                   
                   'create': 'создай', 'cry': 'зарыдай', 'jumpup': 'вскочи', 'above': 'сверх', 
                   'dissonance': 'разлад', 'harm': 'вред', 'dies': 'гибнет', 'nails': 'гвозди', 
                   'rustier': 'ржавее', 'exclude': 'исключи', 'slogan': 'девиз', 'trouble': 'беда', 
                   'newer': 'новее', 'knock': 'стучи', 'blow': 'сдуй'}


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
        '-sr',
        default=16000,
        help='sr',
    )
    args_parser.add_argument(
        '--exp_name',
        default='new',
        help='exp_name',
    )
    args = args_parser.parse_args()
    
    bs = int(args.bs)
    sr = int(args.sr)
    path_to_data = Path(args.inp)
    path_to_save = Path(args.out) / f'res_stt-{args.exp_name}'

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
    #model.to(device)
    
    model.config.forced_decoder_ids = None
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="russian", task="transcribe")

    res = defaultdict(dict)
    for part_pathway in path_to_data.iterdir():
        audio_names, inp_audio_list = [], []
        
        for class_pathway in part_pathway.iterdir():
            if class_pathway.name.lower() not in en_ru_label_enc.keys():
                print(f'Unknown eng word: "{class_pathway.name.lower()}"')
                continue
                
            for fp in class_pathway.iterdir():
                name = f'{class_pathway.name.lower()}_{fp.stem}'
                audio_names.append(name)
                
                res[part_pathway.name][name] = {
                    'true': en_ru_label_enc[class_pathway.name.lower()], 
                    'pred': None
                }
    
                audio, sr_sample = librosa.load(fp, sr=None)
                if sr_sample != sr:
                    audio = librosa.resample(audio, orig_sr=sr_sample, 
                                             target_sr=sr)
                inp_audio_list.append(audio)            
    
        start_time = datetime.datetime.now()
        print(f'Start transcribe {len(inp_audio_list)} audio from {part_pathway.name}')
        
        transcriptions = []
        for batch in tqdm(
            batch_gen(df=inp_audio_list, bs=bs), 
            total=len(inp_audio_list)//bs + \
                  (1 if len(inp_audio_list)%bs!=0 else 0)
        ):
            batch_transcript = get_whisper_transcriptions(
                audio_list=batch, sr=sr)
            transcriptions.extend(batch_transcript)
        print(f'Get transcriptions with: {str(datetime.datetime.now()-start_time)}')
        
        for ind, name in enumerate(audio_names):
            res[part_pathway.name][name]['pred'] = transcription_normalizer(transcriptions[ind])
    
        save_obj(res, path_to_save/'res_stt-whisper_large_v3.pkl')
