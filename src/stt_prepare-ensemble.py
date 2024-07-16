# author: naumov_al <sanya-naumov@mail.ru>
# coding: utf-8


"""
Скрипт `stt_prepare-ensemble.py` — предназначен для получения аггрегированных 
результатов среди всех использованных STT подходов.
"""

import pickle
import argparse as ap
from pathlib import Path
from collections import Counter


def save_obj(obj, path: Path or str = 'object.pkl'):
    '''Function for save object to pickle file
    '''
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path: Path or str = 'object.pkl'):
    '''Function for load object from pickle file
    '''
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    args_parser = ap.ArgumentParser()
    args_parser.add_argument(
        '-inp_vosk',
        default=None,
        help='inp_vosk',
    )
    args_parser.add_argument(
        '-inp_whisper',
        default=None,
        help='inp_whisper',
    )
    args_parser.add_argument(
        '-inp_nvidia',
        default=None,
        help='inp_nvidia',
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

    df_parts = []
    stt_results = []
    path_to_save = Path(args.out) / f'res_stt-{args.exp_name}'
    
    if not args.inp_whisper is None:
        res = load_obj(args.inp_whisper)
        if len(df_parts) == 0:
            df_parts = list(res.keys())
        stt_results.append(res)

    if not args.inp_vosk is None:
        res = load_obj(args.inp_vosk)
        if len(df_parts) == 0:
            df_parts = list(res.keys())
        stt_results.append(res)

    if not args.inp_nvidia is None:
        res = load_obj(args.inp_nvidia)
        if len(df_parts) == 0:
            df_parts = list(res.keys())
        stt_results.append(res)
        
    if len(df_parts) == 0:
        raise ValueError('inputs dicts are empty or None')
    
    res_ensemble = {}
    for df_part in df_parts:
        res_ensemble[df_part] = {'good': [], 'ok': [], 'bad': []}
        
        for name in stt_results[0][df_part].keys():
            uniq_true_labels = set([i[df_part][name]['true'] 
                                    for i in stt_results])
            assert len(uniq_true_labels) == 1, f'err: {name}'
    
            true = uniq_true_labels[0]
            pred_counter = Counter([i[df_part][name]['pred'] 
                                    for i in stt_results])
            
            if true in pred_counter.keys():
                if pred_counter[true] >= 2:
                    res_ensemble[df_part]['good'].append(name)
                else:
                    res_ensemble[df_part]['ok'].append(name)
            else:
                res_ensemble[df_part]['bad'].append(name)

    save_obj(res_ensemble, path_to_save / f'res_stt-{args.exp_name}.pkl')
