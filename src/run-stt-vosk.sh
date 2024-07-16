#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"

export INP_PATH="/s/ls4/groups/g0126/datasets/audio/ru_google_speech_command/xtts_v2/fullspeaker_vadfltr"
export OUT_PATH="/s/ls4/users/naumov/Projects/#18_Audio/3_Notebooks/res_stt-ru_gsc_fullspeaker/vosk"

for DS_PART_PATH in ${INP_PATH}/*/
    do
        export DS_PART_PATH=${DS_PART_PATH%*/}
        export DS_PART="${DS_PART_PATH##*/}"
        
        for DS_NAME_PATH in ${DS_PART_PATH}/*/
        	do
                export DS_NAME_PATH=${DS_NAME_PATH%*/}
                export DS_NAME="${DS_NAME_PATH##*/}"
                
                mkdir -p "${OUT_PATH}/${DS_PART}/${DS_NAME}"
        		
                vosk-transcriber --input ${DS_NAME_PATH} \
                                 --output "${OUT_PATH}/${DS_PART}/${DS_NAME}" \
                                 --model-name "vosk-model-ru-0.42" \
                                 --lang "ru"
            done
    done

python stt_prepare-vosk.py -inp $OUT_PATH -out ${OUT_PATH%vosk*}
