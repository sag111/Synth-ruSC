# Datasets

This is the directory where all the data will be stored.

For example, in our case:
1. `./data/google_speech_command_v002` -- Directory with reference audios where we took sample voices

    We used scripts from NeMo to get google speech commands dataset and create manifests.
    For more details see [this link](https://github.com/NVIDIA/NeMo/blob/main/scripts/dataset_processing/process_speech_commands_data.py) or [this instruction](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speech_classification/datasets.html#google-speech-commands-dataset).

2. `./data/Synt-RuSC` -- Directory with generated synthetic data at different stages (raw/interim/processed)
    
    The Synth-ruSC dataset is available for download at [this link](https://cloud.mail.ru/public/eNA1/FDgVqAbJL).
    
    The structure of this directory:
    ```
    |-raw - Initially generated data using the XTTS model
    |-interim - Synthetic data after processing with VAD and STT models
    |-processed - The final set of filtered audio recordings
        |-train
        |-test 
        |-validation  
        |-real - The collected real-world dataset with 896 real audio recordings from 23 people
    ```