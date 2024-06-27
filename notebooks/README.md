The following jupyter notebooks are available here:
- 1_Stage-STT_Generation.ipynb -- This notebook is designed to generate synthetic audio recordings based on voices from a reference dataset;
- 2_Stage-VAD_Trimming.ipynb -- This notebook is designed to extract a segment of audio with just the spoken word in the generated audio recordings (to generate audio recordings, see notebook #1);
- 3_Stage-STT_Filtering.ipynb -- This notebook is designed to filter out low-quality generated audio recordings that have been trimmed using the VAD model (see notebook #2).

Important Note:
The notebook "2_Stage-VAD_Trimming.ipynb" should be placed in the root of the [sgvad repository](https://github.com/jsvir/vad) and run from there.
