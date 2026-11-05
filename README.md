# Multilingual Lecture-to-Notes & Quiz Generator

CS3235 Mini-Project 5 (AI4Bharat & BharatGen).
Upload a lecture recording (or paste a link) in Hindi / Kannada / Tamil and get a transcript, study notes and 5 multiple-choice questions, optionally translated into another language.

## Pipeline
```
Lecture audio/video (file or link)
  -> ffmpeg / yt-dlp (convert to 16 kHz mono wav, first N minutes)
  -> AI4Bharat IndicConformer (ASR, 20 s chunks)
  -> AI4Bharat IndicTrans2 (lecture language -> English)
  -> BharatGen Param1 (notes + 5 MCQs, generated in English)
  -> AI4Bharat IndicTrans2 (English -> chosen output language, optional)
  -> Gradio frontend
```
English is used as the intermediate language because small LLMs generate more reliable notes and quizzes in English.

## Models
| Stage | Model |
|---|---|
| ASR | `ai4bharat/indic-conformer-600m-multilingual` |
| Translation | `ai4bharat/indictrans2-indic-en-dist-200M`, `ai4bharat/indictrans2-en-indic-dist-200M` |
| Notes + MCQs | `bharatgenai/Param-1-2.9B-Instruct` (replace here if you used another: <FILL IN>) |

Some of these models are gated on Hugging Face: request access with the same account as your token.

## Files
- `app.py` - pipeline (ASR, translation, notes, quiz)
- `ui.py` - Gradio frontend (file upload or link)
- `evaluate_asr.py` - WER evaluation with `jiwer`
- `requirements.txt`

## Setup and run (Google Colab, T4 GPU)
1. Runtime -> Change runtime type -> T4 GPU. Add `HF_TOKEN` in Colab Secrets.
2. Install and log in:
```python
!pip install -q -r requirements.txt
from google.colab import userdata
from huggingface_hub import login
login(userdata.get("HF_TOKEN"))
```
3. Restart the session after installing, then run the frontend:
```python
!python ui.py
```
4. Open the public Gradio link, upload a file or paste a link, choose the lecture language and output language, and click Generate.
