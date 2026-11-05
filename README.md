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

## Evaluation
<FILL IN ONLY WITH WHAT YOU ACTUALLY RAN - do not report numbers you did not measure>
- Test clips: <N>, language: <lang>
- ASR WER (`evaluate_asr.py`): <value>
- Notes factual consistency (manual): <supported facts / total>
- MCQ quality (manual): <value>
- Quiz generation failures: <x of N>

## Failure analysis
<Fill with observed errors and examples>
- Code-mixed speech: the speaker mixes English technical terms (e.g. VS Code, GitHub) into Kannada; the ASR writes them phonetically in Kannada script, which hurts translation and the notes.
- <Add other observed failures>

## Known issues / notes
- Requires `transformers<5`; IndicTransToolkit breaks with newer versions.
- IndicTrans2 is run with `use_cache=False` and greedy decoding to avoid a cache incompatibility with recent `transformers`; this makes translation slower.
- YouTube links may be blocked on Colab; uploading the file is more reliable.
- The first run is slow because models are downloaded.
- Quiz output is requested as JSON; the code retries up to 3 times and may still fail.

## Limitations
- Small test set (<N> clips); a 30-50 case evaluation is future work.
- Errors propagate through the pipeline (ASR -> translation -> notes -> quiz).

## Team
<Names>
