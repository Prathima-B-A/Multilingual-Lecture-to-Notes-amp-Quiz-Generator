import json, re
import torch, torchaudio
from transformers import (AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM)
from IndicTransToolkit.processor import IndicProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ASR_ID = "ai4bharat/indic-conformer-600m-multilingual"
TRANS_INDIC_EN = "ai4bharat/indictrans2-indic-en-dist-200M"
TRANS_EN_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"
LLM_ID = "bharatgenai/Param-1-2.9B-Instruct"   # change if unavailable

LANGS = {"Hindi": ("hi", "hin_Deva"), "Kannada": ("kn", "kan_Knda"),
         "Tamil": ("ta", "tam_Taml"), "English": (None, "eng_Latn")}

_cache = {}
def load(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]

def transcribe(audio_path, lang_code, chunk_s=20):
    asr = load("asr", lambda: AutoModel.from_pretrained(ASR_ID, trust_remote_code=True))
    wav, sr = torchaudio.load(audio_path)
    wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    step = chunk_s * 16000
    parts = []
    for i in range(0, wav.shape[1], step):
        seg = wav[:, i:i + step]
        if seg.shape[1] < 1600:
            continue
        parts.append(asr(seg, lang_code, "ctc"))
    return " ".join(str(p).strip() for p in parts)

def translate(text, src, tgt):
    if src == tgt or not text.strip():
        return text
    name = TRANS_INDIC_EN if tgt == "eng_Latn" else TRANS_EN_INDIC
    tok = load(name + "tok", lambda: AutoTokenizer.from_pretrained(name, trust_remote_code=True))
    mdl = load(name, lambda: AutoModelForSeq2SeqLM.from_pretrained(
        name, trust_remote_code=True).to(DEVICE).eval())
    ip = load("ip", lambda: IndicProcessor(inference=True))
    sents = [s.strip() for s in re.split(r"(?<=[.?!।])\s+", text) if s.strip()]
    out = []
    for i in range(0, len(sents), 8):
        batch = ip.preprocess_batch(sents[i:i + 8], src_lang=src, tgt_lang=tgt)
        inp = tok(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            # use_cache=False works around an IndicTrans2 / new-transformers incompatibility
            gen = mdl.generate(**inp, use_cache=False, num_beams=1, max_length=256)
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        out += ip.postprocess_batch(dec, lang=tgt)
    return " ".join(out)

def llm(prompt, max_new_tokens=700):
    tok = load("ltok", lambda: AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True))
    m = load("llm", lambda: AutoModelForCausalLM.from_pretrained(
        LLM_ID, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True).to(DEVICE).eval())
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

def chunks(text, words=450):
    w = text.split()
    return [" ".join(w[i:i + words]) for i in range(0, len(w), words)]

def make_notes(english):
    partial = [llm("Write concise bullet-point study notes from this lecture excerpt. "
                   "Use only facts stated in it.\n\n" + c, 400) for c in chunks(english)]
    if len(partial) == 1:
        return partial[0]
    return llm("Merge these partial notes into one clean set of notes with short headings, "
               "no repetition:\n\n" + "\n\n".join(partial), 800)

def make_quiz(notes):
    p = ("Create exactly 5 multiple-choice questions from the notes below. Use only facts in the notes. "
         "Return ONLY a JSON list; each item: {\"question\": str, \"options\": [4 strings], "
         "\"answer\": \"A\"|\"B\"|\"C\"|\"D\"}.\n\nNOTES:\n" + notes)
    for _ in range(3):                      # retry up to 3 times
        raw = llm(p, 900)
        m = re.search(r"\[.*\]", raw, re.S)
        try:
            quiz = json.loads(m.group(0))
            if len(quiz) >= 5:
                return quiz[:5]             # keep exactly 5
        except Exception:
            pass
    return None

def format_quiz(quiz):
    if not quiz:
        return "Quiz generation failed (invalid JSON). Try again."
    lines = []
    for i, q in enumerate(quiz, 1):
        lines.append(f"**Q{i}. {q['question']}**")
        lines += [f"- {'ABCD'[j]}. {o}" for j, o in enumerate(q["options"])]
        lines.append(f"*Answer: {q['answer']}*\n")
    return "\n".join(lines)
