"""WER over a test set. Put audio in test_data/ and a CSV: path,reference_text"""
import csv, jiwer
from app import transcribe, LANGS
LANG = "Hindi"
refs, hyps = [], []
for row in csv.DictReader(open("test_data/test.csv", encoding="utf-8")):
    hyp = transcribe(row["path"], LANGS[LANG][0])
    refs.append(row["reference_text"]); hyps.append(hyp)
    print(f"{row['path']}: WER={jiwer.wer(row['reference_text'], hyp):.3f}")
print("Overall WER:", jiwer.wer(refs, hyps))
