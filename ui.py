import subprocess, tempfile, glob, gradio as gr
from app import (transcribe, translate, make_notes, make_quiz,
                 format_quiz, LANGS)

def to_wav(src, max_min):
    out = tempfile.mktemp(suffix=".wav")
    cmd = ["ffmpeg", "-y", "-i", src, "-t", str(int(max_min * 60)),
           "-ac", "1", "-ar", "16000", out]
    subprocess.run(cmd, check=True, capture_output=True)
    return out

def fetch_link(url):
    d = tempfile.mkdtemp()
    subprocess.run(["yt-dlp", "-x", "--no-playlist", "-o", f"{d}/a.%(ext)s", url],
                   check=True, capture_output=True)
    return glob.glob(f"{d}/a.*")[0]

def run(file, link, lang, out_lang, max_min):
    try:
        if file:
            src = file
        elif link and link.strip():
            yield "Downloading from link...", "", ""
            src = fetch_link(link.strip())
        else:
            yield "Please upload a file or paste a link.", "", ""
            return
        wav = to_wav(src, max_min)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode(errors="ignore")[-400:]
        yield f"Could not get/convert the audio. Try uploading the file instead.\n\n{err}", "", ""
        return

    asr_code, src_code = LANGS[lang]
    yield "Transcribing...", "", ""
    transcript = transcribe(wav, asr_code)
    yield transcript, "Generating notes...", ""

    english = translate(transcript, src_code, "eng_Latn")
    notes = make_notes(english)
    yield transcript, notes, "Generating quiz..."

    quiz = make_quiz(notes)
    quiz_md = format_quiz(quiz)
    tgt = LANGS[out_lang][1]
    if tgt != "eng_Latn":
        yield transcript, "Translating output...", quiz_md
        notes = translate(notes, "eng_Latn", tgt)
        quiz_md = translate(quiz_md, "eng_Latn", tgt) if quiz else quiz_md
    yield transcript, notes, quiz_md

with gr.Blocks(title="Lecture -> Notes & Quiz") as demo:
    gr.Markdown("# Multilingual Lecture-to-Notes & Quiz Generator\n"
                "Upload a lecture file **or** paste a link. If both are given, the file is used.")
    with gr.Row():
        with gr.Column():
            file = gr.File(label="Upload audio/video file",
                           file_types=["audio", "video"], type="filepath")
            link = gr.Textbox(label="...or paste a link (YouTube / direct audio URL)")
        with gr.Column():
            lang = gr.Dropdown(["Hindi", "Kannada", "Tamil"], value="Kannada",
                               label="Lecture language")
            out_lang = gr.Dropdown(list(LANGS), value="English", label="Output language")
            max_min = gr.Slider(1, 15, value=3, step=1,
                                label="Use only the first N minutes (faster)")
            btn = gr.Button("Generate", variant="primary")
    t = gr.Textbox(label="Transcript", lines=6)
    n = gr.Markdown()
    q = gr.Markdown()
    btn.click(run, [file, link, lang, out_lang, max_min], [t, n, q])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
