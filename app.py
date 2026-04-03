import gradio as gr
import torch
import nemo.collections.asr as nemo_asr
import librosa
import soundfile as sf
import tempfile
import os
from omegaconf import open_dict

print("Loading model... this might take a minute.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '/home/sriram1o1/au2tex/indicconformer_stt_te_hybrid_rnnt_large.nemo'

# Load the model globally
model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=model_path)
model.freeze()
model = model.to(device)
model.cur_decoder = 'ctc'

# Monkey-patch the tokenizer methods
original_ids_to_text = model.tokenizer.ids_to_text
original_ids_to_tokens = model.tokenizer.ids_to_tokens

def patched_ids_to_text(ids, *args, **kwargs):
    if 'lang' not in kwargs and len(args) == 0:
        kwargs['lang'] = 'te'
    return original_ids_to_text(ids, *args, **kwargs)

def patched_ids_to_tokens(ids, *args, **kwargs):
    if 'lang_id' not in kwargs and len(args) == 0:
        kwargs['lang_id'] = 'te'
    return original_ids_to_tokens(ids, *args, **kwargs)

model.tokenizer.ids_to_text = patched_ids_to_text
model.tokenizer.ids_to_tokens = patched_ids_to_tokens

if hasattr(model, 'decoding') and hasattr(model.decoding, 'tokenizer'):
    model.decoding.tokenizer.ids_to_text = patched_ids_to_text
    model.decoding.tokenizer.ids_to_tokens = patched_ids_to_tokens

# Update the decoding configuration
decoding_cfg = model.cfg.decoding
with open_dict(decoding_cfg):
    decoding_cfg.preserve_alignments = True
    decoding_cfg.compute_timestamps = True
    decoding_cfg.word_seperator = " "
    decoding_cfg.word_separator = " "
model.change_decoding_strategy(decoding_cfg)

print("Model and timestamp configuration loaded successfully!")

# --- NEW: Helper function to generate SRT content ---
def generate_srt(word_timestamps, time_stride, words_per_subtitle=5):
    srt_lines = []
    sub_index = 1
    
    for i in range(0, len(word_timestamps), words_per_subtitle):
        chunk = word_timestamps[i:i + words_per_subtitle]
        if not chunk:
            continue
            
        first_word = chunk[0]
        start_time = first_word.get('start_time', first_word.get('start_offset', 0) * time_stride)
        
        last_word = chunk[-1]
        end_time = last_word.get('end_time', last_word.get('end_offset', 0) * time_stride)
        
        text = " ".join([w.get('word', w.get('char', '')) for w in chunk])
        
        def format_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int(round((seconds - int(seconds)) * 1000))
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            
        srt_lines.append(f"{sub_index}")
        srt_lines.append(f"{format_time(start_time)} --> {format_time(end_time)}")
        srt_lines.append(f"{text}\n")
        
        sub_index += 1
        
    return "\n".join(srt_lines)

def transcribe(audio_path, lang_id):
    if not audio_path:
        return "Error: No audio provided.", "", None
    
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        mono_audio_path = tmp.name
        sf.write(mono_audio_path, y, sr)

    srt_file_path = None

    try:
        hypotheses = model.transcribe(
            [mono_audio_path], 
            batch_size=1, 
            return_hypotheses=True,
            language_id=lang_id
        )
        
        if type(hypotheses) == tuple and len(hypotheses) == 2:
            hypotheses = hypotheses[0]
        hypothesis = hypotheses[0]
        
        ctc_text = hypothesis.text
        
        timestamp_dict = getattr(hypothesis, 'timestep', getattr(hypothesis, 'timestamp', {}))
        word_timestamps = timestamp_dict.get('word', timestamp_dict.get('char', []))
        
        try:
            subsampling = model.cfg.encoder.get('subsampling_factor', 4)
            window_stride = model.cfg.preprocessor.window_stride
            time_stride = subsampling * window_stride
        except Exception:
            time_stride = 0.04 
            
        # --- NEW: Generate and save the SRT file ---
        if word_timestamps:
            srt_content = generate_srt(word_timestamps, time_stride)
            # Create a temporary file to hold the SRT data
            srt_tmp = tempfile.NamedTemporaryFile(suffix=".srt", delete=False, mode="w", encoding="utf-8")
            srt_tmp.write(srt_content)
            srt_file_path = srt_tmp.name
            srt_tmp.close()

        formatted_timestamps = []
        for stamp in word_timestamps:
            if 'start_time' in stamp:
                start = stamp['start_time']
                end = stamp.get('end_time', start)
            else:
                start = stamp.get('start_offset', 0) * time_stride
                end = stamp.get('end_offset', 0) * time_stride
                
            word = stamp.get('word', stamp.get('char', ''))
            if word and word.strip() != "":
                formatted_timestamps.append(f"[{start:0.2f}s - {end:0.2f}s] : {word}")
            
        timestamp_text = "\n".join(formatted_timestamps) if formatted_timestamps else "No timestamps detected."

    finally:
        if os.path.exists(mono_audio_path):
            os.remove(mono_audio_path)
            
    # Return 3 things now: Text, Timestamps, and the SRT File Path
    return ctc_text, timestamp_text, srt_file_path

# Build the Gradio UI
with gr.Blocks(title="Indic ASR Transcription", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Telugu IndicConformer Transcription (With Subtitles)")
    gr.Markdown("Upload a `.wav` file or record from your microphone to get the transcription, timestamps, and an `.srt` subtitle file.")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="Upload or Record Audio")
            lang_input = gr.Textbox(value="te", label="Language ID (LANG_ID)", lines=1)
            transcribe_btn = gr.Button("Transcribe Audio", variant="primary")
            
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Full Transcription", lines=3)
            output_timestamps = gr.Textbox(label="Word-Level Timestamps", lines=6)
            # --- NEW: File output component ---
            output_file = gr.File(label="Download Subtitles (.srt)")

    transcribe_btn.click(
        fn=transcribe, 
        inputs=[audio_input, lang_input], 
        outputs=[output_text, output_timestamps, output_file] # Map the 3 returns to the 3 UI elements
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)