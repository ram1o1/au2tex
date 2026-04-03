import gradio as gr
import torch
import nemo.collections.asr as nemo_asr
import librosa
import soundfile as sf
import tempfile
import os
import gc
from omegaconf import open_dict

from utils import generate_srt
from model_manager import INDIC_MODELS, get_or_download_model

# Global state to manage loaded models and save memory
CURRENT_MODEL = None
CURRENT_LANG_NAME = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(language_name, progress=gr.Progress()):
    """Loads the model into memory. Downloads it first if necessary."""
    global CURRENT_MODEL, CURRENT_LANG_NAME
    
    # Skip if the requested model is already loaded
    if CURRENT_LANG_NAME == language_name and CURRENT_MODEL is not None:
        return CURRENT_MODEL, INDIC_MODELS[language_name]['code']

    # Free up memory if a different model was previously loaded
    if CURRENT_MODEL is not None:
        progress(0, desc="Unloading previous model...")
        del CURRENT_MODEL
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    progress(0.1, desc=f"Checking/Downloading {language_name} model...")
    model_path, lang_code = get_or_download_model(language_name, progress_callback=lambda msg: progress(0.5, desc=msg))
    
    progress(0.7, desc="Loading model into memory (this takes a minute)...")
    model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=model_path)
    model.freeze()
    model = model.to(device)
    model.cur_decoder = 'ctc'

    # Monkey-patch the tokenizer for the specific language dynamically
    original_ids_to_text = model.tokenizer.ids_to_text
    original_ids_to_tokens = model.tokenizer.ids_to_tokens

    def patched_ids_to_text(ids, *args, **kwargs):
        if 'lang' not in kwargs and len(args) == 0:
            kwargs['lang'] = lang_code
        return original_ids_to_text(ids, *args, **kwargs)

    def patched_ids_to_tokens(ids, *args, **kwargs):
        if 'lang_id' not in kwargs and len(args) == 0:
            kwargs['lang_id'] = lang_code
        return original_ids_to_tokens(ids, *args, **kwargs)

    model.tokenizer.ids_to_text = patched_ids_to_text
    model.tokenizer.ids_to_tokens = patched_ids_to_tokens

    if hasattr(model, 'decoding') and hasattr(model.decoding, 'tokenizer'):
        model.decoding.tokenizer.ids_to_text = patched_ids_to_text
        model.decoding.tokenizer.ids_to_tokens = patched_ids_to_tokens

    # Update decoding configuration
    decoding_cfg = model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.preserve_alignments = True
        decoding_cfg.compute_timestamps = True
        decoding_cfg.word_seperator = " "
        decoding_cfg.word_separator = " "
    model.change_decoding_strategy(decoding_cfg)

    # Update globals
    CURRENT_MODEL = model
    CURRENT_LANG_NAME = language_name
    
    return CURRENT_MODEL, lang_code

def transcribe(audio_path, language_name, progress=gr.Progress()):
    if not audio_path:
        return "Error: No audio provided.", "", None
    
    # 1. Ensure Model is Loaded
    try:
        model, lang_code = load_model(language_name, progress)
    except Exception as e:
        return f"Error loading model: {str(e)}", "", None

    progress(0.8, desc="Processing audio...")
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        mono_audio_path = tmp.name
        sf.write(mono_audio_path, y, sr)

    srt_file_path = None

    try:
        progress(0.9, desc="Transcribing...")
        hypotheses = model.transcribe(
            [mono_audio_path], 
            batch_size=1, 
            return_hypotheses=True,
            language_id=lang_code
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
            
        # Generate SRT
        if word_timestamps:
            progress(0.95, desc="Generating subtitles...")
            srt_content = generate_srt(word_timestamps, time_stride)
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
            
    progress(1.0, desc="Done!")
    return ctc_text, timestamp_text, srt_file_path

# Build the Gradio UI
with gr.Blocks(title="Indic ASR Transcription", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ AI4Bharat IndicConformer Transcription")
    gr.Markdown("Select a language, upload audio, and the app will automatically download the required model and transcribe the file.")
    
    with gr.Row():
        with gr.Column(scale=1):
            language_dropdown = gr.Dropdown(
                choices=list(INDIC_MODELS.keys()), 
                value="Telugu", 
                label="Select Language Model",
                info="If the model isn't downloaded yet, it will download automatically upon clicking Transcribe."
            )
            audio_input = gr.Audio(type="filepath", label="Upload or Record Audio")
            transcribe_btn = gr.Button("Transcribe Audio", variant="primary")
            
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Full Transcription", lines=3)
            output_timestamps = gr.Textbox(label="Word-Level Timestamps", lines=6)
            output_file = gr.File(label="Download Subtitles (.srt)")

    transcribe_btn.click(
        fn=transcribe, 
        inputs=[audio_input, language_dropdown], 
        outputs=[output_text, output_timestamps, output_file]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)