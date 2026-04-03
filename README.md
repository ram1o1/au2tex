# Au2Tex: Multilingual IndicConformer ASR App

This project provides a Gradio-based web interface to transcribe audio into text using AI4Bharat's `IndicConformer` models. It supports 22 Indian languages and automatically downloads the necessary `.nemo` model files on demand. It also automatically generates word-level timestamps and `.srt` files for subtitles.

## Prerequisites
Ensure you have `uv` installed. If not, install it using:
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

## Installation

1. **Create and activate a virtual environment:**
```bash
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

2. **Install Python dependencies:**
```bash
uv pip install -r requirements.txt
```

3. **Install NVIDIA NeMo (Required by AI4Bharat):**
```bash
git clone [https://github.com/AI4Bharat/NeMo.git](https://github.com/AI4Bharat/NeMo.git)
cd NeMo
bash reinstall.sh
cd ..
```

## Usage
Start the Gradio web interface by running the application from the `src` directory:
```bash
uv run src/app.py
```

1. Open the provided local URL (usually `http://0.0.0.0:7860`) in your browser. 
2. Select your desired language from the dropdown menu.
3. Upload a `.wav` file or record directly from your microphone.
4. Click **Transcribe**. *Note: If this is your first time using a specific language, the app will automatically download the ~1GB model file into the `models/` directory before processing.*