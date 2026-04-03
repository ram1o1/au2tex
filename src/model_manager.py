import os
import requests
from tqdm import tqdm

# Dictionary mapping friendly names to their language codes and download URLs
INDIC_MODELS = {
    "Multilingual (600M)": {"code": "multi", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_multi_hybrid_rnnt_600m.nemo"},
    "Assamese": {"code": "as", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_as_hybrid_rnnt_large.nemo"},
    "Bengali": {"code": "bn", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_bn_hybrid_rnnt_large.nemo"},
    "Bodo": {"code": "brx", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_brx_hybrid_rnnt_large.nemo"},
    "Dogri": {"code": "doi", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_doi_hybrid_rnnt_large.nemo"},
    "Gujarati": {"code": "gu", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_gu_hybrid_rnnt_large.nemo"},
    "Hindi": {"code": "hi", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_hi_hybrid_rnnt_large.nemo"},
    "Kannada": {"code": "kn", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_kn_hybrid_rnnt_large.nemo"},
    "Konkani": {"code": "kok", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_kok_hybrid_rnnt_large.nemo"},
    "Kashmiri": {"code": "ks", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_ks_hybrid_rnnt_large.nemo"},
    "Maithili": {"code": "mai", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_mai_hybrid_rnnt_large.nemo"},
    "Malayalam": {"code": "ml", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_ml_hybrid_rnnt_large.nemo"},
    "Manipuri": {"code": "mni", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_mni_hybrid_rnnt_large.nemo"},
    "Marathi": {"code": "mr", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_mr_hybrid_rnnt_large.nemo"},
    "Nepali": {"code": "ne", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_ne_hybrid_rnnt_large.nemo"},
    "Odia": {"code": "or", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_or_hybrid_rnnt_large.nemo"},
    "Punjabi": {"code": "pa", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_pa_hybrid_rnnt_large.nemo"},
    "Sanskrit": {"code": "sa", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_sa_hybrid_rnnt_large.nemo"},
    "Santali": {"code": "sat", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_sat_hybrid_rnnt_large.nemo"},
    "Sindhi": {"code": "sd", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_sd_hybrid_rnnt_large.nemo"},
    "Tamil": {"code": "ta", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_ta_hybrid_rnnt_large.nemo"},
    "Telugu": {"code": "te", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_te_hybrid_rnnt_large.nemo"},
    "Urdu": {"code": "ur", "url": "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_ur_hybrid_rnnt_large.nemo"}
}

def get_or_download_model(language_name, progress_callback=None):
    """
    Checks if the model for the given language exists. If not, downloads it.
    Returns the absolute path to the .nemo model file.
    """
    if language_name not in INDIC_MODELS:
        raise ValueError(f"Language '{language_name}' not supported.")

    model_info = INDIC_MODELS[language_name]
    url = model_info["url"]
    file_name = url.split('/')[-1]
    
    # Define models directory at the root of the project
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    file_path = os.path.join(models_dir, file_name)

    # Download if it doesn't exist
    if not os.path.exists(file_path):
        print(f"Downloading {language_name} model to {file_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1 Megabyte
        
        with open(file_path, 'wb') as file:
            for data in tqdm(response.iter_content(block_size), total=total_size//block_size, unit='MB'):
                file.write(data)
                if progress_callback:
                    # Provide a rough progress update to Gradio if passed
                    progress_callback("Downloading model... Please wait.")
                    
        print("\nDownload complete!")
    else:
        print(f"Model for {language_name} already exists at {file_path}.")

    return file_path, model_info["code"]