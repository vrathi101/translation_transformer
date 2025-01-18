from pathlib import Path


def get_config(seq_len, lang_src="en", lang_tgt="{1}"):
    return {
        "batch_size": 16,
        "num_epochs": 50,
        "lr": 10**-4,
        "seq_len": seq_len,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": lang_src,
        "lang_tgt": lang_tgt,
        "model_folder": f"weights/weights_{lang_src}_{lang_tgt}",
        "model_basename": f"tmodel_{lang_src}_{lang_tgt}_",
        "preload": "latest",
        "tokenizer_file_src": f"tokenizers/tokenizer_{lang_src}_{lang_tgt}.json",
        "tokenizer_file_tgt": f"tokenizers/tokenizer_{lang_tgt}_{lang_src}.json",
        "experiment_name": f"runs/tmodel_{lang_src}_{lang_tgt}"
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
