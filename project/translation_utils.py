import re
import torch
from loading_utils import get_model, greedy_decode, get_tokenizers
from config import get_config, get_weights_file_path


language_models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


language_settings = {
    ("en", "it"): {"seq_len": 500, "pt": "final"},
    ("en", "fr"): {"seq_len": 550, "pt": "final"}
}


def initialize_translation_system():
    """
    Loads the model, vocabularies, and configuration once at startup.
    """
    global language_models
    for langs, settings in language_settings.items():
        lang_src, lang_tgt = langs

        # Create a configuration for the specific language pair
        config = get_config(seq_len=settings["seq_len"], lang_src=lang_src, lang_tgt=lang_tgt)

        # Load vocabularies and model
        print(lang_src, lang_tgt)
        vocab_src, vocab_tgt = get_tokenizers(config)
        model = get_model(config, vocab_src.get_vocab_size(), vocab_tgt.get_vocab_size()).to(device)
        print(config)
        # Load model weights
        model_filename = get_weights_file_path(config, settings["pt"])
        state = torch.load(model_filename, map_location=torch.device('cpu'))
        model.load_state_dict(state['model_state_dict'])

        # Store the loaded model and vocabularies
        language_models[(lang_src, lang_tgt)] = {
            "model": model,
            "vocab_src": vocab_src,
            "vocab_tgt": vocab_tgt,
            "seq_len": settings["seq_len"]
        }


def get_language_model(src_lang, tgt_lang):
    """
    Retrieves the model and vocabularies for a specific language pair.
    """
    if (src_lang, tgt_lang) not in language_models:
        raise ValueError(f"Translation system for {src_lang} to {tgt_lang} is not initialized.")
    return language_models[(src_lang, tgt_lang)]


def translate_sentence(src_lang, tgt_lang, custom_sentence, device):
    """
    Translates a custom sentence into the target language using the model.
    """
    language_data = get_language_model(src_lang, tgt_lang)
    model = language_data["model"]
    vocab_src = language_data["vocab_src"]
    vocab_tgt = language_data["vocab_tgt"]
    seq_len = language_data["seq_len"]
    custom_sentence = clean_translation_input(custom_sentence)

    encoder_input = torch.tensor(
        [vocab_src.token_to_id(token) for token in custom_sentence.split()] +
        [vocab_src.token_to_id('[EOS]')],
        dtype=torch.long
    ).unsqueeze(0).to(device)

    decoder_input = torch.tensor(
        [[vocab_tgt.token_to_id("[SOS]")]],
        dtype=torch.long
    ).to(device)

    encoder_mask = (encoder_input != vocab_src.token_to_id("[PAD]")).unsqueeze(1).to(device)
    decoder_mask = (decoder_input != vocab_tgt.token_to_id("[PAD]")).unsqueeze(1).to(device)

    model_out = greedy_decode(
        model, encoder_input, encoder_mask, vocab_src, vocab_tgt, seq_len, device
    )

    model_output_tokens = [vocab_tgt.id_to_token(idx) for idx in model_out.cpu().tolist()]
    translated_sentence = " ".join(model_output_tokens).replace("[EOS]", "").strip()

    return clean_translation_output(translated_sentence)


def clean_translation_output(sentence):
    sentence = sentence.lstrip("[SOS]").lstrip().rstrip().replace("[UNK]", "")
    sentence = sentence[0].upper() + sentence[1:]
    sentence = re.sub(r'\s([?.!,;])', r'\1', sentence)
    return sentence


def clean_translation_input(sentence):
    sentence = sentence.replace("'", "")
    sentence = sentence.lower()
    sentence = re.sub(r'\s([?.!,;])', r'\1', sentence)
    return sentence
