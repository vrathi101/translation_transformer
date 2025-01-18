import warnings
from flask import Flask, request, render_template, jsonify
import torch
from translation_utils import initialize_translation_system, translate_sentence

app = Flask(__name__, static_folder='static')
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOURCE_LANGUAGES = {
    'en': 'English'
}

TARGET_LANGUAGES = {
    'fr': 'French',
    'it': 'Italian'
}

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        sentence = request.form.get("sentence", "").strip()
        src_lang = request.form.get("src_lang", "en")
        tgt_lang = request.form.get("language", "")

        if not sentence:
            return jsonify({"error": "Please enter text to translate."})

        try:
            translated_sentence = translate_sentence(src_lang, tgt_lang, sentence, device)
            return jsonify({"translated_sentence": translated_sentence})
        except ValueError as e:
            # Handle specific model tokenizer errors
            return jsonify({"error": str(e)})
        except Exception:
            # Catch all other unexpected exceptions
            return jsonify({"error": "The sentence is too complex or contains unsupported vocabulary. Please simplify your input."})

    return render_template("index.html",
                           source_languages=SOURCE_LANGUAGES,
                           target_languages=TARGET_LANGUAGES)

if __name__ == "__main__":
    print("Initializing translation system...")
    initialize_translation_system()
    app.run(debug=True)

# import warnings
# from flask import Flask, request, render_template
# import torch
# from translation_utils import initialize_translation_system, translate_sentence


# app = Flask(__name__, static_folder='static')
# warnings.filterwarnings("ignore")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SOURCE_LANGUAGES = {
#     'en': 'English'
# }

# TARGET_LANGUAGES = {
#     'fr': 'French',
#     'it': 'Italian'
# }


# @app.route("/", methods=["GET", "POST"])
# def home():
#     if request.method == "POST":
#         sentence = request.form.get("sentence", "").strip()
#         src_lang = request.form.get("src_lang", "en")
#         tgt_lang = request.form.get("language", "")

#         if not sentence:
#             return render_template("index.html",
#                                    error="Please enter text to translate.",
#                                    source_languages=SOURCE_LANGUAGES,
#                                    target_languages=TARGET_LANGUAGES)

#         try:
#             translated_sentence = translate_sentence(src_lang, tgt_lang, sentence, device)
#             return render_template("index.html",
#                                    translated_sentence=translated_sentence,
#                                    sentence=sentence,
#                                    src_lang=src_lang,
#                                    tgt_lang=tgt_lang,
#                                    source_languages=SOURCE_LANGUAGES,
#                                    target_languages=TARGET_LANGUAGES)
#         except Exception as e:
#             return render_template("index.html",
#                                    error="Please enter a simple sentence. Complex sentences may not translate well with this model.",
#                                    sentence=sentence,
#                                    src_lang=src_lang,
#                                    tgt_lang=tgt_lang,
#                                    source_languages=SOURCE_LANGUAGES,
#                                    target_languages=TARGET_LANGUAGES)

#     return render_template("index.html",
#                            source_languages=SOURCE_LANGUAGES,
#                            target_languages=TARGET_LANGUAGES)


# if __name__ == "__main__":
#     print("Initializing translation system...")
#     initialize_translation_system()
#     app.run(debug=True)
