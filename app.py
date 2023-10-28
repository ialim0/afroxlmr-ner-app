from flask import Flask, request, render_template
from transformers import AutoTokenizer, TFAutoModelForTokenClassification, pipeline
import re
import fitz
from io import BytesIO
import tensorflow as tf

# Forcer TensorFlow à utiliser le CPU
tf.config.set_visible_devices([], 'GPU')


app = Flask(__name__)
app.use_static = True

# Charger le modèle pré-entraîné et le pipeline NER
tokenizer = AutoTokenizer.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
model = TFAutoModelForTokenClassification.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_ner_entities():
    istext = request.form.get('text')
    isfile = request.files.get('file')

    if istext and isfile:
        return "Veuillez saisir du texte ou soumettre un fichier PDF."
        #return render_template('votre_template_d_erreur.html', error=error_message)
    elif istext:
        text = request.form['text']
        print(text)
    elif isfile:
        uploaded_file = request.files['file']
        text = extract_text_from_pdf(uploaded_file)
    else:
        return "Veuillez saisir du texte ou soumettre un fichier PDF."


    word_starts = get_word_starts(text)
    entities = assign_entities_to_words(word_starts, nlp(text))
    ner_results = [{"word": word, "entity": entity} for word, entity in entities.items()]

    # Initialisation des variables
    prediction_text = []  # Pour stocker les entités regroupées
    current_group = {"text": "", "entity": None}  # Pour stocker temporairement les entités en cours de regroupement

    for entity in ner_results:

        if entity['entity']=='O':
            current_group = {"text": entity['word'], "entity":'O'}
        else:
            if entity['entity'].startswith("B-"):
                current_group = {"text": entity['word'], "entity": entity['entity'][2:]}  # Enlever le préfixe "B-"
            elif entity['entity'] == "I-" + current_group['entity']:
                # Ajouter l'entité à la phrase en cours de regroupement
                current_group['text'] += " " + entity['word']

        print(element_existe(current_group, prediction_text))
        if not element_existe(current_group, prediction_text):
            prediction_text.append(current_group)

    return render_template('index.html', prediction_text=[prediction_text, text])

def get_word_starts(sentence):
    words = re.findall(r'\S+', sentence)
    word_starts = {}
    start = 0

    for word in words:
        word_starts[start] = word
        start += len(word) + 1  # Ajouter 1 pour l'espace

    return word_starts


def assign_entities_to_words(word_dict, entity_list):
    entities = {}

    for word_start, word in word_dict.items():
        entities[word] = "O"  # Par défaut, l'entité est 0

        for entity in entity_list:
            if entity['start'] == word_start:
                entities[word] = entity['entity']
                break

    return entities

def element_existe(element, tableau):
    return element in tableau

def extract_text_from_pdf(pdf_file):
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ''
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        pdf_document.close()
        return text
    except Exception as e:
        return f"Erreur lors de la lecture du fichier PDF : {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
