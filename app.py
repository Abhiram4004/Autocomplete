import nltk
from nltk.corpus import brown, reuters, gutenberg
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
from flask import Flask, request, jsonify, render_template
import re
import pickle
import os

app = Flask(__name__)

# --- FILES ---
TRIGRAM_FILE = "final_trigram.pkl"
BIGRAM_FILE = "final_bigram.pkl"
UNIGRAM_FILE = "final_unigram.pkl"

# --- MODELS ---
trigram_model = defaultdict(Counter)
bigram_model = defaultdict(Counter)
unigram_counts = Counter()

# --- CONFIG ---
# We explicitly allow these symbols now
VALID_PUNCTUATION = set(".,!?;:")

# Standard email phrases
EMAIL_PHRASES = [
    "please find attached", "looking forward to", "let me know", 
    "kind regards", "best regards", "thank you for", "hope you are doing well"
] * 50 

def train_on_sentence(sentence_tokens, boost=1):
    """
    Train models, now INCLUDING punctuation.
    """
    # 1. Filter: Allow Alphanumeric OR valid punctuation
    clean_sent = [w for w in sentence_tokens if w.isalnum() or w in VALID_PUNCTUATION or "'" in w]
    
    # Lowercase everything for consistency
    clean_sent = [w.lower() for w in clean_sent]

    for i in range(len(clean_sent)):
        word = clean_sent[i]
        
        unigram_counts[word] += boost
        
        if i > 0:
            prev = clean_sent[i-1]
            bigram_model[prev][word] += boost
            
        if i > 1:
            prev_two = (clean_sent[i-2], clean_sent[i-1])
            trigram_model[prev_two][word] += boost

# --- LOAD / TRAIN ---
if os.path.exists(TRIGRAM_FILE):
    print("⚡ LOADED PUNCTUATION-AWARE MODELS")
    with open(TRIGRAM_FILE, "rb") as f: trigram_model = pickle.load(f)
    with open(BIGRAM_FILE, "rb") as f: bigram_model = pickle.load(f)
    with open(UNIGRAM_FILE, "rb") as f: unigram_counts = pickle.load(f)
else:
    print("🐢 TRAINING PUNCTUATION MODEL...")
    try: nltk.data.find('corpora/brown')
    except LookupError:
        nltk.download('brown'); nltk.download('reuters'); nltk.download('gutenberg'); nltk.download('punkt')

    gen_corpus = list(brown.sents()) + list(reuters.sents())[:5000] + list(gutenberg.sents())[:5000]
    email_corpus = [word_tokenize(sent) for sent in EMAIL_PHRASES]
    
    print("--- LEARNING ---")
    for sentence in email_corpus + gen_corpus:
        train_on_sentence(sentence, boost=1)
        
    print("--- SAVING ---")
    with open(TRIGRAM_FILE, "wb") as f: pickle.dump(trigram_model, f)
    with open(BIGRAM_FILE, "wb") as f: pickle.dump(bigram_model, f)
    with open(UNIGRAM_FILE, "wb") as f: pickle.dump(unigram_counts, f)
    print("✅ Ready!")


# --- PREDICTION LOGIC ---

def get_next_word_prob(text_input):
    words = text_input.strip().split()
    if not words: return None, 0

    # 1. Trigram
    if len(words) >= 2:
        last_two = (words[-2].lower(), words[-1].lower())
        if last_two in trigram_model:
            return trigram_model[last_two].most_common(1)[0][0], 3 

    # 2. Bigram
    last_word = words[-1].lower()
    if last_word in bigram_model:
        return bigram_model[last_word].most_common(1)[0][0], 2 
    
    # 3. Unigram
    if unigram_counts:
        return unigram_counts.most_common(1)[0][0], 1 
        
    return None, 0

def predict_phrase(text, max_length=3):
    current_text = text
    phrase_output = ""
    
    for _ in range(max_length):
        word, score = get_next_word_prob(current_text)
        if not word: break
        if score == 1: break # Stop if guessing randomly
        
        # SMART SPACING LOGIC:
        # If the predicted word is punctuation, DON'T add a space before it.
        # Else, add a space.
        if word in VALID_PUNCTUATION:
            phrase_output += word
            current_text += word # No space in context either
        else:
            phrase_output += " " + word
            current_text += " " + word
        
    return phrase_output

def predict_completion(partial):
    candidates = [w for w in unigram_counts if w.startswith(partial)]
    if not candidates: return ""
    return max(candidates, key=lambda w: unigram_counts[w])[len(partial):]

# --- ROUTES ---

@app.route('/')
def index(): return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    # REGEX UPDATE: Allow letters, numbers, AND punctuation chars in input
    # This prevents the cleaner from stripping the "." you just typed
    clean_text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\']', '', text) 
    
    if text.endswith(' '):
        sugg = predict_phrase(clean_text, max_length=3)
        return jsonify({'suggestion': sugg, 'type': 'next'})
    elif text and text[-1] in VALID_PUNCTUATION:
        # If user just typed ".", predict the next word immediately (e.g. "The")
        sugg = predict_phrase(clean_text, max_length=3)
        return jsonify({'suggestion': sugg, 'type': 'next'})
    else:
        words = clean_text.split()
        if not words: return jsonify({'suggestion': '', 'type': 'none'})
        sugg = predict_completion(words[-1].lower())
        return jsonify({'suggestion': sugg, 'type': 'completion'})

@app.route('/learn_new_sentence', methods=['POST'])
def learn_new_sentence():
    data = request.json
    text = data.get('text', '')
    if text:
        tokens = word_tokenize(text)
        train_on_sentence(tokens, boost=5000) # Super Boost
        
        with open(TRIGRAM_FILE, "wb") as f: pickle.dump(trigram_model, f)
        with open(BIGRAM_FILE, "wb") as f: pickle.dump(bigram_model, f)
        with open(UNIGRAM_FILE, "wb") as f: pickle.dump(unigram_counts, f)
        return jsonify({'status': 'learned'})
    return jsonify({'status': 'error'})

if __name__ == '__main__':
    app.run(debug=True)