import streamlit as st
import nltk
import string
import datetime
import time

# Fallback Dictionary (Top 1000 common words as safety buffer)
FALLBACK_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
}

# NLTK Initialization with Error Handling
@st.cache_resource
def load_nlp_resources():
    try:
        nltk.download('words', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('brown', quiet=True)
        from nltk.corpus import words, brown
        from nltk.tokenize import word_tokenize
        from nltk import bigrams, FreqDist
        
        dictionary = set(words.words())
        
        # Build Bigram Frequency Distribution for context-aware correction
        print("Building bigram frequency distribution...")
        brown_words = [w.lower() for w in brown.words()]
        bg_freq = FreqDist(bigrams(brown_words))
        unigram_freq = FreqDist(brown_words)
        
        return dictionary, word_tokenize, bg_freq, unigram_freq
    except Exception as e:
        print(f"NLTK Load Error: {e}. Using fallback system.")
        return FALLBACK_WORDS, lambda x: x.split(), None, None

DICTIONARY_RAW, word_tokenize, BG_FREQ, UNIGRAM_FREQ = load_nlp_resources()
DICTIONARY = set(DICTIONARY_RAW)
DICTIONARY.update({"i", "im", "ive", "id", "youre", "theyre", "dont", "cant", "wont", "its", "thats", "ok", "okay", "yeah", "hey", "hi"})
print(f"Dictionary loaded: {len(DICTIONARY)} words")

# Session state initialization
if 'corrections_made' not in st.session_state:
    st.session_state.corrections_made = 0
if 'words_checked' not in st.session_state:
    st.session_state.words_checked = 0
if 'correction_history' not in st.session_state:
    st.session_state.correction_history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'last_results' not in st.session_state:
    st.session_state.last_results = None

# Page Configuration
st.set_page_config(
    page_title="GlitchCheck",
    layout="wide",
    page_icon="⚡"
)

# Boot Sequence
if not st.session_state.initialized:
    boot_placeholder = st.empty()
    boot_placeholder.markdown("""
        <div style="height: 80vh; display: flex; flex-direction: column; justify-content: center; align-items: center; background-color: #0d0015; color: #00ffff; font-family: 'Share Tech Mono', monospace;">
            <h2 style="text-shadow: 0 0 10px #00ffff; animation: pulse 1s infinite;">INITIALIZING NEURAL BUFFER...</h2>
            <p style="color: #ff00ff;">LOADING CORE_NLP_ENGINE v1.0</p>
            <style>
                @keyframes pulse { 0% { opacity: 0.4; } 50% { opacity: 1; } 100% { opacity: 0.4; } }
            </style>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(1.2)
    st.session_state.initialized = True
    boot_placeholder.empty()

# Core Engine Functions
import math
import re

def _get_edits1(word):
    """Generate all strings 1 edit away from word."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits   = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [L + R[1:]           for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces   = [L + c + R[1:]       for L, R in splits if R for c in letters]
    inserts    = [L + c + R           for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def _word_freq(word):
    """Return word frequency — higher = more common."""
    if UNIGRAM_FREQ:
        return UNIGRAM_FREQ.get(word.lower(), 0)
    return 0

def get_candidates(word):
    """
    Get correction candidates sorted by frequency (most common first).
    Edit-distance 1 candidates are always preferred over edit-distance 2.
    Within each distance tier, sort by corpus frequency descending.
    This ensures 'typing' beats 'throng' for 'typong'.
    """
    e1 = _get_edits1(word)
    e1_valid = [w for w in e1 if w in DICTIONARY]

    if e1_valid:
        # Sort edit-1 candidates by frequency — most common word wins
        e1_valid = sorted(set(e1_valid), key=_word_freq, reverse=True)
        return e1_valid[:10]
    
    # Only go to edit-2 if NO edit-1 candidates exist
    e2 = set(e2w for e1w in e1 for e2w in _get_edits1(e1w))
    e2_valid = [w for w in e2 if w in DICTIONARY]
    e2_valid = sorted(set(e2_valid), key=_word_freq, reverse=True)
    return e2_valid[:10]

def correct_word(word):
    """Correct a single word. Returns (corrected, was_correct, alternatives, confidence)."""
    clean = word.strip(string.punctuation).lower()
    if not clean:
        return (word, True, [], 100)
    if clean in DICTIONARY:
        return (word, True, [], 100)

    candidates = get_candidates(clean)
    if not candidates:
        return (word, False, [], 0)

    best = candidates[0]

    # Confidence: log-scale frequency ratio vs max frequency word
    if UNIGRAM_FREQ and UNIGRAM_FREQ.values():
        max_freq = max(UNIGRAM_FREQ.values())
        word_freq = UNIGRAM_FREQ.get(best, 1)
        confidence = min(97, max(40, int(math.log(word_freq + 1) / math.log(max_freq + 1) * 100)))
    else:
        confidence = 60

    # Preserve original capitalisation
    if word and word[0].isupper():
        best = best.capitalize()
        candidates = [c.capitalize() for c in candidates]

    return (best, False, candidates[:5], confidence)

def correct_sentence(text):
    """
    Two-pass context-aware correction.
    Pass 1: Correct each word by frequency (most common candidate wins).
    Pass 2: Re-rank using bigram context — if a neighbour pair strongly
            favours a different candidate, switch to it.
    """
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # Pass 1 — frequency-based correction
    results = []
    for token in tokens:
        if all(c in string.punctuation for c in token):
            results.append({'orig': token, 'best': token, 'correct': True, 'alts': [], 'conf': 100})
            continue
        corrected, was_correct, alts, conf = correct_word(token)
        results.append({'orig': token, 'best': corrected, 'correct': was_correct, 'alts': alts, 'conf': conf})

    # Pass 2 — bigram context re-ranking (only overrides if context score is strong)
    if BG_FREQ:
        for i in range(len(results)):
            if results[i]['correct'] or not results[i]['alts']:
                continue
            prev_word = results[i-1]['best'].lower() if i > 0 else '<s>'
            next_word = results[i+1]['best'].lower() if i < len(results) - 1 else '</s>'

            best_cand   = results[i]['best'].lower()
            best_score  = (BG_FREQ.get((prev_word, best_cand), 0) +
                           BG_FREQ.get((best_cand, next_word), 0))

            for cand in results[i]['alts']:
                cand_lower = cand.lower()
                score = (BG_FREQ.get((prev_word, cand_lower), 0) +
                         BG_FREQ.get((cand_lower, next_word), 0))
                # Only override if context score is meaningfully higher (threshold 2)
                if score > best_score + 2:
                    best_score = score
                    best_cand  = cand_lower

            # Apply context winner (restore capitalisation if needed)
            orig = results[i]['orig']
            if orig and orig[0].isupper():
                best_cand = best_cand.capitalize()
            results[i]['best'] = best_cand

    return [(r['orig'], r['best'], r['correct'], r['alts'], r['conf']) for r in results]

# Custom CSS for Cyberpunk Aesthetic
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

    /* Global Overrides */
    * {
        border-radius: 2px !important;
    }
    .stApp {
        background-color: #0d0015;
        color: #ffffff;
        font-family: 'Share Tech Mono', monospace;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Scanline Effect */
    .stApp::before {
        content: " ";
        display: block;
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
        z-index: 100;
        background-size: 100% 2px, 3px 100%;
        pointer-events: none;
    }

    /* Glitch Animation */
    @keyframes glitch {
        0% { transform: translate(0); color: #ff00ff; }
        1% { transform: translate(-4px, 2px); color: #00ffff; }
        2% { transform: translate(4px, -2px); color: #ffff00; }
        3% { transform: translate(0); color: #ff00ff; }
        100% { transform: translate(0); color: #ff00ff; }
    }

    .glitch-header {
        color: #ff00ff;
        font-size: 4rem;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 0;
        text-shadow: 0 0 10px #ff00ff;
        animation: glitch 5s infinite;
        font-family: 'Share Tech Mono', monospace;
    }

    .subtitle {
        color: #00ffff;
        font-size: 1rem;
        letter-spacing: 0.2rem;
        text-transform: uppercase;
        margin-top: -10px;
        text-shadow: 0 0 5px #00ffff;
        font-family: 'Share Tech Mono', monospace;
    }

    /* Surface/Card Styling */
    .stCard {
        background-color: #1a0030;
        border: 1px solid #cc00cc;
        border-radius: 2px;
        padding: 20px;
        box-shadow: 0 0 10px #ff00ff;
    }

    /* Override Streamlit default container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Custom Input Styling */
    div[data-baseweb="textarea"] {
        background-color: #0d0015 !important;
        border: 1px solid #ff00ff !important;
        color: #ffffff !important;
        box-shadow: 0 0 5px rgba(255, 0, 255, 0.2);
    }
    textarea {
        background-color: #0d0015 !important;
        color: #ffffff !important;
        font-family: 'Share Tech Mono', monospace !important;
        caret-color: #ff00ff !important;
    }

    /* Button Styling */
    .stButton > button {
        width: 100%;
        background-color: #ff00ff !important;
        color: #000000 !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-weight: bold !important;
        border: none !important;
        text-transform: uppercase !important;
        box-shadow: 0 0 10px #ff00ff;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        box-shadow: 0 0 20px #ff00ff;
        transform: scale(1.01);
    }

    /* Correction Highlights */
    .word-correct { color: #ffffff; }
    .word-corrected { 
        color: #ff00ff; 
        text-shadow: 0 0 8px #ff00ff;
        cursor: help;
        border-bottom: 1px dashed #ff00ff;
    }
    .word-unknown { 
        color: #ffff00; 
        text-shadow: 0 0 8px #ffff00;
        border-bottom: 1px dashed #ffff00;
    }

    .output-container {
        background-color: #1a0030;
        border: 1px solid #00ffff;
        padding: 20px;
        min-height: 100px;
        margin-top: 10px;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.2);
    }

    .section-header {
        color: #00ffff;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 0.1rem;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

    /* Stats & Log Styling */
    .stats-card {
        background-color: #1a0030;
        border: 1px solid #ffff00;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(255, 255, 0, 0.3);
        animation: pulse-yellow 4s infinite;
    }
    @keyframes pulse-yellow { 0% { box-shadow: 0 0 5px rgba(255, 255, 0, 0.2); } 50% { box-shadow: 0 0 15px rgba(255, 255, 0, 0.4); } 100% { box-shadow: 0 0 5px rgba(255, 255, 0, 0.2); } }
    
    .stat-line {
        font-size: 0.9rem;
        margin: 5px 0;
        display: flex;
        justify-content: space-between;
    }
    .stat-value { color: #00ffff; }

    .log-card {
        background-color: #0d0015;
        border: 1px solid #00ffff;
        padding: 15px;
        max-height: 400px;
        overflow-y: auto;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.2);
    }
    .log-entry {
        border-bottom: 1px solid rgba(255, 0, 255, 0.2);
        padding: 10px 0;
        font-size: 0.85rem;
    }
    .log-timestamp {
        font-size: 0.7rem;
        color: #888;
        margin-bottom: 4px;
    }
    .log-arrow { color: #ff00ff; }
    .alt-pill {
        display: inline-block;
        background: rgba(255, 255, 0, 0.1);
        border: 1px solid #ffff00;
        color: #ffff00;
        padding: 2px 6px;
        font-size: 0.7rem;
        margin-right: 4px;
        margin-top: 4px;
    }

    /* Progress Bar Override */
    div[data-baseweb="progress-bar"] > div > div {
        background-color: #ff00ff !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #00ffff;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.7rem;
        margin-top: 4rem;
        opacity: 0.7;
    }
    </style>
    """, unsafe_allow_html=True)

# Header Section
LOGO_B64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/4QCORXhpZgAATU0AKgAAAAgABAEPAAIAAAAMAAAAPgEyAAIAAAAUAAAASpADAAIAAAAUAAAAXpAEAAIAAAAUAAAAcgAAAABJZGVvZ3JhbSBBSQAyMDI2OjA0OjA5IDAyOjI4OjQwADIwMjY6MDQ6MDkgMDI6Mjg6NDAAMjAyNjowNDowOSAwMjoyODo0MAD/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCAQABAADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD5UoopK+gNxaKKKACikooGLRRRQAUUlFAhaDSClpgFJRRSAWikFLQIKWkooAWiikoAKKKKBBRRRQAtFJRQMWigUUwCiiikIKWkopgLRRRSAWikoNMApKKKQCilpBS0xiE0UUd6BCiiig9KYCUCkpR1oAWlFJRQMWikpeaACigUUxBRRRSAKKKWgBKKKKYwoNFFAhKKKKQBS0lFAC0UZooAWikpaYBR+FFFABRRijtQAUUUUAApaKKACiiimAooNFBoENNFBooAUUUCigAozSUUDFooooEFFFFIApaSlpgJRRRQIKKKKAFpKWimAUUUtACUYopaBXExS0UUAJijFLRQMTFGKWigQ3FGKdTaQxcUYpaKYWGkUAU6ikAhHFVpqtN0qrL3oYinKOagIqxLUJ60gtcYBU0QplSRihCtYtQiraDiqsAq4nSrBDgKUCkFLQUBooopXAMUmBS0GgRTNFBpKzRqhaKSloAKKKKQwooooAKKKKBBRRS0AJRRRQIWiiigAooooAKKKKBBRRRQAUUUUwCiiikAUtJS0wCiiikAUUUtMAoopKAFooooASiiikAUtJRTQC0opMUtABRRRTATFLRRQAUUUUDCloooAKKKKACiiigAoopaAEoopaYBSUtFAhtFKRSUgCijFGKYBS0mKWgAopaKACiiigAoo70UAGKKWigAooopgFLSUUCFopKKACjFFFMAooooEFJS0UhhRRS0wEooopAFFFFMQUlLRigAoooxQAoooApaBBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAwooooAKBRQOtAA3Sq8oqyeRUUi5pMTKEoqFhV2SPNRGI0rgiuq5NTRrTliNTJHQA+FasL0piDFSCqAKUGkooHcdRmkFBoACaKSigCqetJRRWRqgpaSimAtFFFIApaKKBCUUtJQAtFFFMBKWiikAUtJS0CCiiigAopaSgAooooAKKWigBKKKKACloopgGKKWikIBRRRQAGm0402mgFoxQKWgBMUYpaKB2DFFFFABRRRQAUUUUwCiiloAMUUUUAFFFJQAtFJS0AFFFFABS0UUIAooopgFFFFAgoxS0UAJijFLRQAlFFFABS0UUAJS0UUAFFFFMAooopAFFFFABRRRQAUUUUwA0UGkoELRRS0wEpaKKQBRRRTEJRRRQAUuKKWgBMUYpaKACiiigAooooAUUhoFLQAlFFFAgooxS0WGJRRRQAUUtFACUUtFOwCUUtFFgsJRRRSsAtBFFFFgGFQabsFSGilYCMIPSnhaWlosAgFLmkzRTELmikpRTsFgpaSikAUUUUAVKKKKyNkFFFFMBaKKKAFooooEJS0UlAC0UUUAFFFFIQUUUUAKKKKKACiiigAooooAWiiimAlFLSUgClFJRTEOopM0ZpALRSZop2AKSlopgApaQGloAKKKKBhRRRQAUUUUAFFFFABS0UUAFFFFACUUUUAFLRiigBaKKKACiiigApKWimAUUUUCFopKKAFopKKACiiigApaSigBaKKKYBRRRTAKKKKQBRRRRYAooooAKWkpaYhKKKKACiiigQtFFFABRRSUAFFFFACilpKM0ALSUUUALRRRQAUUUUAFFLRQAlLRRTAKKKKAEooooAKWkpaAEpaKKACiiigANJS0UAFJS0hoAQ9aKWkpAFLSUtMAooooAKKKKAFooooASiiigRVooorJGwUtFFABS0UUAFFFFABSUtJQAtFFFAgpaSlpCCiiigAooooAKWikpoAooooAWikFLQAUUUUCCkp1IaAEoooFIAoopaYCUUUUxiiiiigBaKSigBaKKKAClpKWgAooooAKKKKYBRRRQAUUUUgClpKKAFopKWgAooopgFJRRQAtFJS0AFFFFAhaSiimAUUUlIBaWkpaYBRRRTAKWkooAKKKKBC0UUUAFJS0UAFFFFABRRRQIKWiigAooooAKQ0tJQAlFFKBQAUUCloATFLRRQAUUUUxhRRRQAUUUUCFooooAKKKKAEoooosK4ClpKWgYlFFFAC0lLQaBCUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFBoAq0UUtZI2CiiigApaSloAKKKKACkpaKBBRRRQAUtFFIQUUUUAFFFFAC0lLRTASilpKACiiigQUtJS0wFpDR0oxSQMSiiiiwwooop2AKKKXFFgAUtFFABRRRQAUe9FFABS0lFAC0UlFMApRSUtABRRRQAUUlFAC0Ud6KACiiloAKSlpKACiiigAooooEFLSUUALRRRTASloooAKWkFLTAKKKKACloooEJRS0UCCiiigYUUUUAFFFFABS0gooAWiiigQUUlFABRRRQAUCiigBaSlopjCiiigQUUUUDCiiigLhRRRQIKWkFFAC0UUUAJQKKWgAoNFJQJhQKUCigYUGigigBKKKKYgooooAKKKXFIBKKKKACiiigAooooAKKKKACiiigCt3ooorE2CiiimAUUUUCFooooGFFFLQIKKKKBBRS0UgCiiimAlFLRQAUUUUAFFKKKAEpcUUUgExQKWkoAWiiimAhFFLSU7gApcUUUwCiiikAtFFFACUUGigAooooAKKKKLAFFFLTASlpKKAFooooAKSlpKAFooooAKWkpaACkpaKAEooooEFLSUtMAooopALRQKKoAoooqQCiiimAUUUUwFooooEFFLRQIKKKKACkoooAKKKWgBKKKKBi0lLRQISiikoAWigUUAFLRRTQBRRQKAFxRRRQAUUUUAJRQaKACilpKBBRRRQAoopBS0DQUUUUAFFFFMAoooosAUUUU7AFFFFIAoxRRQAUUZooADSUtBoASiiigVgooooCwUUUUAKKKKKQFSilorI2CiiigAooooAWiiimIKKWikAUUUUCClpKKQC0UUUwCiiigAooooEFLSUChjQtFFFKwAaSgmkqkgClzSUUwFooFLSAKKKKACiiigBaKSigAooooAKKKKACiilpgJS0UlABRRRQAUtJRQAUUUUAFLSUtABRRRQAUUUtAhKKKWmAUUUUAFAoxS0AFFFFMBKKKKQBS0lLTAKKKKAFooooELRRRQIKKKKAEooooAKM0lLQAUtJS0AFFFFACUUUtABRRRTAKKKKACgUUUALSUtAoABRRQaAEooooEKKKKKBiUUtIaYC0ZpKMUWAWjNJigCgBaKKKYBRRS0AJRS0UAJRRRQAUUUUgEpaSloC4UhpaQ0AFFFFOwXCiiigLgKBS0UCCiiikBVpaKKxNgoooNMAoopKQhaKKWgAooopCFpKWimAUlFFIBaKKKYBSUUUAFLSUtMQAUZozSUDHUU2iiwC96KKBQAtFFFABRRSUALRRRQAUUUUAFFFFMAooooAWjFFFABRRRQAUUUUAFJS0lABRS0lABRRRQAtFApaYCUUUUgFoopaBCUUtFABRRRVAFFFFIAooopgFFFFABRRRQAUUUUCFpaQUUABozSUtABS0lFABSUppKBC0UUtABRRRQAUUUUAFFFFABRQKXFACUUUU7AApaQUtABRRRQAUUUUAFFFFABRRRQAUhoNLTABRRRTAKKKKAuFFFLQISlpKKQC0UUUAJRRRQAUUUtACUYpaSkAUUtFMBKMUtGKAEpcUUUwCkpaKQBRRRQBVooorI2FpKKKACiiigAoopaBBS0lLSEFFFFACUUUtABRRRQAlFFFABRRRTAKKKKEAUUUUwFpabS0gFooooAKSiigAFLSUtABRRRTAKKKMUAFLRSUALRRRQAUtIKWgApKKKYBSUUtIApKXFFMBKWiigBaKKKBBRRilpAIKWiigAooooAKKKKYBRRSUALSUUUwCiiigBaKKKBBRRRQIKWgUUDEpaKKACiikoEFFFKBQACloooEFFFFAwooFLigBKKXFFMAooooASiiigApaKKACiiimAUUUUAFFFFABRRRQFwooooC4UUtJTEFFFLSAKKKKACiiikAUUUUwEopaSgQUUUUDFooooEFFFFABRRRTAKKKKACiigUgCiig0AVaBRRWJuLRRRTAKKKKACilooEwopKWkIKKKKACiiikAUUtFMBKKUUlABSUUU0AUUUUIAoopaYBS0lFIBaKKDQAUlFAoGFLRRTAUUUCloASjFKKKBCUUUUwCilooASloopAFFApaYCYopaKAExRS0UwEopaKkQCiiimAUUUUgCiiimAUUUUwClpKKAA0lLRQAUUCigBKKU0lABS0lLQAUUUUCFpaKKACiikoEGKMUtFACYpaKKACiiigAoAopaACiiimAGiikoAWiiigAooooAKKKKYBRRRQAUtFFMApKKKQgpaKKACiiigApaKKYCUUUUgFooooAKSlooASilooEJRS0UAJiloopjCiiikIKKWigBDSUtJQAUUUtMAooooAKDRRQBUooorA3ClpKWmAUUUUAFFFFAgooooAWikpaQgoFFFAC0UgooAM0lFFMBaKKSgAooooQC0UUtACUUUtABRRSUAFKKTFLQMKKKKYhRSikooGLRRRTEJRRRQAtFFJQAUtJS0AFKKSloAKKKKYBRRRQJhRRRSYBRRRSAKKKKACiiiqAKKKKACjtS0dqAEoFFFAIKKKKBgaSlooBiUtJS0EhS0lLQAtFJRQIKWkFLQAUUUUAFFFFABRRRQAopKKKYC0UUUAFJS0UAFFFFABRRRQAUUUUwCiiloEFFFFACUUtJQAtFFFMAooopALRRRTASilpKBC0UUUDCiiikIKKKKYBSUtFABRRRQAUtJRSAWiikNAAaSlpKYhaKKKACiiigAoooNAFSiiisToFFFAooAKKKKACiiigQUlFFAC0UCikIWikooAWkoopgFFFFABRRRQAUoopaAAUUUUAFFFFABRRRQAUUUUWAKKWimAlLRRQAtFFFMBKKKKQBS0UUwCloFLQAlFLRTAKSlpKQgooooAKKKKQBRRRTAKKKKAClpKDTAKKKKACiiigAooopALikpRQelMYlFBpKQhaWkpaYBRRQKBC0lLSUCAUtJS0AFFFFABRRRQAUUUtMBBS0UUAFLSUtABSUUUALSUtJQAUUUUCClpKKYC0lFLQAUUUUAFFFFABRRS0wEooopALRQKKYBRRRQIKKKWgBKKKKACiiigQUUUUAFFFFABRRRSAKKKKYBRRRQAUUUUAFFLRQAUGiikBToopayOgKKKKACiiigAooooASilooEJS0CigQUUUUAFFFFABRRRQAUUUtMAFLSUtIAoooNABSUUUDFooopgFFFLQAUtJRQIKKKSgYtFFFMQUUUUALSikpRQAUtJS0wCg0lFABRRRSEFFFFABSClopAFFFFACUtFApgFFFFMYUUUUguFFFFFxBRRRSAUUGiiqGJRS0lAgpRSUtABRQKWgQUlFGKBBS0UUgCiiimAUUlLQAUUUUwFopKWgAooooAKKKKACiiimAUUUUhBRRRTAKKKKAFooopgFFFFIAooopgFFFFABS0UUCCiiigLhRRRQIKSiloAKKKKACiiikAUUUUwCiiikAUUUUIAooopgFFFFABRRRQMWiikpAVe9FFFZG4UtJS0AFGKKKACkpaKAEooooEFFFFAgooooAKKKKACiiloABS0UUAFJRRQAtFJS0AJS0UUxhRRRQIKWikoKFopBS9qBBRSUtABS4ooqhBRS0UrAAooFLTAKSiigAooooEFFFFIAooooQBRQaSkAtFApaYxKWigUAGKMUtFMBDRxQaSlcBcUEUtFMBtKKMUtKwBRRRTAKQ0tIaACigUUCCiiigTCiiigQd6WiigAooooAQ0Cg0ChALS0lLTASiiigBaKBQKACloooAKKKKYBRRSUhC0UUUwEooooAWiiimAUUUUgCiilpiEoFFLSAKKKKYgooopAFFFFABRRRQAUUUUwCiiikAUUUUx2CiiikAUUUUwsLikoooAKKKKYBRRSUgFooooAq0UUVibhS0UlAC0UUUAFFGKKYCUUUUCCilpKQBRRRQIWiiigAoFFAoAWiiigBKKKKACilooGJS0UUxBS0lLTGJRRS4pDAUUUUxCUtFFABS0UUxC0UUUAFFFFABRRRQAUUUUCEoFFFKwC0UlFABS0UUDCiiikFwoopRTAWiiimA2iloxSAWiiimAUUUUCuFFFFABSGlpKBiUtFFAgpKWkoJCloAoxQAtFIKM0gFooopgIaWiimAUUUUAFFFFAAKWilpgFFFFIAooopgJRRS0CCiiigBKKKKAFooooAKKSigBaKKKYBS0lLSEFFFFMAooopAFFFFAhKWiigAooooAKKKKACiiimMKKKKACiiigYUUUUwCiiigQUUUUAFBoopAVB1paTvS1ibhS0gooAWiiigAooopgFFFFABRQKKBBSUUUhC0UUtACUUUUAFFFFABS0lLQMKKKKYBS96BRTAKKWigYlFBpKYC0tJRQAtFFFAgpaKSgApaSigBaKSg0AFGaKSgBaWkooFYKKKWiwWCjFFLQAlFFFABS4pKUUgDFLSUZpgLRSUtACUtFFABRRRQAUopKWgEFBFFFACUUGigAopKWmIKTFLRSJCkNLSGgBKKKKACnU2nUIBKKWkpgLRSUtMApaKKQC0UUUALRRRQAUUUUwEooopCCiiimAUUUUAFFFLQAlFFFABS0lLTEFFJS0AFLSUUALSUUUALRRRQAlFLSUgsLijFFFMBKKKKACiiigYUUUUAFFFFMBaSiikIKWkpaYBRSUtIQUUUUAU+9FApaxOgKKKWgAooooAKKWkpgFFFFABRRRQISloopCCiiigAooooGFFFGKAAUtFFMApe1JS0wCigUtAwoopKYCGilooAKKKKACloooEFFFFAwooooAKKKKACkpaKACiijFABS0UUwCiiikIKKKKBBRRRQAUUUtABRRS0AFFFFABRRRQAClpKWgAoJoNJQAUlLSYoEFFLRQIKKKKBBSUtJmgBKKKWgAFFKKKYBSUtFMYlKKKWgApaSlpCCiiigApaSlpgFFFJQAUUUUCCiiigAooooAWkpaKAEooooAKWkopgFFFFAC0UUUCDvRRRQAUtFFABSUtJQAUUtFACUUUUDCiiigAooopgFFLRQISiilpAFJS0UxCUUUUAFFFFICoKWgdaKxOgKUUlLQAtFFFABRRRQAUlLRTASilooEJRQaKAClpKWgQUUUUhhRS0UwsJS0UlAC0UClpjCiikpgLRRRQAlFFFABSikpaACiiigQUUUUDCiiigBaQ0UUAFFFFABS0lLQAUUUUAFLSUtAhKKKKACiiigBaKBS0AFWtNsbi/nMNt5e8LuO+QIMfUmqtIRmmVDlT95XX3fo/yNz/hGNU7myH1u0/xoHhfVP71h/wCBkf8AjWCQPQU63ikuJRFbwvM5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UUUCCiiigApaSloQxaKQUtMYUUUUAJRRRQAtFJS0CClFJSigYUGig0CEooopgFAoooAWigUUAFFFFMBaSilpAFJS0UAJRRRTAKKKKACiiigBaKSloEFLSUtABRRRQAtFJS0ALSUUtADaKKKQBRRRTAKKKKQDhRRRTAKQUtJQAtFAooAKKKKACiiigQUuabS0wFopKKACilxSUDuFFFLTEJS0UUAFFAooJCiiikAUUUUAFFFFABS0lFAC0UUUALSUUUAFFFFABRRRQAUlFFMAooooAWiiigBaSiimAlLRSUALSUUGgQUUUtACUUtJQAUUUUAFLSUtABS0lFAhaM0UUAJRRRQAUtIKWgAooooAKKKKAKgpaQdaWsTZCUUpopgGKKKKQgooooAKM0UlCAWiiimAtFJS0hhRRRTGLSUtFACUtJS0CClpKKAFpaSloASgilpDQMBRRRQSFFFFAwpaSlpiCiiikMKKKKYBRRRQAUUUUwEopaKACgUUtAgooooAKKKKACiiigBaKSloASiiikAtJS0UwEoFFFADqKQUtABRRQKACil6UGgBKKKKACkpaSgGFFFFAgpaSloAKKKKAFooopgFJRRQIWiiigAooooEFFFFABS0lFABS0lLQAUUUUAFFFFMAoooosAUUUUwEooooAKWkooAWkoooAKWkooAKWkpaACiiigQUUUUgFpKWkpgFFFFABS0lLQAUUlLSAKKKKYCUUGikAUtJS0xBRRSUALRSUtAFQdaWkFLWJsJS0lLQIKKKKACiiigAoxRRQAUUUUAFLSUUxi0UlLQAUtJRQAtFJRQAtFFFIAooopgKDRSUtAIDSUtJQAUtJRQAtFFFAC0UlFAhaKSimAtFJS0DCiiigAoopKBC0tJRTAWikFFAC0UUUAFFFFABRRRQAUUUUAFFFFABQKKKAFopKKAHUUCkoAWikooEGaWkpaBhSUGigGFFJRQIWgUlKKAFooooAKWkopgFFFLQISiiimMKKKKQhaKKKBBRRRQAUUuKMUAJS0lFABS0lFMBaSiii4BRRRRcAooopgFFFFABRRRQAUUUUAFLSUUALRRSUALRRRSAKKKKEIKKKKYBRRRQAUtIKKQC0lLSUwCgUUUgCiiimAUUUUhBRRSUxlYdaWjvRWJqFFFLQIKKKKQBikpaDQAlFFFMAooooAKKKKAClpKWgAooooGFFFAoAWiiigAooooQC0UlLTAWkpaSgBKKU0lAhaKBRQAUUUUAFFFFABS0lFAC0UUUwCiiikMSloopiCikpaEAUtJS0wA0lBopALRRRTAKKKKQBRRRTAKKKKQBRRRQAtFJRTAKWkoFACilFJRQAUGig0CEooooAKUUlLQAtFFFABRRRQAUUUUwCilpKYBRRRSAWlpKKCQoopaQC0UUGmAlJRRQAUUUUAFFFFABRRRQAUmaU0lNALRQKKYBRSZopWAWikzS0AFFGaKACiiimAtJRRSAKWkopiFopKKAFooooAKKSigBaKKKACkoopALRRSUwFpKKKQBRRRSAr0UgpazNQpaSigQtFJS0gCiikoAKKKKLgFFFFMAooooAKKKKYBS0lFIBaKSigBaWkooGFFFFAhaM0UUDFopKKYC5opKWgBKWg0lAC0UCigQUUUUAFFJS0ALRSUtABRRRQMKKKKBCUUUUwFopKWgAoFFFAC0UlLTAKKKKQBRRRTAKKKKQBRRRTAKKKKACiiigApaKKACiikoEFFFFABS0lFAC0tIKWgAooooAKKKKYBRRRQAUUUYoAKM0UUCFzRSUooELmikooAKKDRmgAooooACaSiikAoopKKLgLRSUZphcBS0maKdwCiiigAooopgFFFFAC0Ugo70ALRQaO9ABRRRQAUUUUAFFFFABRRRQIKKKKACilpKACiiigAoopKVgCiiigCuKWkFLWRqgooooAWikpaQgoopKACiiikAUUUU0AUUUGmAUUUUAFFFFAC0UUUAFLRRQAUUUUAFFFFABRRRTAKdTaUUDQtJS0UAIKWiigAoNFFACUCiigQUUUtABSikpRQMKQilooENopaSmACloooAKKKKAFFFFFMAoopKQC0UUUABooooAKKKKYBRRmigAxRRS0AFFFFABRRSUCCiiigApaSloAKWkoFAC0UUUAFFFFABS0lFMBaKKKACjFFFABRRRQIKKKKBCGig9aKQCiigdKDTASiiikAUUUUAFFFFMQUUUUAFFFGKYwooopgFFFFABRRRQAUUUUAApaQUtABRRRQAUUUUAFFFFABRRRQAUUUUCsFFFFAWCiiigAooopMCtS0lKKyNUFFFFABRRRSEFFFFABRRRSAKKKKACg0UUwCiiimAUUUUAFLSUUwFooooAWikpaQBSUtJQAUUUUALRSUtAC0UlFMYtLSUUALRSUUCCiikoAWiiigApaSloAXNJRRQAUUUUAFFJS0wClpKKAFooopgJRRRUgLRRRQAUUUVQBRRSUAFLSUUALS0lFAC0UlKKACkpaSgQUUUUAFLSUUAFLSUtAC0UUUAFFFFABRRRTAKWkpRQAUUUUAFFFFABRRRQSBpKKWgAoNFBoASiiigQUUUUAFFFFABRRRQAUUUUAFFFFO4XCiiii47hRRRTAKKKKACiiigApQaSjFAC0UUZoAKKSlpAFFFFMAooooAKKKKACiig0AFIaKKTEV6WkorI1QtFFFAwoopKBC0UCikIWkoopAFFFFMAooopgFFFFABRRRQAUtJRQAtFJRQAtLSUUALRRRQAUUUUAFFFFABRRRQAoooooGFFFGaYBSClpKAFooooAWikpaBBRRRQAUUlGaQwooopgFLSUUCFoooFMAooopALRRRTAKKSgUALRRRTASiiikAUtJS0AFLSCimAtJS0GgQlFFFABQKKKAFpaSigBaKKKACiiigAoFFFAC0lFFABS0lLTEJmiiikMWigUGmJiUCiikIWg0maDTAKKKKLiCiigUAFFLRRcBKKWii4CUUUUAFFFFABRRRRcAoooouFwooop3C4UUUUXGKBRSUZpXAWgijNFFwDFFHNHNFwCkNLSGncQUtJRRcBaKM0UXGFFFFFwCjFFFAitRRRWJqgpaSlpjEopaKBBRRRQIKKKWkAlFLSUAFFFFMAooooAKKKKACiiigAooooAKWigUALRRRQAUUUUAFFFFABRRRQAUUUUDCiiigQUUUUwFooooAKKKKAFopKWgBDSUppKQBRSiimAUUUUAFFFFAC0CiigBaSiimAUUUUAKKDQKKYCUUUVIgooopgFLSUtMAoNFFABSUtFABRRRQAtFFFAC0UUlABRRRQAtFIKWgAooooAKKKKYBRRS0gAUGig0wEooooEFFFFAgooopCClpKUUAFFFFABQaKKAEooxRigAopaQ07gFFFFIAooooAKKKMUAFKKMUUXAKMUUUAFFFFABRRRQAUhpaKAEooxRimACijFLRcBKWiigAoNFIaAK9FFFZGyClpBS0wCiiigApaSigQtFFJUiFooooAKSlpKACiiimAUUUUwCiiigApaSigBRS0lLQAUUUUAFFAooAKKKKACiiigAopKKAFooooAWikFFABS0lLQAUUUUAFFFFMAooopAApaQUtACUUtFMAooooAKKKKACiiigAooooAUUGgUUwEooooABRS0lAhaKSigBaKKKACiikpgLRRRQAtFJS0AFLSUUAFFFFABS0lFAC0UUUAFFFFMBaKSloAKKKKAENFB60UAGKKWkoJCikpaQgpRSUZoAWiiigAooooAKKKKACiiigBKKDRQAUYoFLQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRTsAUUUUgCiiigAoopKACiiimBXooorI2QUtNpRTAWiiigAooooEwooopMQUtJRSAKKKKACiiimAUUUUAFFFFMAoopaACiiigAooooAWikpaACkpaKACkoooAKKKKAClpKWgAooooAKKKKAClpKKAFooooAKKKKAClpBRQAtFJRTAWiiigAopKKAFopKWgAooooAKWkopgLSUUUAFFFFAgooooAKKKKYBS0lLQAUUUUAFLSUUALRRRQAlFFFAC0CkpaAFopKKBC0UUUXGFKKSimAtFJS0AJRRSigBKDS0GkJjaKU0lAgpaSigQopaQUtABRRRQAUUUUAFFFFAAaSlpB1oAWiijvQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRTAKKKKQBQaDSUIAooopgFJS0lAFeikpayNgpaSlpgLRSUUALRRRQJhRRRUiCiiimAtJRRSAKKKKACiiigApaKKYBRRSUwFooooAWkoopALRSUtMApKKKACiiigAoopKAFooooAKWkooAKWiigAooooAWikooAWikooAWiiigAooooAKKKKYBS0lFABS0lLQAUUUUAFFFFABRRRQIKKKKYBRRQKYBS0UUDCiiigQUUUUAFFFFABS0lLQIDSUUUDFFFAooEFFFFABS0lLQAUUUUDClpKKYC0UlKKQBRRRQAlFFFMQUUUUCFopKKQhaKTNLmgAJpAaKKAFooFGaACigUUAFFFFABRRRQCCiiigYUUUUCCiiigAooooAKKKKACiiigBKKKKYBRRSUALRRRQBVpRSUorI2ClpKUUAFFFFMBaKSigQUUUUhC0UlFAC0UlLSAKKKKACiiigBaKSlqgEooooAKKKKAFoopKAFooooAKKKSgBaKKSgAooooAKKKKYC0UlLSAKKKKAFooooAKKKKYBRRRSAKKKKACiiigAooopgLRRRQAUtJRQAtFFFABRRRQAUUUUCCiiimAlLSUUALSikopgLRSUtAwopKWgQUUUUAFFFFABRRRQAUUUUAFLSUUALRRRQIKWkooAWiiigQUCiigaCiiimMKSlooEFJS0lAgooopALRRRQIKKKKACiiigAFLSUtABQaKDQACigUUAFFFFABRRRQAUGig0AJQKKKBoWigUUBYTvRS0GgQlFLSUwEopaSgBaKKKAKtLSUtYmwUUUUwFopKWgQUUUUAFFFFAhaDSUUAFFFFABS0gpaACiiigBaKKKYCUUGkoAWikooAWiiigAooooAKKKKACiiimgCiiigApcUCloYCUUGigApaQUtABSUZopWAWikFLTAKKTNApALRRRTAKKKKQBS0lFMBaKKKACikpaACjNJRQA6ikFLQAUUUUCCiiimgCiiigAooooAKKKKYBS0lFAC0UlFAC0UlLQAUUUUAFFFFABRRRQAtFAooEFFFFAC0UlLQAUUUUAFFFFMYUUUUiQooopgFJS0UAFFFBoCwUCiikIWiiigANGaSigBaKSigBaKSlBoAKKKKACiikNAC0UZpDQAUUUUDFFFJS0AFFFFAgooopgJRQaKACkpaDQBVFKKKKyNgoopaAEpaKKBBRRRQAUUUUCFpKWigBKKWigAooooAKKKWmAlLRRQAhpKWg0AJRRS0XAMUUUUAFFFFABRRRQAUUtFFwEopaKdwACloopAJSU6kpgFFFFACUUtHFABS0UUgCkpaKACiiigAooooAKKKKYC0UcUUAJS0UUAJRRRQACnUgooELRRS4oASiloNO4CUUUUAFFFFMBKWiikAUUUUXAKKKKLgFFFFABQKKKYC0UUUAFFJS0hBS0lLmmAUUUcUAFLRRQISloooAKKKKBhRRRQIKKSii4BRRRTuAtFJS0AFFFFIQoNBpKWgBKKKKACiiigAoooFAC0UUGgBDRRRQAUUUUAFFFFABRRRQAoNFAooAKDRQaAEooop3AKQ0tIaQH//Z"
st.markdown(f'''
    <div style="display:flex; align-items:center; gap:18px; margin-bottom:4px;">
        <img src="data:image/jpeg;base64,{LOGO_B64}" 
             style="height:54px; width:auto; border-radius:8px; box-shadow: 0 0 16px #ff00ff;">
        <h1 class="glitch-header" style="margin:0;">GLITCH//CHECK</h1>
    </div>
''', unsafe_allow_html=True)
st.markdown('<p class="subtitle">NEURAL SPELLING CORRECTION SYSTEM v1.0</p>', unsafe_allow_html=True)

# Layout
col_left, col_right = st.columns([0.65, 0.35])

with col_left:
    st.markdown('<p class="section-header">// INPUT_STREAM:</p>', unsafe_allow_html=True)
    input_text = st.text_area(
        label="Input Stream",
        placeholder="TYPE OR PASTE TEXT HERE...",
        height=200,
        key="input_text",
        label_visibility="collapsed"
    )

    scan_btn = st.button("⚡ SCAN FOR ERRORS")

    if scan_btn and input_text:
        results = correct_sentence(input_text)
        st.session_state.last_results = results
        
        # Update Stats
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        words_in_scan = 0
        errors_in_scan = 0
        
        for original, corrected, was_correct, alternatives, confidence in results:
            if not all(c in string.punctuation for c in original):
                words_in_scan += 1
                if not was_correct:
                    errors_in_scan += 1
                    st.session_state.correction_history.insert(0, {
                        'original': original,
                        'corrected': corrected,
                        'timestamp': timestamp,
                        'alternatives': alternatives,
                        'confidence': confidence
                    })
        
        st.session_state.words_checked += words_in_scan
        st.session_state.corrections_made += errors_in_scan
        st.session_state.correction_history = st.session_state.correction_history[:20]

    if st.session_state.last_results:
        results = st.session_state.last_results
        st.markdown('<p class="section-header">// CORRECTED_OUTPUT:</p>', unsafe_allow_html=True)
        
        html_output = '<div class="output-container">'
        corrected_plain_text = ""
        words_with_alts = []
        
        for original, corrected, was_correct, alternatives, confidence in results:
            if was_correct:
                html_output += f'<span class="word-correct">{corrected}</span> '
            elif alternatives:
                html_output += f'<span class="word-corrected" title="Original: {original} | Confidence: {confidence}%">{corrected}</span> '
                words_with_alts.append((original, corrected, alternatives))
            else:
                html_output += f'<span class="word-unknown" title="No candidates found">{original}</span> '
            
            corrected_plain_text += corrected + " "
        
        html_output += '</div>'
        st.markdown(html_output, unsafe_allow_html=True)
        
        # Suggestions Panel
        if words_with_alts:
            with st.expander("// ALTERNATIVE_SUGGESTIONS"):
                for orig, curr, alts in words_with_alts:
                    st.markdown(f"**{orig}** → current: `{curr}`")
                    cols = st.columns(len(alts[:4]))
                    for i, alt in enumerate(alts[:4]):
                        if cols[i].button(alt, key=f"alt_{orig}_{alt}"):
                            # Simple replacement logic for the session state result
                            new_results = []
                            for r_orig, r_corr, r_was, r_alts, r_conf in st.session_state.last_results:
                                if r_orig == orig:
                                    new_results.append((r_orig, alt, r_was, r_alts, r_conf))
                                else:
                                    new_results.append((r_orig, r_corr, r_was, r_alts, r_conf))
                            st.session_state.last_results = new_results
                            st.rerun()

        # Copy Button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("COPY CORRECTED TEXT"):
            import streamlit.components.v1 as components
            js_code = f"""
            <script>
            const text = `{corrected_plain_text.strip()}`;
            navigator.clipboard.writeText(text).then(() => {{
                parent.postMessage({{type: 'streamlit:message', message: 'Copied to clipboard!'}}, '*');
            }});
            </script>
            """
            components.html(js_code, height=0)
            st.success("TEXT COPIED TO NEURAL BUFFER")

with col_right:
    # System Stats
    st.markdown('<p class="section-header">// SYSTEM_STATS</p>', unsafe_allow_html=True)
    
    total_words = st.session_state.words_checked
    total_errors = st.session_state.corrections_made
    accuracy = 100.0 if total_words == 0 else ((total_words - total_errors) / total_words) * 100
    
    stats_html = f"""
    <div class="stats-card">
        <div class="stat-line"><span>WORDS SCANNED:</span> <span class="stat-value">{total_words}</span></div>
        <div class="stat-line"><span>ERRORS DETECTED:</span> <span class="stat-value">{total_errors}</span></div>
        <div class="stat-line"><span>CORRECTIONS MADE:</span> <span class="stat-value">{total_errors}</span></div>
        <div class="stat-line"><span>ACCURACY RATE:</span> <span class="stat-value">{accuracy:.1f}%</span></div>
        <div class="stat-line"><span>SESSION TOTAL:</span> <span class="stat-value">{total_errors}</span></div>
    </div>
    """
    st.markdown(stats_html, unsafe_allow_html=True)
    
    # Accuracy Meter
    st.markdown(f'<p style="color:#ff00ff; font-size:0.8rem; margin-bottom:5px;">CORRECTION CONFIDENCE: {accuracy:.1f}%</p>', unsafe_allow_html=True)
    st.progress(accuracy / 100.0)
    
    # Export Button
    if st.session_state.last_results:
        report = f"GLITCHCHECK NEURAL REPORT - {datetime.datetime.now()}\n"
        report += "="*40 + "\n"
        report += f"STATS:\n- Words Checked: {total_words}\n- Errors Found: {total_errors}\n- Accuracy: {accuracy:.1f}%\n\n"
        report += "CORRECTIONS MADE:\n"
        for entry in st.session_state.correction_history:
            report += f"[{entry['timestamp']}] {entry['original']} -> {entry['corrected']}\n"
        
        st.download_button(
            label="💾 EXPORT REPORT",
            data=report,
            file_name=f"glitchcheck_report_{int(time.time())}.txt",
            mime="text/plain"
        )

    # Correction Log
    log_col1, log_col2 = st.columns([0.7, 0.3])
    with log_col1:
        st.markdown('<p class="section-header">// CORRECTION_LOG</p>', unsafe_allow_html=True)
    with log_col2:
        if st.button("CLEAR LOG"):
            st.session_state.correction_history = []
            st.rerun()
            
    if not st.session_state.correction_history:
        st.markdown('<p style="color:#555; font-size:0.8rem;">LOG_EMPTY: NO DATA DETECTED</p>', unsafe_allow_html=True)
    else:
        for entry in st.session_state.correction_history:
            alts_html = "".join([f'<span class="alt-pill">{a}</span>' for a in entry['alternatives']])
            entry_html = f"""
            <div class="log-entry">
                <div class="log-timestamp">[{entry['timestamp']}]</div>
                <div>❌ {entry['original']} <span class="log-arrow">→</span> ✅ {entry['corrected']}</div>
                <div>{alts_html}</div>
            </div>
            """
            st.markdown(entry_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">GLITCHCHECK v1.0 | EDIT-DISTANCE NLP ENGINE | POWERED BY NLTK</div>', unsafe_allow_html=True)
