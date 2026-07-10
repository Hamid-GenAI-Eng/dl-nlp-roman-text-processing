import re
from typing import Dict, List

# Phrase-level variants must be handled before punctuation/token splitting.
PHRASE_NORMALIZATION: Dict[str, str] = {
    r"\bm\s*sha\s*allah\b": "mashallah",
    r"\bmshaallah\b": "mashallah",
    r"\bmshallah\b": "mashallah",
    r"\bmasha\s*allah\b": "mashallah",
    r"\bmashaallah\b": "mashallah",
    r"\bmashallah\b": "mashallah",
    r"\bin\s*sha\s*allah\b": "inshallah",
    r"\binsha\s*allah\b": "inshallah",
    r"\binshaallah\b": "inshallah",
    r"\binshallah\b": "inshallah",
    r"\bal\s*hamdu\s*lillah\b": "alhamdulillah",
    r"\balhumdulillah\b": "alhamdulillah",
    r"\balhamdulillah\b": "alhamdulillah",
}

WORD_NORMALIZATION: Dict[str, str] = {
    # religious/common expressions
    "mshaallah": "mashallah", "mshallah": "mashallah", "mashaallah": "mashallah", "mashallah": "mashallah",
    "inshaallah": "inshallah", "inshallah": "inshallah", "alhumdulillah": "alhamdulillah",
    # common Roman Urdu spellings
    "boht": "bohat", "bht": "bohat", "bohot": "bohat", "bahat": "bohat", "buhat": "bohat",
    "achha": "acha", "achaa": "acha", "accha": "acha", "acha": "acha", "achi": "achi", "achy": "achy",
    "nhi": "nahi", "nai": "nahi", "nahin": "nahi", "ni": "nahi", "nae": "nahi",
    "hy": "hai", "ha": "hai", "hey": "hai", "h": "hai",
    "khrab": "kharab", "kharaab": "kharab", "khrb": "kharab",
    "pasnd": "pasand", "psand": "pasand", "pasand": "pasand",
    "behtreen": "behtareen", "behtareen": "behtareen", "bestt": "best",
    "recommendedd": "recommended", "recomend": "recommend", "recomended": "recommended",
    "sloww": "slow", "fastt": "fast", "servicee": "service", "productt": "product",
    "pyari": "pyari", "piyari": "pyari", "piari": "pyari",
    "zabardst": "zabardast", "zbrdst": "zabardast", "zabrdast": "zabardast",
    "bakwas": "bekar", "bekaar": "bekar", "bkwaas": "bekar",
}

EMOJI_TO_TEXT: Dict[str, str] = {
    "😍": " emoji_positive ", "❤️": " emoji_positive ", "💖": " emoji_positive ", "💕": " emoji_positive ",
    "🔥": " emoji_positive ", "💯": " emoji_positive ", "👍": " emoji_positive ", "😊": " emoji_positive ",
    "👌": " emoji_positive ", "🥰": " emoji_positive ", "😘": " emoji_positive ", "😎": " emoji_positive ",
    "👎": " emoji_negative ", "😭": " emoji_negative ", "😢": " emoji_negative ", "😡": " emoji_negative ",
    "😠": " emoji_negative ", "🤮": " emoji_negative ", "🤢": " emoji_negative ", "😑": " emoji_negative ",
    "😞": " emoji_negative ", "💀": " emoji_negative ", "👿": " emoji_negative ", "💩": " emoji_negative ",
    "😂": " emoji_laughter ", "👏": " emoji_laughter ", "🤣": " emoji_laughter ", "😜": " emoji_laughter ",
    "lol": " emoji_laughter ", "fr fr": " emoji_laughter ", "ngl": " emoji_neutral "
}

STOPWORDS = {
    "ye", "yh", "y", "hai", "hein", "hun", "ho", "tha", "thi", "the", "ko", "ka", "ki", "ke",
    "se", "sy", "par", "per", "aur", "or", "to", "tu", "main", "mein", "me", "mai", "mujhe",
    "mje", "ap", "aap", "is", "iss", "us", "wo", "woh", "bhi", "bi", "hi", "hi", "tha", "hain"
}


def lowercase_text(text: str) -> str:
    return text.lower().strip()


def map_emojis_and_slang(text: str) -> str:
    t = lowercase_text(text)
    for emoji, replacement in EMOJI_TO_TEXT.items():
        t = t.replace(emoji, replacement)
    return t


def reduce_repeated_characters(text: str) -> str:
    # Reduce 3 or more repeating characters to 1
    return re.sub(r'(.)\1{2,}', r'\1', text)


def apply_phrase_normalization(text: str) -> str:
    output = text
    for pattern, replacement in PHRASE_NORMALIZATION.items():
        output = re.sub(pattern, replacement, output, flags=re.IGNORECASE)
    return output


def remove_punctuation(text: str) -> str:
    # Keep numbers and letters. Replace punctuation/symbols with spaces.
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_words(text: str) -> str:
    words = text.split()
    normalized = [WORD_NORMALIZATION.get(word, word) for word in words]
    return " ".join(normalized)


def remove_stopwords(text: str) -> str:
    return " ".join([w for w in text.split() if w not in STOPWORDS])


def preprocess_steps(text: str) -> List[dict]:
    raw = text.strip()
    step1 = map_emojis_and_slang(raw)
    step2 = reduce_repeated_characters(step1)
    step3 = apply_phrase_normalization(step2)
    step4 = remove_punctuation(step3)
    step5 = normalize_words(step4)
    step6 = remove_stopwords(step5)
    return [
        {"step": 0, "title": "Raw Unfiltered Input", "text": raw},
        {"step": 1, "title": "Emoji & Slang Mapping", "text": step1},
        {"step": 2, "title": "Repeated Character Reduction", "text": step2},
        {"step": 3, "title": "Phrase Normalization", "text": step3},
        {"step": 4, "title": "Punctuation & Special Character Removal", "text": step4},
        {"step": 5, "title": "Spelling Standardization", "text": step5},
        {"step": 6, "title": "Stopword Filtering & Final Output", "text": step6},
    ]


def preprocess_text(text: str) -> str:
    return preprocess_steps(text)[-1]["text"]
