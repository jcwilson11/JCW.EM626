# wordcloud_from_hurricane.py
from collections import Counter
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- FILES ---
text_path = Path("hurricane_melissa_clean.txt")   # your text  (:contentReference[oaicite:7]{index=7})
stop_path = Path("stopwords_en.txt")               # your stops (:contentReference[oaicite:8]{index=8})

# --- LOAD STOPWORDS ---
stopwords = {line.strip().lower() for line in stop_path.open("r", encoding="utf-8") if line.strip()}
stopwords |= {"said"}  # optional domain extension to match cloud.py style (:contentReference[oaicite:9]{index=9})

# --- READ TEXT ---
raw_lines = [line.rstrip("\n") for line in text_path.open("r", encoding="utf-8")]

# --- CLEANING: mirrors cloud.py's txt_clean (:contentReference[oaicite:10]{index=10}) ---
def txt_clean(word_list, stopwords_list, min_len: int):
    clean_words = []
    vocab = set()
    for line in word_list:
        for word in line.strip().split():
            w = word.lower()
            if w not in stopwords_list and w.isalpha() and len(w) > min_len:
                clean_words.append(w)
                vocab.add(w)
    return clean_words, sorted(vocab)

tokens, vocab = txt_clean(raw_lines, stopwords, 2)
all_words = " ".join(tokens)

# --- WORD CLOUD ---
# IMPORTANT: give WordCloud a TTF font path for PIL
font_path = font_manager.findfont("DejaVu Sans")  # typically resolves to a valid .ttf locally
wc = WordCloud(background_color="white", max_words=2000, font_path=font_path)
wc.generate(all_words)

out_path = "hurricane_melissa_clean-cloud.png"
wc.to_file(out_path)  # save

plt.imshow(wc)
plt.axis("off")
plt.show()
print(f"Saved word cloud to: {out_path}")
