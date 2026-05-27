import zipfile
import nltk
from nltk.corpus import wordnet as wn
import random
from tqdm import tqdm

# ─────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────
ZIP_PATH      = "para-nmt-50m.zip" # path to your para-nmt-50m.zip
INNER_TXT     = "para-nmt-50m/para-nmt-50m.txt"                                         # path inside the zip
OUT_PARA      = "cwe_paraphrase_sample_pairs_en.txt"                                       # output paraphrase pairs
OUT_CONTRA    = "cwe_contradiction_sample_pairs_en.txt"                                    # output contradiction pairs                                    
MAX_TOKENS    = 7                                                                       # drop very long sentences
RANDOM_SEED   = 1234                                                                    # random seed for reproducibility
nltk.download('wordnet')
random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────
#  Utility functions
# ─────────────────────────────────────────────────────
def get_antonyms(word):
    """Return a list of antonyms for `word` via WordNet."""
    ants = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                ants.add(ant.name().replace('_', ' '))
    return list(ants)

def find_and_substitute_antonym(tokens):
    """
    Find first token that has an antonym; return (idx, antonym).
    If none found, return None.
    """
    for i, w in enumerate(tokens):
        ants = get_antonyms(w.lower())
        if ants:
            return i, random.choice(ants)
    return None

# ─────────────────────────────────────────────────────
#  Main processing
# ─────────────────────────────────────────────────────
with zipfile.ZipFile(ZIP_PATH, 'r') as zf, \
     open(OUT_PARA,   'w', encoding='utf8') as f_para, \
     open(OUT_CONTRA, 'w', encoding='utf8') as f_contra:

    with zf.open(INNER_TXT, 'r') as fin:
        for raw in tqdm(fin, desc="Streaming para-nmt lines"):
            # Each line: b"sent1 \t sent2 \n"
            try:
                line = raw.decode('utf8').strip()
            except UnicodeDecodeError:
                continue
            if not line:
                continue

            # Split into the two paraphrases
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            ph1, ph2 = parts[0].strip(), parts[1].strip()

            # Skip overly long sentences
            if len(ph1.split()) > MAX_TOKENS or len(ph2.split()) > MAX_TOKENS:
                continue

            # 1) Write the paraphrase pair
            f_para.write(f"{ph1}\t{ph2}\t1\n")

            # 2) Try to create a contradiction by antonym substitution
            #    We attempt on ph1 first, then ph2
            for base, other in [(ph1, ph2), (ph2, ph1)]:
                tokens = base.split()
                res = find_and_substitute_antonym(tokens)
                if not res:
                    continue
                idx, ant = res
                tokens[idx] = ant
                new_phrase = " ".join(tokens)
                f_contra.write(f"{new_phrase}\t{other}\t0\n")
                break  # only one substitution per original pair

print("Done! Generated:")
print(f"  • {OUT_PARA}   (paraphrase pairs, label=1)")
print(f"  • {OUT_CONTRA} (contradiction pairs, label=0)")
