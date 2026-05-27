import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.decomposition import PCA

# Load IndicBERT
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
model = AutoModel.from_pretrained("ai4bharat/indic-bert")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Target embedding dimension
TARGET_DIM = 300

# Your vocab
def build_vocab_from_hindi_cwe(cwe_paths):
    vocab = set()
    for path in cwe_paths:
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                ph1, ph2, lbl = line.strip().split('\t')
                vocab.update(ph1.split())
                vocab.update(ph2.split())
    return sorted(vocab)

cwe_paths = [
    "cwe_paraphrase_sample_pairs_hi.txt",
    "cwe_contradiction_sample_pairs_hi.txt"
]

vocab = build_vocab_from_hindi_cwe(cwe_paths)

# Batching params
BATCH_SIZE = 512

# Store original IndicBERT embeddings
all_embeddings = []

with torch.no_grad():
    for i in tqdm(range(0, len(vocab), BATCH_SIZE)):
        batch_words = vocab[i:i+BATCH_SIZE]
        encoded = tokenizer(
            batch_words,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # (batch_size, seq_len, hidden_dim)
        last_hidden = outputs.last_hidden_state

        # Mean pooling over token embeddings
        for j, w in enumerate(batch_words):
            length = attention_mask[j].sum().item()
            if length > 2:
                token_embs = last_hidden[j, 1:length-1, :]
                pooled = token_embs.mean(dim=0)
            else:
                pooled = torch.rand(model.config.hidden_size).uniform_(-0.25, 0.25).to(device)

            all_embeddings.append(pooled.cpu().numpy())

all_embeddings = np.array(all_embeddings)

# Reduce 768d -> 300d using PCA
pca = PCA(n_components=TARGET_DIM)
reduced_embeddings = pca.fit_transform(all_embeddings)

# Embedding setup
word2idx = {"<PAD>": 0, "<UNK>": 1}
embedding_matrix = [
    np.zeros((TARGET_DIM,), dtype=np.float32),
    np.random.uniform(-0.25, 0.25, TARGET_DIM).astype(np.float32)
]
idx = 2

# Build final embedding matrix
for word, emb in zip(vocab, reduced_embeddings):
    embedding_matrix.append(emb.astype(np.float32))
    word2idx[word] = idx
    idx += 1
embedding_matrix = np.stack(embedding_matrix)

# Save
with open("hindi_cwe_word2idx300d.pkl", "wb") as f:
    pickle.dump(word2idx, f)
torch.save(
    torch.from_numpy(embedding_matrix),
    "hindi_cwe_finetuned_emb_300d.pt"
)

print(f"Done. Vocab: {len(word2idx)} | Embedding shape: {embedding_matrix.shape}")
