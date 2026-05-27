import os
import ctranslate2
import transformers
import logging

# ===== Configuration =====
MODEL_NAME = "facebook/nllb-200-distilled-600M"
CT2_MODEL_DIR = os.path.expanduser("trans_models/nllb-600M-ct2")  # Where converted CT2 model will be saved
INPUT_FILE = "cwe_contradiction_sample_pairs_en.txt"  # Tab-separated English pairs
OUTPUT_FILE = "cwe_contradiction_sample_pairs_hi.txt"
CHECKPOINT_FILE = "ckpt_con_hi.txt"
LOG_FILE = "trans_con_hi.log"

# Language codes (Flores-200)
SRC_LANG = "eng_Latn"  # English
TGT_LANG = "hin_Deva"  # Hindi

BATCH_SIZE = 512  # Adjust based on GPU memory
CHECKPOINT_INTERVAL = 100  # Save progress every N lines
DEVICE = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

# ===== Setup =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def convert_model_to_ct2():
    """Convert Hugging Face model to CTranslate2 format"""
    if not os.path.exists(CT2_MODEL_DIR):
        logger.info("Converting model to CTranslate2 format...")
        converter = ctranslate2.converters.TransformersConverter(
            MODEL_NAME,
            low_cpu_mem_usage=True
        )
        # Convert with int8 quantization to reduce size
        converter.convert(CT2_MODEL_DIR, quantization="int8")
        logger.info(f"Model saved to {CT2_MODEL_DIR}")

def initialize_models():
    """Load models with error handling"""
    try:
        # Convert if needed
        if not os.path.exists(CT2_MODEL_DIR):
            convert_model_to_ct2()

        logger.info(f"Loading model from {CT2_MODEL_DIR}...")
        translator = ctranslate2.Translator(CT2_MODEL_DIR, device=DEVICE)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            MODEL_NAME,
            src_lang=SRC_LANG,
            clean_up_tokenization_spaces=True
        )
        return translator, tokenizer
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

def translate_batch(translator, tokenizer, texts):
    """Translate a batch of texts efficiently"""
    encoded = [tokenizer.convert_ids_to_tokens(tokenizer.encode(text)) for text in texts]
    results = translator.translate_batch(
        encoded,
        target_prefix=[[TGT_LANG]] * len(texts),
        batch_type="tokens",
        max_batch_size=2048,
        beam_size=3  # Balance between quality and speed
    )
    return [
        tokenizer.decode(tokenizer.convert_tokens_to_ids(result.hypotheses[0][1:]))
        for result in results
    ]

def process_file():
    """Main processing function with checkpointing"""
    translator, tokenizer = initialize_models()
    start_line = 0

    # Resume from checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            start_line = int(f.read().strip())
        logger.info(f"Resuming from line {start_line}")

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "a" if start_line > 0 else "w", encoding="utf-8") as outfile:

        lines = infile.readlines()
        total_lines = len(lines)
        batch = []
        line_data = []  # Store original line data for writing

        for i in range(start_line, total_lines):
            try:
                line = lines[i].strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) != 3:
                    logger.warning(f"Skipping malformed line {i+1}: {line[:50]}...")
                    continue

                # Store English texts and original line data
                batch.append((parts[0], parts[1]))
                line_data.append((i, parts[2]))  # Store line index and label

                # Process batch when full or at end
                if len(batch) >= BATCH_SIZE or i == total_lines - 1:
                    # Split into separate batches for parallel translation
                    en1_batch = [item[0] for item in batch]
                    en2_batch = [item[1] for item in batch]

                    # Translate both columns in parallel
                    hindi1 = translate_batch(translator, tokenizer, en1_batch)
                    hindi2 = translate_batch(translator, tokenizer, en2_batch)

                    # Write results
                    for j in range(len(batch)):
                        outfile.write(f"{hindi1[j]}\t{hindi2[j]}\t{line_data[j][1]}\n")

                    # Clear batches
                    batch = []
                    line_data = []

                # Save checkpoint
                if (i + 1) % CHECKPOINT_INTERVAL == 0:
                    with open(CHECKPOINT_FILE, "w") as f:
                        f.write(str(i + 1))
                    logger.info(f"Checkpoint: {i+1}/{total_lines} lines processed")

                # Progress update
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i+1}/{total_lines} ({((i+1)/total_lines)*100:.1f}%)")

            except Exception as e:
                logger.error(f"Error at line {i+1}: {str(e)}")
                continue

        # Cleanup
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        logger.info("Translation completed!")

if __name__ == "__main__":
    logger.info(f"Starting translation (device: {DEVICE})")
    process_file()
