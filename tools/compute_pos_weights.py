import json
import torch
import os
import sys
import pandas as pd
import spacy
from tqdm import tqdm
from collections import defaultdict

try:
    import clip
except ImportError:
    print("Error: openai-clip not found. Please run 'pip install openai-clip'")
    exit()

sys.path.append(os.getcwd())

def compute_pos_weights():

    data_path = "data/MSRVTT/MSRVTT_data.json"
    split_file_path = "data/MSRVTT/MSRVTT_train.9k.csv"
    output_path = "data/MSRVTT/pos_weights_new_light1.pt"
    
    vocab_size = 49408
    
    POS_WEIGHTS_MAP = {
        "VERB": 3.0,  
        "NOUN": 2.0,  
        "PROPN": 2.0, 
        "ADJ": 1.5,   
        "ADV": 1.0,   
        "NUM": 1.2,   
        "DET": 0.1,   
        "ADP": 0.3,   
        "CCONJ": 0.3, 
        "PRON": 0.5,  
        "PUNCT": 0.1, 
        "DEFAULT": 0.5 
    }

    print(f"Loading full data from: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    all_sentences = data['sentences'] if 'sentences' in data else data

    print(f"Loading train split from: {split_file_path}")
    train_df = pd.read_csv(split_file_path)
    if 'video_id' in train_df.columns:
        valid_video_ids = set(train_df['video_id'].astype(str).tolist())
    else:
        valid_video_ids = set(train_df.iloc[:, 0].astype(str).tolist())

    train_sentences = [item for item in all_sentences if item['video_id'] in valid_video_ids]
    print(f"Filtered Training Sentences: {len(train_sentences)}")
    
    print("Loading Spacy model...")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        print("Error: Spacy model not found.")
        return

    token_pos_counts = defaultdict(lambda: defaultdict(int))

    print("Processing sentences (POS Tagging)...")
    batch_size = 1000
    captions = [item['caption'] for item in train_sentences]
    
    device = "cpu"

    for doc in tqdm(nlp.pipe(captions, batch_size=batch_size), total=len(captions)):
        for token in doc:
            word = token.text
            pos = token.pos_
            
            try:
                token_tensor = clip.tokenize(word, truncate=True).squeeze(0) # [77]
            except Exception:
                continue

            eot_idx = token_tensor.argmax().item()
            sub_tokens = token_tensor[1:eot_idx].tolist()
            
            if not sub_tokens:
                continue

            for t_id in sub_tokens:
                if t_id < vocab_size:
                    token_pos_counts[t_id][pos] += 1

    print("Computing final weights...")
    
    final_weights = torch.ones(vocab_size, dtype=torch.float32)
    covered_tokens = 0
    
    for t_id in range(vocab_size):
        counts = token_pos_counts[t_id]
        if not counts:
            final_weights[t_id] = 1.0 
            continue
            
        covered_tokens += 1
        dominant_pos = max(counts, key=counts.get)
        weight = POS_WEIGHTS_MAP.get(dominant_pos, POS_WEIGHTS_MAP["DEFAULT"])
        final_weights[t_id] = weight

    print(f"\n--- POS Weight Statistics ---")
    print(f"Vocab Coverage: {covered_tokens}/{vocab_size} ({covered_tokens/vocab_size:.1%})")
    print(f"Max Weight: {final_weights.max().item()}")
    print(f"Mean Weight: {final_weights.mean().item():.4f}")
    
    check_words = ["running", "man", "is", "the", "red"]
    print("\n[Sanity Check]")
    for w in check_words:
        token_tensor = clip.tokenize(w).squeeze(0)
        eot_idx = token_tensor.argmax().item()
        sub_tokens = token_tensor[1:eot_idx].tolist()
        
        if sub_tokens:
            t_id = sub_tokens[0]
            print(f"Word: '{w}' -> TokenID: {t_id} -> Weight: {final_weights[t_id]:.2f}")

    torch.save(final_weights, output_path)
    print(f"\nSuccess! POS weights saved to: {output_path}")

if __name__ == "__main__":
    compute_pos_weights()