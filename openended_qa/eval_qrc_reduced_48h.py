import os
import json
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import difflib

# ======================================================
# CACHE DIRECTORIES (PORTABLE)
# ======================================================
HF_HOME = os.path.join(os.environ["HOME"], ".cache", "huggingface")
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")

for path in [
    os.environ["HF_HOME"],
    os.environ["HF_HUB_CACHE"],
    os.environ["TRANSFORMERS_CACHE"],
    os.environ["HF_DATASETS_CACHE"],
]:
    os.makedirs(path, exist_ok=True)

# ======================================================
# MODEL
# ======================================================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

def load_model():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        use_fast=True,
        cache_dir=os.environ["HF_HUB_CACHE"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        cache_dir=os.environ["HF_HUB_CACHE"]
    )
    return model, tokenizer

# ======================================================
# CORRECTNESS CHECK
# ======================================================
def string_match(pred, answers):
    pred = pred.lower().strip()
    answers = [a.lower().strip() for a in answers]

    if pred in answers:
        return True
    for a in answers:
        if pred in a or a in pred:
            return True
    for a in answers:
        if difflib.SequenceMatcher(None, pred, a).ratio() > 0.65:
            return True
    return False

# ======================================================
# CONFIDENCE
# ======================================================
def compute_conf(logits, sequences, prompt_len):
    logits = torch.stack(logits, dim=1)
    probs = torch.softmax(logits, dim=-1)
    generated_ids = sequences[0, prompt_len:]
    idx = torch.arange(len(generated_ids))
    token_probs = probs[0, idx, generated_ids]
    return float(token_probs.mean().item())

# ======================================================
# GENERATION
# ======================================================
def generate_answer(model, tokenizer, question, temperature, perturb):
    with torch.no_grad():
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        out = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=True,
            temperature=max(0.001, temperature),
            output_scores=True,
            return_dict_in_generate=True
        )

        new_tokens = out.sequences[0][prompt_len:]
        gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        conf = compute_conf(out.scores, out.sequences, prompt_len)
        conf = max(0.0, min(1.0, conf - perturb))
        return gen_text, conf

# ======================================================
# METRICS
# ======================================================
def ece90(confs):
    if not confs:
        return 0.0
    bins = np.linspace(0, 1, 11)
    ece = 0
    for i in range(10):
        lo, hi = bins[i], bins[i+1]
        in_bin = [c for c in confs if lo <= c < hi]
        if not in_bin:
            continue
        acc = np.mean(in_bin)
        mid = (lo + hi) / 2
        ece += abs(acc - mid) * (len(in_bin) / len(confs))
    return float(ece)

def qrc90(confs):
    return float(np.quantile(confs, 0.90)) if confs else 0.0

def cvar90(confs):
    if not confs:
        return 0.0
    t = np.quantile(confs, 0.90)
    tail = [c for c in confs if c >= t]
    return float(np.mean(tail)) if tail else 0.0

# ======================================================
# LOAD REDUCED DATASETS
# ======================================================
def load_reduced():
    data = []

    squad = load_dataset("squad_v2", split="validation[:500]")
    for ex in squad:
        answers = ex["answers"]["text"] or [""]
        data.append({"dataset": "SQuADv2", "question": ex["question"], "answers": answers})

    trivia = load_dataset("trivia_qa", "rc", split="validation[:500]")
    for ex in trivia:
        answers = ex["answer"]["aliases"] if "answer" in ex else [""]
        data.append({"dataset": "TriviaQA", "question": ex["question"], "answers": answers})

    tq = load_dataset("truthful_qa", "generation", split="validation[:500]")
    for ex in tq:
        answers = [ex.get("best_answer", "")]
        data.append({"dataset": "TruthfulQA", "question": ex["question"], "answers": answers})

    print(f"Loaded {len(data)} examples.")
    return data

# ======================================================
# RUN SWEEP
# ======================================================
def run_sweep(model, tokenizer, dataset, T, P):
    confs = []
    acc = {"SQuADv2": [0, 0], "TriviaQA": [0, 0], "TruthfulQA": [0, 0]}

    for item in tqdm(dataset):
        pred, conf = generate_answer(model, tokenizer, item["question"], T, P)
        confs.append(conf)
        is_correct = string_match(pred, item["answers"])
        acc[item["dataset"]][0] += int(is_correct)
        acc[item["dataset"]][1] += 1

    return {
        "temperature": T,
        "perturb": P,
        "ECE90": ece90(confs),
        "QRC90": qrc90(confs),
        "CVAR90": cvar90(confs),
        "accuracy": {ds: correct/total for ds, (correct, total) in acc.items()}
    }

# ======================================================
# MAIN
# ======================================================
def main():
    model, tokenizer = load_model()

    temps = [0.0, 0.3, 0.7, 1.0]
    perturbs = [0.0, 0.5, 0.9]

    data = load_reduced()
    output_dir = "results/reduced48_small"
    os.makedirs(output_dir, exist_ok=True)

    for T in temps:
        for P in perturbs:
            print(f"\n=== T={T}, P={P} ===")
            metrics = run_sweep(model, tokenizer, data, T, P)
            out_file = os.path.join(output_dir, f"T{T}_P{P}_metrics.json")
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=2)
            print(metrics)

if __name__ == "__main__":
    main()
