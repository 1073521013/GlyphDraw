import json
import argparse
import os
import torch
import re
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="LLaMA Factory Inference & Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the checkpoint (e.g., checkpoint-1000)")
    parser.add_argument("--test_file", type=str, default=None, help="Path to a single test JSON file")
    parser.add_argument("--test_dir", type=str, default="/mnt/workspace/majian/vlm_data/input/jsons/test", help="Path to the directory containing test JSON files")
    parser.add_argument("--base_model_path", type=str, default=None, help="Base model path (required if model_path is a LoRA adapter)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    return parser.parse_args()

def get_label(output_str):
    """
    Map output string to class label:
    0: "0"
    1: "1"
    2: Summary (JSON or other text)
    """
    # Remove <think>...</think> content if present, matching across newlines
    s = re.sub(r'<think>.*?</think>', '', output_str, flags=re.DOTALL).strip()
    
    if s == "0":
        return 0
    elif s == "1":
        return 1
    else:
        # Anything else is considered the "Summary" class
        return 2

def load_model_and_tokenizer(model_path, base_model_path=None, device="cuda"):
    print(f"Loading model from {model_path}...")
    
    # Check if it's an adapter (LoRA)
    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    if is_adapter:
        if base_model_path is None:
            raise ValueError("model_path looks like an adapter (contains adapter_config.json), but --base_model_path is not provided.")
        
        print(f"Loading base model from {base_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            torch_dtype=torch.float16, 
            device_map="auto", 
            trust_remote_code=True
        )
        print(f"Loading LoRA adapter from {model_path}...")
        model = PeftModel.from_pretrained(model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    else:
        # Full model
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map="auto", 
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model.eval()
    
    # Reset generation config to avoid warnings with do_sample=False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.top_k = None
        model.generation_config.top_p = None
        model.generation_config.temperature = None
        
    return model, tokenizer

def evaluate_dataset(model, tokenizer, data, args, dataset_name):
    print(f"\nProcessing {dataset_name} ({len(data)} samples)...")
    
    correct = 0
    total = 0
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}
    results = []
    
    for i, item in tqdm(enumerate(data), total=len(data), desc=dataset_name):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        gt_output = item.get("output", "").strip()

        if tokenizer.chat_template:
            messages = [{"role": "user", "content": f"{instruction}\n{input_text}"}]
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        else:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_ids = outputs[0][len(input_ids[0]):]
        pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Remove <think> content from prediction
        pred_text = re.sub(r'<think>.*?</think>', '', pred_text, flags=re.DOTALL).strip()

        gt_class = get_label(gt_output)
        pred_class = get_label(pred_text)
        
        is_match = (gt_class == pred_class)
        if is_match:
            correct += 1
            class_correct[gt_class] += 1
        class_total[gt_class] += 1
        total += 1

        results.append({
            "index": i,
            "instruction": instruction,
            "input": input_text[:100] + "...",
            "gt_raw": gt_output,
            "pred_raw": pred_text,
            "gt_class": gt_class,
            "pred_class": pred_class,
            "match": is_match
        })

        if not is_match:
             print(f"\n[{dataset_name}] Mismatch at index {i}")
             print(f"  Input: {input_text[:200]}..." if len(input_text) > 200 else f"  Input: {input_text}")
             print(f"  GT:   {gt_class} (Raw: {gt_output})")
             print(f"  Pred: {pred_class} (Raw: {pred_text})")

    accuracy = correct / total if total > 0 else 0
    return {
        "dataset": dataset_name,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "class_correct": class_correct,
        "class_total": class_total,
        "results": results
    }

def main():
    args = parse_args()

    # Load Model
    try:
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model_path, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Determine files to process
    files_to_process = []
    if args.test_file and os.path.exists(args.test_file):
        files_to_process.append(args.test_file)
    elif args.test_dir and os.path.exists(args.test_dir):
        # Find all json files in directory
        files_to_process = glob.glob(os.path.join(args.test_dir, "*.json"))
        # Filter out results file if it exists there
        files_to_process = [f for f in files_to_process if "inference_results" not in f]
        files_to_process.sort()
    else:
        print("No valid test file or directory provided.")
        return

    if not files_to_process:
        print(f"No JSON files found in {args.test_dir}")
        return

    print(f"Found {len(files_to_process)} datasets to evaluate.")
    
    all_metrics = []
    all_results = {}

    for file_path in files_to_process:
        dataset_name = os.path.basename(file_path).replace(".json", "")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
            
        metrics = evaluate_dataset(model, tokenizer, data, args, dataset_name)
        all_metrics.append(metrics)
        all_results[dataset_name] = metrics["results"]

    # Final Report
    print("\n" + "="*50)
    print("FINAL EVALUATION REPORT")
    print("="*50)
    print(f"{'Dataset':<20} | {'Total':<8} | {'Accuracy':<10}")
    print("-" * 50)
    
    total_samples = 0
    total_correct = 0

    for m in all_metrics:
        print(f"{m['dataset']:<20} | {m['total']:<8} | {m['accuracy']:.2%}")
        total_samples += m['total']
        total_correct += m['correct']
    
    print("-" * 50)
    if total_samples > 0:
        print(f"{'OVERALL':<20} | {total_samples:<8} | {total_correct/total_samples:.2%}")
    
    # Save detailed results
    output_log = "inference_results.json"
    with open(output_log, 'w', encoding='utf-8') as f:
        # Save simplified results structure
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {output_log}")

if __name__ == "__main__":
    main()
