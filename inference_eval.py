import json
import argparse
import os
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="LLaMA Factory Inference & Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the checkpoint (e.g., checkpoint-1000)")
    parser.add_argument("--test_file", type=str, default="/mnt/workspace/majian/vlm_data/input/jsons/test/Chinese.json", help="Path to the test JSON file")
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
    s = output_str.strip()
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
        
        from peft import PeftModel
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
    return model, tokenizer

def main():
    args = parse_args()

    # Verify input file
    if not os.path.exists(args.test_file):
        print(f"Warning: Test file not found at {args.test_file}")
        # Proceeding anyway as this might be run in a different environment
    
    # Load Model
    try:
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model_path, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load Data
    try:
        with open(args.test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    correct = 0
    total = 0
    
    # Metrics per class
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}

    results = []

    print(f"Starting inference on {len(data)} samples...")

    for i, item in tqdm(enumerate(data), total=len(data)):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        gt_output = item.get("output", "").strip()

        # Construct Prompt
        # Using a standard chat format or concatenating if simple
        # Adjust this template matching your training format
        if tokenizer.chat_template:
            messages = [
                {"role": "user", "content": f"{instruction}\n{input_text}"}
            ]
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        else:
            # Fallback simple template
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_new_tokens=args.max_new_tokens,
                do_sample=False, # Greedy decoding for reproducibility
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        generated_ids = outputs[0][len(input_ids[0]):]
        pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Remove <think> content from prediction
        pred_text = re.sub(r'<think>.*?</think>', '', pred_text, flags=re.DOTALL).strip()

        # Evaluation
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

        # Optional: Print first few mismatches for debugging
        if not is_match and total <= 5:
             print(f"\n[Mismatch] GT: {gt_class} ({gt_output}) | Pred: {pred_class} ({pred_text})")

    # Final Report
    print("\n" + "="*30)
    print("Evaluation Results")
    print("="*30)
    print(f"Total Samples: {total}")
    print(f"Overall Accuracy: {correct/total:.2%} ({correct}/{total})")
    
    print("\nPer-Class Accuracy:")
    labels = {0: "Class 0 (Non-urgent)", 1: "Class 1 (Urgent/Other)", 2: "Class 2 (Summary)"}
    for cls in [0, 1, 2]:
        if class_total[cls] > 0:
            acc = class_correct[cls] / class_total[cls]
            print(f"  {labels[cls]}: {acc:.2%} ({class_correct[cls]}/{class_total[cls]})")
        else:
            print(f"  {labels[cls]}: N/A (0 samples)")

    # Save detailed results
    output_log = "inference_results.json"
    with open(output_log, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {output_log}")

if __name__ == "__main__":
    main()
