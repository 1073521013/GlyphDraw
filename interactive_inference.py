import json
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Inference Script")
    parser.add_argument("--model_path", type=str, default="/home/notebook/code/group/majian/ai-prompt-old/outputs_shouyao/notice_0.6B", help="Path to the checkpoint (e.g., checkpoint-1000)")
    parser.add_argument("--base_model_path", type=str, default=None, help="Path to the base model (required if model_path is an adapter)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max new tokens to generate")
    return parser.parse_args()

def get_label(output_str):
    """
    Map output string to class label:
    0: "0"
    1: "1"
    2: Summary (JSON or other text)
    """
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
            # Try to infer base model path if not provided, or raise error
            # For safety, just raise error or warn
            pass # Will rely on explicit argument or let PeftModel handle it if possible, but code below raises error.
            
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
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.top_k = None
        model.generation_config.top_p = None
        model.generation_config.temperature = None

    return model, tokenizer

def main():
    args = parse_args()

    # Load Model
    try:
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model_path, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n" + "="*50)
    print("Interactive Inference Mode")
    print("Type 'exit', 'quit', or press Ctrl+C to stop.")
    print("="*50)

    while True:
        try:
            print("\nPlease provide input:")
            instruction = input("Instruction (optional): ").strip()
            if instruction.lower() in ['exit', 'quit']:
                break
                
            input_text = input("Input text: ").strip()
            if input_text.lower() in ['exit', 'quit']:
                break
            
            if not instruction and not input_text:
                print("Empty input, skipping...")
                continue

            print("\nGenerating response...")

            # Construct Prompt
            if tokenizer.chat_template:
                messages = [
                    {"role": "user", "content": f"{instruction}\n{input_text}"}
                ]
                # Note: enable_thinking parameter might be specific to certain tokenizer versions/modifications
                # Keeping it as per original code, but wrapping in try/except if it fails
                try:
                    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False).to(model.device)
                except TypeError:
                    # Fallback if enable_thinking is not supported
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
            pred_class = get_label(pred_text)

            print("-" * 30)
            print(f"Raw Prediction: {pred_text}")
            print(f"Predicted Class: {pred_class}")
            print("-" * 30)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError during inference: {e}")

if __name__ == "__main__":
    main()
