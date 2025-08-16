import json
import re

import torch
from query_utils import create_advanced_query_understanding_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer_from_json(response_text):
    """Extract answer from JSON format response."""
    try:
        # Try to find and parse JSON in the response
        json_match = re.search(r'\{.*?"answer".*?\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            return parsed.get("answer", "").strip()
        return ""
    except (json.JSONDecodeError, AttributeError):
        return ""


def process_query(query_text, tokenizer, model):
    """Process a single query and return the response."""
    # Use structured chat format
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": create_advanced_query_understanding_prompt(query_text)},
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)

    # Decode only the new tokens (excluding the input prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response


def process_query_all_models(query_text, models_data):
    """Process a query with all models and return comparison results."""
    results = {}

    for model_name, (tokenizer, model) in models_data.items():
        try:
            print(f"  🔄 Processing with {model_name}...")
            response = process_query(query_text, tokenizer, model)
            extracted_answer = extract_answer_from_json(response)

            results[model_name] = {"response": response, "extracted_answer": extracted_answer}
        except Exception as e:
            results[model_name] = {"response": f"Error: {e}", "extracted_answer": ""}

    return results


def main():
    """Interactive query understanding system."""
    print("🚀 Query Understanding System - Model Comparison")
    print("=" * 80)

    # Define all model paths
    model_paths = {
        "🔥 Fine-tuned (Step 40)": (
            "checkpoints/verl_func_rm_example_gsm8k/qwen2_5_0_5b_gen_rm_docleaderboard/"
            "global_step_40/actor/huggingface"
        ),
        "🌟 Fine-tuned (Step 20)": (
            "checkpoints/verl_func_rm_example_gsm8k/qwen2_5_0_5b_gen_rm_docleaderboard/"
            "global_step_20/actor/huggingface"
        ),
        "🟢 Raw Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B-Instruct",  # Using HF model name
    }

    print("Loading models and tokenizers...")
    models_data = {}

    for model_name, model_path in model_paths.items():
        print(f"  📦 Loading {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto"
            )
            models_data[model_name] = (tokenizer, model)
            print(f"  ✅ {model_name} loaded successfully!")
        except Exception as e:
            print(f"  ❌ Error loading {model_name}: {e}")
            print(f"  ⚠️  Continuing without {model_name}")

    if not models_data:
        print("❌ No models loaded successfully. Exiting.")
        return

    print(f"\n🎯 Loaded {len(models_data)} models successfully!\n")

    # Interactive loop
    while True:
        print("-" * 80)
        query_text = input("\n📝 Enter your query (or 'quit' to exit): ").strip()

        if query_text.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break

        if not query_text:
            print("⚠️  Please enter a valid query.")
            continue

        print(f"\n🔍 Processing query: '{query_text}'")
        print("⏳ Generating responses from all models...")

        try:
            # Get responses from all models
            results = process_query_all_models(query_text, models_data)

            # Display comparison results
            print("\n" + "=" * 80)
            print("📊 MODEL COMPARISON RESULTS:")
            print(f"🎯 Query: {query_text}")
            print("=" * 80)

            # Display extracted answers side by side
            print("\n✨ EXTRACTED ANSWERS:")
            for model_name, result in results.items():
                answer = result["extracted_answer"]
                if answer:
                    print(f"  {model_name}: '{answer}'")
                else:
                    print(f"  {model_name}: ❌ No valid JSON answer found")

            print("\n" + "-" * 80)
            print("📄 FULL RESPONSES:")
            print("-" * 80)

            # Display full responses
            for model_name, result in results.items():
                print(f"\n🤖 {model_name}:")
                print("-" * 40)
                print(result["response"])
                print("-" * 40)

            print("=" * 80)

        except Exception as e:
            print(f"❌ Error processing query: {e}")


if __name__ == "__main__":
    main()
