import os
import json
import random
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq

def load_vqa_questions(path, num_samples):
    with open(path, 'r') as f:
        questions_data = json.load(f)['questions']
    return random.sample(questions_data, num_samples)

def construct_image_path(image_id, base_dir):
    filename = f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
    return os.path.join(base_dir, filename)

def generate_answer(model, processor, tokenizer, device, image, question, max_new_tokens=50):
    inputs = processor(text=question, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def run_batch_inference(model, processor, tokenizer, device, questions_list, images_dir):
    results = []

    for q in tqdm(questions_list, desc="Running inference"):
        question_id = q['question_id']
        image_id = q['image_id']
        question_text = q['question']

        image_path = construct_image_path(image_id, images_dir)
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path} — skipping.")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            answer = generate_answer(model, processor, tokenizer, device, image, question_text)
            results.append({
                "question_id": question_id,
                "answer": answer
            })
        except Exception as e:
            print(f"❌ Error on question {question_id}: {e}")
            continue

    return results

def main(args):
    # Configurable paths
    MODEL_NAME = args.model_path
    questions_file = "datasets/Questions/v2_OpenEnded_mscoco_val2014_questions.json"
    images_dir = args.images_dir

    # Output file name
    output_file_name = "v2_OpenEnded_mscoco_val2014_vqav2_results.json"
    output_json_path = f"datasets/Results/llama_vision/{output_file_name}"

    # Load model, processor, tokenizer
    print(f"🔄 Loading model from: {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load questions
    questions_list = load_vqa_questions(questions_file, args.num_samples)

    # Run inference
    results = run_batch_inference(model, processor, tokenizer, device, questions_list, images_dir)

    # Save results
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Saved {len(results)} results to {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VQA inference with tuned LLaMA vision model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="HuggingFace model ID or local path (default: meta-llama/Llama-3.2-11B-Vision-Instruct)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of image-question pairs to run inference on (default: 50)"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="datasets/Images/mscoco/val2014",
        help="HuggingFace mscoco vqa v2 dataset image directory (default: datasets/Images/mscoco/val2014)"
    )

    args = parser.parse_args()
    main(args)
