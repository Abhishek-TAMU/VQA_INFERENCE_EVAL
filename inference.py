import json
import os
import random
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, A

import json
import os
import random
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm

# Configuration
# Replace with your model's repository or local path.
MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Directories for VQA data (adjust these paths as needed)
questions_file = "datasets/Questions/v2_OpenEnded_mscoco_val2014_questions.json"
images_dir = "datasets/Images/mscoco/val2014"

# Result json path
result_file_name = "v2_OpenEnded_mscoco_val2014_vqav2_results.json"
output_json_path = f"datasets/Results/llama_vision/{result_file_name}"
num_samples = 50  # Number of image-question pairs to run inference on

# Load processor, tokenizer, and model
processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def load_vqa_questions(path, num_samples):
    with open(path, 'r') as f:
        questions_data = json.load(f)['questions']
    return random.sample(questions_data, num_samples)

def construct_image_path(image_id, base_dir):
    filename = f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
    return os.path.join(base_dir, filename)

def generate_answer(image: Image.Image, question: str, max_new_tokens: int = 50) -> str:
    inputs = processor(text=question, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

def run_batch_inference(questions_list):
    results = []

    for q in tqdm(questions_list, desc="Running inference"):
        question_id = q['question_id']
        image_id = q['image_id']
        question_text = q['question']

        image_path = construct_image_path(image_id, images_dir)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path} — skipping.")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            answer = generate_answer(image, question_text)
            results.append({
                "question_id": question_id,
                "answer": answer
            })
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            continue

    return results

if __name__ == "__main__":
    questions_list = load_vqa_questions(questions_file, num_samples)
    results = run_batch_inference(questions_list)

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n Saved {len(results)} results to {output_json_path}")
