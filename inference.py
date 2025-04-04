import os
import json
import random
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
# system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
system_message = "Just give answer in 1 word/number."

def get_formatted_question(question):
    return [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]

def process_generated_answer(answer):
    # "user\n\nWhere is the playing?assistant\n\nThe playing is taking place on a street."
    # Just get the answer after assistant\n\n
    # print("Raw answer:", answer)
    if "assistant" in answer:
        answer = answer.split("assistant")[1]
    # Remove everything after the first period
    if "." in answer:
        answer = answer.split(".")[0]
    # Remove all \n
    answer = answer.replace("\n", "")
    return answer.strip()
    

def load_vqa_questions(path, num_samples):
    with open(path, 'r') as f:
        questions_data = json.load(f)['questions']

    # selected_questions = []
    # for q in questions_data:
    #     if q['question_id'] == 158254000 or q['question_id'] == 201561001:
    #         selected_questions.append(q)     

    # return selected_questions
    
    return random.sample(questions_data, num_samples)

def construct_image_path(image_id, base_dir):
    filename = f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
    return os.path.join(base_dir, filename)


def generate_answer(model, processor, tokenizer, device, image, question, max_new_tokens=50):
    # print("Raw question:", question)
    # print("PIL image:", image)
    question_formatted = get_formatted_question(question)
    question_templated = tokenizer.apply_chat_template(question_formatted, tokenize=False)

    # print("🔄 Chat Templated Question", question_templated)
    inputs = processor(text=question_templated, images=image, return_tensors="pt", padding=True)

    inputs = {k: v.to(device) for k, v in inputs.items()}
    # print("🔄 Generating answer...")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # print("🔄 Done generating answer...")


    generated_answer = tokenizer.decode(output[0], skip_special_tokens=True)
    # print("Generated answer:", generated_answer)
    # processed_response = process_generated_answer(generated_answer)
    # print("Processed answer:", processed_response)
    return generated_answer


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

    # Process results
    for result in results:
        processed_response = process_generated_answer(result['answer'])
        result['answer'] = processed_response

    # Save results
    if results:
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
