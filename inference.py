import json
import os
import random
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM

# Replace with your model's repository or local path.
MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Directories for VQA data (adjust these paths as needed)
questions_file = "datasets/Questions/v2_OpenEnded_mscoco_val2014_questions.json"
images_dir = "datasets/Images/mscoco/val2014"

# Load the processor, tokenizer, and model.
processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_answer(image: Image.Image, question: str, max_new_tokens: int = 50) -> str:
    """
    Runs inference on a tuned LLaMA vision model.
    
    Args:
        image (PIL.Image): Input image.
        question (str): The question about the image.
        max_new_tokens (int): Maximum number of tokens to generate for the answer.
    
    Returns:
        str: The generated answer.
    """
    inputs = processor(text=question, images=image, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return answer

def load_vqa_question(questions_json_path: str):
    """
    Loads the VQAv2 questions JSON and returns a random question.
    
    Returns:
        A tuple (question_text, image_id, question_id).
    """
    with open(questions_json_path, 'r') as f:
        data = json.load(f)

    # VQAv2 questions are typically under the "questions" key.
    questions = data['questions']

    # Pick a random question.
    q = random.choice(questions)
    return q['question'], q['image_id'], q['question_id']

def construct_image_path(image_id: int, images_dir: str) -> str:
    """
    Construct the image filename based on the image_id.
    VQAv2 images are usually named like: "COCO_val2014_000000<image_id>.jpg"
    
    Args:
        image_id (int): The image ID from the VQA dataset.
        images_dir (str): The directory containing the images.
    
    Returns:
        str: The full path to the image.
    """
    # Format the image id with zero-padding (12 digits)
    filename = f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
    return os.path.join(images_dir, filename)

def save_result(question_id: int, answer: str, output_path: str):
    """
    Save the generated answer as a new object in a JSON array.

    Args:
        question_id (int): The question ID.
        answer (str): The generated answer.
        output_path (str): The path to save the JSON result array.
    """
    result_entry = {
        "question_id": question_id,
        "answer": answer
    }

    # Load existing results if the file exists
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            try:
                existing_data = json.load(f)
                assert isinstance(existing_data, list)
            except (json.JSONDecodeError, AssertionError):
                existing_data = []
    else:
        existing_data = []

    # Append the new result
    existing_data.append(result_entry)

    # Save back to file
    with open(output_path, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"Appended result to {output_path}")


if __name__ == "__main__":
    # Load a random VQA question.
    question_text, image_id, question_id = load_vqa_question(questions_file)
    print(f"Question ID: {question_id}\nQuestion: {question_text}\nImage ID: {image_id}")

    # Construct the image path.
    image_path = construct_image_path(image_id, images_dir)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Open the image.
    image = Image.open(image_path).convert("RGB")
    
    # Generate the answer.
    answer = generate_answer(image, question_text)
    print("Generated Answer:", answer)
    
    # Save the answer with the question ID to a JSON file.
    result_file_name = "v2_OpenEnded_mscoco_val2014_vqav2_results.json"
    output_json_path = f"datasets/Results/llama_vision/{result_file_name}"  # Change path/filename if needed.
    save_result(question_id, answer, output_json_path)
