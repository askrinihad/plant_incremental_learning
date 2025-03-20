import os
import json
import torch
import numpy as np
from PIL import Image
import clip
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, recall_score, f1_score

# Paths
JSON_FILE_PATH = "oxford_206_plantclef_features.json"
FINE_TUNED_MODEL_PATH = "finetuned_clip_vit_l14_species_classifier_20E.pth"
VALID_DIR = "gbif_206_classes"
OUTPUT_JSON_PATH = "classification_metrics.json"

# Load the CLIP model and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
model.load_state_dict(torch.load(FINE_TUNED_MODEL_PATH, map_location=device), strict=False)
model.eval()

# Load class embeddings
with open(JSON_FILE_PATH, 'r') as f:
    class_embeddings = json.load(f)
    for species_name, data in class_embeddings.items():
        class_embeddings[species_name]["average"] = np.array(data["average"])

# Helper functions
def extract_features(image_path, model, preprocess, device):
    try:
        image = Image.open(image_path).convert("RGB")
    except OSError as e:
        print(f"Error loading image {image_path}: {e}")
        os.remove(image_path)  # Delete corrupted image
        return None

    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_tensor).cpu().numpy().squeeze()
    return features


def predict(features, class_embeddings):
    max_similarity = -1
    predicted_species = None
    for species_name, class_data in class_embeddings.items():
        features = normalize([features])[0]
        class_embedding = normalize([class_data["average"]])[0]
        similarity = cosine_similarity([features], [class_embedding])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            predicted_species = species_name
    return predicted_species

if __name__ == "__main__":
    valid_classes = [species_name for species_name in os.listdir(VALID_DIR) if os.path.isdir(os.path.join(VALID_DIR, species_name)) and not species_name.startswith('.')]

    y_true = []
    y_pred = []
    class_wise_metrics = {}

    # Initialize metrics for each class
    for species_name in valid_classes:
        class_wise_metrics[species_name] = {
            "total_images": 0,
            "correct_predictions": 0,
            "predictions": [],
            "true_labels": []
        }

    # Process each class directory
    for species_name in valid_classes:
        species_dir = os.path.join(VALID_DIR, species_name)

        for image_name in tqdm(os.listdir(species_dir), desc=f"Processing {species_name}"):
            image_path = os.path.join(species_dir, image_name)
            if image_name.startswith('.') or not os.path.isfile(image_path):
                continue

            class_wise_metrics[species_name]["total_images"] += 1
            features = extract_features(image_path, model, preprocess, device)

            if features is None:
                continue  # Skip this image if it's corrupted and already deleted

            predicted_species = predict(features, class_embeddings)
            y_true.append(species_name.replace("_", " "))
            y_pred.append(predicted_species)

            class_wise_metrics[species_name]["true_labels"].append(species_name.replace("_", " "))
            class_wise_metrics[species_name]["predictions"].append(predicted_species)

            if predicted_species == species_name.replace("_", " "):
                class_wise_metrics[species_name]["correct_predictions"] += 1

    # Calculate global metrics
    global_accuracy = (np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)) * 100 if y_true else 0
    global_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    global_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    global_f1_score = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100

    # Calculate per-class metrics
    for species_name in valid_classes:
        total_images = class_wise_metrics[species_name]["total_images"]
        correct_predictions = class_wise_metrics[species_name]["correct_predictions"]

        if total_images > 0:
            class_accuracy = (correct_predictions / total_images) * 100
            precision = precision_score(class_wise_metrics[species_name]["true_labels"],
                                         class_wise_metrics[species_name]["predictions"],
                                         average='weighted', zero_division=0) * 100
            recall = recall_score(class_wise_metrics[species_name]["true_labels"],
                                   class_wise_metrics[species_name]["predictions"],
                                   average='weighted', zero_division=0) * 100
            f1 = f1_score(class_wise_metrics[species_name]["true_labels"],
                          class_wise_metrics[species_name]["predictions"],
                          average='weighted', zero_division=0) * 100
        else:
            class_accuracy, precision, recall, f1 = 0, 0, 0, 0

        class_wise_metrics[species_name].update({
            "class_accuracy": class_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

    global_metrics = {
        "global_accuracy": global_accuracy,
        "global_precision": global_precision,
        "global_recall": global_recall,
        "global_f1_score": global_f1_score,
        "class_wise_metrics": class_wise_metrics
    }

    with open(OUTPUT_JSON_PATH, "w") as json_file:
        json.dump(global_metrics, json_file, indent=4)

    print(f"Global metrics and class-wise metrics saved to {OUTPUT_JSON_PATH}")
