import os
import json
import torch
import numpy as np
from PIL import Image
import clip
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Load category-to-name mapping
CAT_TO_NAME_PATH = "Oxford_102_Flower_Dataset/cat_to_name.json"
JSON_FILE_PATH = "feature.json"
METRICS_FILE_PATH = "metrics.json"  # New file to save metrics
FINE_TUNED_MODEL_PATH = "finetuned_clip_vit_l14_species_classifier_20E.pth"
TRAIN_DIR = "Oxford_102_Flower_Dataset/dataset/train"  # Path to training folder
VALID_DIR = "Oxford_102_Flower_Dataset/dataset/valid"  # Path to validation folder

# Load category-to-name mapping
with open(CAT_TO_NAME_PATH, 'r') as f:
    cat_to_name = json.load(f)

# Load the CLIP model and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
model.load_state_dict(torch.load(FINE_TUNED_MODEL_PATH, map_location=device), strict=False)
model.eval()  # Set model to evaluation mode


class IncrementalLearner:
    def __init__(self, json_file_path, metrics_file_path, model, preprocess, device):
        self.json_file_path = json_file_path
        self.metrics_file_path = metrics_file_path
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.class_embeddings = {}  # Store average embeddings per class
        self.initialize_metrics_file()

    def initialize_metrics_file(self):
        """Initialize or load the metrics JSON file."""
        if not os.path.exists(self.metrics_file_path):
            with open(self.metrics_file_path, 'w') as f:
                json.dump({"batches": []}, f, indent=4)
            print(f"Created new metrics file at {self.metrics_file_path}")
        else:
            print(f"Metrics file already exists: {self.metrics_file_path}")

    def save_metrics_to_json(self, batch_number, accuracy, precision, recall, f1):
        """Save evaluation metrics for a batch to the metrics JSON file."""
        with open(self.metrics_file_path, 'r') as f:
            data = json.load(f)

        data["batches"].append({
            "batch_number": batch_number,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        with open(self.metrics_file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Metrics for batch {batch_number} saved to {self.metrics_file_path}")

    def extract_features(self, image_path):
        """Extract features from the CLIP model."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_tensor).cpu().numpy().squeeze()
        return features
    def update_class_embeddings(self, species_name, features):
        """Update the stored embeddings for a given species."""
        if species_name not in self.class_embeddings:
            self.class_embeddings[species_name] = {"embeddings": []}
        # Add the new feature to the embeddings list
        self.class_embeddings[species_name]["embeddings"].append(features)
    
        # Update the average embedding for the species
        all_embeddings = np.array(self.class_embeddings[species_name]["embeddings"])
        self.class_embeddings[species_name]["average"] = np.mean(all_embeddings, axis=0)

    def predict(self, features):
        """Predict the class by finding the most similar embedding."""
        max_similarity = -1
        predicted_species = None
    
        for species_name, class_data in self.class_embeddings.items():
            class_embedding = class_data["average"]  # Retrieve the average embedding
            similarity = cosine_similarity([features], [class_embedding])[0][0]  # Compute cosine similarity
            if similarity > max_similarity:
                max_similarity = similarity
                predicted_species = species_name
    
        return predicted_species


    def evaluate(self, valid_dir, cat_to_name):
        """Evaluate the current model on the validation set."""
        y_true, y_pred = [], []

        for species_id in os.listdir(valid_dir):
            species_name = cat_to_name.get(species_id, "Unknown")
            species_dir = os.path.join(valid_dir, species_id)
            for image_name in os.listdir(species_dir):
                image_path = os.path.join(species_dir, image_name)
                true_label = species_name

                # Extract features for the validation image
                features = self.extract_features(image_path)

                # Predict class using current embeddings
                predicted_species = self.predict(features)

                y_true.append(true_label)
                y_pred.append(predicted_species)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_true, y_pred) * 100
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100

        return accuracy, precision, recall, f1

    def incremental_learning(self, train_dir, valid_dir, batch_size=50, eval_every_n=1, test_mode=True, max_images=5):
        """Perform incremental learning."""
        batch_accuracy = []
        train_image_paths = []

        # Collect train image paths
        for species_id in os.listdir(train_dir):
            species_dir = os.path.join(train_dir, species_id)
            for idx, image_name in enumerate(os.listdir(species_dir)):
                train_image_paths.append((os.path.join(species_dir, image_name), species_id))
                if test_mode and len(train_image_paths) >= max_images:
                    break
            if test_mode and len(train_image_paths) >= max_images:
                break

        num_images = len(train_image_paths)
        print(f"Collected {num_images} images for incremental learning.")

        for i in tqdm(range(0, num_images, batch_size)):
            batch_paths = train_image_paths[i:i + batch_size]

            for image_path, species_id in batch_paths:
                # Ground truth label
                species_name = cat_to_name.get(species_id, "Unknown")

                # Extract features using fine-tuned model
                features = self.extract_features(image_path)

                # Update class embeddings
                self.update_class_embeddings(species_name, features)

            # Evaluate model after processing each batch
            if (i // batch_size + 1) % eval_every_n == 0:
                accuracy, precision, recall, f1 = self.evaluate(valid_dir, cat_to_name)
                batch_number = i // batch_size + 1
                batch_accuracy.append((batch_number, accuracy, precision, recall, f1))

                # Save metrics to JSON file
                self.save_metrics_to_json(batch_number, accuracy, precision, recall, f1)

                print(f"Batch {batch_number}:")
                print(f"Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1 Score: {f1:.2f}%")

                if test_mode:
                    break  # Stop after first evaluation in test mode

        print(f"Batch-wise accuracy history: {batch_accuracy}")
        return batch_accuracy


if __name__ == "__main__":
    # Initialize learner
    learner = IncrementalLearner(JSON_FILE_PATH, METRICS_FILE_PATH, model, preprocess, device)

    # Perform incremental learning
    results = learner.incremental_learning(TRAIN_DIR, VALID_DIR, batch_size=100, test_mode=False)