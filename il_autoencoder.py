import os
import json
import torch
import numpy as np
from PIL import Image
import clip
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import shutil

# Import Autoencoder
from autoencoder_training import Autoencoder  # Replace with the actual import path to your autoencoder

# Paths
JSON_FILE_PATH = "autoencoder_206_plantclef_features.json"  # Path to the existing JSON file
METRICS_FILE_PATH = "autoencoder_metrics_of_206_plantclef.json"
FINE_TUNED_MODEL_PATH = "finetuned_clip_vit_l14_species_classifier_20E.pth"
TRAIN_DIR_NEW = "split_dataset/train"
TEST_DIR_NEW = "split_dataset/test"

# Device and model setup
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)
clip_model.load_state_dict(torch.load(FINE_TUNED_MODEL_PATH, map_location=device), strict=False)
clip_model.eval()  # Set CLIP model to evaluation mode

# Initialize Autoencoder
autoencoder = Autoencoder(input_dim=768, latent_dim=128).to(device)  # Adjust input_dim to match CLIP feature size
autoencoder.load_state_dict(torch.load('autoencoder.pth'))  # Load pre-trained autoencoder
autoencoder.eval()  # Set autoencoder to evaluation mode

class IncrementalLearner:
    def __init__(self, json_file_path, metrics_file_path, clip_model, autoencoder, preprocess, device):
        self.json_file_path = json_file_path
        self.metrics_file_path = metrics_file_path
        self.clip_model = clip_model
        self.autoencoder = autoencoder
        self.preprocess = preprocess
        self.device = device
        self.class_embeddings = {}
        self.load_class_embeddings()
        self.initialize_metrics_file()

    def load_class_embeddings(self):
        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, 'r') as f:
                self.class_embeddings = json.load(f)
            for species_name, data in self.class_embeddings.items():
                data['average'] = np.array(data['average'])
                data['embeddings'] = [np.array(e) for e in data['embeddings']]
            print(f"Loaded class embeddings from {self.json_file_path}")
        else:
            print(f"No existing embeddings found. Starting fresh.")

    def save_class_embeddings(self):
        serializable_data = {
            species_name: {
                'average': data['average'].tolist(),
                'embeddings': [e.tolist() for e in data['embeddings']]
            }
            for species_name, data in self.class_embeddings.items()
        }
        with open(self.json_file_path, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        print(f"Saved class embeddings to {self.json_file_path}")



    def initialize_metrics_file(self):
        if not os.path.exists(self.metrics_file_path) or os.path.getsize(self.metrics_file_path) == 0:
            with open(self.metrics_file_path, 'w') as f:
                json.dump({"batches": []}, f, indent=4)
            print(f"Initialized {self.metrics_file_path} with default JSON structure.")
        else:
            print(f"Metrics file {self.metrics_file_path} already initialized.")

    def save_metrics_to_json(self, batch_number, accuracy, precision, recall, f1):
        if os.path.exists(self.metrics_file_path) and os.path.getsize(self.metrics_file_path) > 0:
            with open(self.metrics_file_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: {self.metrics_file_path} is invalid. Reinitializing it.")
                    data = {"batches": []}
        else:
            print(f"{self.metrics_file_path} is empty or missing. Initializing it.")
            data = {"batches": []}

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

    def extract_features_batch(self, image_paths):
        # Extract features using the CLIP model
        images = [self.preprocess(Image.open(img).convert("RGB")) for img in image_paths]
        image_tensor = torch.stack(images).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(image_tensor).cpu().numpy()

        # Pass the CLIP features through the autoencoder to get the compressed representation
        features_tensor = torch.tensor(features).to(self.device)
        with torch.no_grad():
            compressed_features = self.autoencoder(features_tensor.float()).cpu().numpy()  # Get compressed features
        return compressed_features

    def update_class_embeddings(self, species_name, features):
        if species_name not in self.class_embeddings:
            self.class_embeddings[species_name] = {"embeddings": [], "average": np.zeros_like(features)}

        class_data = self.class_embeddings[species_name]
        class_data["embeddings"].append(features)
        n = len(class_data["embeddings"])
        class_data["average"] = (class_data["average"] * (n - 1) + features) / n

        self.save_class_embeddings()

    def predict(self, features):
        max_similarity = -1
        predicted_species = None
        for species_name, class_data in self.class_embeddings.items():
            class_embedding = class_data["average"]
            similarity = cosine_similarity([features], [class_embedding])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                predicted_species = species_name
        return predicted_species

    def evaluate(self, valid_dir):
        y_true, y_pred = [], []
        for species_name in os.listdir(valid_dir):
            species_dir = os.path.join(valid_dir, species_name)
            for image_name in os.listdir(species_dir):
                image_path = os.path.join(species_dir, image_name)
                true_label = species_name
                features = self.extract_features_batch([image_path])[0]  # Get transformed features
                predicted_species = self.predict(features)
                y_true.append(true_label)
                y_pred.append(predicted_species)

        accuracy = accuracy_score(y_true, y_pred) * 100
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        return accuracy, precision, recall, f1

    def incremental_learning(self, train_dir, valid_dir, batch_size=50, eval_every_n=1, test_mode=True, max_images=5):
        train_image_paths = [(os.path.join(train_dir, species_name, img), species_name) 
                             for species_name in os.listdir(train_dir)
                             for img in os.listdir(os.path.join(train_dir, species_name))[:max_images if test_mode else None]]

        print(f"Collected {len(train_image_paths)} images for incremental learning.")

        for i in tqdm(range(0, len(train_image_paths), batch_size)):
            batch_paths = train_image_paths[i:i + batch_size]
            image_paths, labels = zip(*batch_paths)
            features = self.extract_features_batch(image_paths)  # Get transformed features

            for label, feature in zip(labels, features):
                self.update_class_embeddings(label, feature)

            if (i // batch_size + 1) % eval_every_n == 0:
                accuracy, precision, recall, f1 = self.evaluate(valid_dir)
                self.save_metrics_to_json(i // batch_size + 1, accuracy, precision, recall, f1)

        print("Incremental learning complete.")

# Main execution
if __name__ == "__main__":
    learner = IncrementalLearner(JSON_FILE_PATH, METRICS_FILE_PATH, clip_model, autoencoder, preprocess, device)
    learner.incremental_learning(TRAIN_DIR_NEW, TEST_DIR_NEW, batch_size=100, test_mode=False)
