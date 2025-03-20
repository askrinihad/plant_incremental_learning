import os
import json
import torch
import numpy as np
import faiss
from PIL import Image
import clip
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil

# Paths
JSON_FILE_PATH = "oxford_206_plantclef_features_faiss.json"
METRICS_FILE_PATH = "metrics_of_sub_plantclef_faiss.json"
FINE_TUNED_MODEL_PATH = "finetuned_clip_vit_l14_species_classifier_20E.pth"
TRAIN_DIR_NEW = "split_dataset/train"
TEST_DIR_NEW = "split_dataset/test"

# Device and model setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
model.load_state_dict(torch.load(FINE_TUNED_MODEL_PATH, map_location=device), strict=False)
model.eval()  # Set model to evaluation mode

# Incremental Learner Class with FAISS
class IncrementalLearner:
    def __init__(self, json_file_path, metrics_file_path, model, preprocess, device):
        self.json_file_path = json_file_path
        self.metrics_file_path = metrics_file_path
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.class_embeddings = {}
        self.index = None  # FAISS index
        self.species_list = []
        self.load_class_embeddings()
        self.build_faiss_index()
        self.initialize_metrics_file()

    def load_class_embeddings(self):
        """Load class embeddings from JSON file and rebuild FAISS index."""
        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, 'r') as f:
                self.class_embeddings = json.load(f)
            for species_name, data in self.class_embeddings.items():
                data['average'] = np.array(data['average'])
                data['embeddings'] = [np.array(e) for e in data['embeddings']]
            print(f"Loaded class embeddings from {self.json_file_path}")
        else:
            print(f"No existing embeddings found. Starting fresh.")

    def build_faiss_index(self):
        """Build or update the FAISS index dynamically."""
        if len(self.class_embeddings) == 0:
            print("No class embeddings found. FAISS index not built.")
            return

        embedding_dim = len(next(iter(self.class_embeddings.values()))["average"])
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 (Euclidean distance) index
        self.species_list = list(self.class_embeddings.keys())

        all_embeddings = np.array([self.class_embeddings[species]["average"] for species in self.species_list]).astype(np.float32)
        self.index.add(all_embeddings)  # Add embeddings to FAISS
        print(f"FAISS index built with {len(self.species_list)} species.")

    def save_class_embeddings(self):
        """Save class embeddings to JSON and update FAISS index."""
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
        """Initialize the metrics JSON file if it doesn't exist."""
        if not os.path.exists(self.metrics_file_path) or os.path.getsize(self.metrics_file_path) == 0:
            with open(self.metrics_file_path, 'w') as f:
                json.dump({"batches": []}, f, indent=4)
            print(f"Initialized {self.metrics_file_path} with default JSON structure.")

    def save_metrics_to_json(self, batch_number, accuracy, precision, recall, f1):
        """Save evaluation metrics after each batch."""
        if os.path.exists(self.metrics_file_path) and os.path.getsize(self.metrics_file_path) > 0:
            with open(self.metrics_file_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: {self.metrics_file_path} is invalid. Reinitializing.")
                    data = {"batches": []}
        else:
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
        print(f"Metrics for batch {batch_number} saved.")

    def extract_features_batch(self, image_paths):
        """Extract features for a batch of images."""
        images = [self.preprocess(Image.open(img).convert("RGB")) for img in image_paths]
        image_tensor = torch.stack(images).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_tensor).cpu().numpy()
        return features

    def update_class_embeddings(self, species_name, features):
        """Update class embeddings and dynamically rebuild FAISS index."""
        if species_name not in self.class_embeddings:
            self.class_embeddings[species_name] = {"embeddings": [], "average": np.zeros_like(features)}

        class_data = self.class_embeddings[species_name]
        class_data["embeddings"].append(features)
        n = len(class_data["embeddings"])
        class_data["average"] = (class_data["average"] * (n - 1) + features) / n

        self.save_class_embeddings()
        self.build_faiss_index()  # Update FAISS

    def predict(self, features):
        """Use FAISS to find the nearest class."""
        if self.index is None or len(self.species_list) == 0:
            print("FAISS index is empty, falling back to random classification.")
            return None

        features = np.array(features).astype(np.float32).reshape(1, -1)
        _, closest_index = self.index.search(features, 1)  # Search for the nearest neighbor
        return self.species_list[closest_index[0][0]]

    def evaluate(self, valid_dir):
        """Evaluate the model using FAISS-based predictions."""
        y_true, y_pred = [], []
        for species_name in os.listdir(valid_dir):
            species_dir = os.path.join(valid_dir, species_name)
            for image_name in os.listdir(species_dir):
                image_path = os.path.join(species_dir, image_name)
                true_label = species_name
                features = self.extract_features_batch([image_path])[0]
                predicted_species = self.predict(features)
                y_true.append(true_label)
                y_pred.append(predicted_species)

        accuracy = accuracy_score(y_true, y_pred) * 100
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        return accuracy, precision, recall, f1

    def incremental_learning(self, train_dir, valid_dir, batch_size=50):
        """Perform incremental learning and evaluate using FAISS."""
        train_image_paths = [(os.path.join(train_dir, species, img), species) 
                             for species in os.listdir(train_dir)
                             for img in os.listdir(os.path.join(train_dir, species))]

        print(f"Collected {len(train_image_paths)} images for incremental learning.")

        for i in tqdm(range(0, len(train_image_paths), batch_size)):
            batch_paths = train_image_paths[i:i + batch_size]
            image_paths, labels = zip(*batch_paths)
            features = self.extract_features_batch(image_paths)

            for label, feature in zip(labels, features):
                self.update_class_embeddings(label, feature)

            accuracy, precision, recall, f1 = self.evaluate(valid_dir)
            self.save_metrics_to_json(i // batch_size + 1, accuracy, precision, recall, f1)

        print("Incremental learning complete.")

# Main execution
if __name__ == "__main__":
    learner = IncrementalLearner(JSON_FILE_PATH, METRICS_FILE_PATH, model, preprocess, device)
    learner.incremental_learning(TRAIN_DIR_NEW, TEST_DIR_NEW, batch_size=100)
