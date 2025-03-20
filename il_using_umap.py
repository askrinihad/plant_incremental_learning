import os
import json
import torch
import numpy as np
from PIL import Image
import clip
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances

# Paths
JSON_FILE_PATH = "umap_206_plantclef_features.json"  # Path to the existing JSON file
METRICS_FILE_PATH = "metrics_umap_206_sub_plantclef.json"
FINE_TUNED_MODEL_PATH = "finetuned_clip_vit_l14_species_classifier_20E.pth"
TRAIN_DIR_NEW = "split_dataset/train"
TEST_DIR_NEW = "split_dataset/test"

# Device and model setup
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
        self.class_embeddings = {}
        self.load_class_embeddings()
        self.umap_model = joblib.load('fitted_umap_model_512D.pkl')  # Load the pre-fitted UMAP model
       

    def load_class_embeddings(self):
        """Load class embeddings from the JSON file."""
        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, 'r') as f:
                self.class_embeddings = json.load(f)
            for species_name, data in self.class_embeddings.items():
                data['average'] = np.array(data['average'])
                data['embeddings'] = [np.array(e) for e in data['embeddings']]
            print(f"Loaded class embeddings from {self.json_file_path}")
        else:
            print(f"No existing embeddings found. Creating a new empty feature file.")
            self.class_embeddings = {}

    def extract_features_batch(self, image_paths):
        """Extract features for a batch of images."""
        images = [self.preprocess(Image.open(img).convert("RGB")) for img in image_paths]
        image_tensor = torch.stack(images).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_tensor).cpu().numpy()
        return features

    def update_class_embeddings(self, species_name, features):
        """Update the embeddings for a class (species)."""
        if species_name not in self.class_embeddings:
            self.class_embeddings[species_name] = {"embeddings": [], "average": np.zeros_like(features)}

        class_data = self.class_embeddings[species_name]

        # Normalize the embedding before appending (L2 normalization)
        normalized_features = features / np.linalg.norm(features)

        # Use the pre-fitted UMAP model to reduce the dimensions of the new features
        reduced_embedding = self.umap_model.transform([normalized_features])[0]#cuz the res if transform is 2D numpy array [[x1,X2... ]], it's to extract [x1,x2..]

        # Append the reduced embedding
        class_data["embeddings"].append(reduced_embedding)

        # Update the average of the reduced embeddings
        class_data["average"] = np.mean(class_data["embeddings"], axis=0)  # Update the average embedding

        self.save_class_embeddings()

    def save_class_embeddings(self):
        """Save updated class embeddings to the JSON file."""
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

    def evaluate(self, valid_dir):
        """Evaluate model performance on the validation set."""
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
    
    """ Predict the species using cosine similarity.
    def predict(self, features):
       
        reduced_features = self.umap_model.transform([features])  # Reduce the dimension of input features

        max_similarity = -1
        predicted_species = None
        for species_name, class_data in self.class_embeddings.items():
            class_embedding = class_data["average"]
            
            # Ensure both are 2D for cosine similarity
            similarity = cosine_similarity(reduced_features.reshape(1, -1), class_embedding.reshape(1, -1))[0][0]
            
            if similarity > max_similarity:
                max_similarity = similarity
                predicted_species = species_name
        return predicted_species
    
    """
    def predict(self, features):
        """Predict the species using Euclidean distance."""
        reduced_features = self.umap_model.transform([features])  # Reduce the dimension of input features

        min_distance = float("inf")
        predicted_species = None
        for species_name, class_data in self.class_embeddings.items():
            class_embedding = class_data["average"]
            
            # Euclidean Distance
            distance = euclidean_distances(reduced_features.reshape(1, -1), class_embedding.reshape(1, -1))[0][0]
            
            if distance < min_distance:
                min_distance = distance
                predicted_species = species_name
        return predicted_species



    def incremental_learning(self, train_dir, valid_dir, batch_size=50, eval_every_n=1):
        """Run incremental learning on new batches."""
        train_image_paths = [(os.path.join(train_dir, species_name, img), species_name) 
                             for species_name in os.listdir(train_dir)
                             for img in os.listdir(os.path.join(train_dir, species_name))]

        for i in tqdm(range(0, len(train_image_paths), batch_size)):
            batch_paths = train_image_paths[i:i + batch_size]
            image_paths, labels = zip(*batch_paths)
            features = self.extract_features_batch(image_paths)

            for label, feature in zip(labels, features):
                self.update_class_embeddings(label, feature)

            if (i // batch_size + 1) % eval_every_n == 0:
                accuracy, precision, recall, f1 = self.evaluate(valid_dir)
                print(f"Batch {i // batch_size + 1}: Accuracy={accuracy:.2f}% Precision={precision:.2f}% Recall={recall:.2f}% F1={f1:.2f}%")

        print("Incremental learning complete.")

# Main execution
if __name__ == "__main__":
    learner = IncrementalLearner(JSON_FILE_PATH, METRICS_FILE_PATH, model, preprocess, device)
    learner.incremental_learning(TRAIN_DIR_NEW, TEST_DIR_NEW, batch_size=50)
