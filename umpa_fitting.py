import os
import json
import torch
import numpy as np
from PIL import Image
import clip
import joblib
from tqdm import tqdm
import umap

# Paths
JSON_FILE_PATH = "umap_206_plantclef_features.json"  # Path to the existing JSON file
FINE_TUNED_MODEL_PATH = "finetuned_clip_vit_l14_species_classifier_20E.pth"
TRAIN_DIR_NEW = "training"  # Training data directory

# Device and model setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
model.load_state_dict(torch.load(FINE_TUNED_MODEL_PATH, map_location=device), strict=False)
model.eval()  # Set model to evaluation mode

class UMAPFitting:
    def __init__(self, model, preprocess, device, json_file_path):
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.json_file_path = json_file_path
        self.class_embeddings = {}

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
            self.class_embeddings = {}  # Initialize an empty dictionary

    def extract_features_batch(self, image_paths):
        """Extract features for a batch of images."""
        images = [self.preprocess(Image.open(img).convert("RGB")) for img in image_paths]
        image_tensor = torch.stack(images).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_tensor).cpu().numpy()
        return features
    
    def fit_umap(self):
        """Fit UMAP on all data."""
        all_embeddings = []
        for species_name, data in self.class_embeddings.items():
            for emb in data['embeddings']:
                # Log the shape of each embedding
                print(f"Embedding shape before flattening: {emb.shape}")
                
                # Ensure embedding is flattened to 1D (if it's not already)
                if emb.ndim > 1:
                    emb = emb.flatten()
                    print(f"Flattened embedding shape: {emb.shape}")
                
                # Check if embedding is a 1D array
                if emb.ndim == 1 and emb.shape[0] == 768:  # Ensure shape is (768,)
                    all_embeddings.append(emb)
                else:
                    print(f"Skipping embedding with invalid shape: {emb.shape}")

        if all_embeddings:
            try:
                all_embeddings = np.array(all_embeddings)
                print(f"Total number of embeddings: {all_embeddings.shape[0]}")
                if len(all_embeddings) > 1:  # Ensure there are enough data points for UMAP to fit
                    
                    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=512)
                    umap_model.fit(all_embeddings)  # Fit UMAP on the existing embeddings
                    print(f"UMAP model fitted on existing embeddings.")
                    joblib.dump(umap_model, 'fitted_umap_model_512D.pkl')  # Save the fitted UMAP model
                else:
                    print("Not enough data to fit UMAP yet.")
            except ValueError as e:
                print(f"Error while converting to numpy array: {e}")
                print("All embeddings may have inconsistent shapes.")
        else:
            print(f"No valid embeddings available for UMAP fitting yet.")


# Main execution
if __name__ == "__main__":
    umap_fitting = UMAPFitting(model, preprocess, device, JSON_FILE_PATH)
    umap_fitting.load_class_embeddings()

    # Collect all image paths to extract features
    train_image_paths = [(os.path.join(TRAIN_DIR_NEW, species_name, img), species_name)
                         for species_name in os.listdir(TRAIN_DIR_NEW)
                         for img in os.listdir(os.path.join(TRAIN_DIR_NEW, species_name))]

    print(f"Collected {len(train_image_paths)} images for UMAP fitting.")

    # Extract features from all images
    for i in tqdm(range(0, len(train_image_paths), 50)):
        batch_paths = train_image_paths[i:i + 50]
        image_paths, labels = zip(*batch_paths)
        features = umap_fitting.extract_features_batch(image_paths)

        # Update the class embeddings
        for label, feature in zip(labels, features):
            if label not in umap_fitting.class_embeddings:
                umap_fitting.class_embeddings[label] = {"embeddings": [], "average": np.zeros_like(feature)}
            umap_fitting.class_embeddings[label]["embeddings"].append(feature)

    # Now, fit UMAP
    umap_fitting.fit_umap()
