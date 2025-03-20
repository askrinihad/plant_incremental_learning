import os
import torch
import numpy as np
from PIL import Image
import clip
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# Define the Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=128):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# Function to extract features using CLIP
def extract_features_batch(image_paths, model, preprocess, device, batch_size=32):
    # Preprocess and stack the images into batches
    images = [preprocess(Image.open(img).convert("RGB")) for img in image_paths]
    image_tensor = torch.stack(images).to(device)

    with torch.no_grad():
        features = model.encode_image(image_tensor).cpu().numpy()
    return features

class FeatureDataset(Dataset):
    def __init__(self, image_paths, model, preprocess, device):
        self.image_paths = image_paths
        self.model = model
        self.preprocess = preprocess
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        feature = extract_features_batch([image_path], self.model, self.preprocess, self.device)
        return torch.tensor(feature[0], dtype=torch.float32).to(self.device)  # Convert to tensor and move to device

# Function to collect image paths from a directory (train or test)
def collect_image_paths(directory):
    image_paths = []
    for species_name in os.listdir(directory):
        species_dir = os.path.join(directory, species_name)
        if os.path.isdir(species_dir):
            for img_name in os.listdir(species_dir):
                img_path = os.path.join(species_dir, img_name)
                if img_path.lower().endswith(('png', 'jpg', 'jpeg')):
                    image_paths.append(img_path)
    return image_paths

def train_autoencoder(model, dataloader, criterion, optimizer, num_epochs=20, device="cuda"):
    model.to(device)  # Move model to the specified device (GPU or CPU)
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in dataloader:
            data = data.to(device)  # Ensure data is on the correct device (GPU or CPU)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Calculate accuracy at the end of each epoch
        accuracy = calculate_accuracy(model, dataloader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}, Accuracy: {accuracy*100:.2f}%")

def test_autoencoder(model, dataloader):
    model.eval()
    total_mse = 0.0

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)  # Ensure data is on the correct device (GPU or CPU)
            reconstructed_features = model(data)
            
            # Ensure the original and reconstructed features are of the same shape
            mse = mean_squared_error(data.cpu().numpy(), reconstructed_features.cpu().numpy())  # Move data to CPU for compatibility
            total_mse += mse

    average_mse = total_mse / len(dataloader)
    print(f"Mean Squared Error (MSE): {average_mse:.4f}")
    return average_mse


def calculate_accuracy(model, dataloader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)  # Move data to GPU/CPU
            reconstructed_features = model(data)  # Get reconstructed features
            
            # Calculate cosine similarity between original and reconstructed features
            similarity = torch.nn.functional.cosine_similarity(data, reconstructed_features, dim=1)
            
            # Convert similarity to binary classification (threshold-based)
            threshold = 0.9  # You can adjust this based on your validation results
            predicted_labels = (similarity > threshold).long()  # 1 if similar, 0 otherwise

            # Since you're comparing the reconstruction of the same data, it's always true that original matches itself
            y_true.extend([1] * len(data))  # True labels are all 1 (original matches itself)
            y_pred.extend(predicted_labels.cpu().tolist())  # Convert tensor to list

    accuracy = accuracy_score(y_true, y_pred)  # Compute accuracy
    return accuracy

def load_autoencoder(model, path="autoencoder.pth"):
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {path}")
    return model

if __name__ == "__main__":

    
    #train_dir = "split_dataset/train"
    test_dir = "split_dataset/test"

    #train_image_paths = collect_image_paths(train_dir)
    test_image_paths = collect_image_paths(test_dir)

    # Create DataLoader for training and testing datasets
    #train_dataset = FeatureDataset(image_paths=train_image_paths, model=model, preprocess=preprocess, device=device)
    test_dataset = FeatureDataset(image_paths=test_image_paths, model=model, preprocess=preprocess, device=device)

    #train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the Autoencoder model
    autoencoder = Autoencoder(input_dim=768, latent_dim=128).to(device)

    # Load the pre-trained model
    autoencoder = load_autoencoder(autoencoder, path="autoencoder.pth")  # Ensure you specify the correct path

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    # Train the autoencoder
    #print("Starting training...")
    #train_autoencoder(autoencoder, train_dataloader, criterion, optimizer, num_epochs=10, device=device)

    # Save the trained autoencoder
    #torch.save(autoencoder.state_dict(), "autoencoder.pth")
    #print("Autoencoder model saved successfully!")

    print("Testing the autoencoder...")
    test_mse = test_autoencoder(autoencoder, test_dataloader)
    print(f"Final Test MSE: {test_mse}")

    # Calculate accuracy (or similarity-based evaluation)
    accuracy = calculate_accuracy(autoencoder, test_dataloader)
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
