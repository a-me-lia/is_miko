import os
import torch
from PIL import Image
from torchvision import transforms
from model_training.model import create_model
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader
from tqdm import tqdm  

class PredictCLI:
    def __init__(self):
        self.model = None
        self.image_path = None
        self.default_model_path = 'cute.pth'
        self.tests_csv = 'datasetTests.csv'
        self.data_folder = None

    def load_model(self):
        model_path = self.prompt_for_path(f"Enter path to the trained model (default: {self.default_model_path}): ")
        if not model_path:
            model_path = self.default_model_path

        if not os.path.isfile(model_path):
            print("Error: Model file not found, using default.")

        try:
            self.model = create_model().cuda()
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error: Failed to load model: {e}")

    def upload_image(self):
        image_path = self.prompt_for_path("Enter path to the image file: ")
        if not os.path.isfile(image_path):
            print("Error: Image file not found.")
            return
        
        self.image_path = image_path
        print("Image uploaded successfully.")

    def predict(self):
        if not self.model:
            print("Error: Model must be loaded before making predictions.")
            return

        if not self.image_path:
            print("Error: Image must be uploaded before making predictions.")
            return
        
        try:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = Image.open(self.image_path).convert('RGB')
            image = transform(image).unsqueeze(0).cuda()

            with torch.no_grad():
                prediction = self.model(image).item()

            print(f"Predicted interest: {prediction:.2f}")
        except Exception as e:
            print(f"Error: Prediction failed: {e}")
        print(f"Error: Batch validation failed: {e}")

    def batch_validate(self):
        if not self.model:
            print("Error: Model must be loaded before batch validation.")
            return

        # Prompt user for data folder
        data_folder_choice = input("Use data from 'tests' or 'processedData'? ").strip().lower()
        if data_folder_choice not in ['tests', 'processeddata']:
            print("Invalid choice. Please select 'tests' or 'processedData'.")
            return

        self.data_folder = data_folder_choice

        # Prompt user for number of images to sample
        try:
            num_samples = int(input("Enter the number of images to sample: ").strip())
            if num_samples <= 0:
                raise ValueError("Number of samples must be a positive integer.")
        except ValueError as ve:
            print(f"Invalid input: {ve}")
            return

        try:
            thresholds = pd.read_csv(self.tests_csv)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # Dictionary to store statistics for each class
            class_stats = {}

            for _, row in thresholds.iterrows():
                folder_path = os.path.join(self.data_folder, row['TestFolderPath'])
                threshold = row['GreaterOrLessThan']
                expected_output = row['ExpectedOutput']
                class_name = os.path.basename(folder_path)  # Use folder name as class name

                if not os.path.isdir(folder_path):
                    print(f"Warning: Folder {folder_path} does not exist.")
                    continue
                
                image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                num_images = len(image_files)

                print(f"Checking folder: {folder_path}")
                print(f"Found {num_images} images.")
                
                if num_images == 0:
                    print(f"No images found in folder {folder_path}.")
                    continue
                
                random_images = random.sample(image_files, min(num_samples, num_images))  # Sample the specified number of images
                print(f"Sampling {len(random_images)} images.")
                
                # Initialize lists for this class
                predictions = []
                num_passes = 0
                num_fails = 0
                
                for img_path in random_images:
                    print(f"Processing image: {img_path}")
                    image = Image.open(img_path).convert('RGB')
                    image = transform(image).unsqueeze(0).cuda()

                    with torch.no_grad():
                        prediction = self.model(image).item()
                        predictions.append(prediction)
                        
                        if (threshold == 'greater' and prediction > expected_output) or (threshold == 'less' and prediction < expected_output):
                            num_passes += 1
                        else:
                            num_fails += 1
                
                # Store statistics for the class
                if predictions:
                    avg_prediction = np.mean(predictions)
                    median_prediction = np.median(predictions)
                    stdev_prediction = np.std(predictions)
                    min_prediction = np.min(predictions)
                    max_prediction = np.max(predictions)

                    class_stats[class_name] = {
                        'Average Prediction': f"{avg_prediction:.6f}",
                        'Median Prediction': f"{median_prediction:.6f}",
                        'Standard Deviation': f"{stdev_prediction:.6f}",
                        'Min Prediction': f"{min_prediction:.6f}",
                        'Max Prediction': f"{max_prediction:.6f}",  
                        'Passes': num_passes,
                        'Fails': num_fails
                    }
                else:
                    class_stats[class_name] = {
                        'Average Prediction': 'N/A',
                        'Median Prediction': 'N/A',
                        'Standard Deviation': 'N/A',
                        'Min Prediction': 'N/A',
                        'Max Prediction': 'N/A',
                        'Passes': num_passes,
                        'Fails': num_fails
                    }

            # Print the statistics row by row for each class
            print("\nBatch Validation Results:")
            print(f"{'Class':<20}{'Avg Prediction':<20}{'Median Prediction':<20}{'Stdev Prediction':<20}{'Min Prediction':<20}{'Max Prediction':<20}{'Passes':<10}{'Fails':<10}")
            print("="*130)
            
            for class_name, stats in class_stats.items():
                print(f"{class_name:<20}"
                    f"{stats['Average Prediction']:<20}" 
                    f"{stats['Median Prediction']:<20}"
                    f"{stats['Standard Deviation']:<20}"
                    f"{stats['Min Prediction']:<20}"
                    f"{stats['Max Prediction']:<20}"
                    f"{stats['Passes']:<10}"
                    f"{stats['Fails']:<10}")

        except Exception as e:
            print(f"Error: Batch validation failed: {e}")



    def prompt_for_path(self, prompt_message):
        """Prompt for a file or directory path with tab completion."""
        path_completer = PathCompleter()
        return prompt(prompt_message, completer=path_completer).strip()

def run_predictCLI():
    cli = PredictCLI()
    while True:
        print("\nPrediction CLI")
        print(f"1. Load model (Current default: {cli.default_model_path})")
        print("2. Upload image")
        print("3. Predict")
        print("4. Batch Validate")
        print("q. Exit")
        choice = input("Select an option (1-5): ").strip()

        if choice == '1':
            cli.load_model()
        elif choice == '2':
            cli.upload_image()
        elif choice == '3':
            cli.predict()
        elif choice == '4':
            cli.batch_validate()
        elif choice == 'q':
            break
        else:
            print("Invalid option, please choose again.")

