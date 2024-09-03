import os
import matplotlib.pyplot as plt
from data_preprocessing.video_preprocessing import preprocess_videos
from model_training.train import run_training

def plot_loss_chart(train_loss, val_loss, output_path='lossChart.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, marker='', linestyle='-', color='b', label='Training Loss')
    plt.plot(val_loss, marker='', linestyle='--', color='r', label='Validation Loss')
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, format='png')
    plt.close()
    print(f"Training and Validation Loss Chart saved as '{output_path}'.")

def plot_accuracy_chart(val_accuracy, output_path='accuracyChart.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracy, marker='', linestyle='-', color='g', label='Validation Accuracy')
    plt.title("Validation Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, format='png')
    plt.close()
    print(f"Validation Accuracy Chart saved as '{output_path}'.")

def get_folder_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size

def run_trainCLI():
    csv_file = "dataset.csv"
    epochs = 16  # Default value
    batch_size = 32  # Default value
    learning_rate = 0.001  # Default value
    chart_output_path = 'lossChart.png'  # Default path for the loss chart
    accuracy_output_path = 'accuracyChart.png'  # Default path for the accuracy chart
    model_output_dir = 'model'  # Directory to save the model

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    while True:
        # Display the selected CSV file and processedData folder size
        processed_data_folder = 'processedData'
        folder_size = get_folder_size(processed_data_folder) if os.path.exists(processed_data_folder) else 0
        folder_size_mb = folder_size / (1024 * 1024)  # Convert to MB
        
        print("\nTraining CLI")
        print(f"1. Select CSV File (Currently selected: {csv_file})")
        print(f"2. Preprocess Videos (ProcessedData folder size: {folder_size_mb:.2f} MB)")
        print(f"3. Edit Training Hyperparameters (Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate})")
        print(f"4. Train Model ")
        print(f"5. Change Chart Output Paths (Loss Chart: {chart_output_path}, Accuracy Chart: {accuracy_output_path})")
        print("6. Exit")
        choice = input("Select an option (1-6): ").strip()

        if choice == '1':
            new_csv_file = input("Enter path to the CSV file with video paths: ").strip()
            if not os.path.isfile(new_csv_file):
                print("Error: CSV file not found.")
                continue
            csv_file = new_csv_file
            print(f"CSV file selected: {csv_file}")

        elif choice == '2':
            if not os.path.isfile(csv_file):
                print("Error: CSV file must be selected first.")
                continue

            if not os.path.exists(processed_data_folder):
                os.makedirs(processed_data_folder)

            preprocess_videos(csv_file, processed_data_folder)
            print("All videos have been preprocessed into frames.")

        elif choice == '3':
            try:
                epochs = int(input("Enter number of epochs: ").strip())
                batch_size = int(input("Enter batch size: ").strip())
                learning_rate = float(input("Enter learning rate: ").strip())
                print(f"Hyperparameters updated: Epochs={epochs}, Batch Size={batch_size}, Learning Rate={learning_rate}")
            except ValueError as e:
                print(f"Error: {e}. Please enter valid numbers.")

        elif choice == '4':
            if not os.path.isfile(csv_file):
                print("Error: CSV file must be selected.")
                continue

            if not os.path.exists(processed_data_folder):
                print("Error: Processed data folder does not exist.")
                continue

            try:
                # Prompt the user for the model name
                model_name = input("Enter name for the model (default: 'cute'): ").strip() or "cute"
                model_path = os.path.join(model_output_dir, f"{model_name}.h5")

                # Run training and get loss and accuracy data
                train_loss, val_loss, val_accuracy = run_training(csv_file, processed_data_folder, epochs, batch_size, learning_rate, model_path)
                print("Training completed successfully")

                if train_loss and val_loss:
                    print("Training and Validation Loss Data:")
                    plot_loss_chart(train_loss, val_loss, output_path=chart_output_path)
                else:
                    print("No loss data available for plotting.")

                if val_accuracy:
                    print("Validation Accuracy Data:")
                    plot_accuracy_chart(val_accuracy, output_path=accuracy_output_path)
                else:
                    print("No accuracy data available for plotting.")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '5':
            new_loss_chart_path = input(f"Enter new path for the loss chart (Current path: {chart_output_path}): ").strip()
            new_accuracy_chart_path = input(f"Enter new path for the accuracy chart (Current path: {accuracy_output_path}): ").strip()
            if new_loss_chart_path:
                chart_output_path = new_loss_chart_path
            if new_accuracy_chart_path:
                accuracy_output_path = new_accuracy_chart_path
            print(f"Chart output paths updated to: Loss Chart: {chart_output_path}, Accuracy Chart: {accuracy_output_path}")

        elif choice == '6':
            break
        else:
            print("Invalid option, please choose again.")
