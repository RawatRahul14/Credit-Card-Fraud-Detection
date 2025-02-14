import kagglehub
import os
import shutil

# Define the folder where the dataset will be stored
folder_name = "Data"
os.makedirs(folder_name, exist_ok = True)

# Check if the dataset already exists
if not os.listdir(folder_name):  # Only download if the folder is empty
    print("Downloading dataset...")
    dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

    # Move the downloaded dataset contents to the "Data" folder
    if os.path.exists(dataset_path):
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            shutil.move(item_path, folder_name)

    print("Dataset downloaded successfully!")
else:
    print("Dataset already exists. Skipping download.")

print("Path to dataset files:", os.path.abspath(folder_name))