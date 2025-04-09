import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split


# Example dataset
class PropertyDataset(Dataset):
    def __init__(self, texts, supervised_inputs, labels):
        self.texts = texts
        self.supervised_inputs = supervised_inputs
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.supervised_inputs[idx], self.labels[idx]


def load_data(text_folder, csv_file):
        property_data = pd.read_csv(csv_file)
        property_data.columns = ["Filename", "Property"]
        property_data['Filename'] = property_data['Filename'].apply(str)
        data = []
        i=0
        for filename in os.listdir(text_folder):
            if filename.endswith(".txt"):
                i+=1
                file_path = os.path.join(text_folder, filename)
                with open(file_path, "r") as file:
                    text = file.read()
                    # text = combined_text(text)
                file_name = filename.split(".")[0]
                # print("check", file_name)
                # print("check1", property_data["Filename"])
                # Match the text file with its property in the CSV
                property_row = property_data[property_data["Filename"] == file_name]
                if not property_row.empty:
                    data.append({"text": text, "filename": int(file_name), "label": property_row.iloc[0][
                        "Property"]})  # Replace "Property" with your property column name
        return pd.DataFrame(data)




def split_data(dataset, train_ratio=0.8, random_seed=42):
    """
    Split a dataset into train and test sets.

    Args:
        dataset (Dataset): The dataset to split.
        train_ratio (float): The ratio of the dataset to use for training (default: 0.8).
        random_seed (int): Random seed for reproducibility (default: 42).

    Returns:
        train_dataset (Dataset): The training dataset.
        test_dataset (Dataset): The test dataset.
    """
    # Set the random seed for reproducibility
    torch.manual_seed(random_seed)

    # Calculate the sizes of the train and test sets
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset
