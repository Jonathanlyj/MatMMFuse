import datetime
import itertools
import torch.nn as nn
from transformers import AdamW, get_cosine_schedule_with_warmup
# from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer
from utils import summarize_dataset
import process as process
from training import model_setup, loader_setup
from torch.utils.data import  Dataset, random_split,Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from data_combined import PropertyDataset, split_data, load_data
from fusion_model import CombinedEmbeddingModel, SupervisedModel
from torch_geometric.data import Batch
import torch.optim as optim
import torch
import pandas as pd
import os
import numpy as np
import random
import yaml

def get_data(config):
    text_folder = "/scratch/yll6162/MatMMFuse/data/text_data"  # Path to text files
    csv_file = "/scratch/yll6162/MatMMFuse/targets.csv"  # Path to the CSV file
    cif_data_path_new = "/scratch/yll6162/MatMMFuse/data/bulk_data" # Path to the CIF file
    job_parameters, training_parameters, model_parameters = config["Job"]['Training'], config["Training"],config["Models"]["CGCNN_demo"]
    processing_parameters = config["Processing"]
    supervised_model = SupervisedModel(training_parameters, model_parameters,job_parameters,processing_parameters)
    print("Loading Model")
    dataset_cif = supervised_model.get_data(cif_data_path_new)
    loader_cif = supervised_model.load_data(dataset_cif)
    temp_list_cifs =[]
    for batch_cif in loader_cif:
        temp_list_cifs.append(Batch.to_data_list(batch_cif))
    batched_list_cifs = list(itertools.chain.from_iterable(temp_list_cifs))
    batched_list_cifs_sorted = sorted(batched_list_cifs, key=lambda x: int(x.structure_id[0][0]))
    df = load_data(text_folder, csv_file)
    df_sorted = df.sort_values(by=['filename'], ascending=True)
    fnames = set(df["filename"].tolist())
    batched_list_cifs_filter_sorted = [row for row in batched_list_cifs_sorted if int(row.structure_id[0][0]) in fnames]
    batched_list_cifs_fnames = [ int(r.structure_id[0][0]) for r in batched_list_cifs_filter_sorted ]
    df_sorted_filter = df_sorted[df_sorted['filename'].isin(batched_list_cifs_fnames)]
    text = df_sorted_filter["text"].tolist()
    labels = df_sorted_filter["label"].tolist()

    # # Debug: Print lengths of the lists
    print(f"Length of texts: {len(text)}")
    print(f"Length of supervised_inputs: {len(batched_list_cifs_filter_sorted)}")
    print(f"Length of labels: {len(labels)}")
    return text, batched_list_cifs_filter_sorted, labels, dataset_cif, supervised_model


def predict_combined(model_path):
    print("Predicting")
    with open( "./config.yml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    text, batched_list_cifs_filter_sorted, labels, dataset_cif, supervised_model = get_data(config)
    dataset_combined = PropertyDataset(text, batched_list_cifs_filter_sorted, labels)
    train_dataset_combined, test_dataset_combined = split_data(dataset_combined, train_ratio=0.01, random_seed=42)
    train_dataloader_combined = DataLoader(train_dataset_combined, batch_size=8, shuffle=True)
    test_dataloader_combined = DataLoader(test_dataset_combined, batch_size=8, shuffle=False)
    # checkpoint_path = 'Results/Checkpoint/checkpoint_matbert_0.5data_fe_10_2025-09-04-2233.pth'
    checkpoint_path = None
    if model_path is not None:
        checkpoint_path = model_path
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    transformer_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(transformer_name)
    supervised_model  = supervised_model.load_model(dataset_cif)  # Example dimensions

    # Combined model
    model = CombinedEmbeddingModel(transformer_name, supervised_model, supervised_dim=150)
    # Load the model state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_criterion = nn.SmoothL1Loss()
    total_test_loss = 0
  

    for test_batch in test_dataloader_combined:
        texts, supervised_inputs, labels = test_batch
        predictions = model(texts, supervised_inputs, tokenizer)
        test_loss = test_criterion(predictions.squeeze(), labels.cuda())
        total_test_loss += test_loss.item()

    print(f"Test Loss: {total_test_loss / len(test_dataloader_combined)}")
        # lbls.append(labels.detach().numpy().tolist())
    

   

def train_combined():
    print("Training Code Starting")
    with open( "./config.yml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    text, batched_list_cifs_filter_sorted, labels, dataset_cif, supervised_model = get_data(config)
    dataset_combined = PropertyDataset(text, batched_list_cifs_filter_sorted, labels)
    train_dataset_combined, test_dataset_combined = split_data(dataset_combined, train_ratio=0.2, random_seed=42)
    train_dataloader_combined = DataLoader(train_dataset_combined, batch_size=8, shuffle=True)
    test_dataloader_combined = DataLoader(test_dataset_combined, batch_size=8, shuffle=False)
  
    transformer_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(transformer_name)
    supervised_model  = supervised_model.load_model(dataset_cif)  # Example dimensions

    # Combined model
    model = CombinedEmbeddingModel(transformer_name, supervised_model, supervised_dim=150)
      
    num_epochs = 10
    # Optimizer and loss function
    # optimizer = optim.Adam(model.parameters(), lr=5e-5)
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    criterion = nn.MSELoss()
    total_steps = num_epochs * len(train_dataloader_combined)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    # # Training loop
    # # Check which parameters are trainable
    # print("\n===== Trainable Parameters =====")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"TRAINABLE: {name} | Shape: {param.shape}")
    #     else:
    #         print(f"FROZEN:    {name}")
    # print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # print("================================\n")

    model.train
    model_file = None
    for epoch in range(num_epochs+1):  # Number of epochs
        total_loss = 0
        for train_batch in train_dataloader_combined:

            texts, supervised_inputs, labels =train_batch

            # Forward pass
            predictions = model(texts, supervised_inputs, tokenizer)

            loss = criterion(predictions.squeeze(), labels.cuda())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader_combined)}")
        if (epoch % 5 == 0 and epoch>0):
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss.item(),
            }
            fname =datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
            model_file = f"./Results/Checkpoint/checkpoint_matbert_0.5data_fe_{epoch}_{fname}.pth"
            torch.save(checkpoint, model_file)
            print(f"Checkpoint saved for epoch {epoch}")
    

    return model_file
    

