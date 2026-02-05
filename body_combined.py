import datetime
import itertools
import torch.nn as nn
from transformers import AdamW, get_cosine_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer
from utils import summarize_dataset
import process as process
from training import model_setup, loader_setup
from torch.utils.data import Dataset, random_split, Subset
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
from sklearn.metrics import mean_absolute_error


def get_data(config, prop = "Tc_supercon", subset="train", logger=None):
    processing_parameters = config["Processing"]
    if subset == "train":
        file_path = processing_parameters["train_folder"]
    elif subset == "test":
        file_path = processing_parameters["test_folder"]
    else:
        raise ValueError("subset must be 'train' or 'test'")
    
    text_folder = os.path.join(file_path, "text_data")
    csv_file = os.path.join(file_path,f"targets_{prop}.csv")
    cif_data_path_new = os.path.join(file_path, "bulk_data")

    job_parameters, training_parameters, model_parameters = (
        config["Job"]['Training'], 
        config["Training"],
        config["Models"]["CGCNN_demo"]
    )
    
    supervised_model = SupervisedModel(
        training_parameters, 
        model_parameters,
        job_parameters,
        processing_parameters
    )
    
    if logger:
        logger.info("Loading Model")
    dataset_cif = supervised_model.get_data(cif_data_path_new)
    loader_cif = supervised_model.load_data(dataset_cif)
    temp_list_cifs = []
    
    for batch_cif in loader_cif:
        temp_list_cifs.append(Batch.to_data_list(batch_cif))
    
    batched_list_cifs = list(itertools.chain.from_iterable(temp_list_cifs))
    batched_list_cifs_sorted = sorted(batched_list_cifs, key=lambda x: x.structure_id[0][0])
    df = load_data(text_folder, csv_file)
    df_sorted = df.sort_values(by=['filename'], ascending=True)
    fnames = set(df["filename"].tolist())
    batched_list_cifs_filter_sorted = [
        row for row in batched_list_cifs_sorted 
        if row.structure_id[0][0] in fnames
    ]
    batched_list_cifs_fnames = [r.structure_id[0][0] for r in batched_list_cifs_filter_sorted]
    df_sorted_filter = df_sorted[df_sorted['filename'].isin(batched_list_cifs_fnames)]
    text = df_sorted_filter["text"].tolist()
    labels = df_sorted_filter["label"].tolist()

    if logger:
        logger.info(f"Length of texts: {len(text)}")
        logger.info(f"Length of supervised_inputs: {len(batched_list_cifs_filter_sorted)}")
        logger.info(f"Length of labels: {len(labels)}")
    
    return text, batched_list_cifs_filter_sorted, labels, dataset_cif, supervised_model


def predict_combined(model_path, prop = "Tc_supercon", logger=None):
    if logger:
        logger.info("Predicting")
    
    with open(f"./config_{prop}.yml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
    batch_size = config["Training"]["batch_size"]
    text, batched_list_cifs_filter_sorted, labels, dataset_cif, supervised_model = get_data(
        config, prop = prop, subset="test", logger=logger
    )
    test_dataset_combined = PropertyDataset(text, batched_list_cifs_filter_sorted, labels)
    test_dataloader_combined = DataLoader(test_dataset_combined, batch_size=32, shuffle=False)
    
    checkpoint_path = None
    if model_path is not None:
        checkpoint_path = model_path
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    transformer_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(transformer_name)
    supervised_model = supervised_model.load_model(dataset_cif)

    # Combined model
    model = CombinedEmbeddingModel(transformer_name, supervised_model, supervised_dim=150)
    # Load the model state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for texts, supervised_inputs, labels in test_dataloader_combined:
            preds = model(texts, supervised_inputs, tokenizer)
            all_preds.append(preds.squeeze().cpu())
            all_labels.append(labels.cpu())

    # concatenate everything into flat tensors
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # save all_preds and all_labels to a csv file
    df = pd.DataFrame({'preds': all_preds, 'labels': all_labels})
    os.makedirs(f"./Results/Output", exist_ok=True)
    df.to_csv(f"./Results/Output/preds_{prop}.csv", index=False)

    mae = mean_absolute_error(all_labels, all_preds)
    if logger:
        logger.info(f"Test MAE (sklearn): {mae:.12f}")


def train_combined(prop="Tc_supercon", logger=None, timestamp=None):
    if logger:
        logger.info("Training Code Starting")
    
    with open(f"./config_{prop}.yml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
    text, batched_list_cifs_filter_sorted, labels, dataset_cif, supervised_model = get_data(
        config, prop=prop, logger=logger
    )
    batch_size = config["Training"]["batch_size"]
    train_dataset_combined = PropertyDataset(text, batched_list_cifs_filter_sorted, labels)
    train_dataloader_combined = DataLoader(train_dataset_combined, batch_size=batch_size, shuffle=True)
  
    transformer_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(transformer_name)
    supervised_model = supervised_model.load_model(dataset_cif)

    # Combined model
    model = CombinedEmbeddingModel(transformer_name, supervised_model, supervised_dim=150)
      
    num_epochs = 100
    # Optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    criterion = nn.MSELoss()
    total_steps = num_epochs * len(train_dataloader_combined)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    model.train()
    model_file = None
    
    for epoch in range(num_epochs + 1):
        total_loss = 0
        for train_batch in train_dataloader_combined:
            texts, supervised_inputs, labels = train_batch

            # Forward pass
            predictions = model(texts, supervised_inputs, tokenizer)
            loss = criterion(predictions.squeeze(), labels.cuda())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        if logger:
            logger.info(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader_combined)}")
        
        if (epoch % 50 == 0 and epoch > 0):
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss.item(),
            }
            # fname = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
            fname = timestamp if timestamp else datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
            model_file = f"./model/checkpoint_scibert_cgcnn_mbj_bandgap_{epoch}_{fname}.pth"
            torch.save(checkpoint, model_file)
            if logger:
                logger.info(f"Checkpoint saved for epoch {epoch}")

    # Save final model after training completes
    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": num_epochs,
        "loss": loss.item(),
    }
    fname = timestamp if timestamp else datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
    model_file = f"./model/checkpoint_scibert_cgcnn_mbj_bandgap_final_{num_epochs}_{fname}.pth"
    torch.save(final_checkpoint, model_file)
    if logger:
        logger.info(f"Final model saved after {num_epochs} epochs")

    return model_file