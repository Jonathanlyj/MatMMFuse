import datetime

# !pip install pymatgen robocrys
# !pip install torch-geometric
# !pip install torch-scatter
# !pip install ase
# !pip install torch-sparse
# !pip install ogb


def get_desc(f):
  from pymatgen.core import Structure
  from robocrys import StructureCondenser, StructureDescriber
  structure = Structure.from_file(f) # other file formats also supported

  # alternatively, uncomment the lines below to use the MPRester object
  # to fetch structures from the Materials Project database
  # from mp_api.client import MPRester
  # structure = MPRester(api_key=None).get_structure_by_material_id("mp-856")

  condenser = StructureCondenser()
  describer = StructureDescriber()

  condensed_structure = condenser.condense_structure(structure)
  # description = describer.describe(condensed_structure)
  # return description
  try:
      description = describer.describe(condensed_structure)
      return description
  except TypeError as e:
      print(f"Warning: Description generation failed for structure {f}. Details: {e}")
      return {"error": "description_failed", "file": f}



def get_data():

    import os
    # for f in os.listdir("./Data/Mos2_cif"):
    #   if f.endswith(".cif"):
    #     try:
    #         d = get_desc("./Data/Mos2_cif/"+f)
    #         f_name = f.split(".")[0]
    #         with open(f"./Data/Mos2_txt/{f_name}.txt", "w") as f:
    #             f.write(str(d))
    #         if isinstance(d, dict) and "error" in d:
    #             continue
    #     except Exception as e:
    #         print(f"Error processing {f}: {str(e)}")
    #         continue

    # for f in os.listdir("./Data/WSe2_cif"):
    #   if f.endswith(".cif"):
    #     try:
    #         d = get_desc("./Data/WSe2_cif/"+f)
    #         f_name = f.split(".")[0]
    #         with open(f"./Data/WSe2_txt/{f_name}.txt", "w") as f:
    #           f.write(str(d))
    #         if isinstance(d, dict) and "error" in d:
    #                 continue
    #     except Exception as e:
    #         print(f"Error processing {f}: {str(e)}")
    #         continue

    for f in os.listdir("./Data/bulk_data"):
        if f.endswith(".cif"):
            try:
                d = get_desc("./Data/bulk_data/" + f)
                f_name = f.split(".")[0]
                with open(f"./Data/bulk_data_txt/{f_name}.txt", "w") as f:
                    f.write(str(d))
                if isinstance(d, dict) and "error" in d:
                    continue
            except Exception as e:
                print(f"Error processing {f}: {str(e)}")
                continue
    # get_data()

def run_llm(device):
    import os
    import pandas as pd
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from sklearn.model_selection import train_test_split
    from datasets import Dataset
    from evaluate import load

    # Directories
    text_folder_mos2 = "./data/text_data/MP"  # Path to text files
    csv_file_mos2 = "./data/text_data/MP/targets_te.csv"  # Path to the CSV file
    prop = "te"
    # os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

    # Load CSV data
    property_data = pd.read_csv(csv_file_mos2)

    property_data.columns = ["Filename", "Property"]
    property_data['Filename']= property_data['Filename'].apply(str)

    def load_data(text_folder, property_data):
        data = []
        for filename in os.listdir(text_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(text_folder, filename)
                with open(file_path, "r") as file:
                    text = file.read()
                file_name = filename.split(".")[0]
                property_row = property_data[property_data["Filename"] == file_name]
                if not property_row.empty:
                    data.append({"text": text, "label": property_row.iloc[0]["Property"]})  # Replace "Property" with your property column name
        return pd.DataFrame(data)

    data_df = load_data(text_folder_mos2, property_data)

    # Train-test split
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)

    # Convert to Hugging Face dataset format
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)


    # Tokenizer and model
    # model_name = "distilbert-base-uncased"
    model_name = "allenai/scibert_scivocab_uncased"
    name_model = "scibert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Tokenize datasets
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    format = {'type': 'torch', 'format_kwargs' :{'dtype': torch.bfloat16}}
    train_dataset.set_format(**format)
    test_dataset.set_format(**format)

    # Set format for PyTorch
    train_dataset = train_dataset.with_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset = test_dataset.with_format("torch", columns=["input_ids", "attention_mask", "label"])

    num_labels = 1  # Regression (use >1 if classification with multiple classes)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    print("shape of train_dataset")
    print(train_dataset.shape)
    print(test_dataset.shape)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.001,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10
    )

    # Metric for regression
    metric = load("mae")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.squeeze()
        mae = (abs(predictions - labels)).mean().item()
        return {"mae": mae}

    model =model.to(device)
    # train_dataset = train_dataset.to(device)
    # test_dataset = test_dataset.to(device)
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    from datetime import datetime
    fname= datetime.now()
    fname.strftime("%Y_%m_%d_%H_%M_%S")

    # Save the model
    model.save_pretrained(f"./llm_trained_models/trained_model_{prop}_{name_model}_{fname}")
    tokenizer.save_pretrained(f"./llm_trained_models/trained_model_{prop}_{name_model}_{fname}")

    # Evaluate the model on the test dataset
    results = trainer.evaluate()

    # Print the results
    print("Test Set Evaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

def run_supervised():

    import yaml

    from matdeeplearn import training

    with open( "./config.yml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # config["Job"]['Training']['load_model'] = False

    training.train_regular(
                        "cuda",
                        0,
                        "./data/bulk_data/MP_robofail",
                        config["Job"]['Training'],
                        config["Training"],
                        config["Models"]["CGCNN_demo"],
                        )


    # training.predict(data_path= "./data/bulk_data/perovskite/abs3" ,
    #                  training_parameters = config['Training'] ,
    #                  model_parameters=config["Models"]["CGCNN_demo"], job_parameters= config["Job"]['Predict'])

def robo_fail(provided_numbers, start_range=0, end_range=5001):
    # Define the range of numbers (e.g., 0 to 10)
    start_range = start_range
    end_range = end_range
    # Find missing numbers
    all_numbers = set(range(start_range, end_range + 1))
    missing_numbers = sorted(all_numbers - set(provided_numbers))
    return missing_numbers

def robo_fail_folder(old_text_folder, new_text_folder, missing_numbers):
    import os
    import shutil
    import pandas as pd
    for filename in os.listdir(old_text_folder):
    #     if filename.endswith(".cif"):
    #         file_path = os.path.join(old_text_folder, filename)
    #         file_name = filename.split(".")[0]
    #         if file_int not in missing_numbers:
    #             shutil.copy2(file_path,new_text_folder)
        if filename.endswith(".csv"):
            file_path_old = os.path.join(old_text_folder, filename)
            temp_df = pd.read_csv(file_path_old)
            temp_df.columns = ['filename', 'property']
            temp_df['filename'] = temp_df['filename'].astype(str)
            temp_df = temp_df[~temp_df['filename'].str.extract(r'(\d+)', expand=False).astype(int).isin(missing_numbers)]
            file_path_new = os.path.join(new_text_folder, filename)
            temp_df.to_csv(file_path_new, index=False)
    print("finished")


def train_combined():
  
    import itertools
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    import yaml
    from matdeeplearn import models
    import matdeeplearn.process as process
    from matdeeplearn.training.training import model_setup, loader_setup
    from torch.utils.data import  Dataset, random_split,Subset
    from torch_geometric.loader import DataLoader
    from transformers import AdamW, get_cosine_schedule_with_warmup
    from sklearn.metrics import mean_absolute_error
    from torch_geometric.data import Batch
    import torch.optim as optim
    import pandas as pd
    import os
    import numpy as np
    import random
    from utils import summarize_dataset

    def delete_random_chars(text, prob=0.1):
        return ''.join([char for char in text if random.random() > prob])
    def replace_random_words(text, prob=0.1, replacement_words=["foo", "bar", "baz"]):
        words = text.split()
        return ' '.join([word if random.random() > prob else random.choice(replacement_words) for word in words])
    def add_random_punctuation(text, prob=0.1):
        punctuation = ['.', ',', '!', '?', ';', ':']
        return ''.join([char if random.random() > prob else char + random.choice(punctuation) for char in text])

    def combined_text(text):
        text = delete_random_chars(text, prob=0.6)
        text = replace_random_words(text, prob=0.6)
        text = add_random_punctuation(text, prob=0.6)
        return text
    
    def plot_tsne(encoded_features, labels, prop, ind):

        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt


        encoded_features = np.array(encoded_features.detach().cpu().numpy())
        # labels =np.array(labels)

        # if prop=='formation energy':
        #     labels=df['formation_energy']
        # elif prop=='fermi energy':
        #     labels=df['fermi_energy']
        # elif prop =='band gap':
        #     labels=df['band_gap']
        # elif prop =='energy_above_hull':
        #     labels=df['energy_above_hull']

        tsne = TSNE(n_components=2, random_state=42, perplexity = 5)
        tsne_features = tsne.fit_transform(encoded_features)

        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c= labels, cmap='viridis')
        plt.colorbar(label= 'Formation Energy')
        plt.title(f't-SNE Visualization of {prop} Encoded Features')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig(f'./Results/Output/tsne_{prop}_{ind}.png', dpi=300)


    def plot_tsne_all(supervised_emb, bert_emb, combined_emb, labels, prop):
        plot_tsne(encoded_features=supervised_emb, labels= labels, prop=prop, ind="Supervised" )
        plot_tsne(encoded_features=bert_emb,labels=labels, prop=prop, ind="Bert")
        plot_tsne(encoded_features=combined_emb, labels=labels, prop=prop, ind= "Combined")
        

    # torch.cuda.empty_cache()
    with open( "./config.yml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    class AttentionCombiner(nn.Module):
        def __init__(self, dim):
            super(AttentionCombiner, self).__init__()
            self.query = nn.Linear(dim, dim)  # Query for attention
            self.key = nn.Linear(dim, dim)  # Key for attention
            self.value = nn.Linear(dim, dim)  # Value for attention
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, supervised_embedding, transformer_embedding):
            query = self.query(transformer_embedding)
            key = self.key(supervised_embedding)
            value = self.value(supervised_embedding)
            # Compute attention scores
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (key.size(-1) ** 0.5)
            attention_weights = self.softmax(attention_scores)
            # Weighted sum of values
            combined = torch.matmul(attention_weights, value)
            return combined.squeeze(1)

    class MultiHeadAttentionCombiner(nn.Module):
        def __init__(self, dim, num_heads=8):
            super(MultiHeadAttentionCombiner, self).__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // num_heads

            assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.fc_out = nn.Linear(dim, dim)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, supervised_embedding, transformer_embedding):
            batch_size = supervised_embedding.size(0)

            # Linear transformations
            query = self.query(transformer_embedding).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                                                                    2)
            key = self.key(supervised_embedding).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            value = self.value(supervised_embedding).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            # Scaled dot-product attention
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attention_weights = self.softmax(attention_scores)
            combined = torch.matmul(attention_weights, value)

            # Concatenate heads and apply final linear layer
            combined = combined.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
            combined = self.fc_out(combined)

            return combined.squeeze(1)

    class AttentionCombinerWithNorm(nn.Module):
        def __init__(self, dim):
            super(AttentionCombinerWithNorm, self).__init__()
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.softmax = nn.Softmax(dim=-1)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        def forward(self, supervised_embedding, transformer_embedding):
            # Layer normalization
            supervised_embedding = self.norm1(supervised_embedding)
            transformer_embedding = self.norm1(transformer_embedding)

            # Linear transformations
            query = self.query(transformer_embedding)
            key = self.key(supervised_embedding)
            value = self.value(supervised_embedding)

            # Scaled dot-product attention
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (key.size(-1) ** 0.5)
            attention_weights = self.softmax(attention_scores)
            combined = torch.matmul(attention_weights, value)

            # Layer normalization
            combined = self.norm2(combined)

            return combined.squeeze(1)

    class AttentionCombinerWithResidual(nn.Module):
        def __init__(self, dim):
            super(AttentionCombinerWithResidual, self).__init__()
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.softmax = nn.Softmax(dim=-1)
            self.norm = nn.LayerNorm(dim)

        def forward(self, supervised_embedding, transformer_embedding):
            # Save the input for the residual connection
            residual = transformer_embedding

            # Linear transformations
            query = self.query(transformer_embedding)
            key = self.key(supervised_embedding)
            value = self.value(supervised_embedding)

            # Scaled dot-product attention
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (key.size(-1) ** 0.5)
            attention_weights = self.softmax(attention_scores)
            combined = torch.matmul(attention_weights, value)

            # Add residual connection and apply layer normalization
            combined = self.norm(combined + residual)

            return combined.squeeze(1)

    class AttentionCombinerWithDropout(nn.Module):
        def __init__(self, dim, dropout=0.2):
            super(AttentionCombinerWithDropout, self).__init__()
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.softmax = nn.Softmax(dim=-1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, supervised_embedding, transformer_embedding):
            # Linear transformations
            query = self.query(transformer_embedding)
            key = self.key(supervised_embedding)
            value = self.value(supervised_embedding)

            # Scaled dot-product attention
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (key.size(-1) ** 0.5)
            attention_weights = self.softmax(attention_scores)
            attention_weights = self.dropout(attention_weights)  # Apply dropout
            combined = torch.matmul(attention_weights, value)

            return combined.squeeze(1)

    class ImprovedAttentionCombiner(nn.Module):
        def __init__(self, dim, num_heads=8, dropout=0.2):
            super(ImprovedAttentionCombiner, self).__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // num_heads

            assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.fc_out = nn.Linear(dim, dim)
            self.softmax = nn.Softmax(dim=-1)
            self.dropout = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        def forward(self, supervised_embedding, transformer_embedding):
            batch_size = supervised_embedding.size(0)

            # Layer normalization
            supervised_embedding = self.norm1(supervised_embedding)
            transformer_embedding = self.norm1(transformer_embedding)

            # Linear transformations
            query = self.query(transformer_embedding).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                                                                    2)
            key = self.key(supervised_embedding).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            value = self.value(supervised_embedding).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            # Scaled dot-product attention
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
                raise ValueError("NaN or Inf detected in attention_scores")
            attention_weights = self.softmax(attention_scores)
            attention_weights = self.dropout(attention_weights)  # Apply dropout
            combined = torch.matmul(attention_weights, value)

            # Concatenate heads and apply final linear layer
            combined = combined.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
            combined = self.fc_out(combined)

            # Add residual connection and apply layer normalization
            combined = self.norm2(combined + transformer_embedding)

            return combined.squeeze(1)

    class CombinedEmbeddingModel(nn.Module):
        def __init__(self, transformer_name, supervised_model, supervised_dim, combined_dim=512):
            super(CombinedEmbeddingModel, self).__init__()
            self.device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Loading transformer.. {transformer_name}")
            self.transformer = AutoModel.from_pretrained(transformer_name)
            self.transformer = self.transformer.to(self.device)
            self.supervised_model = supervised_model.to(self.device)  # Pretrained supervised model
            self.supervised_proj = nn.Linear(supervised_dim, combined_dim).to(self.device)
            self.transformer_proj = nn.Linear(self.transformer.config.hidden_size, combined_dim).to(self.device)
            self.attention_combiner = ImprovedAttentionCombiner(combined_dim).to(self.device)
            self.fc = nn.Linear(combined_dim, 1).to(self.device)  # Output layer for regression

        def forward(self, text, input_for_supervised, tokenizer):
            # Get transformer embeddings
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs=inputs.to(self.device)
            transformer_output = self.transformer(**inputs).last_hidden_state.mean(dim=1)
            # outputs = self.transformer(**inputs)
            # transformer_output = outputs.pooler_output 
            transformer_proj = self.transformer_proj(transformer_output)


            # Get supervised model embeddings
            input_for_supervised = input_for_supervised.to(self.device)
            _, supervised_embedding = self.supervised_model(input_for_supervised)
            supervised_proj = self.supervised_proj(supervised_embedding)

            # Combine embeddings with attention
            combined_embedding = self.attention_combiner(supervised_proj.unsqueeze(1), transformer_proj.unsqueeze(1))

            # Predict property
            output = self.fc(combined_embedding)
            return output

    class SupervisedModel():
        def __init__(self, training_parameters, model_parameters, job_parameters, processing_parameters):
            self.training_parameters = training_parameters
            self.model_parameters = model_parameters
            self.job_parameters = job_parameters
            self.processing_parameters =processing_parameters

        def get_data(self, data_path):
            dataset = process.get_dataset(data_path=data_path, target_index=self.training_parameters["target_index"],
                                          reprocess=True, model_name=self.model_parameters["model"], processing_args= self.processing_parameters)
            return dataset

        def load_data(self, dataset):
            train_sampler = None
            loader = DataLoader(
                dataset,
                batch_size=self.model_parameters["batch_size"],
                shuffle=(train_sampler is None),
                num_workers=0,
                pin_memory=False,
                sampler=train_sampler,
            )
            return loader

        def load_model(self, dataset, rank='cuda'):
            print(f"Loading model.. {self.model_parameters['model']}")
            ##Set up model
            model = model_setup(
                rank,
                self.model_parameters["model"],
                self.model_parameters,
                dataset,
                False,
                # self.job_parameters["load_model"],
                # self.job_parameters["model_path"],
                "./Results/Models/cgcnn.pth",
                self.model_parameters.get("print_model", False),
            )
            #
            # # ##Set-up optimizer & scheduler
            # optimizer = getattr(torch.optim, self.model_parameters["optimizer"])(
            #     model.parameters(),
            #     lr=self.model_parameters["lr"],
            #     **self.model_parameters["optimizer_args"]
            # )
            # scheduler = getattr(torch.optim.lr_scheduler, self.model_parameters["scheduler"])(
            #     optimizer, **self.model_parameters["scheduler_args"]
            # )
            return model

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

    # Sample Data
    load_model = True
    print("Code Starting")
    text_folder = "/home/abhibhatt/Pycharm_Projects/Data_MatMMFuse/abo3_txt"  # Path to text files
    csv_file = "/home/abhibhatt/Pycharm_Projects/Data_MatMMFuse/targets.csv"  # Path to the CSV file
    # cif_data_path_old = "./data/bulk_data/MP_robofail"
    cif_data_path_new = "/home/abhibhatt/Pycharm_Projects/Data_MatMMFuse/abo3"
    # text_folder = "./data/text_data/Jarvis_txt"  # Path to text files
    # csv_file = "./data/bulk_data/Jarvis/targets.csv"  # Path to the CSV file
    # # # # cif_data_path_old = "./data/bulk_data/MP_robofail"
    # cif_data_path_new = "./data/bulk_data/Jarvis"
    # provided_list = df["filename"].tolist()
    # missing_list = robo_fail(provided_list)
    # robo_fail_folder(cif_data_path_old,cif_data_path_new, missing_list)
    # exit(0)
    # Load CSV data
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

    # Create dataset and dataloader
    if not load_model:
        dataset_combined = PropertyDataset(text, batched_list_cifs_filter_sorted, labels)
        train_dataset_combined, test_dataset_combined = split_data(dataset_combined, train_ratio=0.2, random_seed=42)
        train_dataloader_combined = DataLoader(train_dataset_combined, batch_size=8, shuffle=True)
        test_dataloader_combined = DataLoader(test_dataset_combined, batch_size=8, shuffle=False)
    else:
        print("Predicting")
        dataset_combined = PropertyDataset(text, batched_list_cifs_filter_sorted, labels)
        train_dataset_combined, test_dataset_combined = split_data(dataset_combined, train_ratio=0.01, random_seed=42)
        train_dataloader_combined = DataLoader(train_dataset_combined, batch_size=8, shuffle=True)
        test_dataloader_combined = DataLoader(test_dataset_combined, batch_size=8, shuffle=False)


    # Initialize models
    # transformer_name =  "distilbert-base-uncased"
    # transformer_name ="albert-base-v2"
    # transformer_name = "roberta-base"



    # transformer_name ="microsoft/deberta-v3-base"
    transformer_name = "allenai/scibert_scivocab_uncased"
    # transformer_name = "m3rg-iitd/matscibert" 
    tokenizer = AutoTokenizer.from_pretrained(transformer_name)
    supervised_model  = supervised_model.load_model(dataset_cif)  # Example dimensions

    # Combined model
    model = CombinedEmbeddingModel(transformer_name, supervised_model, supervised_dim=150)
 
    if load_model: # Change for finetuning
        checkpoint_path = '/data/yll6162/matmmfuse/modelcheckpoint_matbert_fe_5_2025-02-07-0216.pth'
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)
        # Load the model state_dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
    num_epochs =10
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
    if not load_model:
        model.train
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
            # if (epoch % 5 == 0 and epoch>0):
            #     checkpoint = {
            #         "model_state_dict": model.state_dict(),
            #         "optimizer_state_dict": optimizer.state_dict(),
            #         "epoch": epoch,
            #         "loss": loss.item(),
            #     }
                # fname =datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
                # torch.save(checkpoint, f"/data/yll6162/matmmfuse/modelcheckpoint_matbert_0.5data_fe_{epoch}_{fname}.pth")
                # print(f"Checkpoint saved for epoch {epoch}")

 
## Testing loop
    model.eval()
    test_criterion = nn.SmoothL1Loss()
    total_test_loss = 0
    # lbls =[]
    # activations = {}

# # Hook function to capture outputs
#     def get_hook(layer_name):
#         activations[layer_name] =[]
#         def hook_fn(module, input, output):
#             activations[layer_name].append(output.detach())  # Store the output
#         return hook_fn

    # Register the hook on the embedding layer
    # hook_supervised = model.supervised_proj.register_forward_hook(get_hook('supervised_proj'))
    # hook_bert = model.transformer_proj.register_forward_hook(get_hook('transformer_proj'))
    # hook_combined = model.attention_combiner.fc_out.register_forward_hook(get_hook('comb_proj'))

    for test_batch in test_dataloader_combined:
        texts, supervised_inputs, labels = test_batch
        predictions = model(texts, supervised_inputs, tokenizer)
        test_loss = test_criterion(predictions.squeeze(), labels.cuda())
        total_test_loss += test_loss.item()

    print(f"Test Loss: {total_test_loss / len(test_dataloader_combined)}")
        # lbls.append(labels.detach().numpy().tolist())
    

    # lbls= list(itertools.chain.from_iterable(lbls))

    # for name in activations:
    #     activations[name] = torch.cat(activations[name], dim=0).squeeze()
        
    # plot_tsne_all(supervised_emb=activations['supervised_proj'], 
    #               bert_emb= activations['transformer_proj'], 
    #               combined_emb= activations['comb_proj'], 
    #               labels=lbls,
    #               prop='formation_energy')
    
    # np.savetxt('./Results/Output/Jarvis_fe_combined.txt',outputs)
    
    # hook_supervised.remove()
    # hook_bert.remove()
    # hook_combined.remove()

def predict_material_properties(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions
    # Process the model outputs as needed
    prediction = outputs.logits.squeeze().item()
    return prediction,attentions

def vis_attention(attentions, tokenizer):
    # Visualize attention for a specific layer
    import matplotlib.pyplot as plt
    import seaborn as sns

    layer_idx = -1  # Last layer
    attention_matrix = attentions[layer_idx][0][0].detach().numpy()  # Shape: [seq_len, seq_len]
    test_text="ad"
    inputs = tokenizer(test_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_matrix, xticklabels=inputs.tokens(), yticklabels=inputs.tokens(), cmap="viridis")
    plt.title("Attention Weights (Last Layer)")

    import sys
    import os
    import argparse
    import time
    import csv
    import sys
    import json
    import random
    import numpy as np
    import pprint
    import yaml
    import os

    import torch
    import torch.multiprocessing as mp
    # import gc
    # import ray
    # from ray import tune

    # print(sys.path)

    import requests
    import tarfile

    # # Download and extract the file
    # url = "http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz"
    # response = requests.get(url, stream=True)
    # file = tarfile.open(fileobj=response.raw, mode="r|gz")
    # file.extractall(path=".")

    # pip install metis
    #
    # !git clone https://github.com/KarypisLab/METIS.git
    #
    # !make config shared=1 cc=gcc prefix=~/local
    # !make install
    #
    from matdeeplearn import models, process, training

    with open( "", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    config["Job"]['Training']['load_model'] = False

    # config["Models"]["GraphJepa_demo"]["metis_enable"]= False
    # config["Models"]["GraphJepa_demo"]["metis_online"]= False

    training.train_regular(
                        "cuda",
                        0,
                        "./Data/Mos2_cif",
                        config["Job"]['Training'],
                        config["Training"],
                        config["Models"]["CGCNN_demo"],
                        )





if __name__ == "__main__":
    import os
    os.chdir("./MatDeepLearn-main")
    #Converting cif to txt
    # get_data()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is: ", device)
    print("Training started")
    # run_llm(device)
    # run_supervised()
    train_combined()
    print("Training completed successfully")

