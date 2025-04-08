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
