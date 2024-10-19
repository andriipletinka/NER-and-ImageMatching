import torch
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")

label_list = ["O", "B-MOUNT", "I-MOUNT"]
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for i, label in enumerate(label_list)}

# Convert labels to IDs
def encode_labels(labels):
    ids = [label_to_id[label] for label in labels]
    padding_length = tokenizer.model_max_length - len(ids)
    ids += [label_list.index('O')] * padding_length
    return ids

# Convert tokens to IDs
def encode_tokens(tokens):
    ids = tokenizer.convert_tokens_to_ids(tokens)
    padding_length = tokenizer.model_max_length - len(ids)
    ids += [tokenizer.pad_token_id] * padding_length
    return ids

def prepare_dataset(file_path):
    """
    Function to read the json file with data and convert
    it to a Dataset object with encoded labels and tokens
    """
    with open(file_path, "r") as f:
        dataset = json.load(f)
    dataset = Dataset.from_list(dataset)
    dataset = dataset.map(lambda x: {'labels': encode_labels(x['labels']),
                                     'input_ids': encode_tokens(x['tokens'])})
    return dataset

if __name__ == "__main__":
    train_dataset = prepare_dataset("train_dataset.json")
    val_dataset = prepare_dataset("val_dataset.json")

    # Define the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER", num_labels=len(label_list), ignore_mismatched_sizes=True).to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()
    # Save the fine-tuned model
    trainer.save_model("./distilbert-ner-tuned")
