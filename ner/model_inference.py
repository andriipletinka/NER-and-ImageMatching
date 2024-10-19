import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer

model_path = "distilbert-ner-tuned"
tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
model = AutoModelForTokenClassification.from_pretrained(model_path)
trainer = Trainer(model=model)

label_list = ["O", "B-MOUNT", "I-MOUNT"]
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for i, label in enumerate(label_list)}

def decode_predictions(predictions, id_to_label):
    predicted_labels = []
    for pred in predictions:
        label_ids = [id_to_label[label_id] for label_id in pred]
        predicted_labels.append(label_ids)
    return predicted_labels

def run_inference(sample):
    tokenized_sample = tokenizer.tokenize(sample)
    ids = tokenizer.convert_tokens_to_ids(tokenized_sample)
    model_input = Dataset.from_list([{'input_ids': ids}])
    predictions = trainer.predict(model_input)
    preds = predictions.predictions.argmax(-1)
    decoded_predictions = decode_predictions(preds, id_to_label)
    return tokenized_sample, decoded_predictions
    
def extract_entities(tokens, labels):
    entities = []
    current_entity = []
    
    for token, label in zip(tokens, labels):
        if label == "B-MOUNT":  # Beginning of a new entity
            if current_entity:  # Add the previous entity to the list if exists
                entities.append(" ".join(current_entity))
            current_entity = [token]  # Start a new entity
        elif label == "I-MOUNT":  # Continuation of the current entity
            current_entity.append(token)
        else:
            if current_entity:  # Add the entity to the list if exists
                entities.append(" ".join(current_entity))
                current_entity = []  # Reset the current entity
    
    # Catch any remaining entity at the end
    if current_entity:
        entities.append(" ".join(current_entity))
    
    return entities

def pretty_print(tokens, labels, entities):
    # Print tokens with corresponding labels
    print("Tokens and Labels:")
    for token, label in zip(tokens, labels):
        if label == "B-MOUNT":
            print()
            print(f"{token:12} -> {label}")
        if label == "I-MOUNT":
            print(f"{token:12} -> {label}")
    
    # Print the extracted entities
    print("\nDetected Entities:\n")
    for entity in entities:
        string = tokenizer.convert_tokens_to_string(entity.split(" "))
        print(f"Entity: {string}")

def predict_and_print(sample):
    """
    Function to run model inference using a single sample
    """
    tokenized_sentence, decoded_predictions = run_inference(sample)
    tokenized_sentence, decoded_predictions = tokenized_sentence, decoded_predictions[0]
    print(f"Tokens: {tokenized_sentence}")
    print(f"Predicted labels: {decoded_predictions}")
    print()
    entities = extract_entities(tokenized_sentence, decoded_predictions)
    pretty_print(tokenized_sentence, decoded_predictions, entities)

if __name__ == "__main__":
    test_sentence = """The highest mountain on Earth is Mount Everest in the Himalayas of Asia, whose summit is 8,850 m (29,035 ft) above mean sea level.
                    The highest known mountain on any planet in the Solar System is Olympus Mons on Mars at 21,171 m (69,459 ft).
                    The tallest mountain including submarine terrain is Mauna Kea in Hawaii from its underwater base at 9,330 m (30,610 ft)
                    and some scientists consider it to be the tallest on earth."""
    
    predict_and_print(test_sentence)
