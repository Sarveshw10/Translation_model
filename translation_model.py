# Import required libraries
import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load the English-Marathi dataset from Hugging Face
def load_dataset_from_hf():
    dataset = load_dataset("anujsahani01/English-Marathi")
    return dataset

# Preprocessing step for text
def preprocess_text(text, language):
    text = text.lower()
    if language == 'en':
        # English-specific text cleaning
        text = text.replace('[^a-zA-Z0-9 ]', '')
    elif language == 'mr':
        # Marathi-specific text cleaning can be added if necessary
        pass
    return text

# Post-processing function to clean up the translated output
def post_process_translations(translated_texts):
    cleaned_translations = [text.strip().capitalize() for text in translated_texts]
    return cleaned_translations

# Transformer Model Setup (Using mBART)
def build_transformer_model():
    # Load pre-trained mBART model for translation
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    return model, tokenizer

# Fine-tuning the translation models
def fine_tune_model(model, tokenizer, dataset):
    def preprocess_function(examples):
        inputs = [preprocess_text(ex['english'], 'en') for ex in examples['translation']]
        targets = [preprocess_text(ex['marathi'], 'mr') for ex in examples['translation']]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Preprocess the dataset and split it
    dataset = dataset.map(preprocess_function, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Split the dataset for training and testing
    train_dataset, test_dataset = train_test_split(dataset['train'], test_size=0.2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=3,
        load_best_model_at_end=True,
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train the model
    trainer.train()

# Function to translate text using mBART
def translate_text_mbart(text, model, tokenizer, src_lang, tgt_lang):
    # Set the source language for tokenization
    tokenizer.src_lang = src_lang
    preprocessed_input = [preprocess_text(text, 'en' if src_lang == 'en_XX' else 'mr')]
    tokenized_input = tokenizer(preprocessed_input, return_tensors='pt', padding=True, truncation=True)

    # Move model and input to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.to(device)
        tokenized_input = tokenized_input.to(device)

    # Generate the translation with forced target language
    translated_tokens = model.generate(**tokenized_input, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
    translated_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

    return post_process_translations(translated_texts)[0]

# Main Function to Build and Use the Translator
def build_bidirectional_translator():
    # Build the mBART model for both directions (Marathi ↔ English)
    model, tokenizer = build_transformer_model()
    return model, tokenizer

# Real-time translation with user-specified direction
def real_time_translation():
    # Build translator for both directions using mBART
    model, tokenizer = build_bidirectional_translator()

    while True:
        # Ask the user for the direction of translation
        direction = input("Select translation direction (1 for English to Marathi, 2 for Marathi to English, q to quit): ")

        if direction == 'q':
            break  # Exit the loop

        # Validate user input for translation direction
        if direction not in ['1', '2']:
            print("Invalid choice! Please choose 1 for English to Marathi or 2 for Marathi to English.")
            continue

        # User inputs text to translate
        input_text = input("Enter the text to translate: ").strip()

        if direction == '1':  # English to Marathi
            translated_text = translate_text_mbart(input_text, model, tokenizer, src_lang='en_XX', tgt_lang='mr_IN')
            print(f"Translated Text (English to Marathi): {translated_text}")

        elif direction == '2':  # Marathi to English
            translated_text = translate_text_mbart(input_text, model, tokenizer, src_lang='mr_IN', tgt_lang='en_XX')
            print(f"Translated Text (Marathi to English): {translated_text}")

# Example usage
if __name__ == "__main__":
    real_time_translation()
