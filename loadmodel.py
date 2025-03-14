import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Detect device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model path (use your actual saved model path)
model_path = "arabic_to_english_translation4.pth"

# Load tokenizer (Ensure correct model name)
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")

# Load trained model
def load_trained_model(model_path):
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    # Set weights_only=False to avoid UnpicklingError
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model

# Load the model once to avoid reloading every time
model = load_trained_model(model_path)