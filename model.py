import torch
from pathlib import Path
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM\

# Using original pre-trained facebook/bart-large-cnn
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Using self finetuned bart-large-cnn on samsum dataset
# checkpoint_dir = "<path to model>"
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
# model = BartForConditionalGeneration.from_pretrained(checkpoint_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)