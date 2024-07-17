import torch
from pathlib import Path
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Original pre-trained facebook/bart-large-cnn
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
# model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Using self finetuned bart-large-cnn on samsum dataset
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained("jngan/bart-large-cnn-samsum")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)