import torch
from pathlib import Path
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Using self finetuned bart-large-cnn on samsum dataset
checkpoint_dir = "/home/jocelyn/nlp/text-sum/results/bart-large-cnn-20240716211006/bart-large-cnn-20240716211006.pth"
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained(checkpoint_dir)

# Original pre-trained facebook/bart-large-cnn
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
# model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)