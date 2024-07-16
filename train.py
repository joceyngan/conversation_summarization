import os
from datetime import datetime
from pathlib import Path
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForSeq2Seq
from datasets import load_dataset

# Naming the run with timestamp
train_name = 'bart-large-cnn' + '-' + datetime.now().strftime("%Y%m%d%H%M%S")
results_path = Path("results/" + train_name)
results_path.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("samsum")

def preprocess(examples):
    inputs = ["summarize: " + dialogue for dialogue in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=150, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenized_datasets = dataset.map(preprocess, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training arguments
training_args = TrainingArguments(
    output_dir=str(results_path),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="tensorboard",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir=str(results_path),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)], # Stop after 3 eval steps w/ no improvement
)

train_log = trainer.train()

# Evaluation
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Saving models and logs
model_save_path = results_path / f"{train_name}.pth"
trainer.save_model(model_save_path)

train_args_save_path = results_path / f"{train_name}-train-args.txt"
with open(train_args_save_path, 'w') as f:
    f.write(str(training_args))

train_log_save_path = results_path / f"{train_name}-train-log.log"
with open(train_log_save_path, 'w') as f:
    f.write(str(train_log))

eval_results_save_path = results_path / f"{train_name}-evaluation.txt"
with open(eval_results_save_path, 'w') as f:
    f.write(str(eval_results))

# Generate summaries for 5 sample texts from the val set for human evaluation
sample_texts = tokenized_datasets["validation"]["dialogue"][:5]
references = tokenized_datasets["validation"]["summary"][:5]

print("\nSample Summaries:\n")
for i, sample_text in enumerate(sample_texts):
    inputs = tokenizer.encode("summarize: " + sample_text, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    print(f"Sample {i+1}:")
    print("Original Text:", sample_text)
    print("Reference Summary:", references[i])
    print("Generated Summary:", summary)
    print("\n")
