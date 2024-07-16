from model import tokenizer, model, device

def summarize_text(text, input_max_length=1024, output_max_length=150, min_length=40, length_penalty=2.0, num_beams=4):

    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=input_max_length, truncation=True).to(device)
    summary_ids = model.generate(inputs, max_length=output_max_length, min_length=min_length, length_penalty=length_penalty, num_beams=num_beams, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # For debugging
    # input_token_length = inputs.shape[1]
    # print("Input Token Length:", input_token_length)
    # print("Encoded Inputs:", inputs)
    # print("Summary IDs:", summary_ids)
    # print("Decoded Summary:", summary)
    
    return summary
