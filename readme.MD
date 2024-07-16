# Conversation Summarization System
This is a simple conversation summarization system that user can interact from a web frontend. There are 3 major parts included in this repository: frontend, API and model fine-tuning.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Finetuning](#finetuning)
- [Samples](#samples)
- [Future-Developments](#future-developments)
- [Resources](#resources)


## Installation
1. Clone the repository:
```bash
 git clone https://github.com/joceyngan/conversation_summarization.git
```
2. Create the environment and install requirements:

```bash
 conda create --name conv-sum-py310 python=3.10
 conda activate conv-sum-py310
 pip install -r requirements.txt
```


## Usage
1. Starting the server.
```bash
 python app.py
```
2. open ```index.html``` in browser.
3. Enter the conversation text in the text area.
4. Press the "Summarize" button.
5. Clear the entered text and generated summary by pressing the "Clear" button at anytime.


## Finetuning
1. Check and adjust the training arguments in ```train.py```, and save the changes.
2. Run the train script
```bash
 python train.py
```
3. Run tensorboard to monitor the performance in realtime
```bash
tensorboard --logdir=./results/<train_id>
```
4. model and logs will be saved in ```./results/<train_id>```

## Samples

1. Result using original pre-train bart-large-cnn through the web frontend
[Image of the result using original pre-train bart-large-cnn](#https://github.com/joceyngan/conversation_summarization/blob/master/sample_results/original-bart-large-cnn-result.png)

2. Result using pre-train bart-large-cnn fine-tuned on SAMsum dataset through the web frontend
[Image of the result using bart-large-cnn finetuned on SAMsum dataset](#https://github.com/joceyngan/conversation_summarization/blob/master/sample_results/finetuned-bart-large-cnn-samsum-result.png)

## Future-Developments
### A glimpse on above results:
After the finetuning, the general idea of the conversation is summarized correctly about what they are discussing. However the result above is not very optimized. The speaker of last sentence is inaccurately identified which result in a wrong conclusion on who is going to make the call. This should be taken note for future fine-tuning.

### Applying in auto generating meeting minutes
Most of the traditional companies will have secretary to attend and summarise the key points that went through in a meeting and pass down to the team to follow up. This whole process can be automated if we implement conversation summarisation model alongside with ASR model.

### Applying in prompt engineering
As generative LLMs is getting more and more popular, the concept of conversation summarization is similar to the idea of prompt enginerring, by giving the models more concise instruction to preform the same task to reduce the cost in terms of time and money by elimiating redundant conversation with the model.


## Resources
1. Text sample used in above demo [source] (https://americanenglish.state.gov/files/ae/resource_files/b_dialogues_everyday_conversations_english_lo_0.pdf)
2. SAMSum Corpus dataset on HuggingFace used in finetuning [source] (https://huggingface.co/datasets/Samsung/samsum)