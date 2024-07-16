# Conversation Summarization System
This is a simple conversation summarization system that user can interact from a web frontend. There are 3 major parts included in this repositry: frontend, API and model fine-tuning.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Finetuning](#finetuning)


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