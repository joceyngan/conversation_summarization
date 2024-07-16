# Conversation Summarization System

## Introduction
This is a simple conversation summarization system that user can interact from a web frontend. There are 3 major parts included in this repository: frontend, API and model fine-tuning.

#### Why BART?
BART is a sequence-to-sequence model which is very suitable for downstream task that needs to output a sequence of tokens like summarization and creating new, coherent output text. Unlike BERT which is trained for understanding text and extracting the contextual embeddings. Besides, traditional algorithms like TextRank and SumBasic algorithms preform extractive summarization on input text which are proned to copy the phases directly from input and creating less satisfied result.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Finetuning](#finetuning)
- [Evaluation](#evaluation)
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
Default is using original BART model on huggingface without finetuning. If already have finetuned model, uncomment the line 11-13 "Using self finetuned bart-large-cnn on samsum dataset" in ```model.py```, replace the path to model to your own path before the following steps. and comment out line 7-8 of the part "Using original pre-trained facebook/bart-large-cnn" 

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

## Evaluation
After the finetuning, 5 sample scripts in the validation set will be used to run an evaluation step. Then the original text, reference summary (ground truth) and generated summary of the sample scripts will be printed on console to be used for preforming human evaluation on the finetuned model.

Example 1:
##### Original Text: 
Emma: I’ve just fallen in love with this advent calendar! Awesome! I wanna one for my kids!  
Rob: I used to get one every year as a child! Loved them!  
Emma: Yeah, i remember! they were filled with chocolates!  
Lauren: they are different these days! much more sophisticated! Haha!  
Rob: yeah, they can be fabric/ wooden, shop bought/ homemade, filled with various stuff
Emma: what do you fit inside?  
Lauren: small toys, Christmas decorations, creative stuff, hair bands & clips, stickers, pencils & rubbers, small puzzles, sweets  
Emma: WOW! That’s brill! X  
Lauren: i add one more very special thing as well- little notes asking my children to do something nice for someone else  
Rob: i like that! My sister adds notes asking her kids questions about christmas such as What did the 3 wise men bring? etc  
Lauren: i reckon it prepares them for Christmas   
Emma: and makes it more about traditions and being kind to other people  
Lauren: my children get very excited every time they get one!  
Emma: i can see why! :)  
##### Reference Summary:
Emma and Rob love the advent calendar. Lauren fits inside calendar various items, for instance, small toys and Christmas decorations. Her children are excited whenever they get the calendar.
##### Generated Summary:
Emma loves the advent calendar. Rob used to get one every year as a child. Emma would like to buy one for her kids. Rob's sister asks her kids questions about Christmas. Lauren's children get very excited every time they get one.

Example 2:  
##### Original Text:
Robert: Hey give me the address of this music shop you mentioned before  
Robert: I have to buy guitar cable  
Fred: <file_other>  
Fred: Catch it on google maps  
Robert: thx m8  
Fred: ur welcome  
##### Reference Summary:
Robert wants Fred to send him the address of the music shop as he needs to buy guitar cable.
##### Generated Summary:
Fred gives Robert the address of the music shop where he needs to buy guitar cable. Robert can find it on google maps. Robert has to buy a guitar cable at the shop. Fred sends him a link to the address. 

## Samples of the System and Results

1. Result using original pre-train bart-large-cnn through the web frontend
![Image of the result using original pre-train bart-large-cnn](https://github.com/joceyngan/conversation_summarization/blob/master/sample_results/original-bart-large-cnn-result.png?raw=true)

2. Result using pre-train bart-large-cnn fine-tuned on SAMsum dataset through the web frontend
![Image of the result using bart-large-cnn finetuned on SAMsum dataset](https://github.com/joceyngan/conversation_summarization/blob/master/sample_results/finetuned-bart-large-cnn-samsum-result.png?raw=true)

## Future-Developments
### A glimpse on above results:
After the finetuning, the general idea of the conversation is summarized correctly about what they are discussing. However the result above is not very optimized. The speaker of last sentence is inaccurately identified which result in a wrong conclusion on who is going to make the call. This should be taken note for future fine-tuning.

### Applying in auto generating meeting minutes
Most of the traditional companies will have secretary to attend and summarise the key points that went through in a meeting and pass down to the team to follow up. This whole process can be automated if we implement conversation summarisation model alongside with ASR model.

### Applying in prompt engineering
As generative LLMs is getting more and more popular, the concept of conversation summarization is similar to the idea of prompt enginerring, by giving the models more concise instruction to preform the same task to reduce the cost in terms of time and money by elimiating redundant conversation with the model.


## Resources
1. Text sample used in above demo [source](https://americanenglish.state.gov/files/ae/resource_files/b_dialogues_everyday_conversations_english_lo_0.pdf)
2. SAMSum Corpus dataset on HuggingFace used in finetuning [source](https://huggingface.co/datasets/Samsung/samsum)