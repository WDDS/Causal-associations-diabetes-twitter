#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from transformers import BertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import random
import os
import torch.nn.functional as F
import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from torch.utils.data import DataLoader, DataLoader
import transformers
from tqdm import tqdm, trange
#from google.colab import drive, files
import io
import sys
from utils import *
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import torch.distributed as dist
import tqdm
import time
import sys
from tqdm import tqdm


diabetes_keywords = [
"glucose", "#glucose","blood glucose", "#bloodglucose",
"insulin", "#insulin", "insulin pump", "#insulinpump",
"diabetes", "#diabetes", "t1d", "#t1d", "#type1diabetes",
"#type1", "t2d", "#t2d", "#type2diabetes", "#type2",
"#bloodsugar", "#dsma", "#bgnow", "#wearenotwaiting",
"#insulin4all", "dblog", "#dblog", "diyps", "#diyps",
"hba1c", "#hba1c", "#cgm", "#freestylelibre",
"diabetic", "#diabetic", "#gbdoc", "finger prick",
"#fingerprick", "#gestational", "gestational diabetes",
"#gdm", "freestyle libre", "#changingdiabetes",
"continuous glucose monitoring", "#continuousglucosemonitoring",
"#thisisdiabetes", "#lifewithdiabetes", "#stopdiabetes",
"#diabetesadvocate", "#diabadass", "#diabetesawareness",
"#diabeticproblems", "#diaversary", "#justdiabeticthings",
"#diabetestest", "#t1dlookslikeme", "#t2dlookslikeme",
"#duckfiabetes", "#kissmyassdiabetes", "#GBDoc",
"#changingdiabetes", "freestyle libre", "#freestylelibre",
"#cgm"
]

# Transform labels + encodings into Pytorch DataSet object (including __len__, __getitem__)
class TweetDataSet(torch.utils.data.Dataset):
#    def __init__(self, text, labels, tokenizer):
    def __init__(self, text, tokenizer):
        self.text = text
        #self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.text, padding=True, truncation=True, return_token_type_ids=True)
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        return {
                "input_ids" : torch.tensor(ids[idx], dtype=torch.long)
              , "attention_mask" : torch.tensor(mask[idx], dtype=torch.long)
              , "token_type_ids" : torch.tensor(token_type_ids[idx], dtype=torch.long)
             # , "labels" : torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.text)


def compute_metrics(pred, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='weighted')
    acc = accuracy_score(labels, pred)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }



class CausalityBERT(torch.nn.Module):
    """ Model Bert"""
    def __init__(self):
        super(CausalityBERT, self).__init__()
        self.num_labels = 2
        self.bert = transformers.BertModel.from_pretrained("vinai/bertweet-base")
        self.dropout = torch.nn.Dropout(0.3)
        self.linear1 = torch.nn.Linear(768, 256)
        self.linear2 = torch.nn.Linear(256, self.num_labels)
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, output_1 = self.bert(input_ids, attention_mask = attention_mask, token_type_ids=token_type_ids, return_dict=False) # if output 1 is our cls token
        output_2 = self.dropout(output_1)
        output_3 = self.linear1(output_2)
        output_4 = self.dropout(output_3)
        output_5 = self.linear2(output_4)
        # cross entory will take care of the logits - we don't need if we are usign cross entropy for loss function
        # if doing yourself - use nll loss and logSoftmax
#         logit = self.softmax(output_5)
        return output_5




softmax = torch.nn.Softmax(-1)

#################### MODEL PARAMETERS #####################

#device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

bert_model = "vinai/bertweet-base"
# data_path = "/home/adrian/PhD/Data/Tweets20210128/Tweets_per_months_personal_noJokes/matching-tweets_diab_noRT-noDupl_20210128_personal_noJokes_withFullText_emotional.parquet"
data_path = "./data/matching-tweets_diab_noRT-noDupl_20210128_personal_noJokes_withFullText_emotional.parquet"
causal_model = "./model-causal-model/model_4_finetuned-8-epochs-lr_1e-05.pth"
# causal_model = "./model-causal-tweet/model_4_finetuned-8-epochs-lr_1e-05.pth"

result_dir = "result_causal_sentence_prediction"

# whatever we will write here - it is each device - in one go it will process batch_size*no of devicve/
batch_size =  1024 #65552 #1024   #512 # # 65552 #32776 #16388 #8194 #2048

######### LOAD TOKENIZER AND MODEL ####################
tokenizer = AutoTokenizer.from_pretrained(bert_model)

model = CausalityBERT()
model.load_state_dict(torch.load(causal_model, map_location='cpu'))


############ load data #######################
data = pd.read_parquet(data_path)#.sample(n=1000, random_state=33)
print("Total count:", data.shape[0])
# data.head()
# print(data.shape)
print("*************")



####### SPLIT TWEETS INTO SENTENCES ######################

TweetsSplit = data["full_text"].map(lambda full_text: split_into_sentences(normalizeTweet(full_text)))
print(TweetsSplit.shape[0])

sentences = TweetsSplit.explode()
print("tweets to sentences:", sentences.shape[0])


######### Exclude questions and sentences with less than 5 words
# and sentences without diabetes related keyword #################

trainingData = sentences[sentences.str.split(" ").str.len() > 5] # keep only sentence with more than 3 tokens
trainingData = trainingData[~trainingData.str.endswith("?")]
trainingData = trainingData[trainingData.str.contains("|".join(diabetes_keywords))]

print("N sentences with > 5 words & no question & all with diabetes keyword:", trainingData.shape)

text = trainingData.values.tolist()



############ Define Trainer and predict ##############

# set inference arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
#    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=batch_size,   # batch size for evaluation
    logging_dir='./logs',            # directory for storing logs
    seed=0,
#    local_rank = rank
)

# we only use Trainer for inference
trainer = Trainer(model=model, args=training_args)
print("build trainer on device:", training_args.device, "with n gpus:", training_args.n_gpu)



###### ITERATE OVER DATASET ###############

for i in range(0, len(text), batch_size)): # takes always batch_size many tweets
    tweet_subset = text[i:i+batch_size]
    test_dataset = TweetDataSet(tweet_subset, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    with tqdm(total = 10000, file = sys.stdout) as pbar:
        pbar.update(10)
        logits = trainer.predict(test_dataset)
        # pbar.update(10)

    predictions = pd.Series(torch.argmax(torch.Tensor(logits.predictions),dim=1).flatten())
    probas = pd.Series(torch.softmax(torch.Tensor(logits.predictions), dim = -1)[...,-1:].to('cpu').numpy().squeeze())


    #causalDF = pd.DataFrame({"text":text, "causal_predictions": predictions, "proba":probas})
    causalDF = pd.DataFrame({"text":tweet_subset, "causal_predictions": predictions, "proba":probas})
    # causalDF = causalDF[causalDF["causal_predictions"] == 1]
    print("causal sentences:", causalDF.shape[0])
    causalDF.head()

    #causalDF.to_csv(save_path, sep=",")
    causalDF.to_csv(result_dir+"/causal_sentences_predictions_part_{}.csv".format(i), sep=",")
