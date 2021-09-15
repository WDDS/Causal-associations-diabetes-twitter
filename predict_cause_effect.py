from transformers import AutoTokenizer, AutoModel
import pandas as pd
import os
import torch
from nltk.tokenize import TweetTokenizer
from emoji import demojize
import re
import glob
import joblib


########### MODEL PARAMETERS ##################


bert_model = "vinai/bertweet-base" # "bert-large-uncased"; "roberta-large"

dataPath = "result_causal_sentence_prediction"
csv_files = glob.glob(os.path.join(dataPath, "*.csv"))

result_dir = "result_cause_effect_prediction"

# Always predicts batch_size many tweets and stores result in result_dir
batch_size =  10000 #65552 #1024   #512 # # 65552 #32776 #16388 #8194 #2048

cause_effect_model = joblib.load("./model-causal-span/bertEmbeddings_simpleCRF.pkl")


################### LOAD BERT TOKENIZER AND MODEL #############################
tokenizer = AutoTokenizer.from_pretrained(bert_model, padding = "max_length", truncation = True, max_length = 512, return_offsets_mapping=True )
model = AutoModel.from_pretrained(bert_model)


################## LOAD DATA ######################
tuples = []
for file in csv_files:
    with open(file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0: # header
                if line.endswith("\n"):
                    line = line[:-2]
                header = line.split(",")[1:]
            else:
                index, ll = line.split(",", 1)
                ll, prob = ll.rsplit(",", 1)
                if prob.endswith("\n"):
                    prob = prob[:-2]
                text, pred = ll.rsplit(",", 1)
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]
                    #text = text[]
                    #print(text)

                tuples.append((text, pred, prob))

print("N tweets from file:", len(tuples))

df = pd.DataFrame(tuples, columns=["text", "pred", "proba"])
df.pred = pd.to_numeric(df.pred)
df.proba = pd.to_numeric(df.proba)
print("Predicted causal sentences:")
print(df.pred.value_counts())


############ ONLY TAKE CAUSAL SENTENCES ##########################
df_causal = df[df["pred"] == 1]#.sample(n=100, random_state=0)  # SAMPLE ONLY FOR TESTING
df_causal["tokenized"] = df_causal["text"].str.split(" ")



########################### Check if cuda available ############################
print("Cuda available: ", torch.cuda.is_available())
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print("Selected {} for this notebook".format(device))


################## GET BERT EMBEDDINGS + OTHER FEATURES ###############################""
def get_word_embeddings(sentence, sentence_tokenised):
    """ Get word embeddings for each word in sentence """
    ids = tokenizer.encode(sentence)
    ids_tensor = torch.tensor(ids).unsqueeze(0) # Batch size: 1
    word_vectors = model(ids_tensor)[0].squeeze()

    word_embeddings_all = []
    for word in sentence_tokenised: # average word embeddings of sub-tokens
        word_encoded = tokenizer.encode(word)
        word_encoded.remove(tokenizer.cls_token_id)
        word_encoded.remove(tokenizer.sep_token_id)

        word_indices = [ids.index(encoded_id) for encoded_id in word_encoded ]

        # average all sub_word vectors of word
        word_vector = torch.zeros((768))
        for sub_token_id in word_indices:
            word_vector += word_vectors[sub_token_id]
        word_vector /= len(word_indices)

        word_embeddings_all.append(word_vector)

    return word_embeddings_all


def word2features(word, i, wordembedding):

    features = {
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.isdigit()': word.isdigit(),
        'wordlength': len(word),
        'wordinitialcap': word[0].isupper(),
        'wordmixedcap': len([x for x in word[1:] if x.isupper()])>0,
        'wordallcap': len([x for x in word if x.isupper()])==len(word),
        'distfromsentbegin': i
    }

    # here you add 300 (fastText) / 768 (Bert) features (one for each vector component)
    for iv,value in enumerate(wordembedding):
        features['v{}'.format(iv)]=value

    return features


def sent2features(sentence, tokenized):
    word_vectors = get_word_embeddings(sentence, tokenized)
    return [word2features(tokenized[i], i, word_vectors[i]) for i in range(len(tokenized))]


######################## Prediction loop ###################################
for i in range(0, df_causal.shape[0], batch_size):
    print(i)
    tweet_subset = df_causal[i:i+batch_size]
    try:
        X = [sent2features(sentence, tokenized)
             for sentence, tokenized in zip(tweet_subset.text.values.tolist()
                                        , tweet_subset.tokenized.values.tolist())]

        predictions = cause_effect_model.predict(X)
        cause_effect_DF = pd.DataFrame({"text":tweet_subset.text,
                                 "tokenized": tweet_subset.tokenized,
                                 "predictions": predictions})

    except:
        print("Error in batch -> execute single tweets...")
        X = []
        texts = []
        tokenized_texts = []
        for sentence, tokenized in zip(tweet_subset.text.values.tolist() , tweet_subset.tokenized.values.tolist()):
            try:
                X.append(sent2features(sentence, tokenized))
                texts.append(sentence)
                tokenized_texts.append(tokenized)
            except:
                print("\nError and ignore tweet:")
                print("sentence:\t", sentence)
                print("tokenized:\t", tokenized)

        predictions = cause_effect_model.predict(X)
        cause_effect_DF = pd.DataFrame({"text":texts,
                                 "tokenized": tokenized_texts,
                                 "predictions": predictions})


    cause_effect_DF.to_csv(result_dir+"/cause_effect_predictions_part_{}.csv".format(i), sep=";")
