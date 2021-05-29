# Causal-associations-diabetes-twitter



Identification of possible causal associations in diabetes related tweets.

## we have two way: 
 - a two step process and a single MTL process 

##### Common files for both the methods: 
  - Dataset 
  - jmir.yml : conda env I used for the code 


### Two step methods: 
#### Causal Tweet classification 
  - Causality_BERT.ipynb : model build on top of TweetBERT for training and evaluation on own created dataset 
  - [Trained Model](https://www.dropbox.com/s/4y4gz66476f6slb/finetuned-model-10-35.zip?dl=0) : Download and extract the models for casusal tweet identificaiton. It has two models, one trained for 0 epochs and another for 35 epochs.    




 #### Casue and Effect span extraction from the causal tweets     

- Causality-10-epochs-NER-BERT.ipynb : Notebook with output while trainign the model for 10 epochs 
[10 epochs Trained NER span MOdel](https://www.dropbox.com/s/imqf9r1bwzy9ug6/finetuned-NER-10-epochs.zip?dl=0)

 
 
 
 
 
 
 
 ### Single Step approach:  Multi Task Learning with a shared encoder for both causal tweet identification and casue-effect span extraction





