# Causal-associations-diabetes-twitter



Identification of possible causal associations in diabetes related tweets.

## we have two way: 
 - a two step process and a single MTL process 

##### Common files for both the methods: 
  - Dataset 
  - jmir.yml : conda env I used for the code 


## Two step methods: 

### Causal Tweet classification 
#### Weighted loss and accuracy measures on different learnign rate:  
Three different notebooks and corresponding models. [Download the model here](https://www.dropbox.com/s/3s4iay6fht110tb/finetuned-35-epochs-weighted-class.zip?dl=0) - extract them and put in the directory based on the code in each notebook. 

- [Causality_BERT_with_weighted_loss_35_epochs_lr_1e3.ipynb](https://github.com/WDDS/Causal-associations-diabetes-twitter/blob/main/Causality_BERT_with_weighted_loss_35_epochs_lr_1e3.ipynb)
- [Causality_BERT_with_weighted_loss_35_epochs_lr_1e5.ipynb](https://github.com/WDDS/Causal-associations-diabetes-twitter/blob/main/Causality_BERT_with_weighted_loss_35_epochs_lr_1e5.ipynb)
- [Causality_BERT_with_weighted_loss_35_epochs_lr_5e5.ipynb](https://github.com/WDDS/Causal-associations-diabetes-twitter/blob/main/Causality_BERT_with_weighted_loss_35_epochs_lr_5e5.ipynb)





 ### Casue and Effect span extraction from the causal tweets        
  
- 
 
 
 
 
 
 
 
 ## Single Step approach: Multi Task Learning with a shared encoder for both causal tweet identification and casue-effect span extraction   
 
 ### Multi task learnign with just decoder training  
 
 ### Multi task learnign with decoder training and encoder (a few layers) fine tunign   
  




