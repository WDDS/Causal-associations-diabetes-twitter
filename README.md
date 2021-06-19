# Causal-associations-diabetes-twitter



Identification of possible causal associations in diabetes related tweets.

# we have two way: 
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





 ### Casue and Effect span extraction from the causal tweets: 
 
 1. Model and notebook trained only on tweets that has both cause and effect
   - Notebook: [Causality-event-span-for-35-epochs-for-cause-and-effect.ipynb](https://github.com/WDDS/Causal-associations-diabetes-twitter/blob/main/Causality-event-span-for-cause-and-effect.ipynb)     
   - Model: [causal span extraction model tained for 35 epochs only on data with both cause and effect in a single tweet](https://www.dropbox.com/s/i3e4y62km7foav4/finetuned-cause-effect-span-cause-and-effect-35-epochs.pth?dl=0)  

 2. Model and notebook trained only on tweets that has either cause or effect or both cause and effect: 
   - Notebook : [Causality-event-span-for-35-epochs-for-cause-or-effect.ipynb](https://github.com/WDDS/Causal-associations-diabetes-twitter/blob/main/Causality-event-span-for-cause-or-effect.ipynb)  
   - Model : [causal span extraction model trained for 35 epochs on tweets with either cause or effect or both](https://www.dropbox.com/s/qlvt1unck4vycps/finetuned-causal-span-cause-or-effect-35-epochs.pth?dl=0)  

 
 
 
 
 
 
 
 
 ## Single Step approach: Multi Task Learning with a shared encoder for both causal tweet identification and casue-effect span extraction   
 
 We have two ways of doing it; either just train the decoder for both of the task on a shared BERT based encoder or train the decoder as well as finetune a few layers of encoder. 
 
 ### Multi task learnign with just decoder training  
  - Notebook : [multi task model tained for 35 epoochs - decoder training - no- encoder finetuning](https://github.com/WDDS/Causal-associations-diabetes-twitter/blob/main/Causality-Multitask-BERT-decoder-training.ipynb)   
  - Model : [trained model for 35 epochs - with trained decoder- no-encoder -fine-tuning](https://www.dropbox.com/s/7ar5c6zniywpz5d/finetuned-decoder-multi-task-35-epochs.pth?dl=0)  
 
 ### Multi task learnign with decoder training and encoder (a few layers) fine tuning   
  - Notebook : TODO   
  - Model : TODO  
  




