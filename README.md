# Causal-associations-diabetes-twitter



Identification of causal relations in diabetes related tweets using Deep learning.

A preprint of this work is published here: https://arxiv.org/abs/2111.01225#

# The workflow is at follows:
 - Filtering only tweets containing an emotional element to focus on psychological factors (e.g. stress, anxiety, emotions.. related to diabetes) -> [Get emotional tweets.ipynb](https://github.com/WDDS/Causal-associations-diabetes-twitter/blob/main/Get%20emotional%20tweets.ipynb)
 - Identify causal sentences: sentences which are likely to contain a cause-effect pair (binary classification)
   - An active learning routine has been applied on this binary classification to augment the training data efficiently over several rounds -> [Active_learning_iterations](https://github.com/WDDS/Causal-associations-diabetes-twitter/tree/main/Active_learning_iterations)
 - Identify cause-effect pairs in causal sentences. Several models tested:
   - Using FastText vectors as features for a single CRF layer -> [model_cause-effect_training_FastText_CRF.ipynb](https://github.com/WDDS/Causal-associations-diabetes-twitter/blob/main/model_cause-effect_training_FastText_CRF.ipynb)
   - Using BERTweet vectors as features for a single CRF layer -> [model_cause-effect_training__BERT_features_CRF.ipynb](https://github.com/WDDS/Causal-associations-diabetes-twitter/blob/main/model_cause-effect_training__BERT_features_CRF.ipynb)
   - Finetuning a pretrained BERTweet model with two feed-forward linear layers and dropouts on top and with a last softmax layer -> [model_cause-effect_training__Finetuned_BERT.ipynb](https://github.com/WDDS/Causal-associations-diabetes-twitter/blob/main/model_cause-effect_training__Finetuned_BERT.ipynb)
   - Finetuning a pretrained BERTweet model with two feed-forward linear layers and dropouts on top and with a last CRF layer [model_cause-effect_training__Finetuned_BERT_last_layer_CRF.ipynb](https://github.com/WDDS/Causal-associations-diabetes-twitter/blob/main/model_cause-effect_training__Finetuned_BERT_last_layer_CRF.ipynb)
 - Semi-supervised clustering of the identified cause-effect pairs -> [Semi-supervised clustering of causes-effects.ipynb](https://github.com/WDDS/Causal-associations-diabetes-twitter/blob/main/Semi-supervised%20clustering%20of%20causes-effects.ipynb)
 - Visualisation of the identified cause-effect clusters in an interactive network -> [Visualisation_cause-effect_clusters.ipynb](https://github.com/WDDS/Causal-associations-diabetes-twitter/blob/main/Visualisation_cause-effect_clusters.ipynb)
    - The resulting files for the nodes and links for the cause-effect cluster network are stored in the directory: [cause-effect_networks_for_D3_visualisation/](https://github.com/WDDS/Causal-associations-diabetes-twitter/tree/main/cause-effect_networks_for_D3_visualisation)
    - The cause-effect cluster network is visualised interactively in D3. The interested reader can play and discover cause-effect relations -> [HERE](https://observablehq.com/@adahne/cause-and-effect-associations-in-diabetes-related-tweets)
