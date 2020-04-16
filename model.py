#!/usr/bin/env python
# coding: utf-8


'''
Event Clustering within News Articles accepted to AESPEN in LREC 2020.

Faik Kerem Örs, Süveyda Yeniterzi, Reyyan Yeniterzi

2nd April 2020 - Version 1

'''


import numpy as np
import pandas as pd
import json
import torch
import time
import datetime
import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import AlbertForSequenceClassification, AlbertTokenizer, AdamW, get_linear_schedule_with_warmup
import io
from ast import literal_eval
import numpy as np

%tensorflow_version 2.x
import tensorflow as tf

SEQ_LEN = 115 # Decided Based on Sentence Lengths

def text_processor(data_prep, column_names, data_type="train"):
  
  # Remove NaN Rows
  if data_type == "train":
    data_prep.dropna(inplace=True)
  
  for col_name in column_names:
    
    # Strip Spaces
    data_prep[col_name] = data_prep[col_name].str.strip()


def convert_data(data_df):

  # Encode sentence pairs using the tokenizer.

  encoded_data = []

  for ind in data_df["label"].index:
    encoded_sents = tokenizer.encode_plus(text=data_df["sent1"][ind], text_pair=data_df["sent2"][ind], add_special_tokens=True, max_length=SEQ_LEN, pad_to_max_length=True, return_token_type_ids=True)
    encoded_sents["label"] = data_df["label"][ind]

    encoded_data.append(encoded_sents)

  return encoded_data


def test_convert_data(data_df):

    # Again we encode the sentence pairs of the test data.
    # Since we don't have label, we stored index.

  encoded_data = []

  for ind in data_df["sent1"].index:
    encoded_sents = tokenizer.encode_plus(text=data_df["sent1"][ind], text_pair=data_df["sent2"][ind], add_special_tokens=True, max_length=SEQ_LEN, pad_to_max_length=True, return_token_type_ids=True)
    encoded_sents["index"] = ind

    encoded_data.append(encoded_sents)

  return encoded_data


def convert_tensors(encoded_df):

  # Convert data to tensors.
  
  encoded_tensors = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'label': []}

  for elt in encoded_df:
    for key in encoded_tensors.keys():
      encoded_tensors[key].append(elt[key])

  for key in encoded_tensors.keys():
    encoded_tensors[key] = torch.tensor(encoded_tensors[key])

  return encoded_tensors


def test_convert_tensors(encoded_df):

  # Convert test data to tensors.
  # This time we don't have the labels but indices.
  
  encoded_tensors = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'index': []}

  for elt in encoded_df:
    for key in encoded_tensors.keys():
      encoded_tensors[key].append(elt[key])

  for key in encoded_tensors.keys():
    encoded_tensors[key] = torch.tensor(encoded_tensors[key])

  return encoded_tensors


def get_dataloader(dict_tensor, batch_size=32, shuffle=False):

  # Generate the data loader for training and testing.

  dataset = TensorDataset(*dict_tensor.values())

  if shuffle:
    # Train data is shuffled.
    sampler = RandomSampler(dataset)
  else:
    # Test data is not shuffled.
    sampler = SequentialSampler(dataset)

  dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

  return dataloader


def flat_accuracy(preds, labels):

    # Function to calculate the accuracy of our predictions vs labels
    # Based on https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1


    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    Based on https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_classes(preds):

    # Function to calculate the accuracy of our predictions vs labels
    # Based on https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1


    pred_flat = np.argmax(preds, axis=1).flatten()
    return pred_flat


def get_probs(logits):

  # Converts logits to probabilities.

  obj = torch.nn.Sigmoid()
  return obj(logits)


def train(model, train_dataloader):

    # Model Training
    # Based on https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]


    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )


    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 4

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)


    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = 113

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (Based on https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_token_type_ids = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (Based on https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, 
                        token_type_ids=b_token_type_ids, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
            
            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)            
        
        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))


    print("")
    print("Training complete!")



def test(model, test_dataloader):

    # Prediction on test set
    # Based on https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1

    print('Predicting the labels...')

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions, true_labels, sent_logits, sent_probs = [], [], [], []

    # Predict 
    for batch in test_dataloader:
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)
      
      # Unpack the inputs from our dataloader
      b_input_ids, b_token_type_ids, b_input_mask, b_labels = batch
      
      # Telling the model not to compute or store gradients, saving memory and 
      # speeding up prediction
      with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = model(b_input_ids, token_type_ids=b_token_type_ids, 
                          attention_mask=b_input_mask)

      logits = outputs[0]
      sent_probs.append(get_probs(logits).detach().cpu().numpy())
      
      if len(sent_probs) % 100 == 0:
        print(len(sent_probs))

      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()
      
      # Calculate the accuracy for this batch of test sentences.
      #tmp_eval_accuracy = flat_accuracy(logits, label_ids)
      
      # Accumulate the total accuracy.
      #eval_accuracy += tmp_eval_accuracy

      # Track the number of batches
      nb_eval_steps += 1

      # Store predictions and true labels
      predictions.append(get_classes(logits)[0])
      #true_labels.append(label_ids)
      sent_logits.append(logits.tolist()[0])

    print('    DONE.')
    #print("  Accuracy: {0:.3f}".format(eval_accuracy/nb_eval_steps))

    return predictions, sent_logits



def sort_clusters(news_clusters):
  for news_index, clusters in news_clusters.items():

    for lst in clusters:

      lst.sort()

    clusters.sort(key=lambda x: x[0])


def check_duplicates(news_clusters):
  # Check an element exists in other lists

  for news_index, clusters in news_clusters.items():

    for lst in clusters:

      for elt in lst:

        for lst2 in clusters:
          if lst != lst2:
            if elt in lst2:
              print("Duplicate EXISTS:", news_index)


def get_scores(post_val_data, reward, penalty):

  '''

  Scoring Algorithm.

  '''

  news_scores = {}
  for news_index in post_val_data["news_index"].unique():

    # Create a dict to store the pairwise predictions.
    news_relationships = {}

    for _, sent_pair in post_val_data[post_val_data["news_index"] == news_index].iterrows():
      
      # Get pair IDs and corresponding pairwise prediction.
      sent1_index = sent_pair["sent1_index"]
      sent2_index = sent_pair["sent2_index"]
      prediction = sent_pair["predictions"]


      # Store the predictions in the format: {sent_id1: {another_sent_id1: pairwise_prediction1, another_sent_id2: pairwise_prediction2}}
      if sent1_index in news_relationships:
        news_relationships[sent1_index][sent2_index] = prediction
      else:
        news_relationships[sent1_index] = {sent2_index: prediction}

      # Store the relationships symmetrically
      # Symmetric case would be: {sent_id: {another_sent_id: pairwise_prediction}, {another_sent_id: {sent_id: pairwise_prediction}}}
      if sent1_index < sent2_index:

        if sent2_index in news_relationships:
          news_relationships[sent2_index][sent1_index] = prediction
        else:
          news_relationships[sent2_index] = {sent1_index: prediction}


    # Create a dict to store the relationship scores.
    # Neighbor terminology used in this code means that the sentence pairs 'main_key' and 'neigh_key' appear to be in the same cluster.
    final_neighbors = {}

    for main_key, main_neighs in news_relationships.items():

      # The first sentence, say 'main_key'
      final_neighbors[main_key] = {}

      for main_neigh in main_neighs.items():

        # 'main_neigh_key': The second sentence that forms the pair together with the first sentence 'main_key'
        # 'main_pred': Model's prediction for the pair (main_key, main_neigh_key)
        main_neigh_key, main_pred = main_neigh

        # Set initial scores based the pairwise predictions.
        # 1 if they are predicted to be in the same cluster.
        # -1 Otherwise (penalize).
        if main_pred == 1:
          neighbor_score = 1
        else:
          neighbor_score = -1

        # Consider common relationships that main_key and main_neigh_key have.
        # Reward their pairwise score if they have common neighbors.
        # Penalize their pairwise if they have neighbors that are not common.
        if main_neigh_key in news_relationships:

          # Iterate over the neighbors of main_neigh_key (the second sentence)
          for helper_neighs in news_relationships[main_neigh_key].items():

            # 'helper_neigh_key': The sentence that forms the pair together with the second sentence 'main_neigh_key'
            # 'main_pred': Model's prediction for the pair (main_neigh_key, helper_neigh_key)
            helper_neigh_key, helper_neigh_pred = helper_neighs

            # Iterate over the neighbors of main_key to see whether it also appears to be in the same cluster with helper_neigh_key
            for x_neigh_key, x_pred in main_neighs.items():

              if x_neigh_key == helper_neigh_key:

                # If main_key (the first sentence) and main_neigh_key (the second sentence) have a common neighbor, reward their pairwise score.
                # If helper_neigh_key is the neighbor of only one of them (the first or second sentence), penalize the pairwise score of main_key and main_neigh_key.
                # Otherwise, do nothing since we might not know.
                if x_pred == 1 and helper_neigh_pred == 1:
                  neighbor_score += reward
                elif x_pred == 1 and helper_neigh_pred == 0:
                  neighbor_score -= penalty
                elif x_pred == 0 and helper_neigh_pred == 1:
                  neighbor_score -= penalty

                break

        # Scores for one news.
        final_neighbors[main_key][main_neigh_key] = neighbor_score
      
    # Store the scores together with the corresponding news.
    news_scores[news_index] = final_neighbors
  return news_scores



def get_clusters(news_scores):

  '''

  Clustering Algorithm
  
  '''

  # Example input
  '''
  scores = {2: {4: 1, 27: 0, 36: 2, 37: 0, 40: -6, 43: -4}, 
            4: {2: 1, 27: 0, 36: -1, 37: -1, 40: -3, 43: -3}, 
            27: {2: 0, 4: 0, 36: 0, 37: -2, 40: -4, 43: -2}, 
            36: {2: 2, 4: -1, 27: 0, 37: 1, 40: -5, 43: -3}, 
            37: {2: 0, 4: -1, 27: -2, 36: 1, 40: -4, 43: -5}, 
            40: {2: -6, 4: -3, 27: -4, 36: -5, 37: -4, 43: 0}, 
            43: {2: -4, 4: -3, 27: -2, 36: -3, 37: -5, 40: 0}}
  '''

  news_clusters = {}

  for news_index, scores in news_scores.items():

    column_names = ["Sen_1", "Sen_2", "Score"]
    df = pd.DataFrame(columns = column_names)

    # Create a dataframe of pairwise sentence scores
    for sentence, scores in scores.items():
        for key in scores:
            df = df.append(pd.DataFrame({"Sen_1":[sentence], "Sen_2":[key], "Score":[scores[key]]}) , ignore_index = True)

    # Sort the dataframe by descending order of score, and the ascending order of sentence 1 and 2
    df.sort_values(by=['Score', 'Sen_1', 'Sen_2'], ascending=[0, 1, 1], inplace = True)

    # Create a sentence list with all currently assigned to group 0
    sentences = pd.DataFrame(set(df['Sen_1'].tolist()), columns =['Sentences']) 
    sentences['Group'] = 0

    # Eliminate all sentence pairs with score <= 0
    df = df[df['Score'] > 0]

    group_count = 0

    if not df.empty:

      # Eliminate duplicate rows
      df['Sen_min'] = df.apply(lambda row: min(row.Sen_1, row.Sen_2), axis=1)
      df['Sen_max'] = df.apply(lambda row: max(row.Sen_1, row.Sen_2), axis=1)
      df.drop(['Sen_1', 'Sen_2'], axis=1, inplace=True)
      df.drop_duplicates(inplace = True) 

      # Iterate over the dataframe and assign sentence pairs to groups based on the below conditions:
      # - If the current sentence pair have both Group = 0 (means they've not yet assigned to any group), then create a new group and assign both sentence to this new group
      # - Else if only one of the sentence has Group = 0 in the pair, then that sentence is assigned to the group of the other sentence
      # - Else sentences are already assigned to other groups, then no need to do anything

      for index, row in df.iterrows():
          if sentences.loc[sentences['Sentences'] == row['Sen_min'], 'Group'].iloc[0] == 0 and sentences.loc[sentences['Sentences'] == row['Sen_max'], 'Group'].iloc[0] == 0:
              group_count = group_count + 1
              sentences.loc[sentences['Sentences'] == row['Sen_min'], 'Group'] = group_count
              sentences.loc[sentences['Sentences'] == row['Sen_max'], 'Group'] = group_count
          elif sentences.loc[sentences['Sentences'] == row['Sen_min'], 'Group'].iloc[0] == 0:
              sentences.loc[sentences['Sentences'] == row['Sen_min'], 'Group'] = sentences.loc[sentences['Sentences'] == row['Sen_max'], 'Group'].iloc[0]
          elif sentences.loc[sentences['Sentences'] == row['Sen_max'], 'Group'].iloc[0] == 0:
              sentences.loc[sentences['Sentences'] == row['Sen_max'], 'Group'] = sentences.loc[sentences['Sentences'] == row['Sen_min'], 'Group'].iloc[0]
          else:
              pass
              
        
    # At the end if there are still sentences that have not been assigned to any group, then assign them to seperate groups individually
    for index, row in sentences.iterrows():
        if row['Group'] == 0:
            group_count = group_count + 1
            sentences.loc[sentences['Sentences'] == row['Sentences'], 'Group'] = group_count

    news_clusters[news_index] = []
    for gr in sentences["Group"].unique():
      news_clusters[news_index].append(sentences[sentences["Group"] == gr]["Sentences"].values.tolist())

  return news_clusters


if __name__ == "__main__":

    # Load data.
    train_data = pd.read_csv("train_data.csv")
    test_data = pd.read_csv("test_data.csv")

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    # Remove NaN values etc.
    text_processor(train_data, train_data.columns.values.tolist()[3:-1])
    text_processor(test_data, test_data.columns.values.tolist()[3:], data_type="test")

    # ALBERT xxlarge-v2
    model_class = AlbertForSequenceClassification
    tokenizer_class = AlbertTokenizer
    pretrained_weights = 'albert-xxlarge-v2'
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)
    model = model_class.from_pretrained(pretrained_weights, num_labels=2, output_attentions=False, output_hidden_states=False)

    # Encode sentence pairs in train and test sets using tokenizer.
    encoded_train = convert_data(train_data)
    encoded_test = test_convert_data(test_data)

    # Convert encoded data to tensors.
    train_tensors = convert_tensors(encoded_train)
    test_tensors = test_convert_tensors(encoded_test)

    # Create data loaders to feed the data batch by batch.
    train_dataloader = get_dataloader(train_tensors, batch_size=16, shuffle=True)
    test_dataloader = get_dataloader(test_tensors, batch_size=1)

    # Send model to the device.
    model.cuda()

    # Train model
    train(model, train_dataloader)

    # SAVE MODEL
    torch.save(model.state_dict(), "model.pt")

    # LOAD MODEL
    #model.load_state_dict(torch.load("model.pt"))
    #model.cuda()

    # Test model and get pairwise predictions and logits.
    pred_lst, logit_lst = test(model, test_dataloader)


    # Fine-tuned rewards and penalties.
    rewards = [0.8]
    penalties = [0.8]

    for reward in rewards:
      for penalty in penalties:
        
        # Use post_val_data to store the predictions and logits for each sentence pair.
        post_val_data = test_data.copy()
        post_val_data["predictions"] = pred_lst
        post_val_data["logits"] = logit_lst
        #post_val_data["probabilities"] = prob_lst

        # Get Scores
        news_scores = get_scores(post_val_data, reward, penalty)

        # Get the Clusters.
        news_clusters = get_clusters(news_scores)

        # There shouldn't be a duplicate, just for debugging...
        check_duplicates(news_clusters)

        # Sort Clusters for evaluation.
        sort_clusters(news_clusters)

        # Put cluster predictions also to post_val_data.
        for news_index, clusters in news_clusters.items():
          for ind in post_val_data[post_val_data["news_index"] == news_index].index:
            post_val_data.loc[ind, "prediction_clusters"] = str(clusters)


    # Get test.json file that doesn't have gold labels.
    orj_test = pd.read_json("Path_to_Data/test.json", lines=True)

    # Put cluster predictions to the json file for evaluation.
    for news_index, clusters in news_clusters.items():
      orj_test.loc[news_index, "prediction_clusters"] = str(clusters)

    # Name cluster predictions as event_clusters for evaluation.
    orj_test.rename(columns={"prediction_clusters": "event_clusters"}, inplace=True)
    orj_test.loc[:,'event_clusters'] = orj_test.loc[:,'event_clusters'].apply(lambda x: literal_eval(x))

    # Save pairwise and cluster predictions in csv and json format.
    post_val_data.to_csv("pairwise_predictions.csv", index=None)
    orj_test.to_json("cluster_predictions.json", orient="records", lines=True)

    # Use json for final evaluation.




