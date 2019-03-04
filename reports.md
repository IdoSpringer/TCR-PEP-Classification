
# Reports

(older reports will be organized soon)

## Report 22/2/19

### Encoding TCRs
In previous attempts to build a model for predicting TCR and peptide
attachment, we used LSTM to encode both TCR and peptide. We tried a
different approach, using TCR autoencoder (based on Shirit's work).

#### The Autoencoder
The autoencoder we use is based on zero-padding and linear layers, not
RNNs. Given a TCR, we first add a stop signal to the end of its amino-acid
sequence. Then, we convert the TCR to a sequence of one-hot vectors.
We make sure that all TCR sequences have the same max-length by
adding zero-padding vectors in the end of the sequences. It is necessary
because we use linear layers with fixed size. We concatenate the one-
hot vectors of the TCR sequence into a large vector, which is fed into the
encoder. The encoder (and the decoder) are composed from some linear
layers with dropout layers between them (and softmax to make one-hot
vectors). We use the encoder to encode the TCR into a fixed size vector,
and the decoder to decode that vector to a sequence of one-hot vectors
representing the TCR.

The autoencoder (implemented in PyTorch) can be found here: 
[autoencoder](https://github.com/IdoSpringer/TCR-PEP-Classification/blob/master/autoencoder_model.py)

#### Autoencoder configurations
(based on Shirit's code in Keras)

**Dimensions:**  
Encoder: after we concatenate the one-hot vectors of the
TCR, we pass it through a linear layer to dense its size to 300, then 100,
and then the encoding dimension which is 30.

Decoder: we go the
reverse way, from the encoding dimension to 100 to 300 to the original
one-hot TCR dimensions. Then we use softmax.

**Activation:**  We use ELU activation between the linear layers.

**Dropout:**  We use dropout rate of 0.1 between the layers.

**Loss function:**  Mean squared error.

**Optimizer:** Adam, with learning rate = 1e-4, beta1 = 0.9, beta2 = 0.999,
epsilon = 1e-8, weight decay = 0.

**Batch size:**  50 (the batch size is fixed).

**Epochs:** the autoencoder is trained for 300 epochs

#### TCR Autoencoder Data
The autoencoder is trained with the new CDR3 data given by Shirit,
which can be found here: 
[BM_data_CDR3s](https://github.com/IdoSpringer/TCR-PEPClassification/tree/master/BM_data_CDR3s)

#### Autoencoder evaluation
To evaluate the autoencoder, we read the decoded one-hot TCRs, and
compare it to the input TCRs (80% of the TCRs are used to train the
autoencoder, and 20% for evaluation). Measuring only completely
correct decoding may be too harsh, so we also measure accuracy when
we allow some decoding mismatches. The results:

Number of mistakes allowed | accuracy
--- | --- 
Zero mistakes | 0.9203
Up to 1 mistake | 0.9826
Up to 2 mistakes | 0.9935
Up to 3 mistakes | 0.9972

#### The new model
After we trained the TCR autoencoder, we used it in our model to
predict TCR-peptide attachment. Instead of encoding the TCRs using
LSTM, we use the pre-trained TCR autoencoder. The peptide is still
encoded using the LSTM. The encodings of the TCR and the peptide are
concatenated and fed into a MLP to a probability prediction as before.
(Notice that we use one-hots to represent the TCR, and an embedding
matrix to represent the peptide). The autoencoder weights are not
trained during this model training. All other configurations are same as
before.

Code: [Autoencoder based model](https://github.com/IdoSpringer/TCR-PEP-Classification/blob/master/tcr_ae_pep_lstm_model.py)

#### Results on Weizmann data
Unfortunately, the new model with the TCR autoencoder did not
improve the results on the Weizmann data.
![Weizmann Results](https://github.com/IdoSpringer/TCR-PEP-Classification/blob/master/ae_auc.png)

## Report 24/2/19

### Different dropouts
Our autoencoder based model for encoding the TCRs gave lower AUC
results than the LSTM based model. We wanted to check the new model
again with different dropout configurations. Dropout is set between the
layers of the LSTM in the peptides encoding, and in the layers of the MLP
classifier in the end. No regularization is used. We also increased the
number of epochs, so the training algorithm will converge well.

![ae different dropouts](https://github.com/IdoSpringer/TCR-PEP-Classification/blob/master/ae_do_auc.png)

It seems like dropout rate of 0.1 gives the best results (about 0.87 auc on
train, 0.78 on test), Yet in out LSTM based previous model we still reach
better results (0.90 on train, 0.83 on test).

## Report 26/2/19

### Better results with the LSTM based model
In the TCR autoencoder the encoding dimension Is 30, which is larger
than the current encoding model in our LSTM based model (it was 10
before). We trained the LSTM based model again with better
hyperparameters (dropout rate of 0.1, some regularization, and
encoding dimension 30) and got better results than before.
We checked it for all datasets, but mainly on the Weizmann data (so
other parameters can be considered for different datasets)

![Weizmann](https://github.com/IdoSpringer/TCR-PEP-Classification/blob/master/pair_sampling/auc_w_hdim30.png)
![Shugay](https://github.com/IdoSpringer/TCR-PEP-Classification/blob/master/pair_sampling/auc_s_hdim30.png)
![cancer](https://github.com/IdoSpringer/TCR-PEP-Classification/blob/master/pair_sampling/auc_c_hdim30.png)

### Autoencoder based model
We also tried some regularization in the autoencoder based model. We
got slightly better results, but not as in the previous model.

![ae weizmann reg](https://github.com/IdoSpringer/TCR-PEP-Classification/blob/master/ae_w_reg.png)

The autoencoder based model also fails on the cancer data.

![ae cancer](https://github.com/IdoSpringer/TCR-PEP-Classification/blob/master/ae_c.png)

## Report 27/2/19

### Autoencoder parameters training
We tried to train the autoencoder based model again, but now we allow
the algorithm to train also the TCR autoencoder parameters. It is more
task specific, therefore we get better results for the TCR-peptide
attachment prediction task.

Results on Weizmann data with different regularizations:

![ae with training weizmann](https://github.com/IdoSpringer/TCR-PEP-Classification/blob/master/ae_tr_w.png)

The improvement we get is significant, and it competes the LSTM based
model (about 0.85 AUC on test, 0.99 on train).

## Report 4/3/19

### Autoencoder based model performance on cancer data
We saw that when we also allow to train the autoencoder parameters in the autoencoder based model,
we get better results, because the parameters are trained to solve the specific TCR-Peptide attachment task.
We checked that model also on the cancer data:

![ae cancer graph, different wd](https://github.com/IdoSpringer/TCR-PEP-Classification/blob/master/ae_tr_c_reg.png)

The results are still low but better than before.

### TCRGP Paper
We have found some papers that uses machine-learning methods for predicting TCR-Peptide binding.
One of them is [TCRGP: Determining epitope specificity of T cell receptors](https://www.biorxiv.org/content/biorxiv/early/2019/02/06/542332.full.pdf) 
The authors used Gaussian process methods.
The paper's code and data are available in [this](https://github.com/emmijokinen/TCRGP) repository. 
We would like to use its data so we can compare our model to theirs.

#### Leave-One-Out Cross Validation
In the TCRGP paper, the data contains only few unique peptides.
The authors used leave one subject out cross validation to evaluate their model performance on every peptide.
For every peptide, the model is trained with all data excluding that peptide,
and the model is evaluated with all TCRs of this specific peptide.

This type of evaluation is not suitable for our model,
because we work with various TCR-peptide pairs, and not with specific peptides.
(Our train-test split is between all of the pairs, not between the peptides)

#### Negative sampling
In the TCRGP paper, the authors used background TCRs that do not attach the peptides for negative examples.
Those TCRs origin is from another dataset, so it might make the task of TCR-peptide attachment prediction easier.
What we do is creating negative examples from our data by taking TCR-peptide pairs that do not bind each other.
We checked our model performance when we replace our negative examples (in domain) with the background TCRs negative examples
(out of domain). We take random peptides and matching them random background TCRs, 
assuming those TCRs do not bind the peptides.
We compared both negative sampling methods for our 2 models (autoencoder based and LSTM based),
with all datasets (Weizmann, Shugay, cancer):

![ae weizmann gp/not](https://github.com/IdoSpringer/TCR-PEP-Classification/blob/master/tcrgp_ae_w.png)
![ae shugay gp/not](https://github.com/IdoSpringer/TCR-PEP-Classification/blob/master/tcrgp_ae_s.png)
![ae cancer gp/not](https://github.com/IdoSpringer/TCR-PEP-Classification/blob/master/tcrgp_ae_c.png)
![lstm weizmann gp/not](https://github.com/IdoSpringer/TCR-PEP-Classification/blob/master/tcrgp_lstm_w.png)
![lstm shugay gp/not](https://github.com/IdoSpringer/TCR-PEP-Classification/blob/master/tcrgp_lstm_s.png)
![lstm cancer gp/not](https://github.com/IdoSpringer/TCR-PEP-Classification/blob/master/tcrgp_lstm_c.png)


Model | Dataset | Negative Sampling Method | Train AUC | Test AUC
--- | --- | --- | --- | ---
Autoencoder | Weizmann | in domain | 0.999| 0.855
Autoencoder | Weizmann | out of domain | 0.999| 0.973
Autoencoder | Shugay | in domain | 0.999| 0.833
Autoencoder | Shugay | out of domain | 0.999| 0.993
Autoencoder | cancer | in domain | 0.999| 0.566
Autoencoder | cancer | out of domain | 1.0| 0.773
LSTM | Weizmann | in domain | 0.995| 0.850
LSTM | Weizmann | out of domain | 0.999| 0.955
LSTM | Shugay | in domain | 0.964| 0.796
LSTM | Shugay | out of domain | 0.999| 0.980
LSTM | cancer | in domain | 0.992| 0.638
LSTM | cancer | out of domain | 0.999| 0.664


### NetTCR paper
important note - I think its data has different sequencing
might be a problem for the trained ae model
