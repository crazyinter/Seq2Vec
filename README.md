# Seq2Vec
Version 1.1 <br>
Authors: Yan Miao, Fu Liu, Yun Liu <br>
Maintainer: Yan Miao miaoyan17@mails.jlu.edu.cn 

# Description
  This package provides a novel codon embedding method for identification of viral contigs from metagenomic data in a fasta file. The method has the ability to identify viral contigs with short length (<500bp) from metagenomic data.

  Gene2Vec generally tries to map a nucleotide sequence using a codon dictionary to a vector. Before embedded into a vector, a nucleotide sequence is preprocessed into a string of codons with a stride of some bases. Take a look at the nucleotide sequence “ATAGCCTGAAAGC” for an example. It is firstly converted into a format of gene sentence “ATA, TAG, AGC, GCC, CCT, CTG, TGA, GAA, AAA, AAG, AGC” with a stride of one base (see `train_500bp.csv`), where each codon can be regarded as a word in a sentence. The list of all 64 unique codons composes the whole complete dictionary, which is then used to codon embedding. Thus, a dictionary may look like – [‘AAA’, ‘AAT’, ‘AAG’, ‘AAC’, ‘ATA’, ‘ATT’, ……]. In the training step, the nucleotide sentence “ATAGCCTGAAAGCTTGGATTG” in the training dataset with one-hot encoded codons are firstly separated into a training set for a skip-gram model with context window of 1. Next, the input codons in the formed training set are continuously inputted to the skip-gram model, and then multiplied by the weights between input layer and hidden layer into the hidden activations, which latter get multiplied by the hidden-output weights to calculate the final outputs. Then the costs calculated by the negative log likelihood between final outputs and targets produced in the training set are back-propagated to learn the weights. Finally, the weights between the hidden layer and the output layer are taken as the codon vector representations of the codons, namely the embedding matrix.
  The prediction model is an attention based LSTM neural network that learns the high-level features of each contig to distinguish virus from host. The model is trained using equal number of known viral and host sequences from NCBI RefSeq database. Then the sequence is predicted by the LSTM model trained with previously known sequences.

# Dependencies
To utilize Gene2Vec, Python packages "tflearn", "sklearn", "numpy" and "matplotlib" are needed to be previously installed. Some other packages are also needed to make sure the code running correctly such as "os", "ast", etc.

In convenience, you can download Anaconda from https://repo.anaconda.com/archive/, where contains most of needed packages. If there are still some special packages that are missed when running, you can use "pip install" to install the specific packages. 

To insatll tensorflow, start "cmd.exe" and enter <br>
```
pip install tensorflow
```
To insatll Keras, start "cmd.exe" and enter <br>
```
pip install Keras
```
Our codes are all edited by Python 3.6.5 with TensorFlow 1.3.0.

# Training the embedding matrix
Gene2Vec has supplied a trained embedding matrix in `embedding matrix.csv`. The training dataset is chosen as the whole RefSeq genomes. The NCBI accession number can be found in `NCBI accession numbers of the whole Refseq genomes.xlsx`. If you would like to train the embedding matrix by youself, jsut run `embedding.py`.

# Usage
It is simple to use Gene2Vec for users' database. <br>
Before training and testing, the query contigs should be preprocessed to an available format using `preprocessing.py`.
There are two ways for users to train the model using `Gene2Vec_train&test.py`.
* Using our original training database (containing 4500 viral sequences and 4500 host sequences of length 500bp) `"train_500bp.csv"`. If you would like to test query contigs with length of 300bp, you can use our training dataset `"train_300bp.csv"`. <br>
Users can retrain the model first, and then test the query contigs. When training, you can make some changes to the hyperparameters to get a better performance.
* Using users' own database in a ".csv" format. <br>
	* Firstly, chose a set of hyperparameters to train your dataset.
	* Secondly, train and refine your model using your dataset according to the performance on a related validation dataset.
	* Finally, utilize the fully trained model to identify query contigs. 
Note: Before training, set the path to where the database is located. 

To make a prediction, users' own query contigs should be edited into a ".csv" file, where every line contains a single query contig, the format of which should be set as discribed above. After training and testing, Gene2Vec will give a set of scores to each query contig, higher of which represents its classification result.

# Copyright and License Information
Copyright (C) 2019 Jilin University

Authors: Yan Miao, Fu Liu, Yun Liu

This program is freely available as Python at https://github.com/crazyinter/Gene2Vec.

Commercial users should contact Mr. Miao at miaoyan17@mails.jlu.edu.cn, copyright at Jilin University.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
