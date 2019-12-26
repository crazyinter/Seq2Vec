# Gene2Vec
Version 1.1 <br>
Authors: Yan Miao, Fu Liu, Yun Liu <br>
Maintainer: Yan Miao miaoyan17@mails.jlu.edu.cn 

# Description
  This package provides a novel codon embedding method for identification of viral contigs from metagenomic data in a fasta file. The method has the ability to identify viral contigs with short length (<500bp) from metagenomic data.

  Gene2Vec format generally tries to map a nucleotide sequence using a codon dictionary to a vector. Before embedded into a vector, a nucleotide sequence is preprocessed into a string of codons with a stride of some bases. Take a look at the nucleotide sequence “ATAGCCTGAAAGC” for an example. It firstly converted into a format of gene sentence “ATA, TAG, AGC, GCC, CCT, CTG, TGA, GAA, AAA, AAG, AGC” with a stride of one base (see `train_500bp.csv`), where each codon can be regarded as a word in a sentence. The list of all 64 unique codons composes the whole complete dictionary, which is then used to codon embedding. So, a dictionary may look like – [‘AAA’, ‘AAT’, ‘AAG’, ‘AAC’, ‘ATA’, ‘ATT’, ……]. In the training step, the nucleotide sentences “ATAGCCTGAAAGCTTGGATT G” in the training dataset with one-hot encoded codons were firstly separated into a training set for a skip-gram model with context window of 1. Next, the input codons in the formed training set were continuously inputted to the skip-gram model, and then multiplied by the weights between input layer and hidden layer into the hidden activations, which latter got multiplied by the hidden-output weights to calculate the final outputs. Then the costs calculated by the negative log likelihood between final outputs and targets produced in the training set were back-propagated to learn the weights. Finally, the weights between the hidden layer and the output layer were taken as the codon vector representations of the codons, namely the embedding matrix.
  The prediction model is a attention based LSTM neural network that learns the high-level features of each contig to distinguish virus from host sequences. The model was trained using equal number of known viral and host sequences from NCBI RefSeq database. For a query sequence shorter than 500bp, it should be first zero-padded up to 500bp. Then the sequence is predicted by the RNN model trained with previously known sequences.

# Dependencies
To utilize Gene2Vec, Python packages "Keras", "tflearn", "sklearn", "numpy" and "matplotlib" are needed to be previously installed. Some other packages that makes sure the code can be run correctly such as "os", "ast", etc.

In convenience, download Anaconda from https://repo.anaconda.com/archive/, which contains most of needed packages. If there still some special packages that are missed when running, you can use "pip install" to install the packages. 

To insatll tensorflow, start "cmd.exe" and enter <br>
```
pip install tensorflow
```
Our codes were all edited by Python 3.6.5 with TensorFlow 1.3.0.

# Usage
It is simple to use Gene2Vec for users' database. <br>
Before training and testing, the query contigs should be preprocessed to an available format using: <br>
`f=open('your_data.fasta','r') <br>
g=open('preprocssed_data.fasta','a') <br>
lines=f.readlines()<br>
contex=3 <br>
for line in lines:<br>
    l=len(line)-1<br>
    for i in range(0,(l-contex+1)):<br>
        a=line[i:i+contex]<br>
        x=str(a).replace("AAA","1").replace("TTT","2").replace("GAA","3").replace("AAG","4").replace("AAT","5").\<br>
        replace("ATT","6").replace("CAA","7").replace("TGA","8").replace("TTC","9").replace("AGA","10").\<br>
        replace("GAT","11").replace("AAC","12").replace("TAA","13").replace("TTA","14").replace("TCA","15").\<br>
        replace("TAT","16").replace("ATG","17").replace("TGG","18").replace("ATC","19").replace("TTG","20").\<br>
        replace("ATA","21").replace("GTT","22").replace("CTG","23").replace("CTT","24").replace("ACA","25").\<br>
        replace("CAG","26").replace("CGA","27").replace("GGT","28").replace("GGC","29").replace("GCA","30").\<br>
        replace("CAT","31").replace("GCG","32").replace("CGC","33").replace("GCT","34").replace("TCT","35").\<br>
        replace("TCG","36").replace("ACC","37").replace("AGC","38").replace("CGG","39").replace("GAC","40").\<br>
        replace("CCG","41").replace("CCA","42").replace("TGC","43").replace("ACG","44").replace("GGA","45").\<br>
        replace("TGT","46").replace("ACT","47").replace("TAC","48").replace("AGT","49").replace("GCC","50").\<br>
        replace("GAG","51").replace("GTA","52").replace("GTG","53").replace("AGG","54").replace("CGT","55").\<br>
        replace("CAC","56").replace("GTC","57").replace("TCC","58").replace("CCT","59").replace("CTC","60").\<br>
        replace("CTA","61").replace("GGG","62").replace("TAG","63").replace("CCC","64").replace(",\n","\n")<br>
        if i<(1-contex+1):<br>
            g.write(str(x)+",")<br>
        else:<br>
            g.write(str(x))<br>
    g.write("\n")<br>
f.close()<br>
g.close()`
There are two ways for users to train the model using `Gene2Vec_train&test.py`.
* Using our original training database (containing 4500 viral sequences and 4500 host sequences of length 500bp) `"train_500bp.csv"`. If you would like to test query contigs with length 300bp, you can use our train dataset `"train_300bp.csv"`. <br>
Users can retrain the model first, and then test the query contigs. When training, you can make some changes to the hyperparameters toget a better performance.
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
