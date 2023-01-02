# Disease-auxiliary-diagnosis-system-based-on-gut-microbes-and-machine-learning-methods
## structure of this project
### back end  
    server.py: the server handles request from front end and access database to store and use data, back end server was implemented by Flask framework. Run this file to start up server at root directory of this project

    basic_classifier.py: containing 4 classifiers( SVM, RF, DNN, 1D-CNN)  

    disease_diagnosizer.py: generate diease treatment, use muti-processes technology to speed up.  

    graph_classifier.py: train GNN and use GNN to diagnosize  

    models.py: models for GNN

    utils.py: utils for loading dataset, preprocessing data and so on. 
### front end 

    use command npm install to install necessary dependencies at this (./front end) directory
    use command npm run dev to run front server and use browser to access this project on localhost:8080  

    front end was implement based on Vue framework, source codes are in src directory. I used Vue framework to encapsulate html css and javascript to a component. Each component consists of html css and javascript codes. So that I can call these components at a main html file to reuse them.
### models   
    best GNN models on each dataset. 
### data   
    datasets
## Background
&emsp;According to existing academic analyses, certain diseases are closely related to the abundance of microbial communities residing in the human body, and more than 80% of the microorganisms in the human body reside in the gut, which is the largest microbial host site in the human body, and the microbiome in the gut has a direct impact on human health and diseases.
Therefore, this study applies machine learning methods to the analysis of microbial-disease associations, and builds a diagnostic aid for physicians based on this study.
## Machine learning-based approach to analyze the association between gut microbes and diseases 
### source of data
&emsp;The study included 2424 samples from eight studies with six diseases and sequencing data from the birdshot method of macrogenomics. 118 diseased and 114 healthy samples were included in the Cirrhosis dataset; 48 patients with rectal cancer and 73 healthy samples with small rectal tumors as control samples in the Colorectal dataset; In the IBD dataset, the disease sample consisted of 25 cases of ulcerative colitis and Crohn's disease, with a total of 85 healthy individuals; the Obesiy dataset contained 164 obese patients and 89 wasted individuals; the T2D dataset contained 170 patients with type 2 diabetes and 174 healthy individuals from China, and the WT2D dataset sample sampled European samples in a similar manner, with a total of 53 patient samples and 43 control samples.
### Methods
&emsp;I have elaborated 5 models and applied them to the mentioned 6 datasets. They were Support Vector Machine, Random Foerst, deep neural network, one-dimensional convolutional neural network and graph neural network. SVM and RF were implemented based on python package sklearn and the parameter settings were inspired by <a href = "https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004977">pasolli et al.</a>   

&emsp;Deep neural network consists of two hidden layers and the number of nodes in each hidden layer various with dataset, e.g. in Cirrhosis dataset two hidden layers contain 512 and 256 nodes respectively.   

&emsp;In one-dimensional convolutional neural network I applied 64 convolution kernels of shape (3,1) on n input sample of shape (m,1), n representing the total number of samples and m means the total number of microbes. Later I flattened the output tensor of convolutional layer and fed it into a fully connected layer consisting of 128 nodes.  

&emsp;The above two neural networks were implemented based on Tensorflow's Keras framework. The optimizer was Adma and loss function was Cross Entropy. 

&emsp;The last method in this study was Graph Neural Network. Initially, I tried to applied Graph Convolutional Network to this work, however GCN is based on spectral domain, which means it has to recalculate the Laplacian Matrix for the never seen new nodes. However, in this study the input graphs were microbial interaction networks, which was generated by the microbial abundance data of each sample. Due to the difference of microbial abundance between samples, the inputing graphs could consist of different nodes and edges, so that GCN didn't work well. Later I applied GraphSAGE and GAT into this study, however the later one didn't work well so that I chose GraphSAGE as final graph convolution operator. In terms of graph pooling operator, I applied DiffPool method to GNN model, it hierarchically clusters neighboring nodes into a cluster and fed the new graph into the next graph convolutional layer. This process repeats several times until the whole graph was clustered into one node, which represents the whole graph.

&emsp;This study then applied the above method to a real disease diagnosis scenario, where the samples in the OTU table uploaded by the operator could be diagnosed as sick or healthy, and if the samples were sick, the inverse causality algorithm could be used to give microbial abundance level treatment recommendations as a disease aid diagnosis system.
## Result
&emsp;In the screenshots folders, the results of 5 models were displayed. In addition, I designed feature selection algorithm to improve the performance of models. Firstly, I trained RF model on the trainst to calculate feature importance, later I retained the top k important features and drop the others, k was selected in {5，10，20，30，40，50，60，70，80，90，100，125，150，175，200}. The result improved by this algorithm was displayed in screenshots folder as well. 

&emsp;

Finally, the user interface of the system was displayed in the screenshots folder. The system contains disease diagnosis module and disease treatment module and history module. 

## Disease auxiliary diagnosis system with separate front and back ends
### Front end
&emsp;At front end there are four modules, which are model evaluation, auxiliary diagnosis, history display and treatment recommendations.
 Model evaluation: In this module, user can evaluate different machine learning models. To evaluate a model, user have to pass necessary parameters, like the epoch and batchsize. For different models, different parameters are passed.
 Auxiliary diagnosis: In this module, system will choose the best model with highest accuray to infer which disease the patient suffer from. User need to pass a csv file with information of input samples.
 History display: User can check history of two above modules. At the page of history of auxiliary diagnosis, user can also see treatment recommendation generated by the system.
 Treatment recommendations: Here the generated treatment for patients are displayed.
### Back end
The server handles Json form request from front end and call corresponding model to infer or evaluate or access database and then send result back to front end in Json form, back end server was implemented by Flask framework.

## Summary
&emsp;This is an abstract of my graduation dissertation, the full version was written in Chinese and had been attached to this page.

&emsp;The whole system was tested and checked by my tutor to make sure every module can work correctly. Finally, this project was rated 89 out of 100.

