# NLP_project
Measure Text Fluency

We took on this problem as 3 class classification
0 : not fluent
1 : neutral
2 : fluent

we trained various modeled with oversampled data. LSTM with Glove embedding is saved for demo

files to download :

1. word2index.pkl      -   https://drive.google.com/file/d/15-fgBtIXiAKzbx1ePOUlwVS0K2fVxcyG/view?usp=share_link
2. model.h5            -   https://drive.google.com/file/d/1s7X5LdarLhFVKdmSSvbzKKHXI12lwAIU/view?usp=share_link
3. glove.6B.50d.txt    -   https://drive.google.com/file/d/1soSflZdGM1pgOPLL-gzb286656x7AMdT/view?usp=sharing
4. data.tsv            -   https://drive.google.com/file/d/1iqTlSqAJY3pAL5HOcNMtaFzXScL7YSmZ/view?usp=share_link

or download whole folder  https://drive.google.com/drive/folders/1Gut5wTk9a9P3kfHYKpBAa77TE-Nm-BbO?usp=sharing

place data and glove file in data folder. If no data folder create new folder.
place word2index and model file in src folder

run demo.py without any arguments
    python demo.py