# CS272-NLP-Project

## Warning 
#### Please use encoded dataset and model in *rex* branch to avoid unmatched embedding index. These are located at [here](https://drive.google.com/open?id=1lB43R24xH6UdU_B4EZyOBQyw2rJVcUY3)
- classifier.hs
- enc_doc.pkl
- label.pkl
- word_index.pkl

### Teammates: ###   
Yueh Wu  
William Kuo  
James Quintero 


#### Reddit Dataset
https://github.com/linanqiu/reddit-dataset 

#### Create Encoder Data
```haskell =
python create_encoder_data.py
```

#### lstmEncoder usage instruction
- Install GloVe 6B token version at [Here](https://nlp.stanford.edu/projects/glove/) 
- Install *pickle* and *configargparse*
- Change *embedding_path* and *data_path* at config/config.txt
- Run lstmEncoder
```haskell =
python lstmEncoder.py --config config/config.txt
```

#### seq2seq usage instruction
- Same as *lstmEncoder* instruction, but it requires *lstmEnc_DNN.py* *seq2seq.py* and *countBLEU.py*
- Run lstmEncoder
```haskell =
 !python seq2seq.py --config config/config.txt
```
