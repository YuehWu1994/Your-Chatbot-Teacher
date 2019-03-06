# CS272-NLP-Project

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


#### Run this repository on Googl Colab
```haskell =
!git clone -b [specific_branch] https://[username]:[password]@github.com/JamesQuintero/CS272-NLP-Project.git
cd CS272-NLP-Project/
!sh colab_require.sh
!python cnn_bilstm_training.py --config config/config_colab.txt
```
