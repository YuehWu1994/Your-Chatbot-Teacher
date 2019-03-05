# CS272-NLP-Project

### Teammates: ###   
Yueh Wu  
William Kuo  
James Quintero 


#### Reddit Dataset
https://github.com/linanqiu/reddit-dataset 


#### lstmEncoder usage instruction
- Install GloVe 6B token version at [Here](https://nlp.stanford.edu/projects/glove/) 
- Install *pickle* and *configargparse*
- Change *embedding_path* and *data_path* at config/config.txt
- Run lstmEncoder
```haskell =
python lstmEncoder.py --config config/config.txt
```