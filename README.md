# CryptoFutureX

## Team members
* Pankaj Khushalani
* Tushar Bauskar
* Utsav Khatu
* Nikita Deokar

## Mentors
* Sanket Barhate
* Dhwani Panjwani
* Suyash Gatkal

## Description
A web application to view futures of various cryptocurrencies. <br>
The predictions are made by using the trends of prices of cryptocurrencies and correlating them with sentiment analysis. Multivariate time series analysis is used to predict the trends and the percentage change in predicted trends is obtained by sentiment analysis of Twitter feed.

* On the ReactJS frontend, user can view candlebar plot of prices of selected cryptocurreny over the past 7 days.

* The futures of the 8th day are predicted and shown in the same plot. The prices can be viewed by toggling the selection in USD, INR, EUR, and JYP.

* Percentage change in predicted price by sentiment is indicated by a speedometer below the graph.

GitHub organization link: [CryptoFutureX](https://github.com/CryptoFutureX)

## Installation

* For the frontend, clone the repository
```bash
    git clone https://github.com/CryptoFutureX/frontend.git
    cd frontend/client
    npm install
```
Get your API key from [CoinLayer](https://coinlayer.com/#:~:text=The%20coinlayer%20API%20was%20built,as%20low%20as%2020%20milliseconds.). Create ```key.js ``` and add your API key 
```bash
    API_KEY = <your-API-key>
```
Then start the server on port 3000 by
```bash
    npm start
```
* For the backend, clone the repository.
```bash
    git clone https://github.com/CryptoFutureX/backend.git
    cd backend
```
Create a virtual environment using venv (for Windows)
```python
     python3 -m venv <your-virtual-env-name>
    .\env\Scripts\activate
```
Run the backend server on port 4000 by
```bash
    python3 manage.py runserver
```
Since the packages to install require several C++ dependencies and PyTorch along with TensorFlow are heavy libraries, an alternative is using Google Colab.    
Open a Google Colaboratory notebook and mount your Google Drive.
```python
    from google.colab import drive
    drive.mount('/content/drive/MyDrive')
```
Now, clone the repository in your Google Drive and install any dependencies which might not already be installed on Google Colab.
```bash
    !git clone https://github.com/CryptoFutureX/backend.git
    %cd backend
    !pip install -r requirements.txt
```
Now expose a port to Google Colab in order to run the Django app
```python
    from google.colab.output import eval_js
    print(eval_js("google.colab.kernel.proxyPort(4000)"))
```
In ```settings.py```
```python
    ALLOWED_HOSTS = ['colab.research.google.com']
```
Finally, run the server on port 4000
```bash
    !python manage.py runserver 4000
```


## Technology stack

1. Django    
<space>A Python framework for web application development. For the backend, Django with Django Rest Framework was used to supply data to the ReactJS frontend.

2. ReactJS    
<space>A JavaScript library for building user interfaces. For data visualization, Apex charts were used along with React components.

3. TensorFlow    
<sapce>A Python library for machine learning. Keras API was used along with TensorFlow to train the CNN and LSTM models.

4. APIs Used
    - Twint - An advanced Twitter scraping tool written in Python that allows for scraping Tweets from Twitter profiles without using Twitter's API.

    - CryptoCompare - It is a Python library for fetching prices both latest as well as historical, coin listing across various exchanges and also trading using the API. 

5. Dataset
    - [Kaggle dataset](https://www.kaggle.com/paul92s/bitcoin-tweets-14m) containing 1.4 million BitCoin tweets collected in 2 weeks of July 2018 with their sentiment. This was used for the baseline model. However, the tweets had emojis with their raw UTF-8 representations, making it tough to clean the data. 

    - [Sentiment140 dataset](https://drive.google.com/drive/folders/1lGNfH7b5G2Qo2e6EUBq9tncV59LZM1vM) containing tweets of 140 categories with their respective sentiment. This is a generalized data for Twitter sentiment analysis created by Stanford and was used to train our final model.  

## Applications

Barring a stock exchange of cryptocurrencies, there is considerable risk in investment in any cryptocurrency. With CryptoFutureX, an investor can visualize the predicted change in prices and make the right investment. 
 
## Skills Learnt

> We explored several technologies pertaining to the field of Natural Language Processing (NLP). Team members explored alternatives to few technologies while some were learnt by all.

### Common technologies 

1. NLTK (Natural Language Toolkit)    
<space>It is a prevalent NLP library in Python which has rich sources of human language data. We learnt concepts such as tokenization, lemmatization, stemming, POS tagging, BOW and n-grams using NLTK. <br>
However, with unavailibilty of vector embeddings and neural network models, other alternatives were explored. 
References:
    * NLTK tutorials [here](https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL)
    *  NLTK [documentation](https://www.nltk.org/)

2. PyTorch    
<space>It is  is an optimized tensor library for deep learning using GPUs and CPUs. <br> We chose this library as it has rich implementations in NLP such as **transformers** library by Hugging Face. It provides thousands of pre-trained models for NLP such as BERT, RoBERT, DistilBERT. **BERT** was extensively used in training our model. <br> 
<space>References: 
    * Learning PyTorch [here](https://www.youtube.com/watch?v=vo_fUOk-IKk&list=PLWKjhJtqVAbm3T2Eq1_KgloC7ogdXxdRa)
    * Hugging Face [documentation](https://huggingface.co/transformers/pretrained_models.html) for pretrained models

### Technologies learnt by team members

* Pankaj Khushalani
    <br> **spaCy** - Though NLTK is great for a beginner, it didn't provide us with the necessary tools to support word vector embeddings and to build our own model. Hence, spaCy, an advanced NLP library was used. <br>
    Being compatible with classes from _scikit-learn_, building models with spaCy becomes easy. Following were the ML algorithms used with TF-IDF vectorized data: <br>
    <br> <space><table>
        <thead>
            <tr>
                <th>ML Algorithm</th>
                <th>Train acc.</th>
                <th>Test acc.</th>
            </tr>
        </thead>
    <tbody>
    <tr>
        <td>LinearSVC</td>
        <td>0.830495</td>
        <td>0.762606</td>
    </tr>
    <tr>
        <td>SGDClassifier</td>
        <td>0.758569</td>
        <td>0.755504</td>
    </tr>
    <tr>
        <td>DecisionTree</td>
        <td>0.6212434</td>
        <td>0.6210907</td>
    </tr>
    <tr>
        <td>MultinomialNB</td>
        <td>0.7971801</td>
        <td>0.7562758</td>
    </tr>
    </tbody>
    </table>
    <br>
    Though spaCy along with support vector machines gave good accuracy, the CNN model gave more reliable and reproducible results.<br><br>

* Tushar Bauskar   <br>
**CNN model for sentiment analysis** <br>
As we were not able to use **BERT** model for sentiment analysis due to large data and less computational power,a CNN model was created. CNN can be used for sentiment analysis by creating vector embeddings of the text. The vectors were created using pretrained **GloVe Twitter Embedding 27B 100D** vector embeddings. It contains 27 billion words and their embeddings in 100 dimension vectors.
<br><br>
CNN Model

    ```python
    class CnnModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_size = 6002
            self.embed_size = 100
            self.num_filters = 100
            self.filter_sizes = [3, 4, 5]
            self.output_classes = 2
            self.dropout = 0.8

            # Embedding layer
            self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

            # Convolutional layer
            self.convs = nn.ModuleList([
                                    nn.Conv2d(
                                        in_channels=1, 
                                        out_channels=self.num_filters,
                                        kernel_size=(fs, self.embed_size)) 
                                    for fs in self.filter_sizes
            ])

            # Fully connected layer
            self.fc = nn.Linear(len(self.filter_sizes) * self.num_filters, self.output_classes)

            # Dropout layer
            self.dropout = nn.Dropout(self.dropout)

        def forward(self, text, text_lengths):
            embedded = self.embedding(text)
            embedded = embedded.unsqueeze(1)

            conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

            cat = self.dropout(torch.cat(pooled, dim=1))
            return self.fc(cat)
    ```
    
<br>We used the Sentiment140 dataset for training the model. <br>
Accuracy achieved:
Training accuracy | Validation accuracy |
--- | --- |
81.52% | 81.16% |
<br>

* Utsav Khatu    
    - D3JS: We initially decided to use D3Js for the frontend to create and plot prcice but soon realized that the D3Js is a low level library making it easier to customize according to the users but even the small changes takes a lot of effort to create and soon switched to ChartJs

    - ChartJS: Next we stumbled on ChartJs since we initially decided to plot a line plot so it was rather easy since ChartJs is Simple, clean and engaging HTML5 based JavaScript charts.

    - React ChartJS: Since ChartJs was mainly used for vanillaJs we needed a library which could configure and can be updated using React. This is where react-chart library was used.

    - React ApexCharts: Later we decided to give more investor oriented look to our webapp and decided to implement Candle Plot & Line Plot in our webapp.This is where ApexCharts Library came in handy.

    - Multivariate Time Series Modelling using LSTM: For price prediction we created Multivariate Time Series Modelling using LSTM in Tensorflow and Keras. Adjusting through different parameters we got an accuary of around ~81% for Bitcoin Data.
    <br><br>

* Nikita Deokar
    - [BERT](https://github.com/google-research/bert) 
stands for Bidirectional Encoder Representations from Transformers. It is pre-trained on a large corpus of unlabelled text including the entire Wikipedia and Book Corpus. Thus, BERT has proven to be a benchmark model for many NLP models.     
However, due to a huge dataset and low computational power a BERT model could not be trained. To address these problems, we explored DistilBERT and ALBERT variations of BERT.

    - [DistilBERT](https://github.com/huggingface/transformers/blob/master/docs/source/model_doc/distilbert.rst) has 40% lesser parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT’s performance.

    - [ALBERT](https://github.com/google-research/albert) is a lite representation of BERT that reduces energy consumption while increasing training speed compared to BERT model. 
<br>
However, none of the above deep learning transformer models could be trained with the freely available resources such as Google Colab or the free tier usage on GCP or AWS. Thus, results could not be produced from any of the above models. 

## Future Scope

* Automating the data collection and prediction pipeline on the backend.

* Improving the accuracy of the current model and exploring other NLP techniques to get better predictions.

* Adding live indicator of cryptocurrency prices along with the candlebar graph.

* Deployment of the data collection script to AWS Lambda and price prediction model to AWS SageMaker.

## Screenshots

