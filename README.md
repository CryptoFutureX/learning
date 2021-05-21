# CryptoFutureX

## Learning from project

> We explored several technologies pertaining to the field of Natural Language Processing (NLP). Team members explored alternatives to few technologies while some were learnt by all.

### Common technologies 

1. NLTK (Natural Language Toolkit) 
> 
2. PyTorch    
<space>It is  is an optimized tensor library for deep learning using GPUs and CPUs. <br> We chose this library as it has rich implementations in NLP such as **transformers** library by Hugging Face. It provides thousands of pre-trained models for NLP such as BERT, RoBERT, DistilBERT. **BERT** was extensively used in training our model. <br> 
<space>References: 
    * Learning PyTorch [here](https://www.youtube.com/watch?v=vo_fUOk-IKk&list=PLWKjhJtqVAbm3T2Eq1_KgloC7ogdXxdRa)
    * Hugging Face [documentation](https://huggingface.co/transformers/pretrained_models.html) for pretrained models

### Technologies learnt by team members
* Pankaj Khushalani
    <br> **spaCy** - Though NLTK is great for a beginner, it didn't provide us with necessary tools to support word vector embeddings and to build our own model. Hence, spaCy, an advanced NLP library was used. <br>
    Being compatible with classes from _scikit-learn_, building models with spaCy becomes easy. Following were the ML algorithms used with TF-IDF vectorized data: 
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


