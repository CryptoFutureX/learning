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
    <style type="text/css">
        .tg  {border-collapse:collapse;border-spacing:0;}
        .tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
        overflow:hidden;padding:10px 5px;word-break:normal;}
        .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
        font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
        .tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
    </style>
    Being compatible with classes from _scikit-learn_, building models with spaCy becomes easy. Following were the ML algorithms used with TF-IDF vectorized data: <br>
    <br> <space><table class="tg">
        <thead>
            <tr>
                <th class="tg-c3ow">ML Algorithm</th>
                <th class="tg-c3ow">Train acc.</th>
                <th class="tg-c3ow">Test acc.</th>
            </tr>
        </thead>
    <tbody>
    <tr>
        <td class="tg-c3ow">LinearSVC</td>
        <td class="tg-c3ow">0.830495</td>
        <td class="tg-c3ow">0.762606</td>
    </tr>
    <tr>
        <td class="tg-c3ow">SGDClassifier</td>
        <td class="tg-c3ow">0.758569</td>
        <td class="tg-c3ow">0.755504</td>
    </tr>
    <tr>
        <td class="tg-c3ow">DecisionTree</td>
        <td class="tg-c3ow">0.6212434</td>
        <td class="tg-c3ow">0.6210907</td>
    </tr>
    <tr>
        <td class="tg-c3ow">MultinomialNB</td>
        <td class="tg-c3ow">0.7971801</td>
        <td class="tg-c3ow">0.7562758</td>
    </tr>
    </tbody>
    </table>


