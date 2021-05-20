# This repository contains the learnings of the members of team CryptoFutureX

### Technologies learnt by team members
* Tushar Bauskar

**CNN model for sentiment analysis**
As we were not able to use **BERT** model for sentiment analysis due to large data and less computational power,a CNN model was created.CNN can be used for sentiment analysis by creating vector embeddings of the text. The vectors were created using pretrained **GloVe twitter embedding 27B 100d** vector embeddings. It contains 27 billion words and their embeddings in 100 dimension vectors.

![picture alt](https://peltarion.com/static/word_embedding_pa1.png "Embedding words to vectors")

CNN model

```
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
We used the Sentiment 140 dataset for training the model.

Accuracy achieved:
Training accuracy | Validation accuracy |
--- | --- |
81.52% | 81.16% |
