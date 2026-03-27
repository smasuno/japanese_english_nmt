# Japanese-English Neural Machine Translation (NMT)

This project is an extension of the Cherokee-to-English translation project from the Stanford Online NLP course (XCS224N).

The model architecture features a bi-directional LSTM encoder, and a uni-directional LSTM decoder with multiplicative attention.

![Model Architecture](flowchart.png)

From the original code base developed by Stanford, a few changes have been made:
* Instead of Cherokee to English, the model is trained to translate Japanese to English (corpus data from OPUS) 
* Models were trained with varying vocabulary sizes (21K, 30K, 40K, 50K) -- I was mainly interested in seeing if changing the vocab size will dramatically improve performance for Japanese (which is a language with many characters)
* Computations are performed using Google's Vertex AI training jobs
* The original code had preset hyperparameters, but I implemented a tuning sweep to trial learning rate, dropout rate, and batch size using Vertex AI hyperparameter tuning jobs










