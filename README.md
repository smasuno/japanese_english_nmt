# Japanese-English Neural Machine Translation (NMT)

This project is an extension of the Cherokee-to-English translation project from the Stanford Online NLP course (XCS224N).

The model architecture features a bi-directional LSTM encoder, and a uni-directional LSTM decoder with multiplicative attention.

![Model Architecture](flowchart.png)

From the original project, a few changes have been made:
* Instead of Cherokee to English, the model is trained to translate Japanese to English 
* Computations are performed using Google's Vertex AI training jobs
* A hyperparameter tuning job is implemented to trial learning rate, dropout rate, and batch size








