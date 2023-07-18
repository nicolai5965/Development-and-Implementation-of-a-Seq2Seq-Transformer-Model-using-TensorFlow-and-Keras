# Transformer Model for Text Generation
This repository contains code for building, training, and evaluating a Transformer neural network for text generation. The model is implemented in TensorFlow and Keras.

## Model Architecture
The Transformer model consists of an encoder and decoder. The encoder is made up of a stack of identical encoder layers. Each encoder layer has two sub-layers:

* Multi-head self attention layer
* Positionwise feedforward network
The decoder is also made up of a stack of identical decoder layers. Each decoder layer has three sub-layers:

* Masked multi-head self attention layer
* Multi-head attention layer (attention over encoder outputs)
* Positionwise feedforward network
There are also embedding layers and positional encoding applied to the input and output sequences.

The model is optimized using the Adam optimizer and a custom learning rate schedule. The loss function is sparse categorical crossentropy.

## Usage
The main entry point is the ModelManager class which handles preprocessing the data, creating the model, training, evaluation, and saving.

To use:

* Configure the preprocessing in preprocessing_config
* Configure the model architecture in model_config
* Instantiate a ModelManager object
* Call preprocess() to preprocess the data
* Call create_model() to create the model
* Call train() to train the model
* Call save() to save the trained model
The training history and metrics can be visualized by passing the history object to the HistoryPlotter class.

## Data
The model is currently trained on a dataset of texts loaded from a CSV file. The data preprocessor handles loading the data, cleaning the texts, tokenizing, splitting into train/validation sets, padding sequences, etc.

The preprocessing_config dictionary contains the settings for preprocessing.

## MultiHeadAttention
The MultiHeadAttention layer is a key component of the Transformer encoder and decoder. It allows the model to jointly attend to information from different representation subspaces at different positions.

MultiHeadAttention applies self-attention multiple times in parallel on the same input sequence. This allows the model to learn different contextual relationships between words in the sequence.

The independent self-attention layers, or "heads", give the model more expressive power to capture different types of connections. Their outputs are then concatenated and linearly transformed into the final values.

Using MultiHeadAttention instead of regular self-attention allows the Transformer model to have greater ability to learn complex relationships in the sequences for better encoding and decoding.

## Encoder & Decoder
The encoder and decoder are the core components of the Transformer model architecture.

### Encoder
The encoder processes the input sequence and generates an encoded representation of the sequence. This encoding captures important contextual information from the input.

The stacked encoder layers allow the model to learn complex relationships in the input by repeatedly applying self-attention and feedforward layers.

### Decoder
The decoder uses the output from the encoder along with its own self-attention layers to generate the target sequence.

The self-attention layers allow the decoder to focus on relevant parts of the input sequence while generating outputs.

The attention over the encoder output gives the decoder access to the full input context for generating accurate outputs.

The encoder and decoder allow the Transformer model to effectively process sequential data while capturing long-range dependencies, which is a key advantage over recurrent neural networks.

## Training
The model is trained for a specified number of epochs, with early stopping if the validation accuracy does not improve after some patience epochs.

The model_config dictionary contains the hyperparameters for model training.

The train() method handles training the model and provides the training history object.

## Evaluation
Model accuracy and loss are monitored during training using the validation dataset.

The training history can be visualized using the HistoryPlotter class.

## Saving
The trained model and tokenizer can be saved to save path by calling save(). The model architecture and weights are saved in an folder which will be generated. The tokenizer is saved separately as a pickle file.

## Requirements
The main requirements are TensorFlow, Keras, and scikit-learn. See the import statements for the full list of dependencies.

The model was developed with Python 3.7 and TensorFlow 2.3.

## Loading and Using the Trained Model
The load_model.py file contains code for loading the saved Transformer model and tokenizer.

The trained model is loaded from disk by calling tf.keras.models.load_model() and passing the model path. Some custom objects need to be provided for loading the model correctly:
```python
model_saved = tf.keras.models.load_model(
    model_path,
    custom_objects={
        "CustomSchedule": CustomSchedule,
        "compute": loss_function.compute,
        "MultiHeadAttention": MultiHeadAttention
    }
)
```
The tokenizer is loaded separately by unpickling it from the saved pickle file:
tokenizer = load_tokenizer(tokenizer_path)

The loaded model can be used to make predictions by passing in input sequences. For example, the StochasticBeamSearch class does beam search decoding to generate text conditioned on a starting sentence.

Key aspects of using the saved model:

* Loading model architecture and weights
* Loading tokenizer from pickle file
* Providing custom objects for custom classes
* Making predictions on new data by passing inputs through model
* Decoding predicted sequences into text
This allows the trained Transformer model to be loaded and deployed for text generation after training is complete.

## References
The Transformer model implementation is based on the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
 by Vaswani et al. (2017).
