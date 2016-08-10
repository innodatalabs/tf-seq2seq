8/8/2016

This is an adaptation of translation seq2seq model from `tensorflow.models.rnn.transalte`.

It is a character-level model that is trained as a Sequence Autoencoder to learn bibliography records.

Due to GPU memory limitations, biggest bucket is 200 characters. All records longer than that are ignored (not used during the training).

After about 3 days of training it learned to memorize the input character sequence and reproduce it.

3 levels, 256 units each, vocabulary size 150, batch size 25 (limited by memory)

Did it cheat? By using attention it may have learned to copy input to output using attention mechanism.

Next steps:

1. Modify model to make a prediction model, not autoencoder
2. Use learned RNN weights as pre-trained weights for supervised learning
3. Create a dataset for supervised training of reference tagger

Problems:

* Bad names of summary graph variables, and model's `global_step` variable (`Variable`, `Variable_1`, `Variable_2`)
* Need summary graph to correctly read pre-trained RNN weights (to reproduce the order of var creation in the train graph, see auto-assigned names above).