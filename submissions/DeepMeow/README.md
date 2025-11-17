# Deep Meow : Labeling the emission context of cat meows

## Context 

The goal of this deep learning model is to identify the context in which a cat meows. For this purpose, we use a dataset of 440 meows from 21 cats (two breeds: Maine Coon and European Shorthair), recorded under three eliciting contexts: Brushing, Isolation in an unfamiliar environment, and Waiting for food. Each stimulus lasted up to 5 minutes and recordings were collected in naturalistic settings with standardized handling to minimize stress. This dataset is provided by the university of Milan at this address : [CatMeows: A Publicly-Available Dataset of Cat Vocalizations](https://zenodo.org/records/4008297). We also based our research on this paper : [Audio Deep Learning : Sound Classification](https://medium.com/data-science/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5)

Audio labeling is a method widely used in machine learning, agronomy, sociology, music, etc. We therefore propose to present an approach for processing audio signals and labeling these signals.

## Dataset
Our example will be cat audio labeling. The dataset contains 440 audio files in .WAV format. Each audio file is a cat meow. The cats were recorded in different situations (being brushed, waiting for food, isolated in an unfamiliar space). The goal here is to determine the emission context of a meow.

## Topics covered
- Preprocessing of audio data
- Feature extraction via a neural network and convolutional layers

*Project by Augustin Antier, Lena Causeur, Louis Prusiewicz-Blondin
for the Third edition of the Autumn School on "Machine Learning in the Life Sciences" by the Institut Agro Rennes-Angers*
