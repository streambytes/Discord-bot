## Discord-Bot
<p align="center"><img src="icons/logo_bot.png" width="300"></p>
This is just another ML Discord-bot. Coded in nodejs using keras and tensorflow modules.
Its purpose is to manage a whole discord chat, analyzing messages, trying to find and remove offensive/hate speech from the channel.

This bot is meant to be used for experimenting stuffs. Streambytes is using this project to learn JS/nodejs and a bit of Machine Learning.
The bot is not well optimized and totally not secured yet, feel free to use it on your own risk.

### Datasets
You need to downlaod and add to dataset/ directory **glove.6B.\*.txt** embedding from [here](https://nlp.stanford.edu/projects/glove/).
You also need to download and add to dataset/ directory **cleaned_tet_data_v1.csv** file from [here](https://github.com/idontflow/OLID/tree/master/own).

### Optimization
#### TODO
1. <del>Remove stopwords from vocabular;</del>
2. <del>Remove zeros from the sequences before the padding process (text_to_sequence function) : it helps to keep position independence;</del>
3. <del>Remove or manage special characters;</del>