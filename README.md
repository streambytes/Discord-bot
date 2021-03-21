## Discord-Bot
<p align="center"><img src="icons/logo_bot.png" width="300"></p>
This is just another ML Discord-bot. Coded in nodejs using keras and tensorflow modules.
Its purpose is to manage a whole discord chat, analyzing messages, trying to find and remove offensive/hate speech from the channel.

This bot is meant to be used for experimenting stuffs. Streambytes is using this project to learn JS/nodejs and a bit of Machine Learning.
The bot is not optimized and totally not secured yet, feel free to use it on your own risk.

### Datasets
You need to downlaod and add to dataset/ directory **glove.6B.\*.txt** embedding.

### Optimization
#### TODO
1. Remove stopwords from vocabular;
2. Remove zeros from the sequences before the padding process (text_to_sequence function) : it helps to keep position independence;
3. Remove or manage special characters;