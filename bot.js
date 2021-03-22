const Discord = require('discord.js');
const client = new Discord.Client();
const config = require('./config.json'); 
const { Sentiment_Analysis_Model } = require("./ml.js");

// Initialize model
var model = null;
var check = 0;
client.on('ready', () => {
    // train model
    (async () => {
        model = await new Sentiment_Analysis_Model();
        check = 1;        
    })();
});

client.on('message', msg => {
    // evaluate
    if (check == 1) {
        (async () => {
            let pred = await model.predict(msg.content);
            if (pred < 0.5) {
            msg.delete();
            msg.channel.send(`${msg.author} you are so rude!`); 
            }
        })();
    }
});

client.login(config.token);