const Discord = require('discord.js');
const client = new Discord.Client();
const config = require('./config.json'); 

// Initialize model
client.on('ready', () => {
    // train model
});

client.on('message', msg => {
    // evaluate
    let pred = 0;
    if (pred > 0.5) {
       msg.delete();
       msg.channel.send(`${msg.author} you are so rude!`); 
    }
});

client.login(config.token);
