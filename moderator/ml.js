const { Console } = require('console');
const parse = require('csv-parse/lib/sync');
const tfjs = require('@tensorflow/tfjs');
const keras = require('keras-js');
const fs = require('fs').promises;

class tokenizer {
    constructor(dataset) {
        this.vocab = [];
        this.buildVocab(dataset)
    }

    // build vocabulary
    buildVocab(dataset) {
        dataset.forEach(el => {
            let words = el['tweet'].replace("\n", " ").split(" ");
            words.forEach(word => {
                word = word.toLowerCase();
                if (this.vocab.find(el => el == word) == undefined) {
                    this.vocab.push(word);
                }
            });
        });
    }

    // text to sequence
    text_to_sequence(text, maxlen=250) {
        text = text.toLowerCase();
        let words = text.replace("\n", " ").split(" ");
        let seq = []
        words.forEach(word => {
            let idx = this.vocab.findIndex(el => el == word);
            seq.push((idx > 0 ? idx : 0));
        });

        if (seq.length > maxlen) seq.substr(0, maxlen);
        let padding = maxlen - seq.length;
        for (let i = 0; i < padding; i++) {
            seq.push(0);
        }

        return seq;
    }

    // texts to sequences
    texts_to_sequences(texts, maxlen=250) {
        let seq = []
        texts.forEach(text => {
            let x = this.text_to_sequence(text);
            if (x.length > maxlen) x.substr(0, maxlen);
            let padding = maxlen - x.length;

            for (let i = 0; i < padding; i++) {
                x.push(0);
            }

            seq.push(x);
        });

        return seq;
    }
}

(async function () {
    const fileContent = await fs.readFile('dataset/labeled_data.csv');
    const dataset = parse(fileContent, {columns : true});

    for (let i = 0; i < dataset.length; i++) {
        let line = {
            class : (dataset[i]['class'] == 2) ? 1 : 0,
            tweet : dataset[i]['tweet'].substr(1,dataset[i]['tweet'].length - 1)
        }
        dataset[i] = line;
    }

    var tk = new tokenizer(dataset);

    s = "This is just a test!";

    var seq = tk.text_to_sequence(s, 10);

    console.log(seq);
})();
