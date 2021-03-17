const seedrandom = require('seedrandom');
const parse = require('csv-parse/lib/sync');
const tfjs = require('@tensorflow/tfjs');
const keras = require('keras-js');
const fs = require('fs').promises;

class tokenizer {
    constructor(dataset) {
        this.vocab = ["ksdafpdsngpdnagpkenpgneapngepng"];
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
            let x = this.text_to_sequence(text, maxlen);
            seq.push(x);
        });

        return seq;
    }
}

function train_test_split(dataset, seed="$$$", maxlen=1000, test_size=0.25) {
    // shuffle dataset
    dataset.sort(() => { 
        if (seed === "$$$")
            return;
        else {
            let rng = seedrandom(); 
            return 0.5 - (rng());
        }
    });

    let n_test = (maxlen * test_size);
    let n_train = maxlen - n_test;

    let train_els = dataset.slice(0, n_train);
    let test_els = dataset.slice(n_train, maxlen);

    let obj = {
        X_train : [],
        X_test : [],
        y_train : [],
        y_test : []    
    }

    train_els.forEach((el) => {
        obj['X_train'].push(el['tweet']);
        obj['y_train'].push(el['label']);
    });

    test_els.forEach((el) => {
        obj['X_test'].push(el['tweet']);
        obj['y_test'].push(el['label']);
    });

    return obj;
}

(async function () {
    let time = Math.floor(Date.now() / 1000);
    const fileContent = await fs.readFile('dataset/labeled_data.csv');
    const dataset = parse(fileContent, {columns : true});

    for (let i = 0; i < dataset.length; i++) {
        let line = {
            class : (dataset[i]['class'] == 2) ? 1 : 0,
            tweet : dataset[i]['tweet']
        }
        dataset[i] = line;
    }

    // split data
    let data = train_test_split(dataset);

    // init Tokenizer and build the vocabulary
    let tk = new tokenizer(dataset);

    // trasform texts to sequences
    let X_train = tk.texts_to_sequences(data.X_train, padding=50);
    let X_test = tk.texts_to_sequences(data.X_test, padding=50);

    time = Math.floor(Date.now() / 1000) - time;

    console.log(`Process ended in ${time} seconds`);
})();
