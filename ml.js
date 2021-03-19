const seedrandom = require('seedrandom');
const parse = require('csv-parse/lib/sync');
const tfjs = require('@tensorflow/tfjs-node');
const fs = require('fs').promises;
const { TLSSocket } = require('tls');

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
        obj['y_train'].push(el['class']);
    });

    test_els.forEach((el) => {
        obj['X_test'].push(el['tweet']);
        obj['y_test'].push(el['class']);
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

    let maxlen = 50;

    //Glove embedding
    let file = (await fs.readFile("./glove.6B.50d.txt")).toString().split("\n");

    let embeddings = {};
    
    for (let i = 0; i < file.length; i++) { 
        let word = file[i].split(" ");
        let x = word.slice(1, );
        x = x.map(a => parseFloat(a));
        embeddings[word[0]] = x;
    }

    embedding_matrix = [];

    for (let o = 1; o < tk.vocab.length; o++) {
        if (embeddings[tk.vocab[o]] && tk.vocab[o].length > 0) {                  
            embedding_matrix.push(embeddings[tk.vocab[o]]);
        } else {
            let arr = new Array(maxlen).fill(0);
            embedding_matrix.push(arr);
        }
    }

    
  
    // trasform texts to sequences
    
    
    let X_train = tk.texts_to_sequences(data.X_train, padding=maxlen);
    let X_test = tk.texts_to_sequences(data.X_test, padding=maxlen);

    let embedding_dim = 50;
    let vocab_size = tk.vocab.length;

    
    // build the model
    let model = tfjs.sequential();
    model.add(tfjs.layers.embedding({inputDim :vocab_size, outputDim : embedding_dim, inputLength : maxlen, weights : embedding_matrix, trainable : false}));
    model.add(tfjs.layers.globalMaxPooling1d());
    model.add(tfjs.layers.dense({units : 10, activation : 'relu', useBias : true}));
    model.add(tfjs.layers.dense({units : 1, activation : 'sigmoid', useBias : true}));

    model.compile({optimizer : 'adam', loss : tfjs.losses.sigmoidCrossEntropy, metrics :['accuracy']});

    // train the model
    let epochs = 50;
    
    let history = await model.fit(tfjs.tensor(X_train), tfjs.tensor(data.y_train), {
        epochs : epochs,
        validationData : (tfjs.tensor(X_test), tfjs.tensor(data.y_test)), 
        verbose : 0
    });

    console.log(history.history);
    
    // check our results
    time = Math.floor(Date.now() / 1000) - time;

    console.log(`Process ended in ${time} seconds`);
})();