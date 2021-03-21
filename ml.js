const parse = require('csv-parse/lib/sync');
const fs = require('fs').promises;
const seedrandom = require('seedrandom');
const tfjs = require('@tensorflow/tfjs-node');

class Tokenizer {
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
                if ((this.vocab.find(el => el == word) == undefined)) {
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
        
        if (seq.length > maxlen) seq = seq.slice(0, maxlen);
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

class Sentiment_Analysis_Model {
    constructor() {
        return (async () => { 
            await this.initializeModel();
            return this;
        })();
    }

    // init function
    async initializeModel() {
        const fileContent = await fs.readFile('dataset/imdb_labelled.txt');
        // const dataset = parse(fileContent, {columns : true});
    
        let lines = fileContent.toString().split("\n");
        let dataset = [];
    
        // Read from dataset
        lines.forEach(line => {
            let tmp = line.split("\t");
            let obj = {
                tweet : tmp[0],
                class : parseInt(tmp[1])
            }
            dataset.push(obj);
        });
    
        // split data
        let data = this.train_test_split(dataset);
    
        // init Tokenizer and build the vocabulary
        this.tk = new Tokenizer(dataset);
    
        let maxlen = 100;
    
        // trasform texts to sequences    
        let X_train = this.tk.texts_to_sequences(data.X_train, maxlen);
        let X_test = this.tk.texts_to_sequences(data.X_test, maxlen);
    
        //Glove embedding
        let glove_len = 100;
        lines = (await fs.readFile("dataset/glove.6B.100d.txt")).toString().split("\n");
    
        let embeddings = {};
        
        lines.forEach(line => { 
            let word = line.split(" ");
            let x = word.slice(1, );
            x = x.map(a => parseFloat(a));
            embeddings[word[0]] = x;
        });
    
        let embedding_matrix = [];
    
        this.tk.vocab.forEach(word => {
            let x;
            if (embeddings[word] && word.length > 0) { 
                x = embeddings[word]
            } else {
                x = new Array(glove_len).fill(0);
            }
            embedding_matrix.push(x);
        });
    
        // build the model
        let embedding_dim = 100;
        let vocab_size = this.tk.vocab.length;
        let emb_tensor = tfjs.tensor(embedding_matrix);
    
        let weights = tfjs.variable(emb_tensor);
    
        this.model = tfjs.sequential();
        this.model.add(tfjs.layers.embedding({inputDim :vocab_size, outputDim : embedding_dim, inputLength : maxlen, weights: [weights], trainable : false}));
        this.model.add(tfjs.layers.globalMaxPooling1d());
        this.model.add(tfjs.layers.dense({units : 10, activation : 'relu', useBias : true}));
        this.model.add(tfjs.layers.dense({units : 1, activation : 'sigmoid', useBias : true}));
    
        this.model.compile({optimizer : 'adam', loss : tfjs.losses.sigmoidCrossEntropy, metrics :['accuracy']});
    
        // train the model
        let epochs = 100;
    
        let history = await this.model.fit(tfjs.tensor(X_train), tfjs.tensor(data.y_train), {
            epochs : epochs,
            validationData : [tfjs.tensor(X_test), tfjs.tensor(data.y_test)],
            verbose : 0
        });
    
        console.log(history.history);
    }

    train_test_split(dataset, seed="$$$", maxlen=1000, test_size=0.25) {
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

    // prediction function
    async predict(phrase) {
        let s = this.tk.text_to_sequence(phrase, 100);
        let pred = this.model.predict(tfjs.tensor(s, [1, 100]));
        pred = await pred.data();
        console.log(`\"${phrase}\" : ${pred}`);        
    }
}

// test class
(async () => {
    let classifier = await new Sentiment_Analysis_Model();

    let phrases = ["This movie is really bad. Poor production, low budget, awful story!", "It is an awesome production"];
    phrases.forEach(async (phrase) => {
        await classifier.predict(phrase);
    });
    
})();