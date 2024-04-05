class WordTokenizer:
    def __init__(self):
        self.word2idx = dict()
        self.idx2word = dict()

        # special tokens 
        self.word2idx['<sos>'] = 0
        self.word2idx['<eos>'] = 1
        self.word2idx['<pad>'] = 2
        self.word2idx['<unk>'] = 3

        self.idx2word[0] = '<sos>'
        self.idx2word[1] = '<eos>'
        self.idx2word[2] = '<pad>'
        self.idx2word[3] = '<unk>'

        self.SOS = '<sos>'
        self.EOS = '<eos>'
        self.PAD = '<pad>'
        self.UNK = '<unk>'

    def load(self, word_path: str):
        # load word from a words.txt file 
        with open(word_path, 'r') as f:
            for line in f:
                word, idx = line.split()
                word, idx = word.strip(), len(self.word2idx)

                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word

                assert len(self.word2idx) == len(self.idx2word)

    def tokenize(self, text):   
        return [self.word2idx[word] if word in self.word2idx else self.word2idx['<unk>'] for word in text.split()]
    
    def detokenize(self, tokens):
        return " ".join([self.idx2word[token] for token in tokens])
    
    def __len__(self):
        assert len(self.word2idx) == len(self.idx2word)
        return len(self.word2idx)
    
word_tokenizer = WordTokenizer()