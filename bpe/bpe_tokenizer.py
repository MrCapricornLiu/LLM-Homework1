import re
from collections import defaultdict

class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.merge_ops = []

    def _get_vocab(self, text):
        """Initialize vocab as a dictionary of character pairs with their frequencies."""
        vocab = defaultdict(int)
        # Split text into words and then further into character-level tokens
        words = text.split()
        for word in words:
            word = ' '.join(list(word)) + ' </w>'  # Add a special end-of-word token </w>
            for i in range(len(word) - 1):
                pair = word[i:i+2]
                vocab[pair] += 1
        return vocab

    def _get_stats(self, vocab):
        """Get the pair frequencies in the vocab dictionary."""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq
        return pairs

    def _merge_vocab(self, pair, vocab):
        """Merge the most frequent pair into a single token."""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        replacement = replacement + ' ' if replacement != '</w>' else replacement

        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def train(self, text, vocab_size):
        """Train the tokenizer using BPE algorithm."""
        # Step 1: Initialize vocab with character pairs
        vocab = self._get_vocab(text)

        # Step 2: Repeatedly merge the most frequent pair until vocab size is reached
        while len(vocab) < vocab_size:
            pairs = self._get_stats(vocab)
            if not pairs:
                break

            # Find the most frequent pair
            most_frequent_pair = max(pairs, key=pairs.get)

            # Merge the most frequent pair
            vocab = self._merge_vocab(most_frequent_pair, vocab)
            self.merge_ops.append(most_frequent_pair)

        # Build the final vocab
        for word in vocab:
            self.vocab[word] = vocab[word]

    def encode(self, text):
        """Encode the input string into a token list."""
        # Split the text into characters and merge with BPE merges
        tokens = list(text)
        for pair in self.merge_ops:
            pair_str = ''.join(pair)
            for i in range(len(tokens) - 1):
                if tokens[i] + tokens[i+1] == pair_str:
                    tokens[i] = pair_str
                    del tokens[i+1]
                    break
        return tokens

    def decode(self, ids):
        """Decode a token list into a string."""
        return ''.join(ids).replace('</w>', ' ').strip()


# # 示例文本
# text = "low lower newest lowest"

# # 初始化并训练tokenizer
# tokenizer = Tokenizer()
# tokenizer.train(text, vocab_size=10)

# # 编码示例文本
# encoded = tokenizer.encode("lowest")
# print("Encoded:", encoded)

# # 解码回文本
# decoded = tokenizer.decode(encoded)
# print("Decoded:", decoded)
