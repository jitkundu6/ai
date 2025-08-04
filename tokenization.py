# pip install tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, Sequence
from tokenizers.processors import TemplateProcessing

# Initialize a tokenizer with BPE model
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Add basic preprocessing
tokenizer.normalizer = Sequence([Lowercase()])
tokenizer.pre_tokenizer = Whitespace()

# Setup training
trainer = BpeTrainer(vocab_size=100, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])

# Training data
corpus = ["This is a simple example of BPE tokenization", "We are training a BPE model from scratch"]

# Train tokenizer on in-memory data
tokenizer.train_from_iterator(corpus, trainer=trainer)

# Test the tokenizer
output = tokenizer.encode("Tokenization with BPE is awesome!")

print("Tokens:", output.tokens)
print("IDs:", output.ids)
print("Offsets:", output.offsets)
print("Attention Mask:", output.attention_mask)
print("Special Tokens:", output.special_tokens_mask)
print("Type IDs:", output.type_ids)
print("Decoded:", tokenizer.decode(output.ids))
# Save the tokenizer to a file
tokenizer.save("bpe_tokenizer.json")
# Load the tokenizer from a file
# loaded_tokenizer = Tokenizer.from_file("bpe_tokenizer.json")
