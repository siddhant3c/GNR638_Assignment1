import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the visual vocabularies
vocab_sizes = [50, 100, 200, 400]
vocab_list = []
print("Loading visual vocabularies...")
for vocab_size in vocab_sizes:
    with open(f'model/vocab_{vocab_size}.pkl', 'rb') as f:
        vocab_list.append(pickle.load(f))
        print("Shape of vocabulary:", vocab_list[-1].shape)  # Should be (vocab_size, 128)

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
reduced_vocab = []
for vocab in vocab_list:
    reduced_vocab.append(tsne.fit_transform(vocab))

# Plot the t-SNE visualization of the visual vocabularies in 4 subplots
plt.figure(figsize=(10, 7))
for i, vocab in enumerate(reduced_vocab):
    plt.subplot(2, 2, i+1)
    plt.scatter(vocab[:, 0], vocab[:, 1], s=10)
    plt.title(f"Visual Vocabulary Size: {vocab_sizes[i]}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
plt.tight_layout()
plt.savefig('plots/t-sne.png')
plt.show()