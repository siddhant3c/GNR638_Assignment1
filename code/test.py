import numpy as np

vocab_sizes = [50, 100, 200, 400]

a = [1]
b = [2]
c = [3]
d = [4]

accuracy_list = []

accuracy_list.append(a)
accuracy_list.append(b)
accuracy_list.append(c)
accuracy_list.append(d)

x = np.mean(accuracy_list, axis=1)
print(x)
y = np.argmax(x)
print(y)
print(vocab_sizes[y])

best_vocab_size = vocab_sizes[np.argmax(np.mean(accuracy_list, axis=1))]

print(best_vocab_size)