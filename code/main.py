from __future__ import print_function
from random import shuffle
import os
import argparse
import pickle

from get_image_paths import get_image_paths
from get_tiny_images import get_tiny_images
from build_vocabulary import build_vocabulary
from get_bags_of_sifts import get_bags_of_sifts
from visualize import visualize

from nearest_neighbor_classify import nearest_neighbor_classify
from svm_classify import svm_classify
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Step 0: Set up parameters, category list, and image paths.

parser = argparse.ArgumentParser()
parser.add_argument('--feature', help='feature', type=str, default='dumy_feature')
parser.add_argument('--classifier', help='classifier', type=str, default='dumy_classifier')
args = parser.parse_args()

DATA_PATH = '../data/'

#This is the list of categories / directories to use

CATEGORIES = [
    'agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings',
    'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse',
    'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass',
    'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks',
    'tenniscourt'
]
CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}
ABBR_CATEGORIES = [
    'Agr', 'Air', 'BD', 'Bch', 'Bldg', 'Chap', 'DR', 'For', 'FWY', 'GC',
    'Harb', 'Int', 'MR', 'MHP', 'OP', 'PL', 'Riv', 'RWY', 'SR', 'STK', 'TC'
]


# FEATURE = args.feature
FEATURE  = 'bag_of_sift'

# CLASSIFIER = args.classifier
CLASSIFIER = 'support_vector_machine'

#number of training examples per category to use
NUM_TRAIN_PER_CAT = 70
VAL_SIZE = 10
NUM_TEST_PER_CAT = 20

def main():
    
    print("Getting paths and labels for all train and test data")
    train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths(DATA_PATH, CATEGORIES)

    # YOU CODE build_vocabulary.py
    print('No existing visual word vocabulary found. Computing one from training images\n')
    vocab_sizes = [50, 100, 200, 400]
    K = 1  # Number of folds for cross-validation
    fold_size_per_class = 80 // K  # Size of each fold per class (10 images per class)
    accuracy_list = []
    accuracy_list_k = []

    # Group image paths and labels by class
    class_indices = {label: [] for label in set(train_labels)}
    for i, label in enumerate(train_labels):
        class_indices[label].append(i)

    for vocab_size in vocab_sizes:
        for k in range(K):
            train_image_paths_cv = []
            val_image_paths_cv = []
            train_labels_cv = []
            val_labels_cv = []

            for label, indices in class_indices.items():
                # Calculate the start and end indices for this fold within the class
                start_idx = k * fold_size_per_class
                end_idx = start_idx + fold_size_per_class

                # Split the indices into validation and training sets
                train_indices = indices[start_idx:end_idx]
                val_indices = indices[:start_idx] + indices[end_idx:]

                # Append paths and labels for training and validation
                val_image_paths_cv.extend([train_image_paths[i] for i in val_indices])
                train_image_paths_cv.extend([train_image_paths[i] for i in train_indices])
                val_labels_cv.extend([train_labels[i] for i in val_indices])
                train_labels_cv.extend([train_labels[i] for i in train_indices])


            print(f"Fold {k + 1}:")
            print(f"Train start: {start_idx}, end: {end_idx}")
            print()

            train_image_paths_cv = train_image_paths
            val_image_paths_cv = test_image_paths
            train_labels_cv = train_labels

            if os.path.exists(f'model/vocab_{vocab_size}.pkl'):
                print('Loading existing vocab with vocab size:', vocab_size, 'fold:', k+1)
                with open(f'model/vocab_{vocab_size}.pkl', 'rb') as f:
                    vocab = pickle.load(f)
            else:
                print('Computing vocab with vocab size:', vocab_size, 'fold:', k+1)
                vocab = build_vocabulary(train_image_paths_cv, vocab_size)
                # save vocab
                print('Saving vocab with vocab size:', vocab_size, 'fold:', k+1)
                with open(f'model/vocab_{vocab_size}.pkl', 'wb') as f:
                    pickle.dump(vocab, f)

            if os.path.exists(f'model/train_image_feats_{vocab_size}.pkl'):
                print('Loading existing image_feats with vocab size:', vocab_size, 'fold:', k+1)
                with open(f'model/train_image_feats_{vocab_size}.pkl', 'rb') as f:
                    train_image_feats_cv = pickle.load(f)
            else:
                # YOU CODE get_bags_of_sifts.py
                print('Computing image_feats with vocab size:', vocab_size, 'fold:', k+1)
                train_image_feats_cv = get_bags_of_sifts(train_image_paths_cv, vocab);
                # save train_image_feats
                print('Saving image_feats with vocab size:', vocab_size, 'fold:', k+1)
                with open(f'model/train_image_feats_{vocab_size}.pkl', 'wb') as f:
                    pickle.dump(train_image_feats_cv, f)

            
            if os.path.exists(f'model/val_image_feats_{vocab_size}.pkl'):
                print('Loading existing val_image_feats with vocab size:', vocab_size, 'fold:', k+1)
                with open(f'model/val_image_feats_{vocab_size}.pkl', 'rb') as f:
                    val_image_feats_cv = pickle.load(f)
            else:
                print('Computing image_feats with vocab size:', vocab_size, 'fold:', k+1)
                val_image_feats_cv  = get_bags_of_sifts(val_image_paths_cv, vocab);
                # save val_image_feats
                print('Saving val_image_feats with vocab size:', vocab_size, 'fold:', k+1)
                with open(f'model/val_image_feats_{vocab_size}.pkl', 'wb') as f:
                    pickle.dump(val_image_feats_cv, f)

            # YOU CODE svm_classify.py
            print('Classifying with vocab size:', vocab_size, 'fold:', k+1)
            predicted_categories = svm_classify(train_image_feats_cv, train_labels_cv, val_image_feats_cv)

            accuracy = float(len([x for x in zip(test_labels,predicted_categories) if x[0]== x[1]]))/float(len(test_labels))
            print('Accuracy for vocab size:', vocab_size, 'fold:', k+1, 'is:', accuracy)
            accuracy_list_k.append(accuracy)

        accuracy_list.append(accuracy_list_k)

        # exit()

    print("#######################################################################################3")

    # Find the best vocab size
    best_vocab_size = vocab_sizes[np.argmax(np.mean(accuracy_list, axis=1))]
    print('Best vocab size:', best_vocab_size)

    # Plot the average accuracy for each vocab size
    plt.plot(vocab_sizes, np.mean(accuracy_list, axis=1))
    plt.xlabel('Vocab Size')
    plt.ylabel('Mean Accuracy')
    plt.title('Accuracy vs Vocab Size')
    plt.tight_layout()
    # save the plot
    plt.savefig('plots/accuracy_vs_vocab_size.png')
    plt.show()


    # Train the model with the best vocab size
    if os.path.exists(f'model/vocab_{best_vocab_size}.pkl'):
        print('Loading existing vocab with vocab size:', best_vocab_size)
        with open(f'model/vocab_{best_vocab_size}.pkl', 'rb') as f:
            vocab = pickle.load(f)
    else:
        print('Computing vocab with vocab size:', best_vocab_size)
        vocab = build_vocabulary(train_image_paths, best_vocab_size)
        # save vocab
        print('Saving vocab with vocab size:', best_vocab_size)
        with open(f'model/vocab_{best_vocab_size}.pkl', 'wb') as f:
            pickle.dump(vocab, f)

    if os.path.exists(f'model/train_image_feats_{best_vocab_size}.pkl'):
        print('Loading existing image_feats with vocab size:', best_vocab_size)
        with open(f'model/train_image_feats_{best_vocab_size}.pkl', 'rb') as f:
            train_image_feats = pickle.load(f)
    else:
        # YOU CODE get_bags_of_sifts.py
        print('Computing image_feats with vocab size:', best_vocab_size)
        train_image_feats = get_bags_of_sifts(train_image_paths, vocab);
        # save train_image_feats
        print('Saving image_feats with vocab size:', best_vocab_size)
        with open(f'model/train_image_feats_{best_vocab_size}.pkl', 'wb') as f:
            pickle.dump(train_image_feats, f)
    
    if os.path.exists(f'model/test_image_feats_{best_vocab_size}.pkl'):
        print('Loading existing test_image_feats with vocab size:', best_vocab_size)
        with open(f'model/test_image_feats_{best_vocab_size}.pkl', 'rb') as f:
            test_image_feats = pickle.load(f)
    else:
        print('Computing image_feats with vocab size:', best_vocab_size)
        test_image_feats  = get_bags_of_sifts(test_image_paths, vocab);
        # save test_image_feats
        print('Saving test_image_feats with vocab size:', best_vocab_size)
        with open(f'model/test_image_feats_{best_vocab_size}.pkl', 'wb') as f:
            pickle.dump(test_image_feats, f)

    predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)

    accuracy = float(len([x for x in zip(test_labels,predicted_categories) if x[0]== x[1]]))/float(len(test_labels))
    print("Accuracy = ", accuracy)
    
    test_labels_ids = [CATE2ID[x] for x in test_labels]
    predicted_categories_ids = [CATE2ID[x] for x in predicted_categories]
    train_labels_ids = [CATE2ID[x] for x in train_labels]
   
    build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES)
    visualize(CATEGORIES, test_image_paths, test_labels_ids, predicted_categories_ids, train_image_paths, train_labels_ids)

def build_confusion_mtx(test_labels_ids, predicted_categories, abbr_categories):
    # Compute confusion matrix
    cm = confusion_matrix(test_labels_ids, predicted_categories)
    np.set_printoptions(precision=2)
    '''
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm, CATEGORIES)
    '''
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print('Normalized confusion matrix')
    #print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, abbr_categories, title='Normalized confusion matrix')

    plt.show()
     
def plot_confusion_matrix(cm, category, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category))
    plt.xticks(tick_marks, category, rotation=45)
    plt.yticks(tick_marks, category)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':
    main()
