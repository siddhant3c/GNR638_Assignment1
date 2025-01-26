PL go through this codebase: https://github.com/lionelmessi6410/Scene-Recognition-with-Bag-of-Words

Now, change the codebase for the UC Merced dataset: http://weegee.vision.ucmerced.edu/datasets/landuse.html

Take 70% data per class for training, remaining 10% for validation (deciding what should be the optimal number of codewords), and testing on the remaining 20%

You can use the k-fold cross-validation strategy.

You need to report the classification accuracy, a graph showing how does the accuracy changes as you use differ number of codewords in clustering, a t-SNE visualization of the keypoints (each is 128 dimension in SIFT)