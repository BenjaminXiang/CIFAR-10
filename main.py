import random

import matplotlib.pyplot as plt

import cifar10
from lab8_ANN import predict, train
from lab8_kNN import kNN


def plot_9images(images, cls_idx_true, cls_idx_pred=None, all_cls_names=None, smooth=True):
    assert len(images) == len(cls_idx_true) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_idx_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        # Plot image.
        ax.imshow(images[i, :, :, :],
                  interpolation=interpolation)

        # Name of the true class.
        cls_true_name = all_cls_names[cls_idx_true[i]]

        # Show true and predicted classes.
        if cls_idx_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = all_cls_names[cls_idx_pred[i]]

            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def main():
    class_names = cifar10.load_class_names()
    images_train, cls_idx_train, labels_train = cifar10.load_training_data()
    images_test, cls_idx_test, labels_test = cifar10.load_test_data()


    #Plot the first 9 training images and labels
    plot_9images(images=images_train[0:9], cls_idx_true=cls_idx_train[0:9],
                 all_cls_names=class_names, smooth=True)

    # Build your predictor
    w1, b1, w2, b2 = train(images_train, labels_train, images_test, cls_idx_test)
    
    # Visualize your prediction
    print('--------------------------------Neural Network--------------------------------')
    samples = random.sample(range(len(images_test)), 9)
    plot_9images(images=images_test[samples], cls_idx_true=cls_idx_test[samples],
                  cls_idx_pred=predict(images_test[samples], w1, b1, w2, b2), all_cls_names=class_names, smooth=True)
    print(f'\nAccuracy: {(predict(images_test, w1, b1, w2, b2) == cls_idx_test).mean() * 100}%\n')
            
    knn_idx = kNN(images_train, cls_idx_train, images_test)
    print('-------------------------------k-Nearest Neighbor-------------------------------')
    samples = random.sample(range(len(images_test)), 9)
    plot_9images(images=images_test[samples], cls_idx_true=cls_idx_test[samples],
                  cls_idx_pred=knn_idx[samples], all_cls_names=class_names, smooth=True)
    print(f'\nAccuracy: {(knn_idx == cls_idx_test).mean() * 100}%\n')
if __name__ == '__main__':
    main()
