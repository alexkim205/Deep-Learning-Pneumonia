import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from matplotlib import pyplot as plt
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import cv2
import keras.backend as K

np.random.seed(42)
tf.enable_eager_execution()
save_path = '/home/final/data/cache/{}.bottle_necks.labels.paths.npz'
plot_path = '/home/final/data/plots'

device = "gpu:0" if tfe.num_gpus() else "cpu:0"

class XceptionClassifier(tf.keras.Model):
    def __init__(self, n_classes, n_layers):
        self.n_layers = n_layers
        self.n_classes = n_classes
        
        super(XceptionClassifier, self).__init__()
#         self.xception_layers = Xception(include_top=False, weights='imagenet', input_shape=(200,200,3))
#         self.pooling_layer = GlobalAveragePooling2D(data_format='channels_last')
        if n_layers  == 3:
            self.dense_layer1 = tf.keras.layers.Dense(units=1024, activation='relu')
            self.dense_layer2 = tf.keras.layers.Dense(units=512, activation='relu')
            self.dense_layer3 = tf.keras.layers.Dense(units=self.n_classes)
        if n_layers  == 6:
            self.dense_layer1 = tf.keras.layers.Dense(units=1024, activation='relu')
            self.dense_layer2 = tf.keras.layers.Dense(units=512, activation='relu')
            self.dense_layer3 = tf.keras.layers.Dense(units=256, activation='relu')
            self.dense_layer4 = tf.keras.layers.Dense(units=128, activation='relu')
            self.dense_layer5 = tf.keras.layers.Dense(units=64, activation='relu')
            self.dense_layer6 = tf.keras.layers.Dense(units=self.n_classes)
        if n_layers  == 9:
            self.dense_layer1 = tf.keras.layers.Dense(units=1024, activation='relu')
            self.dense_layer2 = tf.keras.layers.Dense(units=512, activation='relu')
            self.dense_layer3 = tf.keras.layers.Dense(units=256, activation='relu')
            self.dense_layer4 = tf.keras.layers.Dense(units=128, activation='relu')
            self.dense_layer5 = tf.keras.layers.Dense(units=64, activation='relu')
            self.dense_layer6 = tf.keras.layers.Dense(units=32, activation='relu')
            self.dense_layer7 = tf.keras.layers.Dense(units=16, activation='relu')
            self.dense_layer8 = tf.keras.layers.Dense(units=8, activation='relu')
            self.dense_layer9 = tf.keras.layers.Dense(units=self.n_classes)
            
    def call(self, inputs):
#         xception = self.xception_layers(inputs)
#         pooling = self.pooling_layer(xception)
        if self.n_layers == 3:
            result = self.dense_layer1(inputs)
            result = self.dense_layer2(result)
            result = self.dense_layer3(result)
        if self.n_layers == 6:
            result = self.dense_layer1(inputs)
            result = self.dense_layer2(result)
            result = self.dense_layer3(result)
            result = self.dense_layer4(result)
            result = self.dense_layer5(result)
            result = self.dense_layer6(result)
        if self.n_layers == 9:
            result = self.dense_layer1(inputs)
            result = self.dense_layer2(result)
            result = self.dense_layer3(result)
            result = self.dense_layer4(result)
            result = self.dense_layer5(result)
            result = self.dense_layer6(result)
            result = self.dense_layer7(result)
            result = self.dense_layer8(result)
            result = self.dense_layer9(result)

        return result
        
def calculate_loss(classifier, images, labels):
    logits = classifier(images)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

def calculate_f1(classifier, images, labels):
    logits = tf.argmax(classifier(images), axis=1)
    TP = tf.count_nonzero(logits * labels)
    TN = tf.count_nonzero((logits - 1) * (labels - 1))
    FP = tf.count_nonzero(logits * (labels - 1))
    FN = tf.count_nonzero((logits - 1) * labels)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return f1

def train(train_dataset, valid_dataset,
          learning_rate, batch_size, n_epochs, n_layers, n_classes,
          save_plot_fname,
          plot_graphs=False, save_graphs=True
         ):
    
    def _plot_loss(train, val, plot, save):
        plt.figure(figsize=(9,6))
        plt.plot(train, label='Train Loss')
        plt.plot(val, label='Validation Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss History')
        if save: 
            plt.savefig("{}/{}_{}.png".format(plot_path, save_plot_fname, "loss"), bbox_inches='tight')
            plt.close()
        if plot: 
            plt.show()
            plt.close()
    
    def _plot_acc(train, val, plot, save):
        plt.figure(figsize=(9,6))
        plt.plot(train, label='Train Accuracy')
        plt.plot(val, label='Validation Accuracy')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy History')
        if save: 
            plt.savefig("{}/{}_{}.png".format(plot_path, save_plot_fname, "accr"), bbox_inches='tight')
            plt.close()
        if plot: 
            plt.show()
            plt.close()
    
    def _plot_f1(train, val, plot, save):
        plt.figure(figsize=(9,6))
        plt.plot(train, label='Train F1')
        plt.plot(val, label='Validation F1')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.title('F1 Score History')
        if save: 
            plt.savefig("{}/{}_{}.png".format(plot_path, save_plot_fname, "f1sc"), bbox_inches='tight')
            plt.close()
        if plot: 
            plt.show()
            plt.close()
    
    
    x_classifier = XceptionClassifier(n_classes=n_classes, n_layers=n_layers)
    optimizer = tf.train.AdamOptimizer(learning_rate) 

    # Performance Metrics
    # - F1 Score
    train_F1_history = []
    val_F1_history = []
    # - Loss
    train_loss_history = []
    val_loss_history = []
    # - Accuracy
    train_acc_history = []
    val_acc_history = []
    
    with tf.device(device):
        for epoch in range(n_epochs):
            
            epoch_loss_avg = tfe.metrics.Mean()
            epoch_acc = tfe.metrics.Accuracy()
            epoch_f1 = tfe.metrics.Mean()
            for batch, (tr_img, tr_lbl) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    # Compute logits, logits are the domain/input to softmax
                    tr_loss = calculate_loss(x_classifier, tr_img, tr_lbl)
                
                # Compute gradient and apply gradients
                grads = tape.gradient(tr_loss, x_classifier.variables)
                optimizer.apply_gradients(zip(grads, x_classifier.variables),
                                          global_step=tf.train.get_or_create_global_step())
                
                # Add current batch metrics
                epoch_loss_avg(tr_loss) # calculate avg epoch loss 
                epoch_acc(tf.argmax(x_classifier(tr_img), axis=1), tr_lbl)
                epoch_f1(calculate_f1(x_classifier, tr_img, tr_lbl)) # calculate avg epoch f1 score 
                
                if batch % 10 == 0:
                    print('\rEpoch: {}, Batch: {}, Loss: {}'.format(epoch, batch, tr_loss.numpy()), end='')
            # Add train loss, accuracy, and f1 for epoch
            train_loss_history.append(epoch_loss_avg.result())
            train_acc_history.append(epoch_acc.result())
            train_F1_history.append(epoch_f1.result())

            
            # Run validation loop at end of each epoch
            val_loss_avg = tfe.metrics.Mean()
            val_acc = tfe.metrics.Accuracy()
            val_f1 = tfe.metrics.Mean()
            with tf.device(device):
                for batch, (val_img, val_lbl) in enumerate(valid_dataset):
                    # Compute validation metrics
                    val_loss = calculate_loss(x_classifier, val_img, val_lbl)
                    val_loss_avg(val_loss) # Add current batch loss
                    val_acc(tf.argmax(x_classifier(val_img), axis=1), val_lbl)
                    val_f1(calculate_f1(x_classifier, val_img, val_lbl))
                
                val_loss_history.append(val_loss_avg.result())
                val_acc_history.append(val_acc.result())
                val_F1_history.append(val_f1.result())
            
            # Print progress of epochs
            if epoch % 100 == 0:
                print("\rEpoch {:03d}: Train F1:{:.3f}, Train Loss:{:.3f}, Train Acc:{:.3%}, Val F1:{:.3%}, Val Loss: {:.3f}, Val Acc: {:.3%}".format(
                    epoch, epoch_f1.result(), epoch_loss_avg.result(), epoch_acc.result(), 
                    val_f1.result(), val_loss_avg.result(), val_acc.result()))
    
    _plot_loss(train_loss_history, val_loss_history, plot_graphs, save_graphs)
    _plot_acc(train_acc_history, val_acc_history, plot_graphs, save_graphs)
    _plot_f1(train_F1_history, val_F1_history, plot_graphs, save_graphs)
    
    return x_classifier, train_loss_history, val_loss_history, train_acc_history, val_acc_history, train_F1_history, val_F1_history


def test(x_classifier, test_dataset, save_plot_fname):
    
    def _plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues,
                              show_graphs=False, save_graphs=True):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print(cm)

        plt.figure(figsize=(9,6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        if save_graphs: 
            plt.savefig("{}/{}_{}.png".format(plot_path, save_plot_fname, "conf_norm" if normalize else "conf"), bbox_inches='tight')
            plt.close()
        if show_graphs: 
            plt.show()
            plt.close()
    
    # Performance metrics
    test_loss_avg = tfe.metrics.Mean() # loss
    test_acc = tfe.metrics.Accuracy() # accuracy
    test_f1 = tfe.metrics.Mean() # f1
    predictions = tf.convert_to_tensor([], dtype=tf.int64)
    correct = tf.convert_to_tensor([], dtype=tf.int64)
    
    # Run testing loop batch-wise
    with tf.device(device):
        for batch, (tst_img, tst_lbl) in enumerate(test_dataset):
            tst_loss = calculate_loss(x_classifier, tst_img, tst_lbl)
            test_loss_avg(tst_loss)
            test_acc(tf.argmax(x_classifier(tst_img), axis=1), tst_lbl)
            test_f1(calculate_f1(x_classifier, tst_img, tst_lbl))
            predictions = tf.concat([predictions, tf.argmax(x_classifier(tst_img), axis=1)], 0)
            correct = tf.concat([correct, tst_lbl], 0)
    
    cnf_matrix = confusion_matrix(correct, predictions)
    class_names = ['normal', 'bacterial', 'viral']
    # Plot non-normalized confusion matrix
    _plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    _plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,show_graphs=True,
                      title='Normalized confusion matrix')
    
    print("\nTest dataset metrics: F1: {:.3f}, Loss: {:.3f}, Accuracy: {:.3%}".format(
        test_f1.result(), test_loss_avg.result(), test_acc.result()))
        
def load_data(path, dataset_type='train'):
    
    data = np.load(path.format(dataset_type))
    data_bottle_necks, data_labels, data_file_paths = data['bottle_necks'],  data['labels'], data['paths']
    
    return data_bottle_necks, data_labels, data_file_paths
    
