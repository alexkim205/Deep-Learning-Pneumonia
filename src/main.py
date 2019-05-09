from datamunging import *
from inceptionclassifier import *

def main():
    # Munge data
    zipped_data[]
    balance_data(zipped_data)
    
    train_paths, train_labels = paired_shuffle(zipped_data[0])
    validate_paths, validate_labels = paired_shuffle(zipped_data[1])
    test_paths, test_labels = paired_shuffle(zipped_data[2])
    
    device = "gpu:0" if tfe.num_gpus() else "cpu:0"
    
    cache_path = "/home/final/data/cache/"
    fname = '{}.bottle_necks.labels.paths.npz'

    if not os.path.isdir(cache_path): 
        os.mkdir(cache_path)

    train_bottle_necks = cache_bottleneck_layers(train_paths, batch_size=50, device=device)
    np.savez(os.path.join(cache_path,fname.format('train')), bottle_necks=train_bottle_necks, paths=train_paths, labels=train_labels)

    validate_bottle_necks = cache_bottleneck_layers(validate_paths, batch_size=50, device=device)
    np.savez(os.path.join(cache_path,fname.format('validate')), bottle_necks=validate_bottle_necks, paths=validate_paths, labels=validate_labels)

    test_bottle_necks = cache_bottleneck_layers(test_paths, batch_size=50, device=device)
    np.savez(os.path.join(cache_path,fname.format('test')), bottle_necks=test_bottle_necks, paths=test_paths, labels=test_labels)
    
    
    # Read in data
    train_bottle_necks, train_labels, train_file_paths = load_data(save_path, 'train')
    validate_bottle_necks, validate_labels, validate_file_paths = load_data(save_path, 'validate')
    test_bottle_necks, test_labels, test_file_paths = load_data(save_path, 'test')
    n = len(train_bottle_necks)

    train_images_dataset = tf.data.Dataset.from_tensor_slices(train_bottle_necks)
    train_labels_dataset = tf.data.Dataset.from_tensor_slices(train_labels)
    train_dataset_original = tf.data.Dataset.zip((train_images_dataset, train_labels_dataset))
    validate_images_dataset = tf.data.Dataset.from_tensor_slices(validate_bottle_necks)
    validate_labels_dataset = tf.data.Dataset.from_tensor_slices(validate_labels)
    validate_dataset_original = tf.data.Dataset.zip((validate_images_dataset, validate_labels_dataset))
    test_images_dataset = tf.data.Dataset.from_tensor_slices(test_bottle_necks)
    test_labels_dataset = tf.data.Dataset.from_tensor_slices(test_labels)
    test_dataset_original = tf.data.Dataset.zip((test_images_dataset, test_labels_dataset))
    
    # Parameters to try out
    list_n_layers = [3] * 7 + [6] * 7 + [9] * 7
    list_batch_size = [64, 32] * 3
    list_n_epochs = [50, 50, 50, 50, 50, 100, 200] * 3
    list_learning_rate = [0.01, 0.01, 0.1, 0.001, 0.0001, 0.01, 0.01] * 3
    n_classes = 3
    
    # Train models
    for i, (n_layers, batch_size, n_epochs, learning_rate) in \
        enumerate(zip(list_n_layers, list_batch_size, list_n_epochs, list_learning_rate), 1):

        save_plot_fname = "L{}.BS{}.LR{}.EP{}".format(n_layers, batch_size, learning_rate, n_epochs)
        title = "| Model number {:02d}: {} |".format(i, save_plot_fname)
        print("-"*len(title))
        print(title)
        print("-"*len(title))
    
        train_dataset = train_dataset_original.shuffle(buffer_size=50).batch(batch_size)
        validate_dataset = validate_dataset_original.shuffle(buffer_size=50).batch(batch_size)
        test_dataset = validate_dataset_original.shuffle(buffer_size=50).batch(batch_size)

        x_classifier, tl, vl, ta, va, tf1, vf1 = train(train_dataset, validate_dataset, learning_rate, batch_size, n_epochs, n_layers, n_classes, save_plot_fname)
        test(x_classifier, test_dataset, save_plot_fname)
        
    # Further training
    n_classes, n_layers, batch_size, learning_rate, n_epochs = 3, 3, 1000, 0.001, 400
    save_plot_fname = "L{}.BS{}.LR{}.EP{}".format(n_layers, batch_size, learning_rate, n_epochs)
    title = "| Model: {} |".format(save_plot_fname)
    print("-"*len(title))
    print(title)
    print("-"*len(title))

    train_dataset = train_dataset_original.shuffle(buffer_size=50).batch(batch_size)
    validate_dataset = validate_dataset_original.shuffle(buffer_size=50).batch(batch_size)
    test_dataset = validate_dataset_original.shuffle(buffer_size=50).batch(batch_size)

    x_classifier, tl, vl, ta, va, tf1, vf1 = train(train_dataset, validate_dataset, learning_rate, batch_size, n_epochs, n_layers, n_classes, save_plot_fname)
    test(x_classifier, test_dataset, save_plot_fname)