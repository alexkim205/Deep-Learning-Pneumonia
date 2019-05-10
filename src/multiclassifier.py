from classifier import * 

# Load data for multi classifier
def load_multi_data(save_path):
    
    # Grab data munged bottle necks
    def _load_data():
        data = np.load(path.format(dataset_type))
        data_bottle_necks, data_labels, data_file_paths = data['bottle_necks'],  data['labels'], data['paths']
    
        return data_bottle_necks, data_labels, data_file_paths
    
    # Read in data
    train_bottle_necks, train_labels, train_file_paths = _load_data(save_path, 'train')
    validate_bottle_necks, validate_labels, validate_file_paths = _load_data(save_path, 'validate')
    test_bottle_necks, test_labels, test_file_paths = _load_data(save_path, 'test')
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
    
    return train_dataset_original, validate_dataset_original, test_dataset_original