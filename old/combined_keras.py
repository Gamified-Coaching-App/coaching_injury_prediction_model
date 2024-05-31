import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, initializers
from sklearn.metrics import roc_auc_score
from exploration.Image_Encoding.RunningDataset import RunningDataset
import numpy as np

class Classifier(models.Model):
    def __init__(self):
        super(Classifier, self).__init__()
        # Assuming input shape as [batch_size, 8, 8, 10]
        self.conv1 = layers.Conv2D(filters = 64, kernel_size=2, strides=1, padding='same', activation='selu', kernel_initializer=initializers.LecunNormal(),input_shape=(8, 8, 10))
        self.conv2 = layers.Conv2D(filters = 16, kernel_size=2, strides=1, padding='same', activation='selu', kernel_initializer=initializers.LecunNormal())
        self.pool = layers.MaxPooling2D(pool_size=2, strides=2)
        self.flatten = layers.Flatten()
        self.fc_layers = [layers.Dense(50, activation='selu', kernel_initializer='lecun_normal') for _ in range(5)]
        self.dropout = layers.Dropout(0.3)
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        for fc in self.fc_layers:
            x = fc(x)
            x = self.dropout(x)
        return self.output_layer(x)
    
def train_model(model, train_dataset, epochs=100, learning_rate=0.01):
    model.compile(optimizer=optimizers.Adadelta(learning_rate=learning_rate),
                  loss=losses.BinaryFocalCrossentropy(alpha=0.55, gamma=5.0),
                  metrics=['accuracy'])
        
    history = model.fit(train_dataset, epochs=epochs, verbose=0)
    return history

def evaluate_model(model, test_dataset):
    all_predictions = []
    all_labels = []
    for batch in test_dataset:
        features, labels = batch
        predictions = model.predict(features)
        all_predictions.extend(predictions.flatten())  # flatten predictions if needed
        all_labels.extend(labels.numpy().flatten())     # extract labels and flatten if needed
    
    # Convert list to numpy array for ROC AUC calculation
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    test_loss = model.evaluate(test_dataset, verbose=0)
    auc = roc_auc_score(all_labels, all_predictions)
    return test_loss, auc

if __name__ == "__main__":
    # Assume RunningDataset provides a way to preprocess data suitable for tf.data.Dataset
    dataset = RunningDataset()
    X_train, y_train, X_test, y_test = dataset.preprocess()

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(512)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(512)

    classifier = Classifier()
    history = train_model(classifier, train_dataset, epochs=200, learning_rate=0.01)
    test_loss, auc_score_test = evaluate_model(classifier, test_dataset)
    train_loss, auc_score_train= evaluate_model(classifier, train_dataset)
    print(f"Test Loss: {test_loss}, Test AUC: {auc_score_test}")
    print(f"Train Loss: {train_loss}, Train AUC: {auc_score_train}")