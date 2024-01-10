import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

dataset = pd.read_csv("/kaggle/input/asl-signs/train.csv")
dataset['path'] = "/kaggle/input/asl-signs/" + dataset['path']
dataset.iloc[0]["path"]

ROWS_PER_FRAME = 543
def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def get_data(file_path):
    # Load Raw Data
    data = load_relevant_data_subset(file_path)
    # Process Data Using Tensorflow
    data = preprocess_layer(data)
    
    return data

ROOT_DIR = "/kaggle/input/gislr-dataset-public"
X_train = np.load(f"{ROOT_DIR}/X_train.npy")
Y_train = np.load(f"{ROOT_DIR}/y_train.npy")
X_val = np.load(f"{ROOT_DIR}/X_val.npy")
Y_val = np.load(f"{ROOT_DIR}/y_val.npy")
NON_EMPTY_FRAME_IDXS = np.load(f"{ROOT_DIR}/NON_EMPTY_FRAME_IDXS.npy")

X_train=tf.convert_to_tensor(X_train)
Y_train=tf.convert_to_tensor(Y_train)
X_val=tf.convert_to_tensor(X_val)
Y_val=tf.convert_to_tensor(Y_val)
X.shape, Y.shape, NON_EMPTY_FRAME_IDXS.shape

X_10 = tf.gather(X, np.argwhere(np.isin(Y,np.arange(10))).squeeze() )
Y_10 = tf.gather(Y, np.argwhere(np.isin(Y,np.arange(10))).squeeze() )
X_10.shape, Y_10.shape

BATCH_ALL_SIGNS_N=4
NUM_CLASSES=250
INPUT_SIZE=64
N_COLS=66
N_DIMS=3
# Custom sampler to get a batch containing N times all signs
def get_train_batch_all_signs(X, y, n=BATCH_ALL_SIGNS_N):
    # Dictionary mapping ordinally encoded sign to corresponding sample indices
    CLASS2IDXS = {}
    for i in range(NUM_CLASSES):
        CLASS2IDXS[i] = np.argwhere(y == i).squeeze().astype(np.int32)
            
    while True:
        # Lists to store batch tensors in
        X_batch = []
        y_batch = []
        
        # Fill batch arrays
        for i in range(NUM_CLASSES):
            idxs = np.random.choice(CLASS2IDXS[i], n)
            X_batch.append(tf.gather(X, idxs))
            y_batch.append(tf.gather(y, idxs))
        
        # Stack lists of tensors into a single tensor
        X_batch = tf.concat(X_batch, axis=0)
        y_batch = tf.concat(y_batch, axis=0)
        
        yield X_batch, y_batch

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    featurewise_std_normalization=True,
    shear_range=0.2,
    samplewise_std_normalization=True,
    validation_split=0.2,
)
img_gen.fit(X_train)

frames = tf.keras.layers.Input([64, 66, 3], dtype=tf.float32, name='frames')
model1 = tf.keras.models.Sequential()
model2 = tf.keras.models.Sequential()
resnet = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=frames,
    pooling='max',
    classes=250,
    classifier_activation="softmax",
)

eff_net = tf.keras.applications.EfficientNetB5(
    include_top=False,
    weights="imagenet",
    input_tensor=frames,
    pooling=None,
    classes=250,
    classifier_activation="softmax",
)
for layer in resnet.layers:
    layer.trainable = True
for layer in eff_net.layers:
    layer.trainable = True
model1.add(resnet)
model1.add(tf.keras.layers.Flatten())
model1.add(tf.keras.layers.BatchNormalization())
model1.add(tf.keras.layers.Dense(1024,activation = 'relu'))
model1.add(tf.keras.layers.Dropout(0.5))
model1.add(tf.keras.layers.BatchNormalization())
model1.add(tf.keras.layers.Dense(512,activation = 'relu'))
model1.add(tf.keras.layers.Dense(250,activation = 'softmax'))
model1.summary()

model2.add(eff_net)
model2.add(tf.keras.layers.Flatten())
model2.add(tf.keras.layers.BatchNormalization())
model2.add(tf.keras.layers.Dense(1024,activation = 'relu'))
model2.add(tf.keras.layers.Dropout(0.5))
model2.add(tf.keras.layers.BatchNormalization())
model2.add(tf.keras.layers.Dense(512,activation = 'relu'))
model2.add(tf.keras.layers.Dense(250,activation = 'softmax'))
model2.summary()

loss = tf.keras.losses.SparseCategoricalCrossentropy()
    
# Adam Optimizer with weight decay
optimizer = tf.optimizers.Adam()

# TopK Metrics
metrics = [
    tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_acc'),
    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc'),
]
model1.compile(loss=loss, optimizer=tf.optimizers.Adam(), metrics=metrics)
model2.compile(loss=loss, optimizer=tf.optimizers.Adam(), metrics=metrics)

tf.keras.backend.clear_session()

model1.fit(img_gen.flow(X_train,Y_train),verbose=1, epochs=15, validation_data=(X_val,Y_val))

model2.fit(img_gen.flow(X,Y),verbose=1, epochs=15)

import tensorflow_datasets as tfds
ds = tfds.load("cifar10", as_supervised=True)

ds['train']