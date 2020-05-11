# Set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import backend as K

import matplotlib.pyplot as plt
import numpy as np
import argparse
from imutils import paths
import os


from datasets.simple_dataset_loader import SimpleDatasetLoader

from utilities.cnn.minivggnet import MiniVGGNet
from utilities.preprocessing.simple_preprocessor import SimplePreprocessor
from utilities.preprocessing.imagetoarray_preprocessor import ImageToArrayPreprocessor
from utilities.preprocessing.one_channel import OneChannelLoader
from utilities.callbacks.trainingmonitor import TrainingMonitor
#%%
'''
# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='Path to input dataset')
args = vars(ap.parse_args())
'''
args={}
args['dataset']='datasets'
args['weights']='output'
args['monitor']='output'
#%%
# Get list of image paths
image_paths = list(paths.list_images(args['dataset']))

#%%

# Initialize SimplePreprocessor and SimpleDatasetLoader and load data and labels
print('[INFO]: Images loading....')
pixels=28
sp = SimplePreprocessor(pixels,pixels)
iap = ImageToArrayPreprocessor()
op = OneChannelLoader()

sdl = SimpleDatasetLoader(preprocessors=[sp,iap,op])
 
(data, labels) = sdl.load(image_paths, verbose=5)



#%%
#one-hot encoding

enc = OneHotEncoder(handle_unknown='ignore')
y=enc.fit_transform(labels.reshape(-1,1)).toarray()

#%%
# Scale the input data to the range [0, 1]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X=data.ravel().reshape(-1,1)
X=scaler.fit_transform(X)

#%%
# 'channels_first' ordering
if K.image_data_format() == "channels_first":
    # Reshape the design matrix such that the matrix is: num_samples x depth x rows x columns
    X = X.reshape(data.shape[0], 1, pixels, pixels)
# 'channels_last' ordering
else:
    # Reshape the design matrix such that the matrix is: num_samples x rows x columns x depth
    X = X.reshape(data.shape[0],pixels,pixels, 1)
    
 #%%
 #split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#%%
#construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=False, fill_mode="nearest")

#%%
# Initialize the optimizer and model
print("[INFO]: Compiling model....")
sgd = SGD(lr=0.01)
model = MiniVGGNet.build(width=pixels, height=pixels, depth=1, classes=17)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

#%%
# construct a set of callback to babysit the training process
figPath = os.path.sep.join([args["monitor"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["monitor"], "{}.json".format(os.getpid())])
call = [TrainingMonitor(figPath, json_path=jsonPath)]

#%%

# construct the callback to save all the improved model to disk
# based on the validation loss
fname = os.path.sep.join([args["weights"],"weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
call =call+ [checkpoint]
#%%
# Train the network
print("[INFO]: Training....")
epoch_num=10
H = model.fit_generator(aug.flow(X_train, y_train, batch_size=32), 
                        validation_data=(X_test, y_test), 
                        steps_per_epoch=len(X_train) ,
                        epochs=epoch_num, verbose=1,callbacks=call)

#%%
#save the final model
model.save('output/final_model.hdf5')
#%%
# Evaluate the network
print("[INFO]: Evaluating....")
preds = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names=[str(x) for x in enc.get_feature_names()]))

#%%
# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoch_num), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch_num), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epoch_num), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, epoch_num), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()