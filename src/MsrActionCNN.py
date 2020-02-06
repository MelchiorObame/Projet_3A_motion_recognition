# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt 
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


import DataLoader,os,SPMF


class DataInformation:
    size=(32,32)
    img_width, img_height = size[0], size[1]
    batch_size = 32
    epochs = 1
    
    
#if K.image_data_format() == 'channels_first':
#    input_shape = (3, img_width, img_height) 
#else:
#    input_shape = (img_width, img_height, 3)


dataName=DataLoader.DataName.action
labelTarget='subject'
print('Pictures création...')
SPMF.getPictureDataSet(dataName, labelTarget=labelTarget, normalizeData=False, size=DataInformation.size)
dirPath = DataLoader.ActionCSV.pictureActionDBDirName

train_data_dir = os.path.join(os.path.join(DataLoader.DirPaths.PictureDataSet, dirPath),'Train/')
validation_data_dir = os.path.join(os.path.join(DataLoader.DirPaths.PictureDataSet, dirPath),  'Validation/')

file_counterValidation = sum([len(files) for r, d, files in os.walk(validation_data_dir)])
file_counterTrain = sum([len(files) for r, d, files in os.walk(train_data_dir)])
nb_train_samples = file_counterTrain
nb_validation_samples = file_counterValidation

print('model construction...')
# Initialising the CNN
model = Sequential()
model.add(Conv2D(10, (3, 3), input_shape = (DataInformation.img_width, DataInformation.img_height, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides=2 ))
model.add(Conv2D(20, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
model.add(Flatten())
model.add(Dense(units = 100, activation = 'relu'))
model.add(Dense(units = 10, activation = 'softmax'))

model.summary()
print('Compiling the CNN...')

learning_rate = 0.0001
optimizer_algo = keras.optimizers.SGD(lr=learning_rate)

model.compile(optimizer = optimizer_algo, loss = 'categorical_crossentropy', metrics = ['accuracy'])

print('Checkpoint on validation')
model_checkpoint = keras.callbacks.ModelCheckpoint('best-model.h5', monitor='val_loss', save_best_only=True)

# none Data augmentation
train_datagen = ImageDataGenerator(rescale = None)
test_datagen = ImageDataGenerator(rescale = None)


training_set = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size = DataInformation.size,
                                                    #color_mode='rgb',
                                                    batch_size = DataInformation.batch_size,
                                                    class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(validation_data_dir,
                                                        #color_mode='rgb',
                                                        target_size = DataInformation.size,
                                                        batch_size = DataInformation.batch_size,
                                                        class_mode = 'categorical')
print('fitting...')
history = model.fit_generator(training_set,
                         steps_per_epoch = nb_train_samples // DataInformation.batch_size,
                         epochs=DataInformation.epochs,
                         validation_data = test_set,
                         validation_steps=nb_validation_samples // DataInformation.batch_size,
                         verbose=2,
                         callbacks=[model_checkpoint])


history_dict = history.history
loss_train_epochs = history_dict['loss']
loss_val_epochs = history_dict['val_loss']

plt.figure()
plt.plot(loss_train_epochs,color='blue',label='train_loss')
plt.plot(loss_val_epochs,color='red',label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('epoch-loss.pdf')
plt.show()
plt.close()

modelTest= keras.models.load_model('best-model.h5')
print('model ok :load')
oss,acc = modelTest.evaluate(training_set,verbose=False)

print("L'accuracy sur l'ensemble du train est:",acc)

loss,acc = model.evaluate(test_set,verbose=False)

print("L'accuracy sur l'ensemble du test est:",acc)