import numpy as np
import matplotlib.pyplot as plt 
import keras
import os
import densenet
import itertools,math
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau


import DataLoader,SPMF


class DataInformation:
    size=(42,42)
    img_width, img_height = size[0], size[1]
    batch_size = 32
    epochs = 60
    lr=0.03
    


class modelCNN:
    def __init__(self, dataName, network="densenet",labelTarget="subject", pictureMode='RGB'):
        """network : choosen networ (densenet, medium or small)"""
        self.dataName = dataName
        self.labelTarget= labelTarget
        outPath=DataLoader.DirPaths.savedCNNpictures
             
        if dataName == DataLoader.DataName.dance:
            minNFrames=800
            dirPath = DataLoader.DanceCSV.pictureDanceDBDirName
            if labelTarget.lower()=='subject':
                savingWeight='DanceDB_subject'
                #lr=0.003
                self.Depth= 43
                self.nb_dense_block =3
                self.growth_rate =12
                self.nb_filter = 22
                self.classes = [value for value in DataLoader.DanceCSV.subjects.keys()]
                
            else:
                savingWeight='DanceDB_action'
                self.classes = [value for value in DataLoader.DanceCSV.dance.keys()]                
        elif dataName == DataLoader.DataName.emotional:
            minNFrames=600
            dirPath = DataLoader.EmotionalCSV.pictureEmotionalDBPicture
            if labelTarget.lower()=='subject':
                savingWeight='EmotionalDB_subject'
                self.classes = [value for value in DataLoader.EmotionalCSV.subjects.keys()]
            else:
                savingWeight='EmotionalDB_action'
                self.classes = [value for value in DataLoader.EmotionalCSV.emotions.keys()]
        elif dataName == DataLoader.DataName.action:
            minNFrames=30
            dirPath = DataLoader.ActionCSV.pictureActionDBDirName
            if labelTarget.lower()=='action':
                savingWeight='MSRAction3D_action'
                self.classes = [value for value in DataLoader.ActionCSV.actions.values()]
            else:
                savingWeight='MSRAction3D_subject'
                self.classes = [i for i in range(1,DataLoader.ActionCSV.NumberSubject+1)]
                self.Depth=34            # Depth: int -- how many layers; "Depth must be 3*N + 4"
                self.nb_dense_block =3   # nb_dense_block: int -- number of dense blocks to add to end
                self.growth_rate =12     # growth_rate: int -- number of filters to add
                self.nb_filter = 16      # nb_filter: int -- number of filters

        self.Depth= 43           # Depth: int -- how many layers; "Depth must be 3*N + 4"
        self.nb_dense_block = 4   # nb_dense_block: int -- number of dense blocks to add to end
        self.growth_rate =12     # growth_rate: int -- number of filters to add
        self.nb_filter = 40      # nb_filter: int -- number of filters
                
                
        #######  get number of classes
        self.train_data_dir = os.path.join(os.path.join(DataLoader.DirPaths.PictureDataSet, dirPath),'Train/')
        self.validation_data_dir = os.path.join(os.path.join(DataLoader.DirPaths.PictureDataSet, dirPath),  'Validation/')
        nb_classes=len(self.classes)
        file_counterValidation = sum([len(files) for r, d, files in os.walk(self.validation_data_dir)])
        file_counterTrain = sum([len(files) for r, d, files in os.walk(self.train_data_dir)])
        nb_train_samples = file_counterTrain
        nb_validation_samples = file_counterValidation
        #calcul du batch size
        self.batch_size = nb_validation_samples//20
        #######  create pictures 
        if pictureMode=='RGB':
            print(' PICTURES CREATION...  ')
            SPMF.getPictureDataSet(dataName, labelTarget=labelTarget,minNFrames=minNFrames, size=DataInformation.size)
        
    
        
        if keras.backend.image_data_format() == 'channels_first':
            input_shape = (3, DataInformation.img_width, DataInformation.img_height)
        else:
            input_shape = (DataInformation.img_width, DataInformation.img_height, 3)
        
        
        ###### Personnal Network
        
        def small_model(nb_classes=None,
                        kernel_size = (3,3),
                        pool_size= (2,2),
                        first_filters = 32,
                        second_filters = 64,
                        dense=256):
            model = Sequential()
            model.add(Conv2D(first_filters, kernel_size, input_shape =(DataInformation.img_width, DataInformation.img_height, 3), activation = 'relu'))
            model.add(MaxPooling2D(pool_size =pool_size, strides=2 ))
            model.add(Conv2D(second_filters, kernel_size, activation = 'relu'))
            model.add(MaxPooling2D(pool_size = pool_size, strides=2))
            model.add(Flatten())
            model.add(Dense(units = dense, activation = 'relu'))
            model.add(Dense(units = nb_classes, activation = 'softmax'))
            return model
        
        def model3(nb_classes):
            #Load the VGG model
            vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(DataInformation.img_width, DataInformation.img_height, 3))
            # Freeze the layers except the last 4 layers
            for layer in vgg_conv.layers[:-4]:
                layer.trainable = False
            # Create the model
            model = Sequential()
             
            # Add the vgg convolutional base model
            model.add(vgg_conv)
             
            # Add new layers
            model.add(Flatten())
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(nb_classes, activation='softmax'))
            return model

            
        def create_model(nb_classes=None,
                    kernel_size = (3,3),
                    pool_size= (2,2),
                    first_filters = 32,
                    second_filters = 64,
                    third_filters = 128,
                    first_dense=256,
                    second_dense=128,
                    dropout_conv = 0.3,
                    dropout_dense = 0.3):
            model = Sequential()
            # First conv filters
            model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding="same",
                             input_shape = (DataInformation.img_width, DataInformation.img_height,3) ))
            model.add(Conv2D(first_filters, kernel_size, padding="same", activation = 'relu'))
            model.add(Conv2D(first_filters, kernel_size, padding="same", activation = 'relu'))
            model.add(MaxPooling2D(pool_size = pool_size)) 
            model.add(Dropout(dropout_conv))
            # Second conv filter
            model.add(Conv2D(second_filters, kernel_size, padding="same", activation ='relu'))
            model.add(Conv2D(second_filters, kernel_size, padding="same", activation ='relu'))
            model.add(Conv2D(second_filters, kernel_size, padding="same", activation ='relu'))
            model.add(MaxPooling2D(pool_size = pool_size))
            model.add(Dropout(dropout_conv))
            # Third conv filter
            model.add(Conv2D(third_filters, kernel_size, padding="same", activation ='relu'))
            model.add(Conv2D(third_filters, kernel_size, padding="same", activation ='relu'))
            model.add(Conv2D(third_filters, kernel_size, padding="same", activation ='relu'))
            model.add(MaxPooling2D(pool_size = pool_size))
            model.add(Dropout(dropout_conv))
            model.add(Flatten())
            # First dense
            model.add(Dense(first_dense, activation = "relu"))
            model.add(Dropout(dropout_dense))
            # Second dense
            model.add(Dense(second_dense, activation = "relu"))
            model.add(Dropout(dropout_dense))
            # Out layer
            model.add(Dense(nb_classes, activation = "softmax"))
            return model
             
        
                # learning rate schedule
        def step_decay(epoch):
        	initial_lrate = 0.001
        	drop = 0.1
        	epochs_drop = 100.0
        	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        	return lrate
        
        ######### Data Generators        
        train_datagen = ImageDataGenerator(rescale = None)
        test_datagen = ImageDataGenerator(rescale = None)
        
        train_generator = train_datagen.flow_from_directory(self.train_data_dir,
                                                            target_size = (DataInformation.img_width, DataInformation.img_height),
                                                            batch_size = self.batch_size,
                                                            class_mode = 'categorical')
        
        validation_generator = test_datagen.flow_from_directory(self.validation_data_dir,
                                                                target_size = (DataInformation.img_width, DataInformation.img_height),
                                                                batch_size = self.batch_size,
                                                                class_mode = 'categorical')
        
        
        #nb_classe2= len(train_generator.class_indices)
        print('model creation...')
        ######### Construct DenseNet architeture.
        if network == "medium":
            self.model = create_model(
                    nb_classes=nb_classes,
                    kernel_size = (3,3),
                    pool_size= (2,2),
                    first_filters = 32,
                    second_filters = 64,
                    third_filters = 128,
                    first_dense=256,
                    second_dense=128,
                    dropout_conv = 0.3,
                    dropout_dense = 0.3)
        elif network == 'densenet':
            self.model = densenet.DenseNet(nb_classes,input_shape,
                    self.Depth,				
                    self.nb_dense_block,				
                    self.growth_rate,				
                    self.nb_filter,				
                    dropout_rate=0.2,
                    weight_decay=0.0001)
        elif network == 'small':
             self.model = small_model(nb_classes=nb_classes,
                        kernel_size = (3,3),
                        pool_size= (2,2),
                        first_filters = 32,
                        second_filters = 64,
                        dense=256)
        elif network =='model3':
            self.model = model3(nb_classes)
            
            
        self.model.summary()
        #optimizer_algo = keras.optimizers.SGD(lr=DataInformation.lr)
        
        # Compile the model.
        #loss = 'sparse_categorical_crossentropy',
        
        #optimizer_algo=Adam(lr=DataInformation.lr)
        self.model.compile(optimizer=Adam(lr=DataInformation.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                           loss = 'categorical_crossentropy', metrics = ['accuracy'])
                
 
        ###### name of weight
        if network=='densenet':
            savingWeight = savingWeight+'_D'+str(self.Depth)+'DB'+str(self.nb_dense_block)+'G'+str(self.growth_rate)+'F'+str(self.nb_filter)+str(DataInformation.img_height)
        #Fit model.
        print('Data Generators : OK')	
        #########  saving  options
        weightPath=DataLoader.DirPaths.weightLocation
        if not os.path.exists(weightPath):
            os.makedirs(weightPath)
        savingWeihtFile = os.path.join(weightPath,savingWeight+'_Network'+network+'.h5')
            
        earlystopper = EarlyStopping(monitor='val_acc', patience=15, verbose=2)
        
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=4, 
                                   verbose=2, mode='max', min_lr=0.00001)
        
        # Save the best model during the traning
        checkpointer = ModelCheckpoint(savingWeihtFile,monitor='val_acc',
                                       verbose=2 ,save_best_only=True,
                                       save_weights_only=True)
        callbacks_list = [earlystopper, checkpointer, reduce_lr]

        ######### train model 
        #if os.path.exists(savingWeihtFile):
        #    self.model.load_weights(savingWeihtFile)
        #print('model loaded')
        
        history = self.model.fit_generator(train_generator,
                                      steps_per_epoch= nb_train_samples // self.batch_size,
                                      epochs=DataInformation.epochs,
                                      validation_data=validation_generator,
                                      validation_steps= nb_validation_samples // self.batch_size,
                                      callbacks=callbacks_list,
                                      verbose=2)
        
        ########   Saving weight.
        #self.model.save_weights(savingWeight+'_Network'+network+'.h5')
        # Get the best saved weights
        #self.model.load_weights(savingWeihtFile)
        #print('model loaded')
        
        oss,acc = self.model.evaluate_generator(train_generator, steps=len(train_generator))
        print(" Accuracy on Training Set : ",round(acc*100,2))
        
        loss,acc = self.model.evaluate_generator(validation_generator, steps=len(validation_generator))
        print(" Accuracy on Test Set :",round(acc*100,2))
        
    
        
        print('Predictions on Validation set...')
        ##### ---------------  Validation
        datagen = ImageDataGenerator(rescale =None)
        generator = datagen.flow_from_directory( self.validation_data_dir ,target_size=DataInformation.size,
                                                batch_size=1,
                                                class_mode=None,  # only data, no labels
                                                shuffle=False)    # keep data in same order as labels
        #self.model.load_weights(savingWeight+'.h5')
        y_pred = self.model.predict_generator(generator,nb_validation_samples)
        y_pred  = np.argmax(y_pred, axis=-1)
        label_map = (train_generator.class_indices)
        print('label_map')
        print(label_map)
        
        y_true =generator.classes
        #print('last y_true')
        #y_truetrue=np.array(list(itertools.chain.from_iterable([[int(os.path.basename(r))]*len(files) for r, d, files in os.walk(self.validation_data_dir) if os.path.basename(r)!=''])))
        
        cnf_matrix = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
        print('Confusion Matrix...')
        #print(confusion_matrix(y_true, y_pred))
        
        if not os.path.exists(outPath):
            os.makedirs(outPath)
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, self.classes ,dataName, normalize=True,title='_')
        plt.savefig(outPath+dataName+'_'+labelTarget+'_ConfusionMatrix_epochs'+str(DataInformation.epochs)+'.png')
        # List all data in history.
        #print(history.history.keys())
        # grab the history object dictionary
        H = history.history
        
        last_test_acc = history.history['val_acc'][-1] * 100
        last_train_loss = history.history['loss'][-1] 
        last_test_acc = round(last_test_acc, 2)
        last_train_loss = round(last_train_loss, 6)
        
        train_acc = 'Train Accuracy'
        train_loss = 'Training Loss'
        test_acc = 'Test Accuracy '
        test_loss = 'Test Loss'
        
         
        # plot the training loss and accuracy
        N = np.arange(0, len(H["loss"]))
        plt.style.use("ggplot")
        plt.figure()
        axes = plt.gca()
        axes.set_ylim([0.0,1.8])
        
        plt.plot(N, H['loss'],linewidth=1.5,label=train_loss,color='blue')
        plt.plot(N, H['val_acc'],linewidth=1.5, label=test_acc,color='red')
                
        plt.title('DenseNet '+dataName+' '+labelTarget,fontsize=8, fontweight='bold',color = 'Gray')
        plt.xlabel('Number of Epochs',fontsize=10, fontweight='bold',color = 'Gray')
        plt.ylabel('Training on Data without Rotated Images',fontsize=8, fontweight='bold',color = 'Gray')
        plt.legend()
        # Save the figure.    
        plt.savefig(outPath+dataName+'_'+labelTarget+'_epochs'+str(DataInformation.epochs)+'_val_acc_And_loss.png')
        plt.show()
        # summarize history for loss
        plt.plot(N, H['acc'],linewidth=1.5, label=train_acc,color='blue')
        plt.plot(N, H['val_loss'],linewidth=1.5, label=test_loss,color='red')
        
        plt.title('DenseNet '+dataName+' '+labelTarget,fontsize=8, fontweight='bold',color = 'Gray')
        plt.ylabel('Training on Data without Rotated Images',fontsize=8, fontweight='bold',color = 'Gray')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(outPath+dataName+'_'+labelTarget+'_epochs'+str(DataInformation.epochs)+'_val_loss_And_acc.png')
        plt.show()
        # losses
        plt.plot(N, H['loss'],linewidth=1.5,    label=train_loss, color='blue')
        plt.plot(N, H['val_loss'],linewidth=1.5,label=test_loss, color='red')
        plt.title('DenseNet '+dataName+' '+labelTarget,fontsize=8, fontweight='bold',color = 'Gray')
        plt.ylabel('Loss ',fontsize=8, fontweight='bold',color = 'Gray')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(outPath+dataName+'_'+labelTarget+'_epochs'+str(DataInformation.epochs)+'_Losses.png')
        plt.show()


def plot_confusion_matrix(cm, classes,dataName,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if dataName == DataLoader.DataName.dance:
        title='DanceDB'
    elif dataName == DataLoader.DataName.emotional:
        title='EmotionalDB'
    elif dataName == DataLoader.DataName.action:
        title='MSRAction3D'
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
    plt.tight_layout()
    plt.ylabel('True Action')
    plt.xlabel('Predicted Action')
                
                     
    #en cours en ligne de commande : cnn = modelCNN(DataLoader.DataName.emotional, labelTarget="subject", network="densenet", pictureMode='RGB')
## ---------- TEST -------------
#cnn = modelCNN(DataLoader.DataName.action, network="medium",labelTarget="action", pictureMode='RGB')
#cnn = modelCNN(DataLoader.DataName.dance, labelTarget="subject", network="personal",pictureMode='RGB')  #ok
cnn = modelCNN(DataLoader.DataName.emotional, labelTarget="action", network="densenet", pictureMode='RGB')
#cnn = modelCNN(DataLoader.DataName.action, network="model3",labelTarget="action", pictureMode='RGB')
