from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

labels={1:'high wave', 2:'horizontal wave', 3:'hammer', 4:'hand catch',5:'Foward punch',
   6:'high throw',7:'draw X',8:'draw tick',9:'draw circle',
   10:'hand clap',11:'two hand clap',12:'side boxing',13:'bend',
   14:'forward kick',15:'side kick',16:'jogging',17:'tennis swing',18:'tennis serve',19:'golf swing', 20:'pick-up throw'}



def confusionMatrix(labelTruth, labelPredicted):
    print('----------  Visualisation  ------------')
    
    #print(classification_report(labelTruth,labelPredicted , target_names=[l for l in labels.values()]))

    conf_mat = confusion_matrix(labelTruth, labelPredicted)
    
    fig = plt.figure(figsize=(6,6))
    width = np.shape(conf_mat)[1]
    height = np.shape(conf_mat)[0]
    
    res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            if c>0:
                plt.text(j-.2, i+.1, c, fontsize=16)
                
    cb = fig.colorbar(res)
    plt.title('Confusion Matrix : Actions')
    