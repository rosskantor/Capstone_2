from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input
from keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard, ModelCheckpoint
from datetime import datetime, timedelta
from distutils.dir_util import copy_tree
from sklearn.model_selection import GridSearchCV
from shutil import copyfile, move
import random
import math
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage.transform import resize
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as scaler
import os
from PIL import Image
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

def run_code(im_size = 64, source= 'Prod', c_scale='rgb', epoch=90):

    if source == 'Prod':
        train = '../data/Train_24'
        test = '../data/Test_24_b'
        train_SIZE=78300
        test_SIZE = 8700
        dense = 24
    if source =='Quick':
        train = '../data/quick_Train'
        test = '../data/quick_Test'
        train_SIZE=22
        test_SIZE = 22
        dense = 2
    if source == 'Test':
        train = '../data/Alpha_Small_Train'
        test = '../data/Alpha_Small_Test'
        train_SIZE = 6061
        test_SIZE = 1900
        dense = 24
    if source == 'White':
        train = '../data/asl_render_train'
        test = '../data/asl_render_test'
        train_SIZE = 78300
        test_SIZE = 1000
        dense = 24
    if source == 'Attempt_2':
        train = '../data/Train_24'
        test = '../data/Test_24_b'
        train_SIZE = 78300
        test_SIZE = 1000
        dense = 24

    now = datetime.now()

    filepath = '../Models' + str(now) + ".h5"
    """
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)]
    """

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1),
        TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=16, write_graph=True, write_grads=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
        ]


    batch_size = 16

    model = build_model(dense,im_size, c_scale='rgb')

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            train,  # this is the target directory
            target_size=(im_size, im_size),  # all images will be resized to im_sizexim_size
            batch_size=batch_size,
            color_mode=c_scale,
            class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

    labels = train_generator.class_indices

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            test,
            target_size=(im_size, im_size),
            batch_size=1,
            color_mode=c_scale,
            class_mode='categorical')

    history = model.fit_generator(
            train_generator,
            steps_per_epoch=train_SIZE // batch_size,
            epochs=epoch,
            validation_data=validation_generator,
            callbacks=callbacks,
            validation_steps=test_SIZE//batch_size)

    plotter(history)
    # serialize model to YAML
    model.save('../logs/' + str(now) +'.h5')
def plotter (history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def build_model( dense,im_size = 64, c_scale='rgb'):

    if c_scale == 'grayscale':
        im_dem = 1
    else:
        im_dem = 3

    lrate = LearningRateScheduler(step_decay)

    K.set_image_dim_ordering('th')

    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', input_shape=(im_dem, im_size, im_size)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))


    model.add(Conv2D(64, (3, 3),strides=(1, 1), padding='same'))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Conv2D(64, (3, 3),strides=(1, 1), padding='same'))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Conv2D(64, (3, 3),strides=(1, 1), padding='same'))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(dense))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def step_decay(epoch):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def predict():
    pass

def copy_tree():
    # copy subdirectory example
    rand_holder= [random.randint(1,3000) for i in range(120)]
    #dir_holder = ['A','B','C','D','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        #'Q','R','S','T','U','V','W','X','Y','Z','space','del','nothing']
    dir_holder = ['O', 'P','Q','R','S','T','U','V','W','X','Y','Z','space','del','nothing']
    for Dir in dir_holder:
        for rand in rand_holder:
            toDirectory = '../data/Alpha_Small_Train/{0}/{0}{1}.jpg'.format(Dir,rand)
            fromDirectory = '../data/asl_alphabet_train/{0}/{0}{1}.jpg'.format(Dir,rand)
            copyfile(fromDirectory, toDirectory)

def cut_tree():
    # copy subdirectory example
    rand_holder= [3,4,206,213,540,548,938,941,1327,1634,1646, 2098,2101,2613,2615,2877, 2881,
            2906, 2915, 2914,2997, 3000]
    dir_holder = ['A','B','C','D','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q','R','S','T','U','V','W','X','Y','Z','space','del','nothing']
    #dir_holder = ['O', 'P','Q','R','S','T','U','V','W','X','Y','Z','space','del','nothing']
    for Dir in dir_holder:
        for rand in rand_holder:
            toDirectory = '../data/Train_24/{0}/{0}{1}.jpg'.format(Dir,rand)
            fromDirectory = '../data/Test_24/{0}/{0}{1}.jpg'.format(Dir,rand)
            move(fromDirectory, toDirectory)

def copy_tree_2():
    # copy subdirectory example
    rand_holder= [random.randint(1,3000) for i in range(50)]
    dir_holder = ['A','B','C','D','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q','R','S','T','U','V','W','X','Y','Z','space','del','nothing']
    #dir_holder = ['O', 'P','Q','R','S','T','U','V','W','X','Y','Z','space','del','nothing']
    for Dir in dir_holder:
        for rand in rand_holder:
            toDirectory = '../data/Alpha_Small_Test/{0}/{0}{1}.jpg'.format(Dir,rand)
            fromDirectory = '../data/asl_alphabet_train/{0}/{0}{1}.jpg'.format(Dir,rand)
            copyfile(fromDirectory, toDirectory)

def grid_search(model):

    X, Y= array_builder()

    # define the grid search parameters
    batch_size = [10, 20, 40, 60, 80, 100]
    epochs = [10, 50, 75,100, 150, 200]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X, Y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def array_builder(rand=True):
    # copy subdirectory example
    if rand==True:
        rand_holder= [random.randint(1,3000) for i in range(400)]
    else:
        rand_holder = [i for i in range(1,3001)]

    im_holder = []
    y_holder = []
    dir_holder = ['A','B','C','D','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q','R','S','T','U','V','W','X','Y','Z','space','del','nothing']
    #dir_holder = ['O', 'P','Q','R','S','T','U','V','W','X','Y','Z','space','del','nothing']
    for Dir in dir_holder:
        for i in rand_holder:
            image = cv2.imread('../data/asl_alphabet_train/{0}/{0}{1}.jpg'.format(Dir,i))
            im_holder.append(image.reshape(1,120000))
            y_holder.append(Dir)
    return np.array(im_holder), np.array(y_holder)

def mlp_array_builder(n):
    # copy subdirectory example
    rand_holder= [random.randint(1,3000) for i in range(n)]


    im_holder = []
    y_holder = []
    dir_holder = ['A','B','C','D','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q','R','S','T','U','V','W','X','Y','Z','space','del','nothing']
    #dir_holder = ['O', 'P','Q','R','S','T','U','V','W','X','Y','Z','space','del','nothing']
    for Dir in dir_holder:
        for i in rand_holder:
            image = cv2.imread('../data/asl_alphabet_train/{0}/{0}{1}.jpg'.format(Dir,i))
            img = image[:,:,0]
            im_holder.append(img)
            y_holder.append(Dir)
    le = preprocessing.LabelEncoder()
    le.fit(y_holder)
    y_holder = le.transform(y_holder)
    X = np.array([np.resize(im_holder[i], (96,96)) for i in range(len(im_holder))])
    X = X.reshape(len(y_holder),9216)
    return X, y_holder

def random_forest(lower_dimension, approximation, y_train):
    dir = '../Models2/'
    clf_pca_lower = RandomForestClassifier(n_estimators=100, max_depth=None,
    min_samples_split=2, random_state=0, n_jobs=-1)
    clf_pca_higher = clf_pca_lower
    clf_pca_lower.fit(lower_dimension, y_train.reshape(len(y_train),))
    clf_pca_higher.fit(approximation, y_train.reshape(len(y_train),))
    with open(dir + 'clf_pca_lower', 'wb') as file:
        pickle.dump(clf_pca_lower, file)
    with open(dir + 'clf_pca_higher', 'wb') as file:
        pickle.dump(clf_pca_higher, file)
    return clf_pca_lower, clf_pca_higher

def array_builder_white_out(n):
    # copy subdirectory example
    x = np.empty([1,16384])
    y = np.empty([1,1])
    rand_holder= [random.randint(1,3000) for i in range(n)]
    dir_holder = ['A', 'B','C','D','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q','R','S','T','U','V','W','X','Y','Z','space','del','nothing']
    labels = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
        'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
        'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26,
        'nothing': 27, 'space': 28}
    #dir_holder = ['O', 'P','Q','R','S','T','U','V','W','X','Y','Z','space','del','nothing']
    for Dir in dir_holder:
        for i in rand_holder:
            image = cv2.imread('../data/asl_alphabet_train/{0}/{0}{1}.jpg'.format(Dir,i))
            image = image[:,:,0]
            image = cv2.resize(image, (128,128))
            im2 = image.reshape(1, 16384)
            x = np.vstack((x, im2))
            y = np.vstack((y,labels[Dir]))
    return x, y

def boost(n):
    X, y = mlp_array_builder(n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=100)
    clf.fit(X, y)
    return clf

def grid_search(clf):
    param_grid = {'n_estimators': [200, 500, 700, 1000],
            'max_features': ['auto', 'sqrt', 'log2'],
            'min_samples_leaf': [5,10,15],
            'max_depth': [2,4,6,8,10]}
    random_search = GridSearchCV(clf, param_grid=param_grid
                                   ,cv=5)

    random_search.fit(X_train, y_train)

def predict_builder():

    x = np.empty([1,3,128,128])
    y_holder = []
    labels = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
    'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26,
    'nothing': 27, 'space': 28}
    rand_holder = [2,10,14,54,70,79,398,405,411,429,436,721,738,985,994,1566
        ,1572,1575,1824,1829,1834,2034,2043,2309,2327,2919,2921,2927,2998,2999]
    model = load_model('../logs/LastModel.h5')
    pred_datagen = ImageDataGenerator(rescale=1./255)
    dir_holder = ['A','B','C','D','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q','R','S','T','U','V','W','X','Y','Z','space','del','nothing']
    #dir_holder = ['O', 'P','Q','R','S','T','U','V','W','X','Y','Z','space','del','nothing']
    for Dir in dir_holder:
        for i in rand_holder:
            image = cv2.imread('../data/ASL_Large_Validation/{0}/{0}{1}.jpg'.format(Dir,i))
            image = cv2.resize(image, (128,128))
            image = image.T
            image = np.expand_dims(image,axis=0)
            x = np.vstack((x, image))
            y_holder.append(labels[Dir])
    #return model.predict_classes(x[1:]), np.array(y_holder)
    return x, y_holder

def graph_dist(x, y, chart_title):
    sns.jointplot(x, y , kind="kde")
    plt.title(chart_title)
    plt.show()

def answer_man(im_size, c_scale='rgb'):

    model = load_model('../logs/LastModel.h5')

    pred_datagen = ImageDataGenerator(rescale=1./255
                                    , data_format='channels_first')

    validation_generator = pred_datagen.flow_from_directory(
            '../data/asl_render_validation',
            target_size=( im_size, im_size),
            batch_size=1,
            color_mode=c_scale,
            class_mode='categorical')

    zz= model.predict_generator(validation_generator, steps=870)

    return zz
def summation(zz):
    output = [np.sum(answer[i:(30*i)+30],axis=0) for i in range(29)]

def outline_creator(Dir='F', im_index=8):
    #load image
    image = cv2.imread('../data/asl_alphabet_train/{0}/{0}{1}.jpg'.format(Dir,im_index))

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    ret,thresh1 = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    max_area=0

    for i in range(len(contours)):
                cnt=contours[i]
                area = cv2.contourArea(cnt)
                if(area>max_area):
                    max_area=area
                    ci=i
    cnt=contours[ci]
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(image.shape,np.uint8)
    cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
    plt.imshow(drawing)
    plt.show()

def edges(n):
    x = np.empty([1,9216])
    y = np.empty([1,1])
    rand_holder= [random.randint(1,3000) for i in range(n)]
    dir_holder = ['A','B','C','D','E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q','R','S','T','U','V','W','X','Y']
    labels = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
    'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26,
    'nothing': 27, 'space': 28}
    for Dir in dir_holder:
        for im_index in rand_holder:
            img=cv2.imread('../data/asl_alphabet_train/{0}/{0}{1}.jpg'.format(Dir,im_index))
            img = cv2.resize(img, (96,96))
            edges = cv2.Canny(img,96,96)
            edges = edges.reshape(1, 9216)
            print(Dir, im_index)
            x = np.vstack((x, edges))
            y = np.vstack((y,labels[Dir]))
    return x[1:], y[1:]

def edges_2(n):
    x = np.empty([1,4096])
    y = np.empty([1,1])
    dir_holder = ['A','B','C','D','E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q','R','S','T','U','V','W','X','Y']
    labels = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
    'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26,
    'nothing': 27, 'space': 28}
    counter = 0
    for Dir in dir_holder:
        file_path = '../data/Train_24/'+ Dir + '/'
        t = n
        while t > 0:
            file = random.choice(os.listdir(file_path))
            img=cv2.imread(file_path + file)
            print((file_path + file))
            if '.jpg' in file:
                if img.shape[0] > 64 and img.shape[1] > 64:
                    print((file_path + file))
                    t -= 1
                    print(t)
                    img = cv2.resize(img, (64,64))
                    edges = cv2.Canny(img,64,64)
                    edges = edges.reshape(1, 4096)
                    x = np.vstack((x, edges))
                    y = np.vstack((y,labels[Dir]))

    return x[1:], y[1:]

def to_pca(x):
    dir = '../Models2/'
    pca = PCA(.95)
    s = scaler()
    x = s.fit_transform(x)
    lower_dimension = pca.fit_transform(x)
    approximation = pca.inverse_transform(lower_dimension)
    dims = pca.n_components_
    joblib.dump(s.scale_ , dir + 'scaler')
    with open(dir + 'pca', 'wb') as file:
        pickle.dump(pca, file)
    np.save(dir + 'lower_dimension', lower_dimension)
    np.save(dir + 'approximation', approximation)
    return lower_dimension, approximation
def reload_scaler():
    dir = '../Models2/'
    scaler = joblib.load(dir + 'scaler')
    return scaler
def individual_images_to_pca(ind_image):
    img = cv2.resize(ind_image, (64,64))
    edges = cv2.Canny(img,64,64)
    edges = edges.reshape(1, 4096)
    pca = PCA(.95)
    s = scaler()
    x = s.fit_transform(edges)
    lower_dimension = pca.fit_transform(x)
    approximation = pca.inverse_transform(lower_dimension)
    return lower_dimension, approximation
def variance_graph(pca, dims):

    tot = sum(pca.explained_variance_)
    var_exp = [(i/tot)*100 for i in sorted(pca.explained_variance_, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    # PLOT OUT THE EXPLAINED VARIANCES SUPERIMPOSED
    plt.figure(figsize=(10, 5))
    plt.step(range(1, dims), cum_var_exp, where='mid',label='cumulative explained variance')
    plt.title('Cumulative Explained Variance as a Function of the Number of Components')
    plt.ylabel('Cumulative Explained variance')
    plt.xlabel('Principal components')
    plt.axhline(y = 95, color='k', linestyle='--', label = '95% Explained Variance')
    plt.axhline(y = 90, color='c', linestyle='--', label = '90% Explained Variance')
    plt.axhline(y = 85, color='r', linestyle='--', label = '85% Explained Variance')
    plt.legend(loc='best')
    plt.show()

def orig_versus_pca(approximation, x, dims, id):
    plt.figure(figsize=(8,4));

    # Original Image
    plt.subplot(1, 2, 1);
    plt.imshow(x[id].reshape(96,96),
                  cmap = plt.cm.gray, interpolation='nearest',
                  clim=(0, 255));
    plt.xlabel(str(96**2) +  ' components', fontsize = 14)
    plt.title('Original Image', fontsize = 20);

    # 154 principal components
    plt.subplot(1, 2, 2);
    plt.imshow(approximation[id].reshape(96, 96),
                  cmap = plt.cm.gray, interpolation='nearest',
                  clim=(0, 255));
    plt.xlabel(str(dims) + ' components', fontsize = 14)
    plt.title('95% of Explained Variance', fontsize = 20);

    plt.show()
def open_pickled_model():
    filename = '../Models2/randomforest30'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
def video_extract():
    dir_holder = ['A','B','C','D','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q','R','S','T','U','V','W','X','Y','Z']
    for dir in dir_holder:
        cam = cv2.VideoCapture('../data/Video_Image_3/' + str(dir) +'.mov')
        currentframe = 3001
        starttime = time.time()
        while(currentframe <= 3303):

            # reading from frame
            ret,frame = cam.read()

            if ret:
                # if video is still left continue creating images
                name = '../data/Train_24/' + str(dir) + '/' + str(int_iterator('../data/Train_24/' + str(dir) + '/' )+1000) + '.jpg'
                print ('Creating...' + name)

                # writing the extracted images
                cv2.imwrite(name, frame)

                # increasing counter so that it will
                # show how many frames are created
                currentframe += 1
            else:
                break
        #https://www.geeksforgeeks.org/extract-images-from-video-in-python/


def visible_image():
    visible_dir = '../data/dataset5/'
    visible_out_dir = '../data/Train_24/'
    visible_dir_1 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    visible_dir_2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                    'u', 'v', 'w', 'x', 'y', 'z']
    im_container = []
    for visible in visible_dir_1:
        for visible_2 in visible_dir_2:
            file_path = visible_dir + visible + '/' + visible_2 + '/'
            if os.path.isdir(file_path):
                for file in os.listdir(file_path):
                    if 'color' in file:
                        fromDirectory = file_path + file
                        toDirectory = visible_out_dir  + visible_2.upper()
                        move(fromDirectory, toDirectory + '/' + str(int_iterator(toDirectory)) + '.jpg')
def int_iterator(string_path):
    return len(os.listdir(string_path)) + 1

def blackened_image_unblackened():
    black_dir = '../data/ds9/'
    black_out_dir = '../data/'
    black_dir_1 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    black_dir_2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                    'u', 'v', 'w', 'x', 'y', 'z']
    for black in black_dir_1:
        for black_2 in black_dir_2:
            file_path = black_dir + black + '/' + black_2 + '/'
            if os.path.isdir(file_path):
                for file in os.listdir(file_path):
                    fromDirectory = file_path + '/' + file
                    x = np.array(Image.open(fromDirectory))
                    im = Image.fromarray(x.astype(np.uint8))
                    toDirectory = black_out_dir + '/' + black_dir.upper()
                    move(fromDirectory, toDirectory + '/' + int_iterator(toDirectory) + '.png')

def copy_tree_new():
    # copy subdirectory example
    rand_holder= [random.randint(1,3000) for i in range(120)]
    dir_holder = ['A','B','C','D','E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q','R','S','T','U','V','W','X','Y']
    #dir_holder = ['O', 'P','Q','R','S','T','U','V','W','X','Y','Z','space','del','nothing']
    for visible in visible_dir_1:
        for visible_2 in visible_dir_2:
            file_path = visible_dir + visible + '/' + visible_2 + '/'
            if os.path.isdir(file_path):
                for file in os.listdir(file_path):
                    if 'color' in file:
                        fromDirectory = file_path + file
                        toDirectory = visible_out_dir  + visible_2.upper()
                        move(fromDirectory, toDirectory + '/' + str(int_iterator(toDirectory)) + '.jpg')
            copyfile(fromDirectory, toDirectory)
def save_array(X, y):
    dir = '../Models2/'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    np.save(dir + 'X_all', X)
    np.save(dir + 'y_all', y)
    np.save(dir + 'X_tr', X_train)
    np.save(dir + 'X_te', X_test)
    np.save(dir + 'y_tr', y_train)
    np.save(dir + 'y_te', y_test)
    return X_train, X_test, y_train, y_test
def load_arrays():
    x_te = np.loadtxt('../Models2/X_test')
    x_tr = np.loadtxt('../Models2/X_train')
    y_te = np.loadtxt('../Models2/y_test')
    y_tr = np.loadtxt('../Models2/y_train')
    return x_te, x_tr, y_te, y_tr
def prediction():
    model = open_pickled_model()
    return model
def image_record():

    cap = cv2.VideoCapture(0)
    then = datetime.now() + timedelta(seconds=0.05)
    while(datetime.now() < then):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here

        # Display the resulting frame

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return frame
