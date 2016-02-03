from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD,Adam
import scipy.io as scio
import numpy as np
import h5py
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('Loading dataset...')
compact_data = h5py.File('compact_dataset_101315.mat')
X_train = compact_data['train_data'][()]
Y_train = compact_data['train_label'][()]
#X_train = X_train[1:11,:]
#y_train = y_train[1:11,:]

#X_train.transpose()
#y_train.transpose()

X_test = compact_data['test_data'][()]
Y_test = compact_data['test_label'][()]
compact_data.close()

size_input = X_train.shape[1]
print(size_input, 'dims')
size_label = Y_train.shape[1]
print(size_label, 'classes')

#X_train /= 10
#X_test /= 10

model = Sequential()
model.add(Dense(17100, 25000, init='uniform'))
model.add(Activation('tanh'))
#model.add(Dense(200, 200, init='uniform'))
#model.add(Activation('tanh'))
model.add(Dense(25000, 2128, init='uniform'))
model.add(Activation('tanh'))
#model.add(Dense(7000, 3000, init='uniform'))
#model.add(Activation('tanh'))
#model.add(Dense(3000, 2000, init='uniform'))
#model.add(Activation('tanh'))
#model.add(Dense(2000, 80, init='uniform'))
#model.add(Activation('tanh'))
sgd = SGD(lr=5, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(loss='mean_squared_error', optimizer=adam)

batch_size = 1000
best_score = 1e23
nb_epoch = 200
for e in range(nb_epoch):
    print('Epoch'+str(e))
    print("Training...")
    for e_sub in range(X_train.shape[0]/batch_size):
        X_batch = X_train[e_sub*batch_size:(e_sub+1)*batch_size,:]
        Y_batch = Y_train[e_sub*batch_size:(e_sub+1)*batch_size,:]
        loss = model.train_on_batch(X_batch, Y_batch)
        print('Batch # '+str(e_sub)+' Training loss:'+str(loss))
        #print("Testing...")

    if ((e%5) == 0):
        # test time! 
        #score = model.evaluate(X_test, Y_test, batch_size=1)
        #print('Test score:', score)
        #print('Best score so far:', best_score)
        print('Saving the test predictions... epoch ', e)
        prediction = model.predict(X_test, batch_size = 1)
        
        temp_str = 'prediction_101315_dataset_all_test_epoch_'+str(e)+'.mat'
        temp_str_2 = 'prediction_101315_dataset_all_test_epoch_'+str(e)
        scio.savemat(temp_str, mdict = {temp_str_2: prediction})
# When you load it in matlab, load it as matrix
# Then,
# temp3 = reshape(temp, 80, 100)
# temp4 = temp3'
