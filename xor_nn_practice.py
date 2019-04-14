import numpy as np
import tensorflow as tf
import keras
from keras import optimizers
from keras import regularizers
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Flatten
import matplotlib.pyplot as plt

X = np.array([[0,0],[1,0],[0,1],[1,1],[0,1],[1,1],[0,0],[1,0]])
print(X, X.shape)
y = np.array([[0],[1],[1],[0],[1],[0],[0],[1]])
print(y,y.shape)
test_X = np.array([[1,1],[0,0],[0,1],[1,0]])
test_y = np.array([[0],[0],[1],[1]])
model = keras.Sequential()
model.add(Dense(20,input_dim=(2)))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer = optimizers.Adam(lr=0.01))
history = model.fit(X,y,batch_size = 4, epochs=50, validation_data = (test_X,test_y), shuffle=True)

value= model.predict_classes(np.array([[0,0]]))
print("Output of XOR is %s" %(value[0]))
value= model.predict_classes(np.array([[1,0]]))
print("Output of XOR is %s" %(value[0]))
value= model.predict_classes(np.array([[0,1]]))
print("Output of XOR is %s" %(value[0]))
value= model.predict_classes(np.array([[1,1]]))
print("Output of XOR is %s" %(value[0]))
      
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc = 'upper left')
plt.show()
