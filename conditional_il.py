import gc
import h5py
import numpy as np
from glob import glob
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, concatenate
from keras import metrics
from functools import partial
import tensorflow as tf


#VARIABLES
dataset_train = glob("/home/john/Documentos/Imitation_Learning/SeqTrain/*")
#dataset_test = glob("./dataset/SeqVal/*")
#VARIABLES

# Generator
def batch_generator(file_names, batch_size=200, masks=None):
    """ High level command: 1 -> Speed (Focus on Speed only) // 2 -> Follow lane // 3 -> Go Left // 4 -> Go Right // 5 -> Straight """
    file_idx = -1
    while True:
        if file_idx > len(file_names):
            file_idx = 0
        file_idx = file_idx + 1
        batch_x = []
        batch_y = []
        batch_s = []
        for i in range(0, batch_size):
            data = h5py.File(file_names[file_idx], 'r')
            for mask in masks:
                if mask == 1:
                    batch_x.append(data['rgb'][i]) #Img
                    batch_s.append(data['targets'][i][10]) #Speed
                elif data['targets'][i][24] == mask:
                    batch_x.append(data['rgb'][i]) #Img
                    batch_y.append(data['targets'][i][:3]) #Steer, Gas, Brake (Output for other models)
                    batch_s.append(data['targets'][i][10]) #Speed
            data.close()
        yield ([np.array(batch_x), np.array(batch_s)], [np.array(batch_s) if mask == 1 else np.array(batch_y) for mask in masks])



def batch_generator_improved(file_names, batch_size=200, masks=None):
    """ High level command: 1 -> Speed (Focus on Speed only) // 2 -> Follow lane // 3 -> Go Left // 4 -> Go Right // 5 -> Straight """
    file_idx = -1
    while True:
        if file_idx > len(file_names): #lista com todos os arquivos h5
            file_idx = 0
        file_idx = file_idx + 1
        batch_x = []
        batch_y = []
        batch_s = []

        while len(batch_x) < batch_size: # verifica se já tem os 200 exemplos
            if file_idx > len(file_names):# se o índice for maior que o total de arquivos h5 zera o contador
                file_idx = 0
            for i in range(file_idx, len(file_names)):# do menor índice do arqivo h5 até o total
                for j in range(0, batch_size):
                    #print(len(file_names), " - file_names[" + str(i) + "] " + str(file_names[i]), " j: ", j)
                    data = h5py.File(file_names[i], 'r') #recebe o arquivo a h5 de menor índice
                    for mask in masks:
                        if mask == 1:
                            batch_x.append(data['rgb'][j]) #Img
                            batch_s.append(data['targets'][j][10]) #Speed
                        elif data['targets'][j][24] == mask: # se o ídice 24 do exemplo é do mesmo tipo da máscara escolhida
                            batch_x.append(data['rgb'][j]) #Img
                            batch_y.append(data['targets'][j][:3]) #Steer, Gas, Brake (Output for other models)
                            batch_s.append(data['targets'][j][10]) #Speed
                    data.close()
                    if len(batch_x) >= batch_size:
                        break
                if len(batch_x) >= batch_size:
                    break
            if len(batch_x) >= batch_size:
                break
            file_idx = file_idx + 1
        yield ([np.array(batch_x), np.array(batch_s)], [np.array(batch_s) if mask == 1 else np.array(batch_y) for mask in masks])

# Network
def load_model():
    image_size = (88, 200, 3)
    input_image = (image_size[0], image_size[1], image_size[2])
    input_speed = (1,)

    branch_config = [
        ["Speed"], #Speed
        ["Steer", "Gas", "Brake"], #Follow
        ["Steer", "Gas", "Brake"], #Left
        ["Steer", "Gas", "Brake"], #Right
        ["Steer", "Gas", "Brake"]  #Straight
    ]

    branch_names = ['Speed', 'Follow', 'Left', 'Right', 'Straight']

    branches = []

    def conv_block(inputs, filters, kernel_size, strides):
        x = Conv2D(filters, (kernel_size, kernel_size), strides=strides, activation='relu')(inputs)
        x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        return x

    def fc_block(inputs, units):
        fc = Dense(units, activation='relu')(inputs)
        fc = Dropout(0.5)(fc)
        return fc

    xs = Input(shape=input_image, name='rgb')
    '''inputs, filters, kernel_size, strides'''
    """ Conv 1 """
    x = conv_block(xs, 32, 5, 2)
    x = conv_block(x, 32, 3, 1)
    """ Conv 2 """
    x = conv_block(x, 64, 3, 2)
    x = conv_block(x, 64, 3, 1)
    """ Conv 3 """
    x = conv_block(x, 128, 3, 2)
    x = conv_block(x, 128, 3, 1)
    """ Conv 4 """
    x = conv_block(x, 256, 3, 1)
    x = conv_block(x, 256, 3, 1)
    """ Reshape """
    x = Flatten()(x)
    """ FC1 """
    x = fc_block(x, 512)
    """ FC2 """
    x = fc_block(x, 512)
    """Process Control"""
    """ Speed (measurements) """

    sm = Input(shape=input_speed, name='speed')
    speed = fc_block(sm, 128)
    speed = fc_block(speed, 128)
    """ Joint sensory """
    j = concatenate([x, speed])
    j = fc_block(j, 512)

    for i in range(len(branch_config)):
        if branch_config[i][0] == "Speed":
            branch_output = fc_block(x, 256)
            branch_output = fc_block(branch_output, 256)
        else:
            branch_output = fc_block(j, 256)
            branch_output = fc_block(branch_output, 256)
        fully_connected = Dense(len(branch_config[i]), name=branch_names[i], activation=tf.keras.layers.LeakyReLU(alpha=0.3))(branch_output)
        branches.append(fully_connected)

    models = []
    for branche in branches:
        if "Speed" in branche.name:
            model = Model(inputs=[xs, sm], outputs=[branche], name="Speed")
        elif "Follow" in branche.name:
            model = Model(inputs=[xs, sm], outputs=[branche], name="Follow")
        elif "Left" in branche.name:
            model = Model(inputs=[xs, sm], outputs=[branche], name="Left")
        elif "Right" in branche.name:
            model = Model(inputs=[xs, sm], outputs=[branche], name="Right")
        elif "Straight" in branche.name:
            model = Model(inputs=[xs, sm], outputs=[branche], name="Straight")
        models.append(model)

    return models #retorna os 5 modelos

def dataset_branche_split(file_names):
    follow_dataset = []
    left_dataset = []
    right_dataset = []
    straight_dataset = []
    for i in range(0, len(file_names)):
        data = h5py.File(file_names[i], 'r')
        for j in range(0, len(data['rgb'])):
            if data['targets'][j][24] == 2:
                follow_dataset.append(file_names[i])
            elif data['targets'][j][24] == 3:
                left_dataset.append(file_names[i])
            elif data['targets'][j][24] == 4:
                right_dataset.append(file_names[i])
            elif data['targets'][j][24] == 5:
                straight_dataset.append(file_names[i])

    follow_dataset = list(dict.fromkeys(follow_dataset))
    left_dataset = list(dict.fromkeys(left_dataset))
    right_dataset = list(dict.fromkeys(right_dataset))
    straight_dataset = list(dict.fromkeys(straight_dataset))

    return follow_dataset, left_dataset, right_dataset, straight_dataset

print("Quantidade de arquivos de treino: ", len(dataset_train))
#print("Quantidade de arquivos de teste: ", len(dataset_test))
speed_train_dataset = dataset_train
#speed_test_dataset = dataset_test

follow_train_dataset, left_train_dataset, right_train_dataset, straight_train_dataset = dataset_branche_split(dataset_train)
#follow_test_dataset, left_test_dataset, right_test_dataset, straight_test_dataset = dataset_branche_split(dataset_test)

print("Train: " ,len(follow_train_dataset), " ", len(left_train_dataset), " ", len(right_train_dataset), " ", len(straight_train_dataset))
#print("Test: " ,len(follow_test_dataset), " ", len(left_test_dataset), " ", len(right_test_dataset), " ", len(straight_test_dataset))
# High level command: 1 - Speed, 2 - Follow lane, 3 - Go Left, 4 - Go Right, 5 - Straight


all_models = load_model()
for model in all_models:
    if "Speed" in model.name:
        HighLevel = 1
        print("Speed ", HighLevel)
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0002), metrics=[metrics.mean_squared_error, metrics.mean_absolute_error, metrics.mean_absolute_percentage_error])
        model.fit(x=batch_generator_improved(dataset_train, 120, [HighLevel]), epochs=50, steps_per_epoch=500, verbose=1)
        model.save("50_Speed_model.h5")
        model.fit(x=batch_generator_improved(dataset_train, 120, [HighLevel]), epochs=50, steps_per_epoch=500, verbose=1)
        model.save("100_Speed_model.h5")
        model.fit(x=batch_generator_improved(dataset_train, 120, [HighLevel]), epochs=150, steps_per_epoch=500, verbose=1)
        model.save("250_Speed_model.h5")
    elif "Follow" in model.name:
        HighLevel = 2
        print("Follow ", HighLevel)
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0002), metrics=[metrics.mean_squared_error, metrics.mean_absolute_error, metrics.mean_absolute_percentage_error])
        model.fit(x=batch_generator_improved(follow_train_dataset, 120, [HighLevel]), epochs=50, steps_per_epoch=500, verbose=1)
        model.save("50_Follow_model.h5")
        model.fit(x=batch_generator_improved(follow_train_dataset, 120, [HighLevel]), epochs=50, steps_per_epoch=500, verbose=1)
        model.save("100_Follow_model.h5")
        model.fit(x=batch_generator_improved(follow_train_dataset, 120, [HighLevel]), epochs=150, steps_per_epoch=500, verbose=1)
        model.save("250_Follow_model.h5")
    elif "Left" in model.name:
        HighLevel = 3
        print("Left ", HighLevel)
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0002), metrics=[metrics.mean_squared_error, metrics.mean_absolute_error, metrics.mean_absolute_percentage_error])
        model.fit(x=batch_generator_improved(left_train_dataset, 120, [HighLevel]), epochs=50, steps_per_epoch=500, verbose=1)
        model.save("50_Left_model.h5")
        model.fit(x=batch_generator_improved(left_train_dataset, 120, [HighLevel]), epochs=50, steps_per_epoch=500, verbose=1)
        model.save("100_Left_model.h5")
        model.fit(x=batch_generator_improved(left_train_dataset, 120, [HighLevel]), epochs=150, steps_per_epoch=500, verbose=1)
        model.save("250_Left_model.h5")
    elif "Right" in model.name:
        HighLevel = 4
        print("Right ", HighLevel)
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0002), metrics=[metrics.mean_squared_error, metrics.mean_absolute_error, metrics.mean_absolute_percentage_error])
        model.fit(x=batch_generator_improved(right_train_dataset, 120, [HighLevel]), epochs=50, steps_per_epoch=500, verbose=1)
        model.save("50_Right_model.h5")
        model.fit(x=batch_generator_improved(right_train_dataset, 120, [HighLevel]), epochs=50, steps_per_epoch=500, verbose=1)
        model.save("100_Right_model.h5")
        model.fit(x=batch_generator_improved(right_train_dataset, 120, [HighLevel]), epochs=150, steps_per_epoch=500, verbose=1)
        model.save("250_Right_model.h5")
    elif "Straight" in model.name:
        HighLevel = 5
        print("Straight ", HighLevel)
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0002), metrics=[metrics.mean_squared_error, metrics.mean_absolute_error, metrics.mean_absolute_percentage_error])
        model.fit(x=batch_generator_improved(straight_train_dataset, 120, [HighLevel]), epochs=50, steps_per_epoch=500, verbose=1)
        model.save("50_Straight_model.h5")
        model.fit(x=batch_generator_improved(straight_train_dataset, 120, [HighLevel]), epochs=50, steps_per_epoch=500, verbose=1)
        model.save("100_Straight_model.h5")
        model.fit(x=batch_generator_improved(straight_train_dataset, 120, [HighLevel]), epochs=150, steps_per_epoch=500, verbose=1)
        model.save("250_Straight_model.h5")
