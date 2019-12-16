
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib, time, nnet_survival
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, LSTM, GRU, Embedding, Concatenate, Conv1D, \
    GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, TimeDistributed
from keras import optimizers, layers, regularizers
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
import math
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from scipy import stats


numpy.set_printoptions(threshold=sys.maxsize)
pandas.set_option('display.max_rows', 30, 'display.max_columns', 50)

###################################### CVD 데이터 duration을 category로 변경 사용
dataset_training = read_csv('data/CVD/serial/cvd_serial_training_seq_1_2_conv_duration.csv', header=0)
dataset_testing = read_csv('data/CVD/serial/cvd_serial_testing_seq_1_2_conv_duration.csv', header=0)

################## training 데이터 ##################
cols, names = list(), list()
even = dataset_training.ix[::2, :]
even.reset_index(drop=True, inplace=True)
even.drop(even.columns[24], axis=1, inplace=True)
even.drop(even.columns[23], axis=1, inplace=True)
cols.append(even)


odd = dataset_training.ix[1::2, :]
training_duration_d = odd.ix[:, -1]  # training 정답 데이터를 위한 것
odd.reset_index(drop=True, inplace=True)
odd.drop(odd.columns[24], axis=1, inplace=True)
odd.drop(odd.columns[23], axis=1, inplace=True)
cols.append(odd)

train_reframed = concat(cols, axis=1)
training_values = train_reframed.values
train_values = training_values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(train_values)
#####################################################


################## testing 데이터 ##################
colss, namess = list(), list()
even = dataset_testing.ix[::2, :]
even.reset_index(drop=True, inplace=True)
even.drop(even.columns[24], axis=1, inplace=True)
even.drop(even.columns[23], axis=1, inplace=True)
colss.append(even)

odd = dataset_testing.ix[1::2, :]
testing_duration_d = odd.ix[:, -1]
odd.drop(odd.columns[24], axis=1, inplace=True)
odd.drop(odd.columns[23], axis=1, inplace=True)
odd.reset_index(drop=True, inplace=True)
colss.append(odd)

test_reframed = concat(colss, axis=1)
testing_values = test_reframed.values
test_values = testing_values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
test = scaler.fit_transform(test_values)
#####################################################

############################## GT 데이터 Y ############################
# training  X 는 맨마지막 CVD 의 값을 빼고, Y 는 CVD 값만
train_X = train[:, :]
test_X = test[:, :]
train_y = np_utils.to_categorical(training_duration_d)
test_y = np_utils.to_categorical(testing_duration_d)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape(train_X.shape[0], 2, 23)
test_X = test_X.reshape(test_X.shape[0], 2, 23)
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#######################################################################

def categorical():
    #################################################################################
    model = Sequential()
    model.add(LSTM(8, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_regularizer=regularizers.l2(0.1)))
    # model.add(GRU(128, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(8, activation="relu"))
    # model.add(Dropout(0.5))
    # model.add(Dense(1))
    # model.add(TimeDistributed(Dense(2, activation="sigmoid")))
    model.add(Dense(13, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adadelta(), metrics=['categorical_accuracy'])
    history = model.fit(train_X, train_y, epochs=30, batch_size=500, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    # history = model.fit(train_X, train_y, epochs=3, batch_size=256, validation_data=(train_X, train_y), verbose=2, shuffle=False)
    model.summary()
    # model.layers.Concatenate(axis=-1)
    #################################################################################

    # Final evaluation of the model
    # _loss, _acc, _precision, _recall, _f1score = model.evaluate(test_X, test_y, verbose=2)
    # print('loss: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1score: {:.3f}'.format(_loss, _acc, _precision, _recall, _f1score))
    scores = model.evaluate(test_X, test_y, verbose=0)
    print("categorical Loss: %.2f%%" % (scores[0] * 100))
    print("categorical Accuracy: %.2f%%" % (scores[1] * 100))
    print("precision: %.2f%%" % (scores[2] * 100))
    print("recall: %.2f%%" % (scores[3] * 100))

    #################################################################################
    # y_pred = model.predict_proba(test_X)
    # y_pred = y_pred/10
    # # oneyr_surv = np.cumprod(y_pred[:, 0:np.nonzero(breaks > 365)[0][0]], axis=1)[:, -1]
    # oneyr_surv = numpy.cumprod(y_pred[:, 0:13], axis=1)[:, -1] * 0.1
    #
    # print('================================')
    # print('Test data with concordance_index ')
    # print(concordance_index(testing_values[:, -1], oneyr_surv, testing_values[:, -2]))
    # print('================================')

    # fig, ax = pyplot.subplots()
    # ax.scatter(test_y, y_pred)
    # ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=4)
    # ax.set_xlabel('Measured')
    # ax.set_ylabel('Predicted')
    # pyplot.show()
    ################################################################################

    ################################################################################
    yhat = model.predict(test_X)
    test_y = test_y.reshape((len(test_y), 1))
    rmse = sqrt(mean_squared_error(yhat, test_y))
    print('Test RMSE: %.3f' % rmse)
    ###############################################################################


def LSTM_concate():
    inputShape = (train_X.shape[1], train_X.shape[2])
    input = Input(shape=inputShape)

    # category
    category = LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]))(input)
    category = Activation("relu")(category)
    category = Dropout(0.5)(category)
    category = Dense(13)(category)
    category = Activation("relu", name="category_output")(category)

    # binary
    binary = LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]))(input)
    binary = Activation("relu")(binary)
    binary = Dropout(0.5)(binary)
    binary = Dense(1)(binary)
    binary = Activation("relu", name="binary_output")(binary)

    merge_model = concatenate([category, binary])
    final_model = Dense(13, activation='softmax')(merge_model)
    model = Model(inputs=input, output=final_model, name="survival")

    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy', rmse])

    early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=20)
    history = model.fit(train_X, category_train_y, epochs=1, batch_size=256, validation_data=(test_X, category_test_y),
                        verbose=2, shuffle=False)
    model.summary()
    ################################################################################
    fig, loss_ax = pyplot.subplots()

    loss_ax.plot(history.history['loss'], 'y', label='train loss')
    loss_ax.plot(history.history['val_loss'], 'r', label='test loss')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')

    loss_ax.legend(loc='upper left')
    pyplot.show()

    fig2, loss_ax2 = pyplot.subplots()
    loss_ax2.plot(history.history['categorical_accuracy'], 'b', label='train acc')
    loss_ax2.plot(history.history['val_categorical_accuracy'], 'g', label='test acc')

    loss_ax2.set_xlabel('epoch')
    loss_ax2.set_ylabel('acc')
    loss_ax2.set_ylim([0, 1])

    loss_ax2.legend(loc='upper left')
    pyplot.show()

    fig3, rmse = pyplot.subplots()
    rmse.plot(history.history['rmse'], 'b', label='train rmse')
    rmse.plot(history.history['val_rmse'], 'g', label='test rmse')

    rmse.set_xlabel('epoch')
    rmse.set_ylabel('acc')
    rmse.set_ylim([0, 1])

    rmse.legend(loc='upper left')
    pyplot.show()
    ###################################################################################

    scores = model.evaluate(test_X, category_test_y, verbose=0)
    print('------------------------------------')
    # print("Loss: %.2f%%" % (scores[0] * 100))
    print('------------------------------------')

    yhat = model.predict(test_X)
    # test_y = test_y.reshape((len(test_y), 1))
    rmse = sqrt(mean_squared_error(yhat, category_test_y))
    print('Test RMSE: %.3f' % rmse)



def dynamicstep_build_data(engine, time, x, max_time, is_test, number):
    # y[0] will be days remaining, y[1] will be event indicator, always 1 for this data
    out_y = np.empty((0, 2), dtype=np.float32)

    # A full history of sensor readings to date for each x
    out_x = np.empty((0, max_time, 24), dtype=np.float32)
    for i in range(number):
        if i % 100 == 0:
            print("Engine: " + str(i))

        # 각 환자들의 시퀀스 최대 개수
        max_engine_time = int(np.max(time[engine == i])) + 1
        if is_test:
            start = max_engine_time - 1
        else:
            start = 0

        this_x = np.empty((0, max_time, 24), dtype=np.float32)
        for j in range(start, max_engine_time):
            # 각 1~100 에 해당하는 행렬  (192, 24) / (200, 24) / (186, 24)
            engine_x = x[engine == i]  # 1 인값에 대한 데이터 프레임 불러옴

            # 세로로 두번째 열에 대하여 펼침 (행 늘림)
            # out_y = np.append(out_y, np.array((max_engine_time - j, 1), ndmin=2), axis=0)

            # 1차원 100개 행, 5개 시퀀스, 24 열 행렬 생성
            xtemp = np.zeros((1, max_time, 24))
            xtemp[:, max_time - min(j, 4) - 1:max_time, :] = engine_x[max(0, j - max_time + 1):j + 1, :]

            # 계속 늘어나는 형태 xtemp를 this_x 계속 추가 (192, 100, 24)
            # 100x24 행렬이 192개 있다는 것
            this_x = np.concatenate((this_x, xtemp))
        out_x = np.concatenate((out_x, this_x))

    return out_x, out_y


def LSTM_survival_model():
    # 시퀀스 길이가 동적이니까 reshape 말고 특정지어서 나타냄 reshape는 정적 배열만 가능
    drop_data = data_training.drop(["duration_d", "CVD"], axis=1)
    person = []
    serial = []
    for idx, seq in enumerate(drop_data['seq']):
        if seq == 1:
            if idx == 0:
                serial.append(np.array(drop_data.iloc[idx].values))
                continue
            person.append(np.array(serial))
            serial = []
            serial.append(np.array(drop_data.iloc[idx].values))
        else:
            serial.append(np.array(drop_data.iloc[idx].values))

    x_train = np.array(person)

    model = Sequential()
    # n 개의 차원을 가지는 rnn 층 <- 이건 크게 상관 없는듯 128 정도 셋팅
    # input_shape는 (시퀀스의 길이(동적인데?),x_train.shape[2]) 동적으로 처리하는 부분 확인
    # 동적으로 처리하는 부분 input_shape=(None, x_train.shape[1])
    model.add(LSTM(20, input_shape=(12, 1)))
    model.add(Dense(7, activation='sigmoid'))

    # 이것은 그냥 위에랑 똑같은 것일까?
    model.add(Activation("relu"))
    model.add(Dense(1))
    model.summary(90)

    '''
    stateful 파라메터를 통해서 하나의 sequence 가 종료되면
    현재의 샘플 학습 상태가 다음 샘플의 초기 상태로 전달
    stateful 상태 초기화 true 로
    상태유지 모드에서는 모델 학습 시에 상태 초기화에 대한 고민이 필요
    현재 샘플 학습 상태가 다음 샘플 학습의 초기상태로 전달되는 식인데,
    현재 샘플과 다음 샘플 간의 순차적인 관계가 없을 경우에는 

    마지막 샘플 학습이 마치고, 새로운 에포크 수행 시에는 새로운 샘플 학습을 해야하므로 상태 초기화 필요
    한 에포크 안에 여러 시퀀스 데이터 세트가 있을 경우, 
    새로운 시퀀스 데이터 세트를 학습 전에 상태 초기화 필요

    현재 코드에서는 한 곡을 가지고 계속 학습을 시키고 있으므로 
    새로운 에포크 시작 시에만 상태 초기화를 수행
    에포크 수행시에만 상태 초기화 수행 코드 작성
    '''

    model.add(LSTM(20, batch_input_shape=(10000000, 12, 24), stateful=False))
    model.add(Dropout(0.5))
    early_stopping = EarlyStopping(monitor='loss', patience=20)
    # 예측하고자 하는 target의 개수에 따라 마지막에 dense 추가 해줌
    # 시퀀스 길이를 동적으로 해주기 위한 방법
    # LSTM(32, return_sequences=True, input_shape=(None, 5))
    model.compile(loss='categorical_crossentropy', optimizer='adam')