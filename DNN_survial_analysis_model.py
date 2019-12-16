from __future__ import print_function

running_time_test = 0
if running_time_test:  # disable GPU, set Keras to use only 1 CPU core
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    import keras.backend as K
    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1, allow_soft_placement=True,
                            device_count={'CPU': 1, 'GPU': 0})
    session = tf.Session(config=config)
    K.set_session(session)
else:
    import keras.backend as K

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, LSTM, GRU, Embedding, Concatenate, Conv1D, \
    GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, TimeDistributed
from keras import optimizers, layers, regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import math
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from scipy import stats
import time
import nnet_survival
import other_code.cox_nnet as cox_nnet  # for cox-nnet baseline model
import tensorflow as tf

import sys

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 5, 'display.max_columns', 1000)

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

import logging

logging.getLogger().setLevel(logging.CRITICAL)

def calib_plot(fu_time, n_bins, pred_surv, time, dead, color, label, error_bars=0, alpha=1., markersize=1.,
               markertype='o'):
    cuts = np.concatenate(
        (np.array([-1e6]), np.percentile(pred_surv, np.arange(100 / n_bins, 100, 100 / n_bins)), np.array([1e6])))
    bin = pd.cut(pred_surv, cuts, labels=False, duplicates='drop')
    kmf = KaplanMeierFitter()
    est = []
    ci_upper = []
    ci_lower = []
    mean_pred_surv = []
    for which_bin in range(max(bin) + 1):
        kmf.fit(time[bin == which_bin], event_observed=dead[bin == which_bin])
        est.append(np.interp(fu_time, kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate))
        ci_upper.append(np.interp(fu_time, kmf.survival_function_.index.values,
                                  kmf.confidence_interval_.loc[:, 'KM_estimate_upper_0.95']))
        ci_lower.append(np.interp(fu_time, kmf.survival_function_.index.values,
                                  kmf.confidence_interval_.loc[:, 'KM_estimate_lower_0.95']))
        mean_pred_surv.append(np.mean(pred_surv[bin == which_bin]))
    est = np.array(est)
    ci_upper = np.array(ci_upper)
    ci_lower = np.array(ci_lower)
    if error_bars:
        plt.errorbar(mean_pred_surv, est, yerr=np.transpose(np.column_stack((est - ci_lower, ci_upper - est))), fmt='o',
                     c=color, label=label)
    else:
        plt.plot(mean_pred_surv, est, markertype, c=color, label=label, alpha=alpha, markersize=markersize)
    return (mean_pred_surv, est)


def AUC(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def Load_data():
    # baseline
    path_training = 'data/cvd_serial_testing_seq_test_2seq.csv'
    path_testing = 'data/cvd_serial_testing_seq_test_2.csv'

    # serial
    # path_training = 'data/cvd_serial_training_seq.csv'
    # path_testing = 'data/cvd_serial_testing_seq.csv'

    data_training = pd.read_csv(path_training)
    print(" training file name : " + path_training)
    data_testing = pd.read_csv(path_testing)
    print(" testing file name : " + path_testing)
    print('-------------------------------------------\n')

    # proportion of patients to place in training set (7 : 3)
    train_prop = 0.7
    np.random.seed(0)

    # 학습용 데이터를 0.7의 비율로 랜덤으로 샘플링
    # # (총 데이터의 개수 중에서 몇개의 데이터만 뽑고, 중복 허용 여부로 False는 중복 안됨 )
    # train_indices = np.random.choice(len(support_data), int(train_prop * len(support_data)), replace=False)

    # # 첫번재 배열로 부터 np.arange(개수) 만큼 배열 생성 후, train_indices에서 추출된 값들은 삭제
    # test_indices = np.setdiff1d(np.arange(len(support_data)), train_indices)

    # # 학습용으로 data_all 중에서도 0.7프로 비율로 뽑은 랜덤 숫자로 랜덤 row값들을 선택
    # data_train = support_data.iloc[train_indices]
    # data_test = support_data.iloc[test_indices]

    data_train = data_training
    data_test = data_testing

    # time열과 dead 열을 제거, 독립변수만을 사용하기 위해서
    # 그리고 matrix로 convert(pandas)
    x_train = data_train.drop(["duration_d", "CVD"], axis=1).as_matrix()
    x_test = data_test.drop(["duration_d", "CVD"], axis=1).as_matrix()

    ########### 일반 표준화
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    ########### 이상치, 특이값 제거(중앙값 IQR) 표준화
    # x_train = RobustScaler().fit_transform(x_train)
    # x_test = RobustScaler().fit_transform(x_test)

    ########## minmax 표준화
    # minmaxscaler = MinMaxScaler().fit(x_train)
    # x_train = minmaxscaler.transform(x_train)
    # x_test = minmaxscaler.transform(x_test)


def cox_Proportional_hazard_model():
    ##################################################################
    print('---------------------------------------')
    print('Standard Cox proportional hazards model')
    print('---------------------------------------')
    '''
    Standard Cox proportional hazards model
    '''
    cph = CoxPHFitter()
    cph.fit(data_train, duration_col='duration_d', event_col='CVD', show_progress=True)
    # cph.print_summary()

    # Cox model discrimination train set
    prediction = cph.predict_partial_hazard(data_train)
    print(
        "\ntrain data c-index = " + str(concordance_index(data_train.duration_d, -prediction, data_train.CVD)))

    # Cox model discrimination test set
    prediction = cph.predict_partial_hazard(data_test)
    print("\ntest data c-index = " + str(concordance_index(data_test.duration_d, -prediction, data_test.CVD)))


def ANN_survival_model():
    #################################################################
    print('-------------------------------------------------------------')
    print('start cross-validation to pick L2 regularization strength for training')
    print('-------------------------------------------------------------')
    halflife = 365. * 2.8

    breaks = -np.log(1 - np.arange(0.0, 0.96, 0.05)) * halflife / np.log(2)
    n_intervals = len(breaks) - 1
    timegap = breaks[1:] - breaks[:-1]

    # y_train = nnet_survival.make_surv_array(data_train.time.values, data_train.dead.values, breaks)
    # y_test = nnet_survival.make_surv_array(data_test.time.values, data_test.dead.values, breaks)

    y_train = nnet_survival.make_surv_array(data_train.duration_d.values, data_train.CVD.values, breaks)
    y_test = nnet_survival.make_surv_array(data_test.duration_d.values, data_test.CVD.values, breaks)

    # uncensored 데이터와 censored 데이터를 구분
    # uncensored 데이터는 2번째 배열에서 dead 인터벌에 1값
    # censored 데이터는 2번째 배열에서 0 값
    hidden_layers_sizes = 7  # Using single hidden layer, with this many neurons

    from sklearn.model_selection import KFold

    n_folds = 10
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    early_stopping = EarlyStopping(monitor='loss', patience=20)

    # l2_array = np.concatenate(([0.],np.power(10.,np.arange(-6,-2))))
    l2_array = np.power(10., np.arange(-4, 1))
    grid_search_train = np.zeros((len(l2_array), n_folds))
    grid_search_test = np.zeros((len(l2_array), n_folds))
    print('execution of 10-fold validation for five times\n')
    for i in range(1):
    # for i in range(len(l2_array)):
        print(str(i + 1) + ' / ' + str(len(l2_array)) + " times")
        j = 0
        cv_folds = kf.split(x_train)
        for traincv, testcv in cv_folds:
            x_train_cv = x_train[traincv]
            y_train_cv = y_train[traincv]
            x_test_cv = x_train[testcv]
            y_test_cv = y_train[testcv]

            # 활성함수는 렐루, 마지막 레이어에 시그모이드, iterator 1000, 7차원 hidden layer
            model = Sequential()
            # model.add(Dense(n_intervals,input_dim=x_train.shape[1],bias_initializer='zeros',kernel_regularizer=regularizers.l2(l2_array[i])))

            # 입력층 개수는 변수의 개수
            model.add(Dense(hidden_layers_sizes, input_dim=x_train.shape[1], bias_initializer='zeros', activation='relu',
                            kernel_regularizer=regularizers.l2(l2_array[i])))
            # model.add(Activation('relu'))
            model.add(Dense(n_intervals))
            model.add(Activation('sigmoid'))

            model.compile(loss=nnet_survival.surv_likelihood(n_intervals), optimizer=optimizers.RMSprop())  # lr=0.0001))

            history = model.fit(x_train_cv, y_train_cv, batch_size=256, epochs=100000, callbacks=[early_stopping],
                                verbose=0)
            # model.summary()
            print(model.metrics_names)
            grid_search_train[i, j] = model.evaluate(x_train_cv, y_train_cv, verbose=0)
            print(grid_search_train[i, j])
            grid_search_test[i, j] = model.evaluate(x_test_cv, y_test_cv, verbose=0)
            print(grid_search_test[i, j])
            j = j + 1


    print(np.average(grid_search_train, axis=1))
    print(np.average(grid_search_test, axis=1))
    l2_final = l2_array[np.argmax(-np.average(grid_search_test, axis=1))]

   ############################### plot ######################################
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(history.history['loss'], 'y', label='train loss')
    loss_ax.plot(history.history['val_loss'], 'r', label='test loss')

    acc_ax.plot(history.history['acc'], 'b', label='train acc')
    acc_ax.plot(history.history['val_acc'], 'g', label='test acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.show()
    ###########################################################################
    score = model.evaluate(x_test, y_test, batch_size=2, verbose=1)
    # print('Test loss: ', score[0])
    # print('Test accuracy: ', score[1])

    # Discrimination performance
    y_pred = model.predict_proba(x_train, verbose=1)
    oneyr_surv = np.cumprod(y_pred[:, 0:np.nonzero(breaks > 365)[0][0]], axis=1)[:, -1]

    print('================================')
    print('Training data with concordance_index ')
    print(concordance_index(data_train.duration_d, oneyr_surv, data_train.CVD))
    print('================================')

    y_pred = model.predict_proba(x_test, verbose=1)
    oneyr_surv = np.cumprod(y_pred[:, 0:np.nonzero(breaks > 365)[0][0]], axis=1)[:, -1]

    print('================================')
    print('Test data with concordance_index ')
    print(concordance_index(data_test.duration_d, oneyr_surv, data_test.CVD))
    print('================================')
    ##########################################################################


def cox_nnet_model():
    print('------------------------')
    print('start cox-nnet modeling \n')

    # cross validation on training set to pick L2 regularization strength
    model_params = dict(node_map=None, input_split=None)
    search_params = dict(method="nesterov", learning_rate=0.01, momentum=0.9,
                         max_iter=10000, stop_threshold=0.995, patience=1000, patience_incr=2,
                         rand_seed=123, eval_step=23, lr_decay=0.9, lr_growth=1.0)
    cv_params = dict(L2_range=np.arange(-6, 2.1))

    likelihoods, L2_reg_params, mean_cvpl = cox_nnet.L2CVProfile(x_train, data_train.duration.as_matrix(),
                                                                 data_train.CVD.as_matrix(),
                                                                 model_params, search_params, cv_params, verbose=False)

    L2_reg = L2_reg_params[np.argmax(mean_cvpl)]  # Best L2_reg is -5

    # train final model
    L2_reg = -5.
    model_params = dict(node_map=None, input_split=None, L2_reg=np.exp(L2_reg))
    cox_nnet_model, cox_nnet_cost_iter = cox_nnet.trainCoxMlp(x_train, data_train.duration.as_matrix(),
                                                              data_train.CVD.as_matrix(), model_params, search_params,
                                                              verbose=False)
    cox_nnet_theta_train = cox_nnet_model.predictNewData(x_train)
    cox_nnet_theta_test = cox_nnet_model.predictNewData(x_test)

    # discrimination on train, test sets
    print(' evaluate cox-nnet modeling ')
    print(concordance_index(data_train.duration, -cox_nnet_theta_train, data_train.CVD))
    print(concordance_index(data_test.duration, -cox_nnet_theta_test, data_test.CVD))


def binary_ANN_surviavl():
    breaks = np.arange(0, 5000, 50)
    n_intervals = len(breaks) - 1
    timegap = breaks[1:] - breaks[:-1]

    halflife1 = 200
    halflife2 = 400
    halflife_cens = 400
    n_samples = 5000
    np.random.seed(seed=0)
    t1 = np.random.exponential(scale=1 / (np.log(2) / halflife1), size=int(n_samples / 2))
    t2 = np.random.exponential(scale=1 / (np.log(2) / halflife2), size=int(n_samples / 2))
    t = np.concatenate((t1, t2))
    censtime = np.random.exponential(scale=1 / (np.log(2) / (halflife_cens)), size=n_samples)
    f = t < censtime
    t[~f] = censtime[~f]

    y_train = nnet_survival.make_surv_array(t, f, breaks)
    x_train = np.zeros(n_samples)
    x_train[int(n_samples / 2):] = 1

    model = Sequential()
    # Hidden layers would go here. For this example, using simple linear model with no hidden layers.
    model.add(Dense(1, input_dim=1, use_bias=0, kernel_initializer='zeros'))
    model.add(nnet_survival.PropHazards(n_intervals))
    model.compile(loss=nnet_survival.surv_likelihood(n_intervals), optimizer=optimizers.RMSprop())
    # model.summary()
    early_stopping = EarlyStopping(monitor='loss', patience=2)
    history = model.fit(x_train, y_train, batch_size=32, epochs=1000, callbacks=[early_stopping])
    y_pred = model.predict_proba(x_train, verbose=0)

    kmf = KaplanMeierFitter()
    kmf.fit(t[0:int(n_samples / 2)], event_observed=f[0:int(n_samples / 2)])
    plt.plot(breaks, np.concatenate(([1], np.cumprod(y_pred[0, :]))), 'bo-')
    plt.plot(kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate, color='k')
    kmf.fit(t[int(n_samples / 2) + 1:], event_observed=f[int(n_samples / 2) + 1:])
    plt.plot(breaks, np.concatenate(([1], np.cumprod(y_pred[-1, :]))), 'ro-')
    plt.plot(kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate, color='k')
    plt.xticks(np.arange(0, 2000.0001, 200))
    plt.yticks(np.arange(0, 1.0001, 0.125))
    plt.xlim([0, 2000])
    plt.ylim([0, 1])
    plt.xlabel('Follow-up time (days)')
    plt.ylabel('Proportion surviving')
    plt.title('One covariate. Actual=black, predicted=blue/red.')
    plt.show()

    myData = pd.DataFrame({'x_train': x_train, 't': t, 'f': f})
    cf = CoxPHFitter()
    cf.fit(myData, 't', event_col='f')
    # x_train = x_train.astype(np.float64)
    # cox_coef = cf.hazards_.x_train.values[0]
    cox_coef = cf.hazards_.x_train
    nn_coef = model.get_weights()[0][0][0]
    print('Cox model coefficient:')
    print(cox_coef)
    print('Cox model hazard ratio:')
    print(np.exp(cox_coef))
    print('Neural network coefficient:')
    print(nn_coef)
    print('Neural network hazard ratio:')
    print(np.exp(nn_coef))


def kaplanmeierfilter():
    kmf = KaplanMeierFitter()
    kmf.fit(data_train.duration_d, data_train.CVD, label='KaplanMeier Estimate')
    kmf.plot(title='CVD', ci_show=False)
    plt.xlabel('timeline')
    plt.ylabel('survival')
    plt.show()

    kmf = KaplanMeierFitter()
    kmf.fit(support_data.time, support_data.dead, label='KaplanMeier Estimate')
    kmf.plot(title='support_data', ci_show=False)
    plt.xlabel('timeline')
    plt.ylabel('survival')
    plt.show()


def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())
    return recall


def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())
    return precision


def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    _f1score = (2 * _recall * _precision) / (_recall + _precision + K.epsilon())
    return _f1score


if __name__ == "__main__":
    start_time = time.time()

    ############# COX model ############
    # cox_Proportional_hazard_model()
    # e = int(time.time() - start_time)
    # print('\nfinish cox proportional hazard rate - {:02d}:{:02d}:{:02d}\n\n'.format(e // 3600, (e % 3600 // 60), e % 60))
    ##############################

    ANN_survival_model()

    e = int(time.time() - start_time)
    print('\nfinish nnet-survival modeling and c-index evaluation - {:02d}:{:02d}:{:02d}'.format(e // 3600,
                                                                                                 (e % 3600 // 60),
                                                                                                 e % 60))

    # cox_nnet_model()

    binary_ANN_surviavl()

    # kaplanmeierfilter()
