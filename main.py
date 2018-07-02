# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import preprocessing as pp
import magic as mc

from matplotlib import pyplot as plt
train = pd.read_csv('train.csv',sep=',')
test = pd.read_csv('test.csv',sep=',')
y_DF = train['value']
#print(y_DF)
train = train.drop(['value'],axis=1)
categorical = [
    'Огорожена территория','Входные группы',
    'Спортивная площадка','Автомойка',
    'Кладовые','Колясочные','Система мусоротведения',
    'Подземная парковка','Двор без машин','Класс объекта',
]

train = train.drop(["date1"], axis=1)
train = train.drop(['bulk_id'], axis=1)
test = test.drop(['date1'], axis=1)
test = test.drop(['bulk_id'], axis=1)
train = train.drop(['start_square'], axis=1)
train = train.drop(['plan_s'], axis=1)
train = train.drop(['plan_m'], axis=1)
train = train.drop(['plan_l'], axis=1)
train = train.drop(['vid_0'], axis=1)
train = train.drop(['vid_1'], axis=1)
train = train.drop(['vid_2'], axis=1)


for index, feature in enumerate(categorical):
    train = pp.transform_to_one_hot(train, feature, dummy_na=True)
    test = pp.transform_to_one_hot(test, feature, dummy_na=True)
    print('  ' + str(index + 1) + '/' + str(len(categorical)) + '\t' + str(feature))
#pp.transform_to_one_hot(train,'Огорожена территория',dummy_na=True)
column = list(train)
print(column)
train.info()
print(train['Поликлиника'].sort_values)

train.spalen.replace(to_replace = 0, value = -1, inplace = True)
train['Поликлиника'].replace(to_replace = 0, value = -1, inplace= True)

kek = [
    'Поликлиника','spalen'
]
for index, feature in enumerate(kek):
    train = pp.transform_to_one_hot(train, feature, dummy_na=True)
    test = pp.transform_to_one_hot(test, feature, dummy_na=True)
    print('  ' + str(index + 1) + '/' + str(len(categorical)) + '\t' + str(feature))


# train.to_csv('r.csv', index = True)
mc.train_and_test(train, y_DF, test)
