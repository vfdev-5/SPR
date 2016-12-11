import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from utils import TARGET_LABELS

def visualize_train_test(X_train, X_test, profile, Y_train=None, Y_test=None):
    plt.figure(figsize=(12,6))        
    ll = len(profile)
    if Y_train is not None and Y_test is not None:
        ll += 1
    
    def _nb_bins(x):
        r = np.max(x) - np.min(x)
        return max(min(r, 150), 1)
    
    for i, feature in enumerate(profile):
        plt.subplot(2,ll, i+1)
        plt.title("Train '%s' hist" % (feature))
        
        x = X_train[feature]
        nbbins = _nb_bins(x)

        X_train[feature].hist(bins=nbbins)
        
        plt.subplot(2,ll, ll + i + 1)
        plt.title("Test '%s' hist" % (feature))
        X_test[feature].hist(bins=nbbins)

    if Y_train is not None and Y_test is not None:
        plt.subplot(2, ll, ll)
        plt.title("Train targets hist")
        Y_train[TARGET_LABELS].sum(axis=0).plot.bar()

        plt.subplot(2, ll, 2*ll)
        plt.title("Test targets hist")
        Y_test[TARGET_LABELS].sum(axis=0).plot.bar()
    

def visualize_folds(x_df, nb_folds, profile, y_df=None, select_active_clients=True):
    kf = KFold(n_splits=nb_folds)
    for train_index, test_index in kf.split(range(x_df.shape[0])):
        X_train, X_val = x_df.loc[x_df.index[train_index], :], x_df.loc[x_df.index[test_index], :]
        if y_df is not None:
            Y_train, Y_val = y_df.loc[y_df.index[train_index], :], y_df.loc[y_df.index[test_index], :]            
        else:
            Y_train, Y_val = None, None
            
        if select_active_clients:
            mask = X_train['diff'] > 0
            visualize_train_test(X_train[mask], X_val, profile, Y_train[mask], Y_val)            
        else:
            visualize_train_test(X_train, X_val, profile, Y_train, Y_val)    
            
            
def compare_two_datasets(X1, X2, profile, Y1=None, Y2=None, title=""):
    
    x1 = X1[profile].values
    x2 = X2[profile].values
    
    plt.figure(figsize=(12,4))  
    plt.suptitle(title)
    ll = len(profile)
    if Y1 is not None and Y2 is not None:
        ll += 1
    
    for i, feature in enumerate(profile):
        r1 = np.max(x1[:, i]) - np.min(x1[:, i])
        r2 = np.max(x2[:, i]) - np.min(x2[:, i])
        r = max(r1, r2)
        nbbins = max(min(r, 100), 1)
        h1 = np.histogram(x1[:, i], bins=nbbins)
        h2 = np.histogram(x2[:, i], bins=nbbins)
#         print "%s : %f, %f" % (feature, np.min(h1[0]), np.max(h1[0]))
#         print "%s : %f, %f" % (feature, np.min(h2[0]), np.max(h2[0]))
        left = h1[1][:-1]
        h1 = h1[0]*1.0/np.max(h1[0])    
        h2 = h2[0]*1.0/np.max(h2[0])    
        diff = h1 - h2
        plt.subplot(1, ll, i+1)
        plt.title("x1-x2 '%s' hist" % (feature))
#         plt.bar(left-0.2, h1, width=0.2, color='r', align='center')
#         plt.bar(left+0.2, h2, width=0.2, color='b', align='center')
        plt.bar(left, diff, width=0.2, color='g', align='center')

    if Y1 is not None and Y2 is not None:
        y1 = Y1[TARGET_LABELS].sum(axis=0).values
        y2 = Y2[TARGET_LABELS].sum(axis=0).values
        h1 = np.histogram(y1, bins=24)
        h2 = np.histogram(y2, bins=24)
        left = h1[1][:-1]
        h1 = h1[0]*1.0/np.max(h1[0])    
        h2 = h2[0]*1.0/np.max(h2[0]) 
        diff = h1 - h2
        plt.subplot(1, ll, ll)
        plt.title("y1-y2 targets hist")
#         plt.bar(left-0.2, h1, width=0.2, color='r', align='center')
#         plt.bar(left+0.2, h2, width=0.2, color='b', align='center')
        plt.bar(left, diff, width=0.2, color='g', align='center')
    
    
    
def compare_folds(x_df, nb_folds, profile, y_df=None, select_active_clients=True):
    
    kf = KFold(n_splits=nb_folds)
    for train_index, test_index in kf.split(range(x_df.shape[0])):
        X_train, X_val = x_df.loc[x_df.index[train_index], :], x_df.loc[x_df.index[test_index], :]
        if y_df is not None:
            Y_train, Y_val = y_df.loc[y_df.index[train_index], :], y_df.loc[y_df.index[test_index], :]            
        else:
            Y_train, Y_val = None, None

        if select_active_clients:
            mask = X_train['diff'] > 0
            X_train = X_train[mask] 
            Y_train = Y_train[mask]
            
        compare_two_datasets(X_train, X_val, profile, Y_train, Y_val, 'Compare between Train/Val')


def compare_folds2(x_df, nb_folds, profile, y_df=None, select_active_clients=True):
    
    train_indices = []
    test_indices = []
    
    kf = KFold(n_splits=nb_folds)
    for train_index, test_index in kf.split(range(x_df.shape[0])):
        train_indices.append(train_index)
        test_indices.append(test_index)

    ll = len(train_indices)
    count = 0
    for i in range(ll):
        
        train_index, test_index = train_indices[count], test_indices[count]
        X_train_1, X_val_1 = x_df.loc[x_df.index[train_index], :], x_df.loc[x_df.index[test_index], :]
        if y_df is not None:
            Y_train_1, Y_val_1 = y_df.loc[y_df.index[train_index], :], y_df.loc[y_df.index[test_index], :]            
        else:
            Y_train_1, Y_val_1 = None, None
        count += 1
        count = count % ll
        
        if select_active_clients:
            mask = X_train_1['diff'] > 0
            X_train_1 = X_train_1[mask] 
            Y_train_1 = Y_train_1[mask]
        
        train_index, test_index = train_indices[count], test_indices[count]
        X_train_2, X_val_2 = x_df.loc[x_df.index[train_index], :], x_df.loc[x_df.index[test_index], :]
        if y_df is not None:
            Y_train_2, Y_val_2 = y_df.loc[y_df.index[train_index], :], y_df.loc[y_df.index[test_index], :]            
        else:
            Y_train_2, Y_val_2 = None, None
        
        if select_active_clients:
            mask = X_train_2['diff'] > 0
            X_train_2 = X_train_2[mask] 
            Y_train_2 = Y_train_2[mask]

        compare_two_datasets(X_train_1, X_train_2, profile, Y_train_1, Y_train_2, 'Compare between Train folds %i and %i' % ((count-1) % ll, count))
        compare_two_datasets(X_val_1, X_val_2, profile, Y_val_1, Y_val_2, 'Compare between Val folds %i and %i' % ((count-1) % ll, count))