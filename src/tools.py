import numpy as np
import pandas as pd

def init_feature_kind(train_X):
    feature_kind = {}
    for feature in train_X.columns:
        count = train_X.value_counts(subset=[feature],sort=True)
        if len(count.index)>10:
            feature_kind[feature] = 'continuous'
        else:
            feature_kind[feature] = list(count.index[:])
    return feature_kind
def get_feature_kind(train_X,feature_kind):
    continuous = []
    categorical = []
    for feature in train_X.columns:
        if feature_kind[feature] == 'continuous':
            continuous.append(feature)
        else:
            categorical.append(feature)
    return continuous,categorical


def get_performance(true,result):
    from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score,matthews_corrcoef
    
    performance={}
    matrix = confusion_matrix(true,result)

    #print("confusion matrix")
    #print(matrix)
    performance["confusion matrix"] = matrix

    #print("acc:")
    #print(accuracy_score(true,result))
    performance["acc"] = accuracy_score(true,result)

    #print("precision:")
    #print(precision_score(true,result))
    performance["precision"] = precision_score(true,result,zero_division=0)

    #print("f1_score:")
    #print(f1_score(true,result))
    performance["f1_score"] = f1_score(true,result)

    #print("recall:")
    #print(recall_score(true,result))
    performance["recall"] = recall_score(true,result,zero_division=0)
    
    performance["matthews_corrcoef"] = matthews_corrcoef(true,result)
   

    return performance
    
def drop_overmissing_feature(threshold,train_X,val_X=None,test_X=None):
    over_missing_col = []
    for feature in train_X.columns:
        cnt = train_X[feature].value_counts(dropna = False)
        
        try:
           # print(cnt[np.nan])
            if(cnt[np.nan]>len(train_X)*threshold):
                over_missing_col.append(feature)
        except:
            pass

    train_X.drop(labels=over_missing_col, axis=1, inplace=True)
    if val_X:
        val_X.drop(labels=over_missing_col, axis=1, inplace=True)
    if test_X:
        test_X.drop(labels=over_missing_col, axis=1, inplace=True)
    
    print('Drop {} features with too many missing value'.format(len(over_missing_col)))
    print(train_X.shape)
    #print(test_X.shape)

    return train_X,val_X,test_X

def drop_predominant_feature(threshold,train_X,val_X=pd.DataFrame(),test_X=pd.DataFrame()):

    
    quasi_constant_feature = []
    for feature in train_X.columns:

        # 計算比率
        predominant = (train_X[feature].value_counts() /
        np.float64(len(train_X))).sort_values(ascending=False).values[0]

        # 假如大於門檻 加入 list
        if predominant >= threshold:
            quasi_constant_feature.append(feature)   
    print("Drop predominant {} feature:".format(len(quasi_constant_feature)))        
    #print(quasi_constant_feature)

    # 移除半常數特徵

    train_X.drop(labels=quasi_constant_feature, axis=1, inplace=True)
    if len(val_X):
        val_X.drop(labels=quasi_constant_feature, axis=1, inplace=True)
    if len(test_X):
        test_X.drop(labels=quasi_constant_feature, axis=1, inplace=True)
    print(train_X.shape)
    
    return train_X,val_X,test_X

def data_transform(train_X):
    train_X['SEX'] = train_X['SEX'].replace(
        {
            'F':0.0,
            'M':1.0
        }
    )
    train_X['Joint'] = train_X['Joint'].replace(
        {
            'TKA':0.0,
            'TKA':1.0,
            'THA':2.0
        }
    )

    # Data editing
    train_X['OP_time'] = train_X['OP_time_hour']*60+train_X['OP_time_minute']
    train_X = train_X.drop(columns=['OP_time_hour','OP_time_minute'])

    return train_X

def wrapper_approach(clf,train_X,train_y,val_X = pd.DataFrame(),val_y=pd.DataFrame()):
    from sklearn.feature_selection import RFE

    record = {}
    for i in range(len(train_X.columns)):
        print(i+1)
        rfe = RFE(estimator=clf,n_features_to_select=i+1)
        train_wrapper_X = rfe.fit_transform(train_X,train_y['outcome'])
        val_wrapper_X = rfe.transform(val_X)


        clf.fit(train_wrapper_X, train_y['outcome'])
        result = clf.predict(val_wrapper_X)

        #print("train")
        get_performance(train_y,clf.predict(train_wrapper_X))
        #print("val")
        record[i+1]=get_performance(val_y,result)
        #print("\n\n")

    return record

def set_pandas_display_options() -> None:
    display = pd.options.display
    display.max_columns = 100
    display.max_rows = 100
    display.max_colwidth = 199
    display.width = None