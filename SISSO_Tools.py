import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
from compressed_sensing.sisso import SissoRegressor
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)



def SISSO(df, label_column, unit_list, allowed_operations, n_operations=2, dim=2, features_per_iter=10, n_best_subs=50):
    '''
    This function creates a SISSO model based on given primary features towards a target property
    Input:
            df:                     pandas dataframe which contains only features as columns and one column with target property
            label_column:           string of target property column name
            unit_list:              list of strings with units of the features in the same order as columns of df
            allowed_operations:     list of applied operations (possible: ['+','-','^2','^0.5','exp'])
            n_operations:           maximum number of operations applied to features
            dim:                    dimension of model
            features_per_iteration: number of features per iteration
            n_best_subs:            amount of highest correlated complex features for further computation
    Output:
            sisso:                  trained SissoRegressor Object
            rmse:                   RMSE value of model
            features_list:          list of all features given to SISSORegressor Object
    '''

    #make brackets around every column head
    data_set = df.copy()
    columns = df.columns
    new_columns=[]
    for col in df.columns:
        new_columns.append('('+col+')')
        
    data_set.columns = new_columns

    #label column 
    y_df = data_set['('+label_column+')']
    #descriptor columns
    X_df = data_set.drop(['('+label_column+')'], axis=1)

    #mathematic operations
    X_combinations = link_descriptors(X_df, unit_list, n_operations, y_df.to_frame(), ['('+label_column+')'], n_best_subs, allowed_operations)
    X_combinations = X_combinations.loc[:, (X_combinations != X_combinations.iloc[0]).any()] 

    D = X_combinations.values
    P = y_df.values
    features_list = X_combinations.columns.tolist()

    sisso = SissoRegressor(n_nonzero_coefs=dim, n_features_per_sis_iter=features_per_iter)

    sisso.fit(D, P)
    sisso.print_models(features_list)
    rmse = sisso.rmses[-1]
    return sisso, rmse, features_list


def evaluate_sisso(sisso_model, Z, features_list):
    '''
    This function applies a tained SISSO model to a dataset with primary features. A prediction is made
    for every row of the data-set
    Input:
            sisso_model:        trained SissoRegressor Object
            Z = predict_df:         pandas DataFrame with all primary features as columns
            features_list:      list of all features given to SISSORegressor Object
    Output:
            res:                pandas series (one column) with predicted target property
    '''

    res = np.zeros((len(Z)))#pd.DataFrame(np.zeros((len(test))), index=test.index)

    for coef_id, feat_id in enumerate(sisso_model.l0_selected_indices[-1]):
        feature_str = str(features_list[feat_id])
        for var in list(Z.columns):
            feature_str = feature_str.replace(var, f'Z["{var}"]')
        
        res = res + sisso_model._list_of_coefs[-1][coef_id]*eval(feature_str)
    res = res + sisso_model.list_of_intercepts[-1]

    return res


def SISSO_leave_one_cross_validation(df, label_column, unit_list, allowed_operations,
                                     n_operations=2, dim=2, features_per_iter=10, n_best_subs=50):
    '''
    This function makes a cross-validation via leafe-one-out method where one data-point is always not
    used for training the model and is used for vaidation
    Input:
            df:                     pandas dataframe which contains only features as columns and one column with target property
            label_column:           string of target property column name
            unit_list:              list of strings with units of the features in the same order as columns of df
            allowed_operations:     list of applied operations (possible: ['+','-','^2','^0.5','exp'])
            n_operations:           maximum number of operations applied to features
            dim:                    dimension of model
            features_per_iteration: number of features per iteration
            n_best_subs:            amount of highest correlated complex features for further computation
    Output:
            avr_pred_err:           average error of all left-out data points
    '''

    total_diff = 0
    n = 0
    for i in range(len(df)):
        train, test = split_train_test(df, method="leave_one_out", index=i)
        sisso_model, rmse, features_list = SISSO(train, label_column, unit_list, 
                                            allowed_operations, n_operations=n_operations, dim=dim,
                                            features_per_iter=features_per_iter, n_best_subs=n_best_subs)
        print('RMSE: '+str(rmse))

        res = evaluate_sisso(sisso_model, test, features_list)
        print('Predicted: '+str(res.iloc[0]))
        print('Measured: '+str(test[label_column].iloc[0]))
        if not np.isnan(res.iloc[0]) and not np.isinf(res.iloc[0]):
            total_diff = total_diff + abs(res.iloc[0]-test[label_column].iloc[0])
            n = n+1
        if n!=0:
            avr_pred_err = total_diff/n
            print(f'avr. pred. error: {avr_pred_err}')

    return avr_pred_err


def link_descriptors(df, unit_list, n_rep, df_label, label_list, n_best, allowed_operations=['+','-','^2','^0.5','exp']):
    '''
    This function applies mathematical operations to link primary features and create complex features
    Input:
            df:                     pandas dataframe which contains only features as columns and one column with target property
            df_label:               string of target property column name
            label_list:             string name of target property as list
            unit_list:              list of strings with units of the features in the same order as columns of df
            allowed_operations:     list of applied operations (possible: ['+','-','^2','^0.5','exp'])
            n_rep:                  maximum number of operations applied to features
            n_best:                 amount of highest correlated complex features for further computation
    Output:
            total_combinations:     pandas dataframe with all (primary and complex) features
    '''

    df1 = df
    unit_list1 = unit_list
    df2 = df
    unit_list2 = unit_list

    total_combinations = pd.DataFrame(df)
    
    for n in range(n_rep):
        df_comb = pd.DataFrame()
        new_unit_list = []

        for c1, col1 in enumerate(df1.keys()):
            for c2, col2 in enumerate(df2.keys()):
                if unit_list1[c1]==unit_list2[c2] and col1!=col2:
                    if '+' in allowed_operations:
                        df_comb['('+col1+'+'+col2+')'] = df1[col1]+df2[col2]
                        new_unit_list.append(unit_list1[c1])

                    if '-' in allowed_operations:
                        df_comb['('+col1+'-'+col2+')'] = df1[col1]-df2[col2]
                        new_unit_list.append(unit_list1[c1])
                        df_comb['('+col2+'-'+col1+')'] = df2[col2]-df1[col1]
                        new_unit_list.append(unit_list1[c1])


                if '*' in allowed_operations:
                    df_comb['('+col1+'*'+col2+')'] = df1[col1]*df2[col2]
                    new_unit_list.append('('+unit_list1[c1]+')*('+unit_list2[c2]+')')

                if '/' in allowed_operations:
                    df_comb['('+col1+'/'+col2+')'] = df1[col1]/df2[col2]
                    new_unit_list.append('('+unit_list1[c1]+')/('+unit_list2[c2]+')')
                    df_comb['('+col2+'/'+col1+')'] = df2[col2]/df1[col1]
                    new_unit_list.append('('+unit_list2[c2]+')/('+unit_list1[c1]+')')

        
        for c1, col1 in enumerate(df1.keys()):
            if '^2' in allowed_operations:
                df_comb['('+col1+')**2'] = (df1[col1]**2).values
                new_unit_list.append('('+unit_list1[c1]+')**2') 

            if 'exp' in allowed_operations:
                if unit_list1[c1]=='':
                    df_comb['np.exp('+col1+')'] = (np.exp(df1[col1])).values
                    new_unit_list.append('exp('+unit_list1[c1]+')')

            if '^0.5' in allowed_operations:
                if not np.isnan(df1[col1]**0.5).any():
                    df_comb['('+col1+')**0.5'] = (df1[col1]**0.5).values
                    new_unit_list.append('('+unit_list1[c1]+')**0.5')    

        for c2, col2 in enumerate(df2.keys()):
            if '^2' in allowed_operations:
                df_comb['('+col2+')**2'] = (df2[col2]**2).values
                new_unit_list.append('('+unit_list2[c2]+')**2') 

            if 'exp' in allowed_operations:
                if unit_list2[c2]=='':
                    df_comb['np.exp('+col2+')'] = (np.exp(df2[col2])).values
                    new_unit_list.append('exp('+unit_list2[c2]+')')

            if '^0.5' in allowed_operations:
                if not np.isnan(df2[col2]**0.5).any():
                    df_comb['('+col2+')**0.5'] = (df2[col2]**0.5).values
                    new_unit_list.append('('+unit_list2[c2]+')**0.5') 

        df_comb.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_comb = df_comb.dropna(axis=1)
        df_comb = df_comb.loc[:, (df_comb != df_comb.iloc[0]).any()] 
        df_comb, new_unit_list = get_best_corr(df_comb, df_label, label_list, n_best, new_unit_list)
        df1 = df_comb
        unit_list1 = new_unit_list

        total_combinations = pd.concat([total_combinations, df_comb], axis=1)

    return total_combinations


def get_best_corr(des_df,label_df,column_names, n_best, unit_list):
    '''
    Get subset of dataset with only highest correlated features to target property
    Input:
        des_df:         pandas dataframe with features
        label_df:       pandas dataframe with target property
        column_names:   string name of target property as list
        n_best:         amount of highest correlated features returned
        unit_list:      string list of units for features
    Output:
        best_des_df:    pandas dataframe with 'n_best' correlated features
        best_units:     list of units for best_des_df
    '''

    des_df = des_df.reindex(label_df.index)

    lab = []
    for i in range(len(label_df.columns)):
        if label_df.columns[i] in column_names:
            lab.append(i)

    corr_dic = {}
    unit_list_dic = {}
    for des in range(len(des_df.columns)):

        corr_coev = 0
        for l in lab:
            corr_coev = corr_coev + abs(pearsonr(label_df[label_df.columns[l]].values,des_df[des_df.columns[des]])[0])

        corr_coev = corr_coev/len(lab)
        
        if np.isnan(corr_coev): corr_coev=0
        corr_dic[des_df.columns[des]] = float(abs(corr_coev))
        unit_list_dic[des_df.columns[des]] = unit_list[des]


    marklist = sorted(corr_dic.items(), key=lambda x:x[1], reverse=True)
    sorted_corr_dict = dict(marklist)

    n=0
    best_des_df = pd.DataFrame()
    best_units = []
    best_model = []
    for key, value in sorted_corr_dict.items():
        # check if des not already in list
        already_in_list = False
        for col in best_des_df.columns:
            if np.allclose(best_des_df[col].values, abs(des_df[key])): already_in_list = True 

        if not already_in_list:
            best_des_df[key] = des_df[key]
            best_units.append(unit_list_dic[key])
            n=n+1
        if n>n_best: break

    return best_des_df, best_units


def split_train_test(data_set, method="random", index=0):
    '''
    Method for splitting a data-set into training and testing data
    Input:
        data-set:       pandas dataframe 
        method:         string name of method that should be applied
        index:          index of test data for 'leave_one_out' method
    Output:
        train:          pandas train dataset
        test:           pandas test dataset
    '''

    if method=="random":
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(data_set, test_size=0.2)

        return train, test


    if method=="leave_one_out":
        test = data_set.iloc[[index]]
        train = data_set.drop(data_set.index[[index]])

        return train, test