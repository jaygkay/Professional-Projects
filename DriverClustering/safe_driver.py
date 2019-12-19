import numpy as np 
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from imblearn.under_sampling import RandomUnderSampler # undersampling
from sklearn.ensemble import BaggingClassifier # ensemble
from sklearn import tree # decision tree
from sklearn.decomposition import PCA # PCA
from sklearn.grid_search import GridSearchCV
from operator import itemgetter
from sklearn import svm # svm classifier function
from sklearn.model_selection import cross_val_score # model selection with k-fold cv
from sklearn.metrics import classification_report, confusion_matrix


import os
import sys
from clfFunction import svmclf as scf

np.set_printoptions(precision = 6)

# =============================================================================
# Rotation Forests function
# =============================================================================

# to select a random subset of attributes called gen_random_subset:
def get_random_subset(iterable,k):
    subsets = []
    iteration = 0
    np.random.shuffle(iterable)
    subset = 0
    limit = len(iterable)/k
    while iteration < limit:
        if k <= len(iterable):
            subset = k
        else:
            subset = len(iterable)
        subsets.append(iterable[-subset:])
        del iterable[-subset:]
        iteration+=1
    return subsets

def build_rotationtree_model(x_train,y_train,d,k):
    models = []
    r_matrices = []
    feature_subsets = []
    for i in range(d):
        x,_,_,_ = train_test_split(x_train,y_train,test_size=0.3,random_state=7)
        # Features ids
        feature_index = list(range(x.shape[1]))
        # Get subsets of features
        random_k_subset = get_random_subset(feature_index,k)
        feature_subsets.append(random_k_subset)
        # Rotation matrix
        R_matrix = np.zeros((x.shape[1],x.shape[1]),dtype=float)
        each_subset = random_k_subset[0]
        for each_subset in random_k_subset:
            pca = PCA()
            x_subset = x.iloc[:,each_subset]
            pca.fit(x_subset)
            for ii in range(0,len(pca.components_)):
                for jj in range(0,len(pca.components_)):
                    R_matrix[each_subset[ii],each_subset[jj]] = pca.components_[ii,jj]
                
        x_transformed = x_train.dot(R_matrix)
        
        model = tree.DecisionTreeClassifier()
        model.fit(x_transformed,y_train)
        models.append(model)
        r_matrices.append(R_matrix)
    return models,r_matrices,feature_subsets


    
def model_worth(models,r_matrices,x,y):
    
    predicted_ys = []
    for i,model in enumerate(models):
        x_mod =  x.dot(r_matrices[i])
        predicted_y = model.predict(x_mod)
        predicted_ys.append(predicted_y)
    
    predicted_matrix = np.asmatrix(predicted_ys)
    final_prediction = []
    for i in range(len(y)):
        pred_from_all_models = np.ravel(predicted_matrix[:,i])
        non_zero_pred = np.nonzero(pred_from_all_models)[0]  
        is_one = len(non_zero_pred) > len(models)/2
        final_prediction.append(is_one)
    report = classification_report(y, final_prediction)
    print (classification_report(y, final_prediction))
    
    return report, final_prediction

# create evaluation tables
def Confusion_matrix_evaluation(actual_cl, predicted_cl, cl_title='', dataset_title='',train_test='',view_matrix=False):
        
    #Create a confusion matrix
    con_matrix = pd.DataFrame(confusion_matrix(actual_cl, predicted_cl))
    con_matrix.index.name='Actual'
    con_matrix.columns.name='Predicted'
    TN, FP, FN, TP = confusion_matrix(actual_cl, predicted_cl).ravel()

    if (TP+FN)==0: 
        Sens=0.0
        Recall=0.0
    else: 
        Sens = TP/(TP+FN)
        Recall = Sens
    
    if (FP+TN)==0: Spec=0.0
    else: Spec = TN/(FP+TN)    
    
    if (TP+FP)==0: Prec=0.0
    else: Prec = TP/(TP+FP)
    
    evaluation_table = pd.DataFrame([[Sens,Recall,Spec,Prec]], columns=['Sensitivity','Recall','Specificity','Precision'])
    print('\n')
    print('----------------------------------------------------')
     
    print('*Classifier: {}'.format(cl_title))
    print('*Data: {}_{}'.format(dataset_title, train_test))
    print('\n')
    print('<Confusion Matrix Evaluation>')
    print(evaluation_table)
    if view_matrix==True: 
        print('\n')
        print('<Confusion Matrix>')
        print(con_matrix)
        
    print('\n')
    print('Score (Accuracy): %.4f' %((TP+TN)/(TN+FP+FN+TP)))
    
    #print(con_matrix)

    #R1 = Sensitivity/Precision
    #R2 = Specificity/Sensitivity
    return evaluation_table, con_matrix

def classifaction_report_plot(report, option_title ='', classifiername = ''):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.strip().split('   ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.drop(['class', 'support'], axis=1, inplace=True)

    # heatmap
    plt.suptitle('Classification Report ({})'.format(classifiername))
    plt.title('Dataset = {}'.format(option_title))
    
    
    #sns.plt.ylabel('Target (0, 1)')
    #sns.plt.xlabel('performances')
    sns.heatmap(dataframe, annot=True, annot_kws={"size": 15})
    plt.show()
    return dataframe


# =============================================================================
# 1. Data import 
# =============================================================================
# input the original dataset
safedriver = pd.read_csv('train.csv', sep=',')
safedriver.drop(['id'], axis=1, inplace=True)
'''
safedriver = pd.read_csv('cleaned.csv', sep=',')
safedriver.drop(['Unnamed: 0'], axis=1, inplace=True)
'''
safedriver.name = 'safedriver'

safedriver.head(10)
safedriver.shape
safedriver.columns

# Target variable (independent) 
plt.title('Target variable')
plt.ylabel('# of cases')
plt.xlabel('Target (0, 1)')
safedriver['target'].value_counts().plot(kind='bar')
safedriver['target'].value_counts()/safedriver.shape[0]


#### missing Values ####
# replace missing value (-1) to 'Nan'
datalist = [safedriver]

for i in datalist:
    i.replace(-1, np.nan, inplace=True)

# record ( by target )
grp_safedriver = safedriver.groupby('target')
grp_safedriver.get_group(1)
datalist = [grp_safedriver.get_group(1), grp_safedriver.get_group(0)]

for i in datalist:
    n = scf.Missing_value_count(i)

28226/(21694*58)
818232/(573518*58)

for i in datalist:
    print()
    #print('Dataset Name: %s' %i.name)
    print(i.apply(lambda x: x.count(), axis=0))

#ps_car_03_cat, ps_car_05_cat and ps_reg_03 have lots of missing values

missing_features = ['ps_car_03_cat', 'ps_car_05_cat', 'ps_reg_03']
for f in missing_features:
    n = safedriver[f].nunique()
    print("'%s' has %d different value(s)" %(f, n))

pd.DataFrame(safedriver[missing_features].agg(['min','max']).T, dtype='float') # check min max on features

# bar chart for Categorical features
for f in safedriver[missing_features[:2]]:
    plt.title("Bar chart of '{}'".format(f))
    plt.ylabel('Categories')
    safedriver[f].value_counts().plot(kind='barh')
    plt.show()

# put each of them in lists
cat_columns = list(filter(lambda x: x.endswith("cat"),
                                     safedriver.columns))
bin_columns = list(filter(lambda x: x.endswith("bin"),
                                     safedriver.columns))
conti_columns = list(filter(lambda x: x not in bin_columns + cat_columns,
                                     safedriver.columns))

for i in cat_columns:
    safedriver[i].fillna(value = safedriver[i].mode()[0],inplace=True)
for i in bin_columns:
    safedriver[i].fillna(value=safedriver[i].mode()[0],inplace=True)
for i in conti_columns:
    safedriver[i].fillna(value=safedriver[i].mean(),inplace=True)

for i in datalist:
    n = scf.Missing_value_count(i)
    
safedriver_dum = pd.get_dummies(safedriver, columns = cat_columns) 
safedriver_dum = safedriver_dum.drop_duplicates()
print ("After created dummies for categorical variables Dataset is:", safedriver_dum.shape)

# Divide the data into Trainand test    
safedriver_x = safedriver.drop(['target'], axis=1)
safedriver_y = safedriver['target']
safedriver_x.shape
safedriver_y.shape

############## using 10% part of data #################
x_train_part,x_test_part,y_train_part,y_test_part = train_test_split(safedriver_x,safedriver_y,test_size = 0.10,random_state=9)
x_test_part.shape
y_test_part.value_counts()
#######################################################

x_test_part = safedriver_x
y_test_part = safedriver_y

# =============================================================================
# [sampling] option 1. no sampling
# =============================================================================
datatitle = 'Raw data (no sampling)'

#x_train,x_test_all,y_train,y_test_all = train_test_split(safedriver_x,safedriver_y,test_size = 0.34,random_state=9)
x_train,x_test_all,y_train,y_test_all = train_test_split(x_test_part,y_test_part,test_size = 0.34,random_state=9)

x_train.shape
x_test_all.shape


# =============================================================================
# [sampling] option 2. undersampling
# =============================================================================
datatitle = 'Raw data (undersampling)'

# Apply the random under-sampling
rus = RandomUnderSampler(return_indices=True)
sampled_x, sampled_y, sampled_idx = rus.fit_sample(x_train, y_train)

sampled_x.shape
sampled_y.shape

# Target variable (independent) 
plt.title('Target variable (undersampling)')
plt.ylabel('# of cases')
plt.xlabel('Target (0, 1)')
pd.DataFrame(sampled_y)[0].value_counts().plot(kind='bar')
pd.DataFrame(sampled_y)[0].value_counts()/sampled_y.shape[0]

x_train = sampled_x
y_train = sampled_y

# =============================================================================
# [preprocessing] option 1. normalization
# =============================================================================
datatitle = 'Normalizated data (no sampling)'
datatitle = 'Normalizated data (undersampling)'

from sklearn import preprocessing

#preprocession-min_max normalization
min_max_scaler = preprocessing.MinMaxScaler().fit(x_train)
x_train_norm = min_max_scaler.transform(x_train)
x_test_norm = min_max_scaler.transform(x_test_all)

# check the min-max normalized dataset
np.mean(x_train_norm, axis=0) # new train set means
np.mean(x_test_norm, axis=0) # new val set means
x_train.mean() # original dataset means
x_train_norm.mean()

# =============================================================================
# [preprocessing] option 2. Feature Selection
# =============================================================================
datatitle = 'PCA data (no sampling)'
datatitle = 'PCA data (undersampling)'

n_comp = 20
# PCA
print('\nRunning PCA ...')
pca = PCA(n_components=n_comp)
X_pca = pca.fit_transform(x_train)
print('Explained variance: %.4f' % pca.explained_variance_ratio_.sum())

print('Individual variance contributions:')
for j in range(n_comp):
    print(pca.explained_variance_ratio_[j])

X_test_pca = pca.transform(x_test_all)
X_pca.shape
X_test_pca.shape


# =============================================================================
# [Bagging][Rotation Forest][lineaer SVM][non-lineaer SVM]
# =============================================================================

# Raw data
trainX = x_train
trainY = y_train
testX = x_test_all
testY = y_test_all

# normalization
trainX = x_train_norm
trainY = y_train
testX = x_test_norm
testY = y_test_all

# feature selection
trainX = X_pca
trainY = y_train
testX = X_test_pca
testY = y_test_all


# [Bagging] Bagging with Decision Tree------------------------------------------
classifiername = 'Bagging'
# select params (# of functions)
dt_clf = tree.DecisionTreeClassifier()
accuracy_tab = scf.Ensemble_clf(trainX, trainY, dt_clf, 20, 'Decision Tree') 
accuracy_tab.columns = ['Errorrate', 'Num_function'] 

# perform Bagging with selected params (number of function = 20)
dt_clf = tree.DecisionTreeClassifier()
bagclf = BaggingClassifier(base_estimator=dt_clf,n_estimators=20)
bagclf = bagclf.fit(trainX, trainY)
acc_train = bagclf.score(trainX, trainY)
acc_test = bagclf.score(testX, testY)
pred_y_test_all = bagclf.predict(testX)
bag_matix = confusion_matrix(testY, pred_y_test_all)
print(classification_report(testY, pred_y_test_all))


report = classification_report(testY, pred_y_test_all)
report_df = classifaction_report_plot(report,datatitle,classifiername)
print(report)

# 10-cv score
scores = cross_val_score(bagclf, trainX, trainY, cv=10)
score = scores.mean()

# report on Bagging
Confusion_matrix_evaluation(testY, pred_y_test_all, cl_title=classifiername, 
                            dataset_title=datatitle,train_test='test',view_matrix=True)



def classifaction_report_plot(report, option_title ='', classifiername = ''):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.strip().split('    ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.drop(['class', 'support'], axis=1, inplace=True)

    # heatmap
    plt.suptitle('Classification Report ({})'.format(classifiername))
    plt.title('Dataset = {}'.format(option_title))
    
    
    #sns.plt.ylabel('Target (0, 1)')
    #sns.plt.xlabel('performances')
    sns.heatmap(dataframe, annot=True, annot_kws={"size": 15})
    plt.show()
    return dataframe



# [Rotation Forests]-----------------------------------------------------------
classifiername = 'Rotation Forests'

# Divide the data into Train, dev and test    
# x_dev,x_test,y_dev,y_test = train_test_split(testX,testY,test_size=0.34,random_state=9)

# # of features in subset = 5
# # of functions = 20
feature_index = np.arange(trainX.shape[1])
trainX_df= pd.DataFrame(trainX)
models,r_matrices,features = build_rotationtree_model(trainX_df,trainY,20,5)
model_worth(models,r_matrices,trainX,trainY)
report_rf, pred_y = model_worth(models,r_matrices,testX,testY)

report_rf_df = classifaction_report_plot(report_rf,datatitle,classifiername)
print(report_rf)

# report on Rotation Forests
Confusion_matrix_evaluation(testY, pred_y, cl_title=classifiername, dataset_title=datatitle,train_test='test',view_matrix=False)

(118809+3893)/(118809+76090+3579+3893)
# [linear SVM]-----------------------------------------------------------------
classifiername = 'linear SVM'

trainX = x_train
trainY = y_train
testX = x_test_all
testY = y_test_all

trainX.shape

# create lists for C
crange=['0.01','0.1','1','10'] 

# plot error rates
compare_DF, method=scf.SVM_Error_plot(trainX, trainY, crange, ensemble=True)
scf.Errorrate_plot(compare_DF, method)
svmclf = svm.SVC(kernel='linear', C=1)
svmclf.fit(trainX, trainY)
clf_pred = svmclf.predict(testX)
# test the svm classifiers
svm, svm_y_pred = scf.Testing_svm(svmclf, trainX, trainY, testX, testY, True, 
                                     clf_title='linear SVM with best parameters')


svmclf = svm.SVC(kernel='linear', C=1, probability=False)
svmclf.fit(trainX, trainY)

clf_pred_prob = svmclf.predict_proba(testX)
from sklearn import metrics

def Plot_ROC(y, x, clf, roc_title='', pos_class=0):
    y_class = np.array(y) # true labels
    y_scores = clf.predict_proba(x)[:,pos_class] # predicted results
    fpr, tpr, thresholds = metrics.roc_curve(y_class, y_scores, pos_label=pos_class)
    roc_auc = metrics.auc(fpr, tpr)
    asd_threshold = list(reversed(thresholds))
    
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' %roc_auc)
    for i in range(len(tpr)):
        y = tpr[i]
        x = fpr[i]
        plt.plot(x, y, '.-')
        plt.text(x * (1 + 0.01), y * (1 + 0.01) , format(asd_threshold[i],'.2f'), fontsize=12)
     
    plt.plot(fpr, tpr, '.')
    #plt.text(fpr * (1 + 0.01), tpr * (1 + 0.01) , thresholds, fontsize=12)
    plt.xlabel('False positive Rate')
    plt.ylabel('True positive Rate')
    plt.title('ROC curve {}'.format(roc_title))
    plt.legend()
    plt.show()
    return fpr, tpr, thresholds

# ROC curves of training
fpr, tpr, thresholds = Plot_ROC(y_train, x_train, nbclf, 'Training')


clf_title='linear SVM with best parameters'

#Classification report
report = classification_report(testY, clf_pred)
report_df = classifaction_report_plot(report,datatitle,classifiername)
print(report)
print ()
print ()






      
# [non-linear SVM] Gaussian kernel
classifiername = 'RBF SVM'

# range of parameters
from sklearn.grid_search import GridSearchCV
from operator import itemgetter
from sklearn import svm # svm classifier function

def SVM_best_param(trainX, trainY, grid_paramters, n_top):
    #y_arr = trainY.values
    clf = svm.SVC(kernel='rbf')
    grid_search = GridSearchCV(clf, param_grid=grid_paramters)
    grid_search.fit(trainX, trainY)
    grid_scores_list = sorted(grid_search.grid_scores_, key=itemgetter(1), reverse=True)

    for i, score in enumerate(grid_scores_list[:n_top]):
        print("Model Rank: {0}".format(i + 1))
        print("Mean score: {0:.4f}".format(score.mean_validation_score))
        print("Std score: {0:.4f}".format(np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
    return grid_scores_list
    
    
grange=['0.001','0.01','0.1','1.0','10.0']
crange=['0.01','0.1','10.0','100.0'] 

RBFSVM_table = scf.RBF_SVM_Error_plot(trainX,trainY, crange, grange, ensemble=True)

param_grid = {"gamma": list(map(float, grange)), "C": list(map(float, crange))}
svm_grid_scores = SVM_best_param(trainX,trainY, param_grid, 4)


# selected parameters
# RBF SVM - c:10.0, gamma:0.01
# test set
rbf_svmclf = svm.SVC(kernel='rbf', gamma=0.01, C=10.0)

# test the svm classifiers
rbfsvm, rbf_y_pred = scf.Testing_svm(rbf_svmclf, trainX, trainY, testX, testY, True, 
                                     clf_title='RBF SVM with best parameters')


#Classification report
report = classification_report(testY, rbf_y_pred)
report_df = classifaction_report_plot(report,datatitle,classifiername)
print(report)
print ()
print ()
print("*Classification report of Classifiers")
#print(classification_report(testY, clf_pred))





classifiers = ["Decision Tree","Gaussian Naïve Bayese","Logistic Regression",
               "K-Nearest Neighbors","Support Vecotr Machines (linear)",
               "Support Vecotr Machines (radial)","Random Forest","AdaBoost",
              "Gradient Boosting"]

models = [DecisionTreeClassifier(),GaussianNB(),LogisticRegression(),
          KNeighborsClassifier(),SVC(kernel='linear',C=0.1,gamma=0.1, probability = True),
          SVC(kernel='rbf',C=1,gamma=0.1,probability = True),RandomForestClassifier(),
         AdaBoostClassifier(n_estimators = 300, random_state = 0),
         GradientBoostingClassifier(n_estimators = 500, random_state = 0)]


test_lst = []
train_lst = []
clfs = []

for i, model in zip(classifiers,models):
    clf = model.fit(x_train2, y_train)
    
    clfs.append(clf)  
    train_lst.append(clf.score(x_train2, y_train))
    test_lst.append(clf.score(x_test2, y_test))
    
    
model_raw = pd.DataFrame({"Train Accuracy": train_lst, "Test Accuracy": test_lst}, index = classifiers)


























# =============================================================================
# train a non-linear SVM with soft margin and Gaussian kernel(sigma)
# =============================================================================

grange=['0.000000001','0.0000001','0.000001', '0.00001','0.0001','0.1','1.0','10.0']
crange=['0.001','0.01','0.1','1.0','2.0','4.0','6.0','10.0','100.0'] 

RBFSVM_table = scf.RBF_SVM_Error_plot(x_data, y_data, crange, grange, CV_eval=True, n_cv=10)
RBFSVM_table = scf.RBF_SVM_Error_plot(x_data, y_data, crange, grange, CV_eval=False)

#list(map(float, grange))
param_grid = {"gamma": list(map(float, grange)), "C": list(map(float, crange))}
svm_grid_scores = scf.SVM_best_param(x_data, y_data, param_grid, 4)

np.set_printoptions(suppress=True)

# selected parameters
# RBF SVM - c:400.0, gamma:0.0001
# test set
rbf_svmclf = svm.SVC(kernel='rbf', gamma=0.0001, C=400.0)

# test the svm classifiers
rbfsvm, rbf_y_pred = scf.Testing_svm(rbf_svmclf, x_data, y_data, x_test, y_test, True, clf_title='RBF SVM with best hyper-parameters')











# =============================================================================
# 1. PCA (30)
# =============================================================================

n_comp = 30
# PCA
print('\nRunning PCA ...')
pca = PCA(n_components=n_comp, svd_solver='full', random_state=1001)
X_pca = pca.fit_transform(safedriver_x)
print('Explained variance: %.4f' % pca.explained_variance_ratio_.sum())

print('Individual variance contributions:')
for j in range(n_comp):
    print(pca.explained_variance_ratio_[j])


# =============================================================================
# train a non-linear SVM with soft margin and Gaussian kernel(sigma)
# =============================================================================
# range of parameters
from sklearn.grid_search import GridSearchCV
from operator import itemgetter
from sklearn import svm # svm classifier function

def SVM_best_param(x_train, y_train, grid_paramters, n_top):
        #y_arr = y_train.values
        clf = svm.SVC(kernel='rbf')
        grid_search = GridSearchCV(clf, param_grid=grid_paramters)
        grid_search.fit(x_train, y_train)
        grid_scores_list = sorted(grid_search.grid_scores_, key=itemgetter(1), reverse=True)
    
        for i, score in enumerate(grid_scores_list[:n_top]):
            print("Model Rank: {0}".format(i + 1))
            print("Mean score: {0:.4f}".format(score.mean_validation_score))
            print("Std score: {0:.4f}".format(np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")
        return grid_scores_list
    
    
grange=['0.001','0.01','0.1','1.0','10.0']
crange=['0.001','0.01','0.1','1.0','10.0','100.0', '200.0','300.0','400.0','500.0','600.0'] 

# train set
RBFSVM_table = scf.RBF_SVM_Error_plot(x_train,y_train, crange, grange, ensemble=True)

param_grid = {"gamma": list(map(float, grange)), "C": list(map(float, crange))}
svm_grid_scores = SVM_best_param(x_train,y_train, param_grid, 4)

