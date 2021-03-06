#KDD Cup 2009: Customer Relationship Prediction

#next steps 
#check other papers
#check other algorithms!!
#try grid search!!
#check sklearn other functions
#replicate the medium article
#start writing
#build figures
#check how to oversample
#cross validation for grid models 

#today: gridsearch, NN with lasso

#grid results:
#Upselling: {'max_depth': 7, 'n_estimators': 400}
#0.9761404711173494
#Churn:{'max_depth': 7, 'n_estimators': 450}
#0.9482977402813088
# Appatency {'max_depth': 7, 'n_estimators': 300}
#0.9827817623766519

#best solution: n_estimators=300, max_depth=2


#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score,roc_curve, ConfusionMatrixDisplay
from sklearn.utils import resample
#classifier imports
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.naive_bayes import GaussianNB
#neutal Network imports
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Input
from tensorflow.keras.optimizers import Adam, SGD

#load data
train = pd.read_csv("orange_small_train.data", sep="\t")
test = pd.read_csv("orange_small_test.data", sep="\t")

#load labels
appetency = pd.read_table('orange_small_train_appetency.labels', header = None)
churn = pd.read_table('orange_small_train_churn.labels', header = None)
upselling= pd.read_table('orange_small_train_upselling.labels', header = None)

#convert labels to 0 and 1
appetency = (appetency + 1)/2
churn = (churn + 1)/2
upselling = (upselling + 1)/2

#merge labels
labels = pd.concat([appetency, churn, upselling], axis=1)
labels.columns = ['Appatency', 'Churn', 'Upselling']
#labels.drop(['Churn','Upselling'],inplace=True, axis=1)

#-----------------Data Exploration and Preprocessing-----------------
#print(train)
#train.describe()

#check distribution of labels
# fig = plt.figure()
# axis=[None]*3
# for i in range (3):
#     axis[i]=fig.add_subplot (1,3,i+1)
# axis[0].hist(appetency,bins=3,color="blue",label="Appetency")
# axis[1].hist(churn,bins=3,color="orange",label="Churn")
# axis[2].hist(upselling,bins=3,color="red",label="Upselling")
# for i in range (3):
#     axis[i].legend()
# fig.suptitle("Label Distribution")
# print(appetency.value_counts())
# print(churn.value_counts())
# print(upselling.value_counts())
# plt.show()


#discovery of #N/As
#sns.heatmap(data=train.isna(),cmap='viridis',cbar=False)
#sns.lineplot(train.columns,np.sort(np.sum(train.isna()).values))
#plt.show()

#filter variables out that have more than 2/3 #N/As
#print(np.sum(train.isna())<2/3*test.shape[0])
test = test.loc[:,np.sum(train.isna())<2/3*test.shape[0]]
train = train.loc[:,np.sum(train.isna())<2/3*test.shape[0]]

#remove categorical variables for numeric preprocessing
train_categorical = train[train.select_dtypes(exclude='number').columns.values]
train = train[train.select_dtypes(include='number').columns.values]

#Numeric Preprocessing
#filter out variables with low variance 
variances= [np.var(x) for x in [train[col] for col in train]]
constant_filter = VarianceThreshold(threshold=np.percentile(variances, 10))
constant_filter.fit(train)
columns=train.columns[constant_filter.get_support()]
train=constant_filter.transform(train)
#format the data
train = pd.DataFrame(train)
train.columns=columns
#number of features left
#print(train.shape)

#check distributions
# fig = plt.figure(figsize=(100,20))
# axis=[None]*train.shape[1]
# for i in range(train.shape[1]):
#     axis[i]=fig.add_subplot (7,6,i+1)
#     axis[i].hist(train.iloc[:,i])
#     axis[i].set_title(f"{train.columns[i]}",fontsize=5)
#     axis[i].axes.xaxis.set_visible(False)
#     axis[i].axes.yaxis.set_visible(False)
# plt.subplots_adjust(hspace=0.5)
# plt.show()

#check outliers
# fig = plt.figure(figsize=(100,100))
# axis=[None]*train.shape[1]
# for i in range(train.shape[1]):
#     axis[i]=fig.add_subplot (8,5,i+1)
#     sns.boxplot(x=train.iloc[:,i])
#     axis[i].set_title(f"{train.columns[i]}",fontsize=5)
#     axis[i].axes.xaxis.set_visible(False)
#     axis[i].axes.yaxis.set_visible(False)
# plt.subplots_adjust(hspace=0.5)
# plt.show()

#Normalize Data with a RobustScaler to consider Outliers
columns=train.columns.values
#scaler = MinMaxScaler()
scaler = StandardScaler()
#scaler = RobustScaler()
train = scaler.fit_transform(train)
#format the data
train = pd.DataFrame(train)
train.columns=columns

#add categorical variables back
train = pd.concat([train,train_categorical],axis=1) 

#feature engineer #NAN values
#Count of NAN values
train["nan_count"] = train.apply(lambda x: x.isna().sum(),axis=1)
#Binary Flag of NAN Values
train["nan"] = train["nan_count"]!=0
train["nan"] = train["nan"].astype(int) #convert False/True to 0/1

#Replacing categorical NAN values by zero
train[train.select_dtypes(exclude='number').columns.values] = train[train.select_dtypes(exclude='number').columns.values].fillna(0)

#Replacing numeric NAN values by mean
train[train.select_dtypes(include='number').columns.values] = train[train.select_dtypes(include='number').columns.values].fillna(train[train.select_dtypes(include='number').columns.values].mean())
#train[train.select_dtypes(include='number').columns.values] = train[train.select_dtypes(include='number').columns.values].fillna(train[train.select_dtypes(include='number').columns.values].max())

#Frequency Encoding of categorical features
for c in train[train.select_dtypes(exclude='number').columns.values]:
    count = train.groupby(c).size() / len(train)
    train[c] = train[c].apply(lambda x : count[x])

#print(train)
#--------------Model--------------
scores = pd.DataFrame(columns=labels.columns.values)

#run for each label once
for label in labels.columns.values:

    #split train and test set
    X_train , X_test , y_train , y_test = train_test_split (train, labels[label], shuffle =True , stratify=labels[label], test_size =int(0.2*len(labels[label])))

    #upsample cases
    # X_train = pd.concat([X_train, y_train],axis=1)
    # train_neg = X_train[X_train.iloc[:,-1]==0]
    # train_pos = X_train[X_train.iloc[:,-1]==1]
    # train_pos_upsampled = resample(train_pos, replace=True, n_samples=int(train_neg.shape[0]/2))
    # X_train = pd.concat([train_neg, train_pos_upsampled])
    # y_train = X_train.iloc[:,-1]
    # X_train = X_train.drop(X_train.columns[-1],axis=1)
    
    #upsample with SMOTE
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    #Feature Selection with Lasso

    #find adequate alpha value
    # coefficient = pd.DataFrame()
    # for n in np.linspace(0.001,0.01,100):
    #     model = Lasso(alpha=n)
    #     model.fit(X_train,y_train)
    #     coefficient[n]= model.coef_

    # #plot
    # coefficient=coefficient.T
    # fig = plt.figure()
    # print(coefficient)
    # for n in range(70):
    #     plt.plot(np.linspace(0.001,0.01,100),coefficient[n])
    # plt.show()

    # reg = Lasso(alpha = 0.0025)
    # reg = reg.fit(X_train, y_train)
    # coefs = pd.DataFrame(data=reg.coef_, columns = ['coef'])
    # coefs['variable'] = X_train.columns
    # selected_features = list(coefs[abs(coefs['coef'])!=0]['variable'])
    # X_train = X_train[selected_features]
    # X_test = X_test[selected_features]

    # # #Gridsearch Model
    # y_train= np.ravel(y_train)

    # parameters = {
    #     "max_depth":[2,3,4,5,6,7],
    #     "n_estimators":[100, 200, 300, 400, 450, 500]
    #     }

    # # parameters = {'bootstrap': [True, False],
    # # 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    # # 'max_features': ['auto', 'sqrt'],
    # # 'min_samples_leaf': [1, 2, 4],
    # # 'min_samples_split': [2, 5, 10],
    # # 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

    # hypermodel= RandomForestClassifier(random_state=42, n_estimators=300, max_depth=2)
    # #hypermodel = GaussianNB()
    # #hypermodel= GradientBoostingClassifier(learning_rate=0.03,max_depth=3,n_estimators=450)
    # #hypermodel = GridSearchCV (RandomForestClassifier() ,parameters , cv =StratifiedKFold(n_splits=5), scoring ="roc_auc",n_jobs=-1, verbose=2)
    # #hypermodel = RandomizedSearchCV(RandomForestClassifier() ,parameters , cv =StratifiedKFold(n_splits=5), scoring ="roc_auc",n_jobs=-1, verbose=2)
    # hypermodel.fit(X_train,y_train)

    # #predict labels
    # y_test_pred = hypermodel.predict(X_test)
    # y_train_pred = hypermodel.predict(X_train)
    # #print(hypermodel.best_params_)
    # #print(hypermodel.best_score_)

    # #get fpr and tpr
    # fpr_train, tpr_train, thresholds_train = roc_curve(y_train , y_train_pred, pos_label=1)
    # fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_pred, pos_label=1)
    

    #-------Neural Network Model-------------
    #create validation set
    X_train , X_val , y_train , y_val = train_test_split (X_train, y_train, shuffle =True ,stratify=y_train, test_size =int(0.2*len(y_train)))

    #give more weight to positives cases
    neg = list(y_train.values).count(0)
    pos = list(y_train.values).count(1)
    total = neg + pos
    #weight_for_0 = (1 / neg)*(total)/2.0 
    #weight_for_1 = (1 / pos)*(total)/2.0
    weight_for_0 = (neg/total)**-1 
    weight_for_1 = (pos/total)**-1 

    class_weight = {0: weight_for_0, 1: weight_for_1}
    #class_weight = {0: 1, 1: 1}

    #format labels
    y_train = y_train.to_numpy().reshape(-1, 1)


    #Neural network with layer (128,64,32,16,8)
    def create_NN_model(input_dim, output_dim, act_funct):
        #create structure
        model = Sequential()
        #input and first hidden layer
        model.add(Dense(2, input_dim=input_dim, activation=act_funct))
        #additional layers
        #model.add(Dense(16, activation= act_funct))
        #model.add(Dense(8, activation= act_funct))
        #model.add(Dense(4, activation= act_funct))
        #model.add(Dense(2, activation= act_funct))
        #final layer
        model.add(Dense(output_dim, activation='sigmoid'))

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])
        return model


    #create model 
    model = create_NN_model(X_train.shape[1], 1, act_funct= 'relu')
    #early stopping
    callback = EarlyStopping(monitor='val_loss', patience=3)

    #predict
    history = model.fit(X_train, y_train, batch_size=32,epochs=30,verbose=1,validation_data=(X_val, y_val) ,class_weight=class_weight,callbacks=[callback])
    
    #check overfitting
    nn_fig, nn_ax = plt.subplots()
    nn_ax.plot(history.history['AUC'])
    nn_ax.plot(history.history['val_AUC'])
    nn_ax.set_title('model accuracy')
    nn_ax.set_ylabel('accuracy')
    nn_ax.set_xlabel('epoch')
    nn_ax.legend(['train', 'val'], loc='upper left')
    #plt.show()

    #find best threshold with lowest (tpr-fpr) from the ROC Curve
    #predict with probabilities
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    #get fpr and tpr
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train , y_train_pred, pos_label=1)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_pred, pos_label=1)

    #find best threshold and round with it
    optimal_idx = np.argmax(tpr_train - fpr_train)
    optimal_threshold = thresholds_train[optimal_idx]
    print("Threshold value is:", optimal_threshold)
    y_train_pred = np.where(y_train_pred > optimal_threshold, 1, 0)
    y_test_pred  = np.where(y_test_pred > optimal_threshold, 1, 0)

    ###evaluation###
    print(label)
    #confusion Matrix
    print(confusion_matrix(y_test, y_test_pred))
    cm = confusion_matrix(y_test, y_test_pred)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    #plt.show()

    #classification report
    print(classification_report(y_test,y_test_pred))
    clf_report = classification_report(y_test,y_test_pred,labels=[0,1], target_names=["False","True"],output_dict=True)
    #.iloc[:-1, :] to exclude support
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    #plt.show()

    #ROC Curve
    #y_score = hypermodel.decision_function(X_test)
    roc_fig, roc_ax = plt.subplots()
    roc_ax.plot(fpr_test,tpr_test, label = "test")
    roc_ax.plot(fpr_train,tpr_train, label = "train")
    roc_ax.plot(np.linspace(0,1),np.linspace(0,1), alpha=0.6, linestyle="--")
    roc_ax.set_xlim(0,1)
    roc_ax.set_ylim(0,1.1)
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.set_title("ROC Curve")
    roc_ax.legend()
    #plt.show()
    
    scores[label]=[roc_auc_score(y_test,y_test_pred)]
    print(scores[label])

#calulate scores
print(scores)
result = scores.iloc[0].mean()
print(result)
plt.show()
