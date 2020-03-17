import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import matplotlib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
matplotlib.use('Agg')

def predict_news(code, budget, time):
    url = "https://www.business-standard.com/rss/markets-106.rss"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content, features="xml")
    items = soup.findAll('item')
    item = items[0]
    news_items = []
    for item in items:
        news_item = {}
        news_item['title'] = item.title.text
        news_items.append(news_item)
    news_items
    import pandas as pd 
    df = pd.DataFrame(news_items)
    df.to_csv('upes2.csv', index=False)
    df.transpose()


    # df=pd.read_csv('upload_DJIA_table.csv',encoding='ISO-8859-1')
    #df.head()

    df1=pd.read_csv("Combined_News_DJIA.csv")
    #df1.head()
    df1.Label
    fname = code + '.csv'
    df3=pd.read_csv(fname)
    # df4=df3[df3['Date']>'20080807':df3['Date']<'20160808']
    k=0
    for i in range(len(df3.Date)-1):
        if k<1989:
            if df3.Date[i]==df1.Date[k]:
                if df3.Open[i]-df3.Close[i] >0:
                    df1.Label[k]=0
                else:
                    df1.Label[k]=1 
                k+=1
            else:
                continue

    train=df1[df1['Date']<'20150101']# train set for news headlines
    test=df1[df1['Date']>'20141231'] # test set for news headlines
    #test

    data=train.iloc[:,2:27]
    data.replace("[^a-zA-Z]"," ",regex=True,inplace=True)

    #Renaming column names for ease of access
    list1=[i for i in range(1,26)]

    #new_Index=[str(i) for i in range(25)]
    data.columns=list1
    #data.head()


    for index in list1:
        data[index]=data[index].str.lower()
    #data.head()

    headlines=[]
    for row in range(0,len(data.index)):
        headlines.append(''.join(str(x) for x in data.iloc[row,0:26]))

    #implement bag of words
    countvector=CountVectorizer(ngram_range=(1,1))
    traindataset=countvector.fit_transform(headlines)

    #implement Randomforest Classifier
    randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
    randomclassifier.fit(traindataset,train['Label'])

    test_transform=[]
    for row in range(0,len(test.index)):
        test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
    test_dataset=countvector.transform(test_transform)
    predictions=randomclassifier.predict(test_dataset)

    #import library to check accuracy
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

    matrix=confusion_matrix(test['Label'],predictions)
    #print(matrix)

    score=accuracy_score(test['Label'],predictions)
    print(score)
    report=classification_report(test['Label'],predictions)
    #print(report)

    score=score*100
    print("the accuracy is ",score,"%")


    df2=pd.read_csv("upes2.csv")

    # new_headlines=[]
    # for row in range(0,len(df2.index)):
    #   new_headlines.append(''.join(str(x) for x in data.iloc[row,0:26]))
    # predict=randomclassifier.predict(test_dataset)

    test_transform1=[]

    test_transform1.append(' '.join(str(x) for x in df2.iloc[:27]))
    test_dataset1=countvector.transform(test_transform1)
    predictions1=randomclassifier.predict(test_dataset1)
    if predictions1[0]==1:
        response = "You should invest in this stock!"
    else:
        response = "You should not invest in this stock!"
        
    plt.plot(df3.Date,df3.Open)
    plt.savefig('modules/static/graph3.png')
    df5=pd.read_csv(fname)

    df5=df5.sort_values("Date")
    df5['Label']=df1.Label
    # df5.append(df1.Label)

    high_prices = df5.loc[:,'High'].to_numpy()
    low_prices = df5.loc[:,'Low'].to_numpy()
    mid_prices = (high_prices+low_prices)/2.0
    train_data = mid_prices[:2000]
    test_data = mid_prices[2000:]
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)
    smoothing_window_size = 500
    for di in range(0,1500,smoothing_window_size):
        scaler.fit(train_data[di:di+smoothing_window_size,:])
        train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])
    # print(len(train_data))
    # You normalize the last bit of remaining data
    scaler.fit(train_data[di+smoothing_window_size:,:])
    train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])
    train_data = train_data.reshape(-1)

    # Normalize test data
    test_data = scaler.transform(test_data).reshape(-1)
    EMA = 0.0
    gamma = 0.1
    # print("Train data at 166: ", train_data[166])
    for ti in range(2000):
        EMA = gamma*train_data[ti] + (1-gamma)*EMA
    # if ti < 166:
        # print(EMA)
    train_data[ti] = EMA
    # print(train_data)

    # Used for visualization and test purposes
    all_mid_data = np.concatenate([train_data,test_data],axis=0)

    # window_size = 100
    # N = train_data.size
    # std_avg_predictions = []
    # std_avg_x = []
    # mse_errors = []

    # for pred_idx in range(window_size,N):

    #     if pred_idx >= N:
    #         date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
    #     else:
    #         date = df.loc[pred_idx,'Date']

    #     std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
    #     mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
    #     std_avg_x.append(date)

    # print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))

    window_size = 100
    N = train_data.size

    run_avg_predictions = []
    run_avg_x = []

    mse_errors = []

    running_mean = 0.0
    run_avg_predictions.append(running_mean)

    decay = 0.5

    for pred_idx in range(1,N):

        if pred_idx >= N:
            date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
        else:
            date = df5.loc[pred_idx,'Date']
        running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1]
        run_avg_predictions.append(running_mean)
        mse_errors.append((run_avg_predictions[-1]-train_data[pred_idx])**2)
        run_avg_x.append(date)

    print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))


    plt.figure(figsize = (18,9))
    # plt.plot(range(df5.shape[0]),all_mid_data,color='b',label='True')
    # plt.plot(range(0,N),run_avg_predictions,color='orange', label='Prediction')
    # #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
    # plt.xlabel('Date')
    # plt.ylabel('Mid Price')
    # plt.legend(fontsize=18)
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.legend(fontsize=18)
    plt.plot(df5.Open,df5.Label, color='red')

    plt.savefig('modules/static/graph4.png')

    return response