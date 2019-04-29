x_labs = ['OI_a1', 'OI_a2', 'OI_a3', 'OI_a4', 'OI_a5',
          'OI Ratio','OI_10ma','OI_20ma']

def predict_time(estimators, df, fitted,xlabs=x_labs,ylab='signal6',n_iter=50,n_points = 200000):
    '''
    estimators - a list of estimators to be compared for predicting
    
    df - pandas data frame that we've been suing
    
    n_points - int: how many points to predict on for comparison
    
    return - data frame comparing prediction times for each estimator on
            n_points of data
    '''
    import random
    import numpy as np
    import pandas as pd
	import time
    x = xlabs
    tmp_df = df[['Date'] + xlabs + [ylab]].dropna()

    if not fitted:
        d = random.choice(np.unique(tmp_df['Date'].values))
        test = tmp_df[tmp_df['Date']==d]
        X = test[x].values
        y = test[ylab].values
        for estim in estimators:
            estim.fit(X,y)
    
    times = np.zeros((len(estimators),n_iter))
    
    test_idx = np.random.randint(0,tmp_df.shape[0],size=n_points)
    X_test = tmp_df.iloc[test_idx,:][x].values
    
    for k in range(n_iter):
        for idx,estim in enumerate(estimators):
            start = time.time()
            pred = estim.predict(X_test)
            #print(pred)
            end = time.time()
            times[idx,k] = end-start
    
    columns = ['Trial ' + str(i+1) for i in range(n_iter)]
    return pd.DataFrame(times, columns = columns, index = range(1,len(estimators)+1))