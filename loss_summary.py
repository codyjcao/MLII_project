import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sns
from sklearn.metrics import confusion_matrix

def my_loss_summmary(ytruetest, ytruetrain, ypredtest, ypredtrain):
    loss = np.array([[0,1,2],[1,0,1],[2,1,0]])
    days = len(ytruetest)
    losstrain = np.zeros(days)
    losstest = np.zeros(days)
    acctrain = np.zeros(days)
    acctest = np.zeros(days)
    acctrain1 = np.zeros(days)
    acctest1 = np.zeros(days)
    
    for i in range(days):
        conftrain = confusion_matrix(ytruetrain[i], ypredtrain[i])
        acctrain[i] = (conftrain[0,0] + conftrain[2,2])/np.sum(conftrain[0] + conftrain[2])
        acctrain1[i] = (conftrain[0,0] + conftrain[2,2])/np.sum(
            conftrain[0,0] + conftrain[0,2] + conftrain[2,0] + conftrain[2,2])
        losstrain[i] = np.sum(conftrain * loss)
        
        conftest = confusion_matrix(ytruetest[i], ypredtest[i])
        acctest[i] = ((conftest[0,0] + conftest[2,2])/
                      np.sum(conftest[0] + conftest[2]))
        acctest1[i] = (conftest[0,0] + conftest[2,2])/np.sum(
            conftest[0,0] + conftest[0,2] + conftest[2,0] + conftest[2,2])
        losstest[i] = np.sum(conftest * loss)
    
    
    ind = [i+1 for i in range(days)]
    col = ['Loss Train','Loss Test','Accuracy Train 1',
           'Accuracy Train 2','Accuracy Test 1',
           'Accuracy Test 2']
 
    plt.figure(figsize=(15,5))
    plt.plot(ind, losstrain, label='Loss (Train)')
    plt.plot(ind, losstest, label='Loss (Test)')
    plt.title('Loss by Day', fontsize=20)
    plt.xlabel("Day")
    plt.ylabel("Loss")
    plt.legend(prop={'size':15})
    plt.show()
    
    plt.figure(figsize=(15,5))
    plt.plot(ind, acctrain, label='Trading Accuracy 1 (Train)')
    plt.plot(ind, acctest, label='Trading Accuracy 1 (Test)')
    plt.title('Trading Accuracy 1 by Day', fontsize=20)
    plt.xlabel("Day")
    plt.ylabel("Trading Accuracy 1")
    plt.legend(prop={'size':15})
    plt.show()
    
    plt.figure(figsize=(15,5))
    plt.plot(ind, acctrain1, label='Trading Accuracy 2 (Train)')
    plt.plot(ind, acctest1, label='Trading Accuracy 2 (Test)')
    plt.xlabel("Day")
    plt.ylabel("Trading Accuracy 2")
    plt.title('Trading Accuracy 2 by Day', fontsize=20)
    plt.legend(prop={'size':15})
    plt.show()
    
    retv = zip(col,(losstrain, losstest, acctrain, acctrain1, acctest, acctest1))
    retdf = pd.DataFrame(dict(retv),index=ind)
    retdf = retdf.round({'Loss Train': 0, 'Loss Test': 0, 'Accuracy Train 1': 4,
                        'Accuracy Train 2': 4, 'Accuracy Test 1': 4, 'Accuracy Test 2': 4})
    
    avedf = pd.DataFrame(retdf.mean())
    avedf.columns = ['Averages']
    avedf = avedf.round(4)
    
    lossdf = retdf.iloc[:,0:2]
    accudf = retdf.iloc[:,[2,4]]
    accudf1 = retdf.iloc[:,[3,5]]
    
    ax1 = sns.heatmap(lossdf)
    ax1.set_title('Loss Heatmap')
    plt.show()
    ax2 = sns.heatmap(accudf)
    ax2.set_title('Trading Accuracy 1 Heatmap')
    plt.show()
    ax3 = sns.heatmap(accudf1)
    ax3.set_title('Trading Accuracy 2 Heatmap')
    plt.show()
    
    return (retdf, avedf)
