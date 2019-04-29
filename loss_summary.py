import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sns
from sklearn.metrics import confusion_matrix

def my_loss_summary(ytruetest, ytruetrain, ypredtest, ypredtrain):
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
    
    
    ind = [('Day ' + str(i+1))for i in range(days)]
    col = ['Loss Train','Loss Test','Accuracy Train 1',
           'Accuracy Train 2','Accuracy Test 1',
           'Accuracy Test 2']
    
    plt.plot(ind, losstrain, label='loss train')
    plt.plot(ind, losstest, label='loss test')
    plt.legend()
    plt.show()
    
    plt.plot(ind, acctrain, label='accuracy train')
    plt.plot(ind, acctest, label='accuracy test')
    plt.legend()
    plt.show()
    
    plt.plot(ind, acctrain1, label='accuracy1 train')
    plt.plot(ind, acctest1, label='accuracy1 test')
    plt.legend()
    plt.show()
    
    retv = zip(col,(losstrain, losstest, acctrain, acctrain1, acctest, acctest1))
    retdf = pd.DataFrame(dict(retv),index=ind)
    
    avedf = pd.DataFrame(retdf.mean())
    avedf.columns = ['Averages']
    
    lossdf = retdf.iloc[:,0:2]
    accudf = retdf.iloc[:,[2,4]]
    accudf1 = retdf.iloc[:,[3,5]]
    
    ax1 = sns.heatmap(lossdf)
    ax1.set_title('heatmap for loss function')
    plt.show()
    ax2 = sns.heatmap(accudf)
    ax2.set_title('heatmap for accuracy1')
    plt.show()
    ax3 = sns.heatmap(accudf1)
    ax3.set_title('heatmap for accuracy2')
    plt.show()

    return (retdf.round(2), avedf.round(2))