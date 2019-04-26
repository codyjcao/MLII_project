import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm # wrap your iterable in tqdm() to see progress bar
from datetime import datetime

start = 8*3600*1000 # 8AM
end = 15*3600*1000  # 3PM


###################### Read in data ###########################
fin = 'ESH2019'
ext = '.csv'
df = pd.read_csv(fin+ext) # input correct csv file
###################### ############ ###########################

# drop unwanted columns
df.drop(['Price','Volume','Type'], axis = 1, inplace = True)

# NaNs occur in same row
df.dropna(inplace = True)

# midpoint
df['Mid'] = (df['Bid Price'].values + df['Ask Price'].values)/2

# mseconds
tmp = pd.to_timedelta(df['Time'].values)
tmp2 = (tmp.seconds*1000 + tmp.microseconds/1000)
df['msecs'] = np.float64((tmp2*2/1000)).round(0)*1000/2

# sec10
tmp = pd.to_timedelta(df['Time'].values)
tmp2 = (tmp.seconds)

# US market hours
df = df[(df['msecs'].values >= start) & (df['msecs'].values <= end)]

# take the latest bid-ask price for each 500 millisecond interval
df = (df.sort_values(by='Time', ascending=True)
  .groupby(['Date','msecs'])
  .head(1)
  .sort_index())

# generating each 500 ms interval for a day
msec_range = np.arange(start,end+.0001,500)
ndt = msec_range.shape[0]
ndate = np.unique(df['Date'].values).shape[0]

# each date within our data set
length = ndt*ndate
date = []
for idx, val in tqdm(enumerate(np.unique(df['Date'].values))):
    date += [val]*ndt

# creating the data frame with the correct dates and ms intervals
DF = pd.DataFrame({'Date':date,
                  'msecs': list(msec_range)*ndate})

# merging the bid-ask data with new time intervals
DF = pd.merge(DF, df, how='left', on=['Date', 'msecs'])

# take out Sundays
dates = np.unique(DF['Date'].values)
day_p = np.zeros(DF.shape[0])
# weekends in dates
wknd = [day for day in dates if datetime.strptime(day,'%m/%d/%Y').weekday()>=5]
for idx, day in enumerate(wknd):
    day_p += DF['Date'].values == day
DF = DF[day_p == 0]

###############################################################################

#M Measures
DF['dM']      = np.zeros(DF.shape[0])
DF['M_10ma']  = np.zeros(DF.shape[0])
DF['M_20ma']  = np.zeros(DF.shape[0])
DF['M_20fma'] = np.zeros(DF.shape[0])
DF['M_10fma'] = np.zeros(DF.shape[0])
DF['signal']  = np.zeros(DF.shape[0])
DF['signal2'] = np.zeros(DF.shape[0])
DF['signal3'] = np.zeros(DF.shape[0])
DF['signal4'] = np.zeros(DF.shape[0])
DF['signal5'] = np.zeros(DF.shape[0])
DF['signal6'] = np.zeros(DF.shape[0])


###
DF['B eq ind'] = np.zeros(DF.shape[0])
DF['B > ind']  = np.zeros(DF.shape[0])
DF['A eq ind'] = np.zeros(DF.shape[0])
DF['A < ind']  = np.zeros(DF.shape[0])
###
DF['dVB'] = np.zeros(DF.shape[0])
DF['dVA'] = np.zeros(DF.shape[0])
DF['OI']  = np.zeros(DF.shape[0])
###
DF['OI_a1']  = np.zeros(DF.shape[0])
DF['OI_a2']  = np.zeros(DF.shape[0])
DF['OI_a3']  = np.zeros(DF.shape[0])
DF['OI_a4']  = np.zeros(DF.shape[0])
DF['OI_a5']  = np.zeros(DF.shape[0])
###
DF['OI_5ma']   = np.zeros(DF.shape[0])
DF['OI_10ma']  = np.zeros(DF.shape[0])
DF['OI_20ma']  = np.zeros(DF.shape[0])

Q = .25 # minimum bid-ask spread

########### Generating the features #################

# iterate through each unique day in data set
for idx, val in tqdm(enumerate(np.unique(DF['Date']))):
    i = DF['Date'] == val

    # fill in NaN in price columns with most recent price
    DF.loc[i,['Bid Price','Ask Price','Ask Size', 'Bid Size', 'Mid']] = (
            DF[i][['Bid Price','Ask Price','Ask Size', 'Bid Size','Mid']].interpolate(
            method='pad',axis=1,limit_direction='forward'))

    ###############################################
    # delta M Measures (Response Variable)
    ## change in mid price
    DF.loc[i,'dM'] = DF.loc[i,'Mid']- DF.shift(1).loc[i,'Mid']

    ## 20 period moving average of mid price
    DF.loc[i,'M_20ma'] = DF.loc[i,"Mid"].rolling(window=20,min_periods=20).mean().shift(1)

    ## 10 period moving average of mid price
    DF.loc[i,'M_10ma'] = DF.loc[i,"Mid"].rolling(window=10,min_periods=10).mean().shift(1)

    ## 20 period forward moving average of mid price
    DF.loc[i,'M_20fma'] = DF.loc[i,"Mid"].rolling(window=20,min_periods=20).mean().shift(-20)

    ## 10 period forward moving average of mid price
    DF.loc[i,'M_10fma'] = DF.loc[i,"Mid"].rolling(window=10,min_periods=10).mean().shift(-10)


    ###### signals ######

    ## signal with 20 period moving averages (RESPONSE VARIABLE 1)
    DF.loc[i,'signal'] =1*(DF.loc[i,'M_20fma']-DF.loc[i,'Mid']>=Q) + -1*(DF.loc[i,'M_20fma']-DF.loc[i,'Mid']<=-Q)

    ## signal with 10 period moving average (RESPONSE VARIABLE 2)
    DF.loc[i,'signal2'] =1*(DF.loc[i,'M_10fma']-DF.loc[i,'Mid']>=Q) + -1*(DF.loc[i,'M_10fma']-DF.loc[i,'Mid']<=-Q)

    ## signal with mid/20p MA (RESPONSE VARIABLE 3)
    DF.loc[i,'signal3'] =1*(DF.loc[i,'Mid']-DF.loc[i,'M_20ma']>=Q) + -1*(DF.loc[i,'Mid']-DF.loc[i,'M_20ma']<=-Q)

    ## signal with mid/10p MA (RESPONSE VARIABLE 4)
    DF.loc[i,'signal4'] =1*(DF.loc[i,'Mid']-DF.loc[i,'M_10ma']>=Q) + -1*(DF.loc[i,'Mid']-DF.loc[i,'M_10ma']<=-Q)

    ## signal with 10f MA/ 10p MA (RESPONSE VARIABLE 5)
    DF.loc[i,'signal5'] =1*(DF.loc[i,'M_10fma']-DF.loc[i,'M_10ma']>=Q) + -1*(DF.loc[i,'M_10fma']-DF.loc[i,'M_10ma']<=-Q)

    ## signal with 20f MA/10p MA (RESPONSE VARIABLE 6)
    DF.loc[i,'signal6']=1*(DF.loc[i,'M_20fma']-DF.loc[i,'M_20ma']>=Q) + -1*(DF.loc[i,'M_20fma']-DF.loc[i,'M_20ma']<=-Q)
    ###############################################

    ###############################################
    # OI Measures
    DF.loc[i,'B eq ind'] = DF.loc[i,'Bid Price'] == DF.shift(1).loc[i,'Bid Price']
    DF.loc[i,'B > ind'] = DF.loc[i,'Bid Price'] > DF.shift(1).loc[i,'Bid Price']
    DF.loc[i,'A eq ind'] = DF.loc[i,'Ask Price'] == DF.shift(1).loc[i,'Ask Price']
    DF.loc[i,'A < ind'] = DF.loc[i,'Ask Price'] < DF.shift(1).loc[i,'Ask Price']

    ## change in ask/bid volumes
    DF.loc[i,'dVA']= (DF.loc[i,'A eq ind']*(DF.loc[i,'Ask Size']-DF.shift(1).loc[i,'Ask Size'])
                      +DF.loc[i,'A < ind']*DF.loc[i,'Ask Size'])
    DF.loc[i,'dVB']= (DF.loc[i,'B eq ind']*(DF.loc[i,'Bid Size']-DF.shift(1).loc[i,'Bid Size'])
                      +DF.loc[i,'B > ind']*DF.loc[i,'Bid Size'])

    ## order imbalance
    DF.loc[i,'OI'] = DF.loc[i,'dVB'] - DF.loc[i,'dVA']
    ###############################################

    ###############################################
    # OI Lags
    DF.loc[i,'OI_a1']=DF.loc[i,'OI'].shift(1)
    DF.loc[i,'OI_a2']=DF.loc[i,'OI'].shift(2)
    DF.loc[i,'OI_a3']=DF.loc[i,'OI'].shift(3)
    DF.loc[i,'OI_a4']=DF.loc[i,'OI'].shift(4)
    DF.loc[i,'OI_a5']=DF.loc[i,'OI'].shift(5)
    ###############################################

    ###############################################
    # OI Moving Averages
    DF.loc[i,'OI_5ma']  = DF.loc[i,"OI"].rolling(window=5,min_periods=5).mean()
    DF.loc[i,'OI_10ma'] = DF.loc[i,"OI"].rolling(window=10,min_periods=10).mean()
    DF.loc[i,'OI_20ma'] = DF.loc[i,"OI"].rolling(window=20,min_periods=20).mean()
    ###############################################


# order imbalance ratio
DF['OI Ratio'] = ((DF['Bid Size'] - DF['Ask Size'])/
                 (DF['Bid Size'] + DF['Ask Size']))

# fixed an issue where Order Imbalance values were objects
DF.loc[:,'OI'] = DF.loc[:,'OI'].astype(np.float64)
DF.loc[:,'Time'] = DF.loc[:,'Time'].replace(np.nan, '', regex=True)

DF.to_csv(fin+'_final'+ext)
