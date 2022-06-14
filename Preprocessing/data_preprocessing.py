import pandas as pd
import numpy as np
import datetime
import os
import unicodedata
import holidays


def time_confirm(df,time_column):
    df[time_column] = pd.to_datetime(df[time_column])

    for i in range(len(df))[:-1]:
        if df[time_column].iloc[i]+pd.Timedelta(1,'h')!=df[time_column].iloc[i+1]:
            raise Exception(str(i)+"<<<<<error here!!!!!")
            print(i)

    if (df[time_column].max()-df[time_column].min())/pd.Timedelta(1,'h')+1 ==len(df):
        raise Exception("날짜 기간이 맞지 않습니다.")

    
def socar_df_preprocessing(df):
  df.loc[df['column2'].str.contains('\u3000'), ['column2']] = df['column2'].str.replace('\u3000',' ')

  df.loc[df['column3'] < df['column4'], ['column3', 'column4']] = df.loc[df['column3'] < df['column4'],['column4','column3']].values
  df['column3'] = pd.to_datetime(df['column3'].str.slice(stop = -6), format="%Y-%m-%d %H:%M:%S")
  df['column4'] = pd.to_datetime(df['column4'].str.slice(stop = -6), format="%Y-%m-%d %H:%M:%S")


  df['column5'] = df['column3'] - df['column4']
  df['column6'] = pd.to_datetime(df['column3']).dt.weekday
  df['column7'] = pd.to_datetime(df['column3']).dt.month
  df['column8'] = pd.to_datetime(df['column3']).dt.day
  df['column9'] = pd.to_datetime(df['column3']).dt.hour

  df['column10'] = pd.to_datetime(df['column4']).dt.weekday
  df['column11'] = pd.to_datetime(df['column4']).dt.month
  df['column12'] = pd.to_datetime(df['column4']).dt.day
  df['column13'] = pd.to_datetime(df['column4']).dt.hour
  df['column14']= df['column5']/np.timedelta64(1, 's')

  df.drop(df[df['column5'] ==datetime.timedelta(0)].index,inplace = True)



def create_aws_data_by_gugun(location,path):
  aws_dir = os.path.join(path,location)
  substrings = ['기온','강수','풍속','습도']
  substrings = ['_' + unicodedata.normalize('NFD',x) + '_' for x in substrings]
  data_set = pd.DataFrame([])

  for substring in substrings:
    loc_list = list(set([string[:string.find(substring)] for string in os.listdir(aws_dir) if substring in string]))
    first = True
    for loc in loc_list:
      file_list = sorted([string for string in os.listdir(aws_dir) if (substring in string)&(loc in string)])
      data_1 = pd.read_csv(os.path.join(aws_dir,file_list[0]),header=None,skiprows=1)[-9:]
      data_2 = pd.read_csv(os.path.join(aws_dir,file_list[1]),header=None,skiprows=1)[:-9]
      data = pd.concat((data_1,data_2),axis = 0,ignore_index=True)
      data[0] = data[0].astype(str)
      data[0] = data[0].str.strip()

      date_list = [date for date in data[0].unique() if not date.startswith('Start')]
      data = data[data[0].isin(date_list)]
      
      if substring == substrings[0]:
        data.loc[data[2]==-50,2]=np.nan
      else:
        data.loc[data[2]<0,2]=np.nan
      if first:
        data_set_1 = data
        first = False
      else:
        data_set_1 = pd.concat((data_set_1,data.iloc[:,-1]),axis=1)
    data_set[substring] = data_set_1.iloc[:,2:].mean(axis=1,skipna=True)
    data_set[substring] = data_set[substring].interpolate()
  data_set.columns = ['temp','precipitation','windspeed','humidity']
  data_set['datetime'] = [datetime.datetime(20xx, 1, 1, 0, 0, 0) + pd.Timedelta(x,'h') for x in range(len(data_set))]
  data_set.drop(data_set[data_set['datetime']>=datetime.datetime(20xx, 11, 30, 0, 0, 0)].index,inplace = True)
  data_set.reset_index(drop=True,inplace=True)
  
  return data_set



def create_aws_forecast_data_by_gugun(path, location):
  aws_dir = os.path.join(path,location)
  substrings = ['3시간기온','6시간강수량','풍속','습도']
  substrings = ['_' + unicodedata.normalize('NFD',x) + '_' for x in substrings]
  data_set = pd.DataFrame([])
  for substring in substrings:
      loc_list = [string[:string.find(substring)] for string in os.listdir(aws_dir) if substring in string]
      first = True
      for loc in loc_list:
        file_list = [string for string in os.listdir(aws_dir) if (substring in string)&(loc in string)]
        data = pd.read_csv(os.path.join(aws_dir,file_list[0]),header=None,skiprows=1)
        data[0] = data[0].astype(str)
        data[0] = data[0].str.strip()
        date_list = [date for date in data[0].unique() if not date.startswith('Start')]
        data = data[data[0].isin(date_list)]
        data[0] = data[0].astype(int)
        data.reset_index(drop=True, inplace =True)

        indexes = data.loc[data[0].shift(1) > data[0]].index.values
        first_1 = True
        for i in range(len(indexes)):
          if first_1:
            data[0][indexes[i]:]=data[0][indexes[i]:]+data[0][indexes[i]-1]
            first_1=False
          else :
            data[0][indexes[i]:]=data[0][indexes[i]:]+data[0][indexes[i]-1]-data[0][indexes[i-1]-1]
        
        data['datetime'] = datetime.datetime(20xx, 11, 30, 0, 0, 0) + pd.to_timedelta(data[0].astype(int),unit='d')+pd.to_timedelta(data[1]/100+ data[2]+9,unit='h')
        data.drop_duplicates(['datetime'],keep ='last',inplace=True)
        data.index = data['datetime']
        data = data.resample(rule='H').last()
        data[3] = data[3].interpolate()
        data.drop([0,1,2,'datetime'],axis=1,inplace=True)

        if first:
          data_set_1 = data
          first = False
        else:
          data_set_1 = pd.concat((data_set_1,data[3]),axis=1)
        assert len(data_set_1[data_set_1.isna().any(axis=1)])==0, "-삐빅- 데이터 오류입니다."
      data_set[substring] = data_set_1['20xx-01-01 00':'20xx-11-29 23'].mean(axis=1).values
  data_set.reset_index(drop=True,inplace=True)
  data_set['datetime'] = [datetime.datetime(20xx, 1, 1, 0, 0, 0) + pd.Timedelta(x,'h') for x in range(len(data_set))]
  data_set.drop(data_set[data_set['datetime']>=datetime.datetime(20xx, 11, 30, 0, 0, 0)].index,inplace = True)
  return data_set



def asos_df_prerprocessing(df):
    df['일시'] = pd.to_datetime(df['일시'])

    df.drop(['일조(hr)','일사(MJ/m2)','풍향(16방위)','해면기압(hPa)','전운량(10분위)'],axis=1,inplace=True)
    df.loc[df['강수량(mm)'].isna(),['강수량(mm)']]=0

    df['풍속(m/s)'].interpolate(inplace=True)

    df.drop(df[df['일시']<datetime.datetime(20xx, 1, 1, 0, 0, 0)].index, inplace = True)
    df.drop(['지점', '지점명'],axis=1,inplace=True)

    df.columns = ['datetime','temp','precipitation','windspeed','humidity']


def finedust_df_prerprocessing(df):
    df.drop(columns=df.columns[:2],inplace=True)
    df.replace('-',np.nan,inplace=True)
    positions = len(df['1'].unique())
    
    for col in df.columns[1:]:
        df[col] = df[col].astype('float')
    
    result_df = pd.DataFrame([])
    for i in range(int(len(df)/positions))[4:]:
        date_df = pd.DataFrame([])
        for j in ['지역A','지역B','지역C','지역D','지역E']:
            position_df = df.iloc[i*positions:(i+1)*positions,1:][df.iloc[i*positions:(i+1)*positions,0].str.contains(j)].mean(axis=0,skipna=True)
            position_df.name = j
            date_df = pd.concat((date_df,position_df),axis = 1)
        result_df = pd.concat((result_df,date_df),axis=0)
    result_df.reset_index(drop=True,inplace=True)
    

    return result_df.iloc[:-2]


def holiday_add(ts_df):
  kr_holidays = holidays.KR()
  start_date = 'startdate'
  end_date = 'startdate'
  date_list = pd.date_range(start=start_date, end=end_date, freq='D')
  holiday_df = pd.DataFrame(columns=['datetime','holiday'])
  holiday_df['datetime'] = sorted(date_list)
  holiday_df['holiday'] = holiday_df.datetime.apply(lambda x: 1 if x in kr_holidays else 0)


  holiday_df['datetime'] = pd.to_datetime(holiday_df['datetime']).dt.strftime("%Y-%m-%d")
  holiday_df.index = date_list

  holiday_df = holiday_df.resample(rule='H').last()

  holiday_df['datetime']=holiday_df.index.values
  holiday_df.ffill(inplace=True)

  ts_df['weekday'] = ts_df['datetime'].dt.weekday
  ts_df_1 = pd.merge(ts_df,holiday_df,how='left')

  ts_df_1['multi_dayoff'] = np.zeros(len(ts_df))
  ts_df_1['one_dayoff'] = np.zeros(len(ts_df))
  
  list_1 = []
  a = 1
  for i in range(int(len(ts_df)/24)):

      if (ts_df_1.iloc[i*24]['holiday']==1)|(ts_df_1.iloc[i*24]['weekday']==5)|(ts_df_1.iloc[i*24]['weekday']==6):
          list_1.append(a)
          a+=1
      elif len(list_1)>=2:
          ts_df_1['multi_dayoff'][(i-len(list_1))*24:i*24] = np.repeat(list_1,24)
          a=1
          list_1=[]

      elif len(list_1)==1:
          ts_df_1['one_dayoff'][(i-1)*24:i*24] = np.repeat(list_1,24)
          a=1
          list_1=[]
  return ts_df_1



def cnt_rate_per_age(df):
  df_1=df.copy()
  week_total_count = []
  cnt=0
  for i in range(24*7,len(df_1),24*7):
    week_sum = 0
    for j in range(i-24*7,i):
      week_sum+=df_1["count"][j]
    week_total_count.append(week_sum)
    cnt+=1
  week_total_count.append(df_1["count"][24*7*cnt:].sum())
  week_total_count = pd.DataFrame({"count":week_total_count})

  age_rate = pd.DataFrame({"dummy":[0 for i in range(48)]})

  for k in df_1.iloc[:,2:7].columns:
    age_lst = []
    cnt=0
    for i in range(24*7,len(df_1),24*7):
      week_sum = 0
      for j in range(i-24*7,i):
        week_sum+=df_1[k][j]
      age_lst.append(week_sum)
      cnt+=1
    age_lst.append(df_1[k][24*7*cnt:].sum())
    age_lst = pd.DataFrame(age_lst)
    age_rate = pd.concat([age_rate,age_lst],axis=1)
  age_rate = pd.concat([age_rate,week_total_count],axis=1)
  age_rate.columns=["dummy",'age_group_1','age_group_2',"age_group_3",'age_group_4','age_group_5',"count"]

  for i in age_rate.columns[:-1]:
    age_rate[f"{i}_rate"]=age_rate[i]/age_rate["count"]*100
  age_rate = age_rate.drop(["dummy","dummy_rate",'age_group_1',   'age_group_2',   'age_group_3',   'age_group_4',   'age_group_5',   'count'],axis=1)
  
  age_rate.index = pd.date_range("startdate","enddate",freq="7D")
  age_rate = age_rate.resample('H').last()
  age_rate = age_rate.fillna(method='ffill')
  dummy_array = pd.DataFrame(pd.date_range('startdate','enddate',freq = 'H')[1:-1],columns=['datetime'])
  for i in age_rate.columns:
    dummy_array[i] = np.nan
  dummy_array =dummy_array.drop("datetime",axis=1)
  age_rate = pd.concat([age_rate,dummy_array],axis=0)
  age_rate = age_rate.fillna(method='ffill')
  age_rate['datetime'] = pd.date_range('startdate','enddate',freq = 'H')[:-1]
  age_rete = age_rate.reset_index(drop=True)
  df_1 = df_1.merge(age_rate)
  return df_1




def timeseries_df_create_2(socar_file_path, weather_file_path, finedust_file_apth, standard_time, location=None):
  socar_df = pd.read_csv(os.path.join(socar_file_path,'20211022_수요예측_hackathon_data.csv'), encoding = 'cp949')
  socar_df_preprocessing(socar_df)
  socar_df = socar_df.loc[socar_df['column1']=='region1']
  socar_df = pd.get_dummies(socar_df,columns = ['age_group','gender','car_model'])
  cols = [col for col in socar_df.columns if (col.startswith('age'))|(col.startswith('gender'))|(col.startswith('car'))]

  finedust25_df = pd.read_csv(os.path.join(finedust_file_apth,'pm25.csv'), encoding = 'cp949')
  finedust10_df = pd.read_csv(os.path.join(finedust_file_apth,'pm10.csv'), encoding = 'cp949')
  finedust25_df = finedust_df_prerprocessing(finedust25_df)
  finedust10_df = finedust_df_prerprocessing(finedust10_df)

  if location:
    weather_df = create_aws_data_by_gugun(location,weather_file_path) 
    socar_df = socar_df.loc[socar_df['column2']==location]
    finedust10_df = finedust10_df[location]
    finedust25_df = finedust25_df[location]
  else:
    weather_df = pd.read_csv(os.path.join(weather_file_path,'region1ASOS_기상자료.csv'), encoding = 'cp949')
    asos_df_prerprocessing(weather_df)
    finedust10_df = finedust10_df.mean(axis=1)
    finedust25_df = finedust25_df.mean(axis=1)

  date_begin = datetime.datetime(20xx, 1, 1, 0, 0, 0)
  date_end = datetime.datetime(20xx, 11, 30, 0, 0, 0)

  dummy_size = int((date_end - date_begin)/pd.Timedelta(1,'h'))
  socar_df.drop(['column1', 'column2','column5', 'column6','column7','column8', 'column9',
                  'column10', 'column11', 'column12', 'column13', 'column14'],axis = 1,inplace = True)
  
  socar_ts_df = pd.DataFrame([],columns =['datetime','count']+cols)
  for i in range(dummy_size):
    data_1 = socar_df.loc[(date_begin+pd.Timedelta(i+1,'h') > socar_df['column4']) &
                        (date_begin+pd.Timedelta(i,'h') < socar_df['column3'])].sum()
    data_1_len = data_1[:5].sum()
    data_1['count'] = data_1_len
    data_1['datetime'] = date_begin+pd.Timedelta(i,'h')
    socar_ts_df = socar_ts_df.append(data_1, ignore_index = True)
  ts_df = socar_ts_df
  ts_df = ts_df.merge(weather_df, how='left')
  ts_df = pd.concat((ts_df, finedust10_df[:len(socar_ts_df)]),axis=1)
  ts_df.rename(columns = {ts_df.columns[-1] : 'finedust10'}, inplace = True)
  ts_df = pd.concat((ts_df, finedust25_df[:len(socar_ts_df)]),axis=1)
  ts_df.rename(columns = {ts_df.columns[-1] : 'finedust25'}, inplace = True)

  ts_df.fillna(0,inplace=True)
  ts_df['count'] = ts_df['count'].astype(int)
  ts_df['sensible_temp'] = 13.12 + 0.6215 * ts_df['temp'] - 11.37 * (ts_df['windspeed']* 3600 / 1000)**0.16 + 0.3965 * (ts_df['windspeed']* 3600 / 1000)**0.16 * ts_df['temp']
  return ts_df

def timeseries_df_create_1(socar_file_path, weather_file_path, finedust_file_apth, standard_time, location=None):
  
  socar_df = pd.read_csv(os.path.join(socar_file_path,'20211022_수요예측_hackathon_data.csv'), encoding = 'cp949')
  socar_df_preprocessing(socar_df)
  socar_df = socar_df.loc[socar_df['column1']=='region1']

  finedust25_df = pd.read_csv(os.path.join(finedust_file_apth,'pm25.csv'), encoding = 'cp949')
  finedust10_df = pd.read_csv(os.path.join(finedust_file_apth,'pm10.csv'), encoding = 'cp949')
  finedust25_df = finedust_df_prerprocessing(finedust25_df)
  finedust10_df = finedust_df_prerprocessing(finedust10_df)

  date_begin = socar_df[standard_time].min().floor('h')
  date_end = socar_df[standard_time].max().ceil('h')

  if location:
    weather_df = create_aws_data_by_gugun(location,weather_file_path) 
    socar_df = socar_df.loc[socar_df['column2']==location]
    finedust10_df = finedust10_df[location]
    finedust25_df = finedust25_df[location]
  else:
    weather_df = pd.read_csv(os.path.join(weather_file_path,'region1ASOS_기상자료.csv'), encoding = 'cp949')
    asos_df_prerprocessing(weather_df)
    finedust10_df = finedust10_df.mean(axis=1)
    finedust25_df = finedust25_df.mean(axis=1)
  
  socar_df = pd.get_dummies(socar_df,columns = ['age_group','gender','car_model'])

  dummy_size = int((date_end - date_begin)/pd.Timedelta(1,'h'))
  dummy_array = pd.DataFrame(np.zeros(dummy_size),columns=['datetime'])
  dummy_array['datetime'] = [date_begin + pd.Timedelta(x,'h') for x in range(dummy_size)]

  socar_df[standard_time] = socar_df[standard_time].dt.floor('h')

  loop_count = socar_df[standard_time].unique()

  socar_df.drop(['column1', 'column2', 'column5', 'column6', 'column7', 'column8', 'column9',
                 'column10', 'column11', 'column12', 'column13', 'column14'], axis=1, inplace=True)

  socar_ts_df = pd.DataFrame([])
  for i in range(len(loop_count)):
    data_1 = socar_df.loc[socar_df[standard_time]==loop_count[i],socar_df.columns[2:]].sum()
    data_1_len = data_1[1:6].sum()
    data_1['column10'] = data_1['column10'] / data_1_len
    data_1['count'] = data_1_len
    data_1['datetime'] = loop_count[i]
    socar_ts_df = socar_ts_df.append(data_1, ignore_index = True)
  
  ts_df = dummy_array.merge(socar_ts_df, how = 'left')
  ts_df = ts_df.merge(weather_df, how='left')
  ts_df = pd.concat((ts_df, finedust10_df),axis=1)
  ts_df.rename(columns = {ts_df.columns[-1] : 'finedust10'}, inplace = True)
  ts_df = pd.concat((ts_df, finedust25_df),axis=1)
  ts_df.rename(columns = {ts_df.columns[-1] : 'finedust25'}, inplace = True)

  ts_df.fillna(0,inplace=True)
  ts_df['count'] = ts_df['count'].astype(int)
  ts_df['sensible_temp'] = 13.12 + 0.6215 * ts_df['temp'] - 11.37 * (ts_df['windspeed']* 3600 / 1000)**0.16 + 0.3965 * (ts_df['windspeed']* 3600 / 1000)**0.16 * ts_df['temp']

  ts_df.drop(ts_df[ts_df['datetime']>=datetime.datetime(20xx, 11, 30, 0, 0, 0)].index,inplace = True)

  return ts_df

  
def timeseries_df_create(socar, asos, finedust10, finedust25, standard_time, location=None):
  socar_df = socar.copy()
  asos_df = asos.copy()
  finedust10_df = finedust10.copy()
  finedust25_df = finedust25.copy()

  date_begin = socar_df[standard_time].min().floor('h')
  date_end = socar_df[standard_time].max().ceil('h')

  socar_df = pd.get_dummies(socar_df,columns = ['age_group','gender','car_model'])

  if location:
    socar_df = socar_df.loc[socar_df['column2']==location]
    finedust10_df = finedust10_df[location]
    finedust25_df = finedust25_df[location]
  else:
    finedust10_df = finedust10_df.mean(axis=1)
    finedust25_df = finedust25_df.mean(axis=1)

  dummy_size = int((date_end - date_begin)/pd.Timedelta(1,'h'))
  dummy_array = pd.DataFrame(np.zeros(dummy_size),columns=['datetime'])
  dummy_array['datetime'] = [date_begin + pd.Timedelta(x,'h') for x in range(dummy_size)]

  socar_df[standard_time] = socar_df[standard_time].dt.floor('h')

  socar_df.drop(socar_df.columns[5:13],axis = 1,inplace = True)

  loop_count = socar_df[standard_time].unique()

  socar_ts_df = pd.DataFrame([])
  for i in range(len(loop_count)):
    data_1 = socar_df.loc[socar_df[standard_time]==loop_count[i],socar_df.columns[5:]].sum()
    data_1_len = data_1[1:6].sum()
    data_1['column10'] = data_1['column10'] / data_1_len
    data_1['count'] = data_1_len
    data_1['datetime'] = loop_count[i]
    socar_ts_df = socar_ts_df.append(data_1, ignore_index = True)
  
  ts_df = dummy_array.merge(socar_ts_df, how = 'left')
  ts_df = ts_df.merge(asos_df, how='left')
  ts_df = pd.concat((ts_df, finedust10_df),axis=1)
  ts_df.rename(columns = {ts_df.columns[-1] : 'finedust10'}, inplace = True)
  ts_df = pd.concat((ts_df, finedust25_df),axis=1)
  ts_df.rename(columns = {ts_df.columns[-1] : 'finedust25'}, inplace = True)

  ts_df.fillna(0,inplace=True)
  ts_df['count'] = ts_df['count'].astype(int)
  ts_df['sensible_temp'] = 13.12 + 0.6215 * ts_df['temp'] - 11.37 * (ts_df['windspeed']* 3600 / 1000)**0.16 + 0.3965 * (ts_df['windspeed']* 3600 / 1000)**0.16 * ts_df['temp']

  ts_df.drop(ts_df[ts_df['datetime']>=datetime.datetime(20xx, 11, 30, 0, 0, 0)].index,inplace = True)

  return ts_df


        
