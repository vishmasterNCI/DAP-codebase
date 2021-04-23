#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:13:21 2021

@author: venugopal
"""


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup as bs
from time import sleep
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
from webdriver_manager.chrome import ChromeDriverManager
import ast
import re
import pymongo
from pymongo import MongoClient  
import datetime
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyodbc 
from matplotlib import dates
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.inspection import permutation_importance
import numpy as np




class vishnu_code():
    def __init__(self,client,db):
        self._client=client#MongoClient(host="localhost", port=27017)
        self._db=db#
        self._mydb = self._client[self._db]
        self._mongo_df=pd.DataFrame()
        self._postgredf=pd.DataFrame()
        self._result=[]
        self._get=[]
        self._date_dict={}
        self._month=''
        self._df_list=[]
        self._countries=['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Anguilla', 'Antigua and Barbuda', 'Argentina', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Bermuda', 'Bhutan', 'Bolivia', 'Brazil', 'Brunei', 'Bulgaria', 'Cambodia', 'Canada', 'Cape Verde', 'Cayman Islands', 'Chile', 'China', 'Colombia', 'Costa Rica', "Cote d'Ivoire", 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Estonia', 'European Union', 'Faeroe Islands', 'Falkland Islands', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Greenland', 'Grenada', 'Guatemala', 'Guernsey', 'Guinea', 'Guyana', 'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jersey', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Laos', 'Latvia', 'Lebanon', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macao', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Mauritania', 'Mauritius', 'Mexico', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Montserrat', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nigeria', 'North America', 'North Macedonia', 'Northern Cyprus', 'Norway', 'Oceania', 'Oman', 'Pakistan', 'Palestine', 'Panama', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'Saint Helena', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'South Africa', 'South America', 'South Korea', 'Spain', 'Sri Lanka', 'Suriname', 'Sweden', 'Switzerland', 'Taiwan', 'Thailand', 'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turks and Caicos Islands', 'Uganda', 'Ukraine', 'United Arab Emirates', 'UK', 'US', 'Uruguay', 'Venezuela', 'Vietnam', 'World', 'Zimbabwe']
        self._collections=self._mydb['cases_deaths_recovery']
        self._options=[]
        self._options = webdriver.ChromeOptions()
        self._options.add_argument("--disable-notifiactions")
        self._driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(),options=self._options)
        self._mongo_file=[]
        self._date_dict={"Jan":"01","Feb":"02" ,"Mar":"03" , "Apr":"04", "May":"05", "Jun":"06","Jul":"07" , "Aug":"08","Sep":"09" , "Oct":"10", "Nov":"11","Dec":"12"}      
        self._script=''
        self._cases=''
        self._deaths=''
        self._active_cases=''
        self._death_rate=''
        self._recovery_rate=''
        self._grouped_df=pd.DataFrame()
        self._sqlserver = 'tcp:dapgroup12.database.windows.net' 
        self._sqldb = 'dap' 
        self._username = 'dap' 
        self._password = 'Admin@123'
        self._con =None
        self._myresult=None
        self._q_data=None
        self._vdf=pd.DataFrame()
        self._final_df=pd.DataFrame()

    
    
    def insert_into_mongo(self,data):        
        self._result = self._collections.insert_many(data)
        print("inserted data into mongo"*10)
        
        
    def get_data_mongo(self):
        self._get=self._collections.find()
        return self._get
    
    def crud(self,conn, query):
        c = conn.cursor()
        c.execute(query)
        c.commit()
    
    def insert_into_table(self,con,query,data):
        cur = con.cursor()
        cur.execute(query.format(*data))
        cur.commit()
        
    def get_data_mysql(self,con,query):
        cur = con.cursor()
        cur.execute(query)
        self._myresult=cur.fetchall()
        return self._myresult
        
    
    def date_cleaner(self,date_string):
        month=[key for key in self._date_dict.keys() if key==date_string.split()[0]]   
        month=month[0]
        date_string=date_string.replace(month,self._date_dict[month])
        date_string=date_string.replace(" ","/")
        date_string=datetime.strptime(date_string, '%m/%d/%Y')
        date_string = pd.to_datetime(date_string,format='%Y-%b-%d')
        return date_string
    
    def scrape_data(self):
        self._df_list=[]
        for country in self._countries:
            country=country.lower()
            country=country.replace(' ','-')
            print(country)
            try:
                self._driver.get(r'https://www.worldometers.info/coronavirus/country/{}/'.format(country))
            
                soup=bs(self._driver.page_source,'html.parser')
                self._script=soup.find_all('script',type="text/javascript")
                for s in self._script:
                    if "Highcharts.chart('graph-cases-daily'" in str(s):                        
                                l1=re.findall(r'\[(.*?)\]',str(s))[0]
                                self._cases=re.findall(r'\[(.*?)\]',str(s))[1] 
                    if "Highcharts.chart('graph-active-cases-total'" in str(s):
                                self._active_cases=re.findall(r'\[(.*?)\]',str(s))[1]    
                    
                    
                    if "Highcharts.chart('graph-deaths-daily'" in str(s):
                        try:            
                            self._deaths=re.findall(r'\[(.*?)\]',str(s))[1]
                        except:
                            self._deaths="0"
                        
                    try:
                        if "Highcharts.chart('deaths-cured-outcome-small'" in str(s):
                                    self._death_rate=re.findall(r'\[(.*?)\]',str(s))[1]
                                    self._recovery_rate=re.findall(r'\[(.*?)\]',str(s))[2]
                    except:
                        self._death_rate=[]
                        self._recovery_rate=[]
                l1=re.sub(r'"[^"]*"', lambda m: m.group(0).replace(',', ''), l1)               
                l1=list(ast.literal_eval(l1))
                self._cases=self._cases.replace('null','None')
                try:
                    self._deaths=self._deaths.replace('null','None')
                except:
                    self._deaths=self._deaths
                    
                self._active_cases=self._active_cases.replace('null','None')
                
                self._cases=list(ast.literal_eval(self._cases))
                try:
                    self._deaths=list(ast.literal_eval(self._deaths))
                except:
                    self._deaths=self._deaths
                
                self._active_cases=list(ast.literal_eval(self._active_cases))
                    
               
                try:
                    self._death_rate=self._death_rate.replace('null','None')
                    self._death_rate=list(ast.literal_eval(self._death_rate))
                except:
                    self._death_rate=self._death_rate
                
                    
                
                try:
                    self._recovery_rate=self._recovery_rate.replace('null','None')
                    self._recovery_rate=list(ast.literal_eval(self._recovery_rate))
                except:
                    self._recovery_rate=self._recovery_rate
              
                d={"country":[],"date":[],"active_cases":[],"daily_cases":[],"deaths":[],"death_rate":[],"recovery_rate":[]}
                d["date"]=l1
                d["daily_cases"]=self._cases
                d["active_cases"]=self._active_cases
                if self._deaths==0:
                    d["deaths"]=[self._deaths for i in range(0,len(d["date"]))]
                else:
                    d["deaths"]=self._deaths  
                d["death_rate"]=self._deaths
                d["recovery_rate"]=self._recovery_rate
                d["country"]=[country for i in range(0,len(d["date"]))]
                df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items()]))
                self._df_list.append(df)
            except Exception as e:
                print(e)
                print("No data for {}".format(country))
                break
            
            
                
        return pd.concat(self._df_list)
    
    def recovery_rate_clean(self,x):
        if x[1]=='':
            x[1]=100.0
        elif x[0]<100.0:
            x[1]=100.0
        
        return x[1]
        
        

    def main_one(self):
        self._client = MongoClient(host="localhost", port=27017)
        self._db="dap_project"
        self._mydb = self._client[self._db]
        df=self.scrape_data()    
        df=df.to_dict(orient='records')
        print("*"*20)
        self.insert_into_mongo(df)
        print("*"*20)
        self._mongo_file=self.get_data_mongo()
        print("*"*30)
        print("retrieved data from Mongo")
        print("*"*30)
        self._df_list=[n for n in self._mongo_file]
        self._mongo_df=pd.DataFrame().from_dict(self._df_list)
        print("Cleaning data has started")
        self._mongo_df=self._mongo_df[self._mongo_df.date.isnull()==False]
        self._mongo_df["date"]=self._mongo_df["date"].apply(lambda x:self.date_cleaner(x))
        print("*"*20+"Date Cleaned"+"*"*20)
        
        self._mongo_df['deaths']=self._mongo_df['deaths'].fillna(0)
        self._mongo_df['death_rate']=self._mongo_df['death_rate'].fillna(0)
        self._mongo_df['daily_cases']=self._mongo_df['daily_cases'].fillna(0)
                
        print("*"*20+"Deaths,death rate ,total cases cleaned"+"*"*20)
        
        print("imputing recovery rate based on number of active cases")

        self._mongo_df['recovery_rate']=self._mongo_df[['active_cases','recovery_rate']].apply(lambda x:self.recovery_rate_clean(x) ,axis=1)
        self._mongo_df['recovery_rate']=self._mongo_df['recovery_rate'].fillna(self._mongo_df['recovery_rate'].median())
        print("*"*20+"Recovery Rate Cleaned"+"*"*20)
        self._mongo_df['recovery_rate']=self._mongo_df['recovery_rate'].astype('float')
        
        self._mongo_df.drop('_id',inplace=True,axis=1)
    
    def main_two(self):

        self._con = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+self._sqlserver+';DATABASE='+self._sqldb+';UID='+self._username+';PWD='+self._password) 
        
        drop_table = ("""   
                                        drop table if exists covidData;
                                        
                                    
                                    """)
        self.crud(self._con,drop_table)
        sql_create_covid_table = ("""   
                                        CREATE TABLE covidData (
                                        active_cases DECIMAL (38,4) NOT NULL ,
                                        death_rate DECIMAL (38,4) NOT NULL ,
                                        deaths DECIMAL (38,4) NOT NULL ,
                                        recovery_rate DECIMAL (38,4) NOT NULL ,
                                        daily_cases DECIMAL (38,4) NOT NULL ,
                                        date DATE NOT NULL,
                                        country varchar(50) NOT NULL
                                        
                                    );
                                    """)#PRIMARY KEY (country,date)
        
        self.crud(self._con,sql_create_covid_table)
        
        sql_insert_values = ''' INSERT INTO covidData(active_cases,death_rate,deaths,recovery_rate,daily_cases,date,country)
                              VALUES({},{},{},{},{},'{}','{}'); '''
        
        
        self._mongo_df=self._mongo_df[["active_cases","death_rate","deaths","recovery_rate","daily_cases","date","country"]]                      
        d=self._mongo_df.to_dict(orient="records")
        print("*"*20 +"adding data into MySql DB" +"*"*20)
        for data in d:
            self.insert_into_table(self._con,sql_insert_values,tuple(data.values()))

        
        
        query = """ select country ,sum(deaths) as total_deaths
                    from covidData 
                    group by country
                    order by total_deaths desc ; """
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country', 'total_deaths'])
        
        
        h=sns.barplot(x='country',y='total_deaths',data=self._vdf[0:5])# only 1 column is passed ie x or y
        h.set_xticklabels(rotation=90,labels = list(self._vdf.country[0:5]))
        plt.title("Top 5 countries with most deaths")
        h.legend()
        plt.show()
        
        
        
        
        
        query = """ select country ,avg(death_rate) as average_death_rate
                    from covidData 
                    group by country
                    order by average_death_rate  ; """
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country', 'average_death_rate'])
        
        
        h=sns.barplot(x='country',y='average_death_rate',data=self._vdf[0:5])# only 1 column is passed ie x or y
        h.set_xticklabels(rotation=90,labels = list(self._vdf.country[0:5]))
        plt.title("Top 5 countries with least average death rate")
        h.legend()
        plt.show()
        
        
        
        query = """ select country ,sum(daily_cases) as total_cases
                    from covidData 
                    group by country
                    order by total_cases desc  ; """
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country', 'total_cases'])
        
        
        h=sns.barplot(x='country',y='total_cases',data=self._vdf[0:5])# only 1 column is passed ie x or y
        h.set_xticklabels(rotation=90,labels = list(self._vdf.country[0:5]))
        plt.title("Top 5 countries with most cases")
        h.legend()
        plt.show()
        
        
        
        
        query = """ select country ,sum(active_cases) as active_cases
                    from covidData 
                    group by country
                    order by active_cases desc; """
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country', 'active_cases'])
        
        
        h=sns.barplot(x='country',y='active_cases',data=self._vdf[0:5])# only 1 column is passed ie x or y
        h.set_xticklabels(rotation=90,labels = list(self._vdf.country[0:5]))
        plt.title("Top 5 countries with most Active Cases")
        h.legend()
        plt.show()
        
        
        
        
        query = """ select country ,date ,max(daily_cases) as max_daily_cases
                    from covidData
                    group by date,country
                    order by max_daily_cases desc ; """
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country','date', 'max_daily_cases'])
        
        
        h=sns.barplot(x='date',y='max_daily_cases',hue='country',data=self._vdf[0:5])# only 1 column is passed ie x or y           
        h.set_xticklabels(rotation=0,labels = list(self._vdf.date[0:5]))
        plt.title("Top 5 days maximum reported cases")
        h.legend()
        plt.show()
        
        
        
        query = """ select date,sum(daily_cases) as total_cases_per_day
                    from covidData
                    group by date
                    order by total_cases_per_day;"""
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['date', 'total_cases_per_day'])
        
        
        h=sns.lineplot(x='date',y=self._vdf.total_cases_per_day.astype('float'),data=self._vdf)# only 1 column is passed ie x or y
        h.set_xticklabels(rotation=90,labels = self._vdf.date)
        h.xaxis.set_major_formatter(dates.DateFormatter("%d-%b-%Y"))
        plt.title("Worldwide total cases")
        h.legend()
        plt.show()
        
        query = """ select date,sum(deaths) as total_deaths_per_day
                    from covidData
                    group by date
                    order by total_deaths_per_day;"""
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['date', 'total_deaths_per_day'])
        
        
        h=sns.lineplot(x='date',y=self._vdf.total_deaths_per_day.astype('float'),data=self._vdf)# only 1 column is passed ie x or y
        h.set_xticklabels(rotation=90,labels = self._vdf.date.values)
        h.xaxis.set_major_formatter(dates.DateFormatter("%d-%b-%Y"))
        plt.title("Worldwide total deaths")
        h.legend()
        plt.show()
        
        
        

    def main_three(self):
        self._con = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+self._sqlserver+';DATABASE='+self._sqldb+';UID='+self._username+';PWD='+self._password) 
        query = """ select * from covidData c,vaccineData v,hospitalData h
                    where c.date=v.date
                    and v.date=h.date
                    and c.country=v.country
                    and v.country=h.country; """
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._final_df = pd.DataFrame.from_records(self._q_data, columns =["active_cases","death_rate","deaths","recovery_rate","daily_cases","date","country"
        ,"total_vaccinations","people_vaccinated","people_fully_vaccinated","new_vaccinations","total_vaccinations_per_hundred"
        ,"people_vaccinated_per_hundred","people_fully_vaccinated_per_hundred","population","date","country","Daily_ICU_occupancy",
        "Daily_hospital_occupancy","country","date","level","new_cases","population","positivity_rate","region","region_name","testing_rate",'tests_done'])
        
        
        

        
        self._mldf= self._final_df 
        self._mldf.drop(["date"],axis=1,inplace=True) 
        self._mldf.active_cases=self._mldf.active_cases.astype('float')
        self._mldf.death_rate=self._mldf.death_rate.astype('float')
        self._mldf.deaths=self._mldf.deaths.astype('float')
        self._mldf.recovery_rate=self._mldf.recovery_rate.astype('float')
        self._mldf.daily_cases=self._mldf.daily_cases.astype('float')
        self._mldf.total_vaccinations=self._mldf.total_vaccinations.astype('float')
        self._mldf.people_vaccinated=self._mldf.people_vaccinated.astype('float')
        self._mldf.people_fully_vaccinated=self._mldf.people_fully_vaccinated.astype('float')
        self._mldf.new_vaccinations=self._mldf.new_vaccinations.astype('float')
        self._mldf.total_vaccinations_per_hundred=self._mldf.total_vaccinations_per_hundred.astype('float')
        self._mldf.people_vaccinated_per_hundred=self._mldf.people_vaccinated_per_hundred.astype('float')
        self._mldf.people_fully_vaccinated_per_hundred=self._mldf.people_fully_vaccinated_per_hundred.astype('float')
        self._mldf.population=self._mldf.population.astype('float')
        self._mldf.Daily_ICU_occupancy=self._mldf.Daily_ICU_occupancy.astype('float')
        self._mldf.Daily_hospital_occupancy=self._mldf.Daily_hospital_occupancy.astype('float')
        self._mldf.positivity_rate=self._mldf.positivity_rate.astype('float')
        self._mldf.testing_rate=self._mldf.testing_rate.astype('float')
        self._mldf.tests_done=self._mldf.tests_done.astype('float')
        self._mldf.new_cases=self._mldf.new_cases.astype('float')
        self._mldf=self._mldf.loc[:,~self._mldf.columns.duplicated()]
        print(self._mldf.info())
        
        #inter dataset visualization
        #correlation
        cor_df=self._mldf.select_dtypes(exclude=['object'])
        sns.heatmap(cor_df.corr())
        #strong negative correlation between testing rate and positivity rate,which means if the testing rate increases positivity rate tends to decrease
        #strong positive correlation between recovery rate and daily icu and hospital admissions
        #strong negative correlation between people vaccinated per hundred and deaths,which means with more vaccintions the death rate tends to decrease
        
        #feature selection
        #remove highly correlated variables: total_vaccinations,people_vaccinated,people_fully_vaccinated,people_vaccinated_per_hundred,people_fully_vaccinated_per_hundred,death_rate
        
        
        
        X=self._mldf[self._mldf.columns.difference(['level','country','region','region_name','people_vaccinated','total_vaccinations,people_vaccinated','people_fully_vaccinated','people_vaccinated_per_hundred','total_vaccinations','total_vaccinations_per_hundred','deaths','death_rate','new_cases'])]
        y=self._mldf[["deaths"]]
        
        
        
        #le = preprocessing.LabelEncoder()
        #X['country']=le.fit_transform(X['country'])
        #X['level']=le.fit_transform(X['level'])
        #X['region']=le.fit_transform(X['region'])
        #X['region_name']=le.fit_transform(X['region_name'])
        
        
        
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13)
        params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}
        
        reg = ensemble.GradientBoostingRegressor()#**params
        reg.fit(X_train, y_train)
        
        mse = mean_squared_error(y_test, reg.predict(X_test))
        #rmse = root_mean_squared_error(y_test, reg.predict(X_test))
        mae = mean_absolute_error(y_test, reg.predict(X_test))
        r2 = r2_score(y_test, reg.predict(X_test))
        adjr2 = r2_score(y_test, reg.predict(X_test),multioutput='variance_weighted')
        
        print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
        print("The mean absolute error (mae) on test set: {:.4f}".format(mae))
        print("The r2 on test set: {:.4f}".format(r2))
        print("The adjr2 on test set: {:.4f}".format(adjr2))
        
        feature_importance = reg.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, np.array(X_test.columns)[sorted_idx])
        plt.title('Feature Importance (MDI)')

        result = permutation_importance(reg, X_test, y_test, n_repeats=10,
                                        random_state=42, n_jobs=2)
        sorted_idx = result.importances_mean.argsort()
        plt.subplot(1, 2, 2)
        plt.boxplot(result.importances[sorted_idx].T,
                    vert=False, labels=np.array(X_test.columns)[sorted_idx])
        plt.title("Permutation Importance (test set)")
        fig.tight_layout()
        plt.show()
        

   
        
        
        
        
        
        
        
        