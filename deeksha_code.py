# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 22:32:29 2021

@author: Lenovo
"""


import pandas as  pd
import pymongo
from pymongo import MongoClient
import seaborn as sns
import matplotlib.pyplot as plt
import pyodbc 
import numpy as np
from datetime import datetime

class deeksha_code():
    def __init__(self,client,db):
        self._client=client
        self._db=db
        self._mydb = self._client[self._db]
        self._mongo_df=pd.DataFrame()
        self._result=[]
        self._get=[]
        self._date_dict={}
        self._month=''
        self._df_list=[]
        self._collections=self._mydb['hospital_data']
        self._options=[]
        self._mongo_file=[]
        self._date_dict={"Jan":"01","Feb":"02" ,"Mar":"03" , "Apr":"04", "May":"05", "Jun":"06","Jul":"07" , "Aug":"08","Sep":"09" , "Oct":"10", "Nov":"11","Dec":"12"}  
        self._Country=''
        self._grouped_df=pd.DataFrame()
        self._sqlserver = 'tcp:dapgroup12.database.windows.net' 
        self._sqldb = 'dap' 
        self._username = 'dap' 
        self._password = 'Admin@123'
        self._con =None
        self._myresult=None
        self._q_data=None
        self._vdf=pd.DataFrame()
        self._df=pd.DataFrame()
        self._df1=pd.DataFrame()
        self._df2=pd.DataFrame()
        self._df3=pd.DataFrame()
    
        #inserting data into Mongo db
    def insert_into_mongo(self,data):        
        self._result = self._collections.insert_many(data)
        print("inserted data into mongo")
    
    def get_data_mongo(self):
        self._get=self._collections.find()
        return self._get
    
    def crud(self,conn, query):
        c = conn.cursor()
        print(query)
        c.execute(query)
        c.commit()
        
    def insert_into_table(self,con,query,data):
        cur = con.cursor()
        print(query.format(*data))
        cur.execute(query.format(*data))
        cur.commit()
        
    def get_data_mysql(self,con,query):
        cur = con.cursor()
        cur.execute(query)
        self._myresult=cur.fetchall()
        return self._myresult
         
    
     
    def main_1(self,df1,df2):  
        self._df=df1#
        df_hosp=self._df[self._df["indicator"]=="Daily hospital occupancy"]
        df_icu=self._df[self._df["indicator"]=="Daily ICU occupancy"]
        df_hosp["Daily_hospital_occupancy"]=df_hosp["value"]
        df_icu["Daily_ICU_occupancy"]=df_icu["value"]
        df_hosp.drop(["indicator","source","url","value"],axis=1,inplace=True)
        df_icu.drop(["indicator","source","url","value"],axis=1,inplace=True)
        self._df1=pd.merge(df_hosp,df_icu,left_on=["country","date"],right_on=["country","date"],how='inner')
        self._df2=df2
        self._df3=pd.merge(self._df1,self._df2,left_on=["country","year_week_x"],right_on=["country","year_week"],how='inner')
        self._df3.drop(["year_week_x","year_week_y","testing_data_source"],axis=1,inplace=True)
        df=self._df3.to_dict(orient='records')   
        self.insert_into_mongo(df)
        df=self.get_data_mongo()
        self._df_list=[n for n in df]
        self._client = MongoClient(host="localhost", port=27017)
        self._db = "dap_project"
        self._mydb = self._client[self._db]
        self._mongo_file = self.get_data_mongo()
        print("-" * 20)
        print("retrieved data from MongoDb")
        print("-" * 20)
        self._df_list = [n for n in self._mongo_file]
        self._mongo_df = pd.DataFrame().from_dict(self._df_list)
        print("Date Cleaning Started")
        self._mongo_df = self._mongo_df.replace([np.inf, -np.inf], np.nan)
        self._mongo_df = self._mongo_df.fillna(0)
        print("Null/NaN values replaced from 0")
        self._mongo_df.country = self._mongo_df.country.replace().replace({"Cote d'Ivoire": "Ivory Coast"}) 
        self._mongo_df.country = self._mongo_df.country.replace().replace({"United States": "US"})
        self._mongo_df.country = self._mongo_df.country.replace().replace({"United Kingdom": "UK"})
        self._mongo_df.country=self._mongo_df.country.str.lower()
        self._mongo_df.country=self._mongo_df.country.str.replace(' ','-')
        print("Replacing country name with proper format")
        print(self._mongo_df.columns)
        print("Date Cleaning done")
    
    def main_2(self):

        self._con = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+self._sqlserver+';DATABASE='+self._sqldb+';UID='+self._username+';PWD='+self._password) 
        
        drop_table = ("""   
                                        drop table if exists hospitalData ;
                                        
                                    
                                    """)
        self.crud(self._con,drop_table)
        sql_create_currency_table = ("""   
                                        CREATE TABLE hospitalData (
                                        Daily_ICU_occupancy DECIMAL (38,4) NOT NULL ,
                                        Daily_hospital_occupancy DECIMAL (38,4) NOT NULL ,
                                        country VARCHAR(50) NOT NULL ,
                                        
                                        date DATE NOT NULL,
                                        level VARCHAR(50) NOT NULL,
                                        
                                        
                                        new_cases DECIMAL (38,4) NOT NULL ,
                                        population DECIMAL (38,4) NOT NULL ,
                                        positivity_rate DECIMAL (38,4) NOT NULL ,
  
                                        
                                        region VARCHAR(50) NOT NULL,
                                        region_name VARCHAR (50) NOT NULL,
                                        
                                        testing_rate DECIMAL (38,4) NOT NULL ,
                                        tests_done DECIMAL (38,4) NOT NULL ,
                                        
                                        
                                        
                                    );
                                    """)
        print(self._mongo_df.columns)
        self._mongo_df.drop(["_id","year_week","country_code"],axis=1,inplace=True)                          
        self._mongo_df.date=self._mongo_df.date.apply(lambda x:datetime.strptime(x, '%d/%m/%y'))
        #self._mongo_df.date=self._mongo_df.date.apply(lambda x:datetime.strptime(x, '%d/%m/%Y'))
        self.crud(self._con,sql_create_currency_table)
               
        sql_insert_values = ''' INSERT INTO hospitalData(
                                Daily_ICU_occupancy,Daily_hospital_occupancy,country,
                                date,level,new_cases,population,
                                positivity_rate,region,region_name,
                                testing_rate,tests_done)
                                VALUES({},{},'{}','{}','{}',{},{},{},'{}','{}',{},{}); '''
        self._mongo_df.date=self._mongo_df.date.astype('str')
        print(self._mongo_df.isnull().sum())
        self._mongo_df=self._mongo_df[['Daily_ICU_occupancy', 'Daily_hospital_occupancy', 'country',
                                        'date', 'level', 'new_cases', 'population',
                                        'positivity_rate', 'region', 'region_name', 'testing_rate',
                                        'tests_done']]

        d=self._mongo_df.to_dict(orient="records")
        for data in d:
            self.insert_into_table(self._con,sql_insert_values,tuple(data.values())) 
            
        # Countries with maximum hospitalisation Average rate
        query = """ select country ,avg(Daily_hospital_occupancy) as Total_Hospitalisation
                    from hospitalData 
                    group by country
                    order by Total_Hospitalisation desc ; """
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country', 'Total_Hospitalisation'])
        
        
        h=sns.barplot(x='country',y='Total_Hospitalisation',data=self._vdf[0:5])
        h.set_xticklabels(rotation=90,labels = list(self._vdf.country[0:5]))
        plt.title("Top 5 countries with Highest Hospitalisation average rate")
        h.legend()
        plt.show()
        
        # Countries with maximum ICU hospitalisation Average rate
        query = """ select country ,avg(Daily_ICU_occupancy) as Total__ICU_Hospitalisation
                    from hospitalData 
                    group by country
                    order by Total__ICU_Hospitalisation desc ; """
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country', 'Total__ICU_Hospitalisation'])
        
        
        h=sns.barplot(x='country',y='Total__ICU_Hospitalisation',data=self._vdf[0:5])
        h.set_xticklabels(rotation=90,labels = list(self._vdf.country[0:5]))
        plt.title("Top 5 countries with Highest ICU Hospitalisation")
        h.legend()
        plt.show()
        
        # Top Countries with average corona cases per day
        query = """ select country,avg(new_cases) as average_New_Cases_per_day
                    from hospitalData
                    group by country
                    order by average_New_Cases_per_day;"""
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country', 'average_New_Cases_per_day'])
        
        
        h=sns.lineplot(x='country',y=self._vdf.average_New_Cases_per_day.astype('float'),data=self._vdf)# only 1 column is passed ie x or y
        h.set_xticklabels(rotation=90,labels = self._vdf.country)
        plt.title("Country Wise average New cases per day")
        h.legend()
        plt.show() 
        
        # Top Countries with average testing rate per day
        query = """ select country ,avg(testing_rate) as Avg_Testing_Rate
                    from hospitalData 
                    group by country
                    order by Avg_Testing_Rate desc ; """
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country', 'Avg_Testing_Rate'])
        
        
        h=sns.barplot(x='country',y='Avg_Testing_Rate',data=self._vdf[0:10])# only 1 column is passed ie x or y
        h.set_xticklabels(rotation=90,labels = list(self._vdf.country[0:10]))
        plt.title("Top 10 countries with highest Testing rate")
        h.legend()
        plt.show()
        
         # Top Countries with average Corona positive rate per day
        query = """ select country ,avg(positivity_rate) as Avg_Corona_Positive_Rate
                    from hospitalData 
                    group by country
                    order by Avg_Corona_Positive_Rate desc ; """
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country', 'Avg_Corona_Positive_Rate'])
        
        
        h=sns.barplot(x='country',y='Avg_Corona_Positive_Rate',data=self._vdf[0:10])# only 1 column is passed ie x or y
        h.set_xticklabels(rotation=90,labels = list(self._vdf.country[0:10]))
        plt.title("Top 10 countries with highest Corona Positive rate")
        h.legend()
        plt.show()
              