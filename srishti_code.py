import json
from pymongo import MongoClient
import pandas
import pandas as pd
import numpy as np
import time
import datetime
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import n_colors
from IPython.display import Image
from plotly.offline import init_notebook_mode, iplot, plot
import pyodbc


class srishti_code():
    def __init__(self, client, db):
        self._client = client  # MongoClient(host="localhost", port=27017)
        self._db = db  #
        self._mydb = self._client[self._db]
        self._mongo_file = []
        self._result = []
        self._get = []
        self._df_list = []
        self._date_dict = {}
        self._month = ''
        self._mongo_df = pd.DataFrame()
        self._collections = self._mydb['covidVaccine']
        self._grouped_df=pd.DataFrame()
        self._sqlserver = 'tcp:dapgroup12.database.windows.net' 
        self._sqldb = 'dap' 
        self._username = 'dap' 
        self._password = 'Admin@123'
        self._con =None
        self._myresult=None
        self._q_data=None
        self._vdf=pd.DataFrame()

    def insert_into_mongo(self, data):
        self._result = self._collections.insert_many(data)
        print("inserted data into MongoDb")
        print("-" * 20)
        return self._result

    def get_data_mongo(self):
        self._get = self._collections.find()
        return self._get

    def scrape_data(self):
        self._df_list=[]
        # self._df_list.append(df)
        return pd.concat(self._df_list)
    
    def date_cleaner(self,date_string):
        date_string = pd.to_datetime(date_string,format='%Y-%b-%d')
        return date_string
    
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


    def main_1(self):
        self._client = MongoClient(host="localhost", port=27017)
        self._db = "dap_project"
        self._mydb = self._client[self._db]
        self._mongo_file = self.get_data_mongo()
        print("-" * 20)
        print("Retrieved data from MongoDb")
        print("-" * 20)
        self._df_list = [n for n in self._mongo_file]
        self._mongo_df = pd.DataFrame().from_dict(self._df_list)
        ## DATA PRE PROCESSING AND CLEANING PART ##
        print("Date Cleaning Started....")
        print(self._mongo_df.columns)
        self._mongo_df = self._mongo_df[self._mongo_df.date.isnull() == False]
        # self._mongo_df["date"] = self._mongo_df["date"].apply(lambda x: self.date_cleaner(x))
        print("Date Cleaning done!")
        # Renaming location with country for proper name
        self._mongo_df.rename(columns={'location': 'country'}, inplace=True)
        print("Replaced 'location' column to 'country'")
        # Replacing Null and NAN values with 0
        self._mongo_df = self._mongo_df.replace([np.inf, -np.inf], np.nan)
        self._mongo_df = self._mongo_df.fillna(0)
        print("Null/NaN values replaced from 0")
        # let's drop new_vaccinations_smoothed and new_vaccinations_smoothed_per_million
        self._mongo_df.drop(['new_vaccinations_smoothed', 'new_vaccinations_smoothed_per_million'], axis=1, inplace=True)
        print("Dropped 'new_vaccinations_smoothed' and 'new_vaccinations_smoothed_per_million'")
        # Removing World data from the country column
        self._mongo_df = self._mongo_df[self._mongo_df.country != "World"]
        print("Removed World data from country column as it is irrelavant")
        # Replacing country "Cote d'Ivoire" with "Ivory Coast, United States with US and United Kingdom with UK"
        self._mongo_df.country = self._mongo_df.country.replace().replace({"Cote d'Ivoire": "Ivory Coast"})
        self._mongo_df.country = self._mongo_df.country.replace().replace({"United States": "US"})
        self._mongo_df.country = self._mongo_df.country.replace().replace({"United Kingdom": "UK"})
        # Making all countries lower case and replacing '' with - to match with other datasets
        self._mongo_df.country=self._mongo_df.country.str.lower()
        self._mongo_df.country=self._mongo_df.country.str.replace(' ','-')
        print("Replacing country name with proper format")
        

        
    def main_2(self):
        # MySql PART #
        self._con = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+self._sqlserver+';DATABASE='+self._sqldb+';UID='+self._username+';PWD='+self._password) 
        
        drop_table = ("""   
                                        drop table if exists vaccineData;
                                        
                                    
                                    """)
        self.crud(self._con,drop_table)
        sql_create_covid_table = ("""   
                                        CREATE TABLE vaccineData (
                                        total_vaccinations DECIMAL (38,4) NOT NULL ,
                                        people_vaccinated DECIMAL (38,4) NOT NULL ,
                                        people_fully_vaccinated DECIMAL (38,4) NOT NULL ,
                                        new_vaccinations DECIMAL (38,4) NOT NULL ,
                                        total_vaccinations_per_hundred DECIMAL (38,4) NOT NULL ,
                                        people_vaccinated_per_hundred DECIMAL (38,4) NOT NULL ,
                                        people_fully_vaccinated_per_hundred DECIMAL (38,4) NOT NULL ,
                                        population DECIMAL (38,4) NOT NULL ,
                                        date DATE NOT NULL,
                                        country varchar(50) NOT NULL
                                        
                                    );
                                    """)#PRIMARY KEY (country,date)
        
        self.crud(self._con,sql_create_covid_table)
        
        sql_insert_values = ''' INSERT INTO vaccineData(total_vaccinations,people_vaccinated,people_fully_vaccinated,new_vaccinations,total_vaccinations_per_hundred,people_vaccinated_per_hundred,people_fully_vaccinated_per_hundred,population,date,country)
                              VALUES({},{},{},{},{},{},{},{},'{}','{}'); '''
        self._mongo_df=self._mongo_df[["total_vaccinations","people_vaccinated","people_fully_vaccinated","new_vaccinations","total_vaccinations_per_hundred","people_vaccinated_per_hundred","people_fully_vaccinated_per_hundred","population","date","country"]]                      
        self._mongo_df = self._mongo_df.replace([np.inf, -np.inf], np.nan)
        self._mongo_df = self._mongo_df.fillna(0)
        print(self._mongo_df.isnull().sum())

        d=self._mongo_df.to_dict(orient="records")
        for data in d:
            self.insert_into_table(self._con,sql_insert_values,tuple(data.values()))
        
        # correlation of plot
        plt.subplots(figsize=(8, 8))
        sns.heatmap(self._mongo_df.corr() , annot=True, square=True )
        plt.show()
        plt.savefig('correlation.png')
        
        ## VISUALISATION PART ##

    # Countries with maximum vaccination Average rate
        query = """ select country ,avg(people_vaccinated) as total_vaccinations
                    from vaccineData 
                    group by country
                    order by total_vaccinations desc ; """
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country', 'total_vaccinations'])
        
        
        h=sns.barplot(x='country',y='total_vaccinations',data=self._vdf[0:5])
        h.set_xticklabels(rotation=90,labels = list(self._vdf.country[0:5]))
        plt.title("Top 5 countries with Highest Vaccination average rate")
        h.legend()
        plt.show()
        plt.savefig('Top 5 countries with Highest Vaccination average rate.png')
        
    #   What country is vaccinated more people?   
        query = """ select country ,max(people_fully_vaccinated) as fully_vaccinated
                    from vaccineData 
                    group by country
                    order by fully_vaccinated desc ; """
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country', 'fully_vaccinated'])
        
        h=sns.barplot(x='country',y='fully_vaccinated',data=self._vdf[0:20])
        h.set_xticklabels(rotation=90,labels = list(self._vdf.country[0:20]))
        plt.title("Top 20 countries with full vaccination")
        h.legend()
        plt.show()
        plt.savefig('Top 20 countries with full vaccination.png')

        
        
    # Analyzing which country has the largest amount of vaccinations
        query = """ select country ,max(total_vaccinations) as Total_vaccinations
                    from vaccineData 
                    group by country
                    order by Total_vaccinations desc ; """
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country', 'Total_vaccinations'])
        
        h=sns.barplot(x='country',y='Total_vaccinations',data=self._vdf[0:15])
        h.set_xticklabels(rotation=90,labels = list(self._vdf.country[0:15]))
        plt.title("Top 15 countries with largest amount of vaccinations")
        h.legend()
        plt.show()
        plt.savefig('Top 15 countries with largest amount of vaccinations.png')
        
    # Max people Vaccinated for top 50 countries 
        query = """ select country ,max(people_vaccinated) as people__vaccinated
                    from vaccineData 
                    group by country
                    order by people__vaccinated desc ; """
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country', 'people__vaccinated'])
        
        h=sns.barplot(x='country',y='people__vaccinated',data=self._vdf[0:50])
        h.set_xticklabels(rotation=90,labels = list(self._vdf.country[0:50]))
        plt.title("Max people Vaccinated for top 50 countries")
        h.legend()
        plt.show()
        plt.savefig('Max people Vaccinated for top 50 countries.png')
        
    
    # People Vaccinated per hundred for top 50 countries
        query = """ select country ,max(people_vaccinated_per_hundred) as people_vaccinated__per_hundred
                    from vaccineData 
                    group by country
                    order by people_vaccinated__per_hundred desc ; """
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country', 'people_vaccinated__per_hundred'])
        
        h=sns.barplot(x='country',y='people_vaccinated__per_hundred',data=self._vdf[0:50])
        h.set_xticklabels(rotation=90,labels = list(self._vdf.country[0:50]))
        plt.title("People Vaccinated per hundred for top 50 countries")
        h.legend()
        plt.show()
        plt.savefig('People Vaccinated per hundred for top 50 countries.png')
        
        
    # people_fully_vaccinated_per_hundred
        query = """ select country ,max(people_fully_vaccinated_per_hundred) as people_fully__vaccinated_per_hundred
                    from vaccineData 
                    group by country
                    order by people_fully__vaccinated_per_hundred desc ; """
        
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country', 'people_fully__vaccinated_per_hundred'])
        
        h=sns.barplot(x='country',y='people_fully__vaccinated_per_hundred',data=self._vdf[0:50])
        h.set_xticklabels(rotation=90,labels = list(self._vdf.country[0:50]))
        plt.title("People fully vaccinated per hundred")
        h.legend()
        plt.show()
        plt.savefig('People fully vaccinated per hundred.png')
        
        
    # Analysis on data for India
    # Select data entries for India 
        # query = """ select country, date, sum(total_vaccinations) as total_Vaccinations
        #             from vaccineData v
        #             where v.country = 'India'
        #             group by country, date
        #             order by total_Vaccinations;"""
                    
        # self._q_data=self.get_data_mysql(self._con,query)
        
        # self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country','date', 'total_Vaccinations'])
        
        # h=sns.lineplot(x='date',y='total_Vaccinations',hue='country',data=self._vdf)
        # h.set_xticklabels(rotation=90,labels = list(self._vdf.date))
        # plt.title("Total vaccinations date-wise trend in India")
        # h.legend()
        # plt.show()
        
        # plt.savefig('Total vaccinations date-wise trend in India.png')
        
        # query = """ select country, date, sum(people_fully_vaccinated) as people_fully_vaccinated
        #             from vaccineData v
        #             where v.country = 'India'
        #             group by country, date
        #             order by people_fully_vaccinated;"""
                    
        # self._q_data=self.get_data_mysql(self._con,query)
        
        # self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country','date', 'people_fully_vaccinated'])
        
        # h=sns.lineplot(x='date',y='people_fully_vaccinated',hue='country',data=self._vdf)
        # h.set_xticklabels(rotation=90,labels = list(self._vdf.date))
        # plt.title("People fully vaccinated - date wise trend in India")
        # h.legend()
        # plt.show()
        # plt.savefig('People fully vaccinated - date wise trend in India.png')
        
        # query = """select country, date, sum(people_fully_vaccinated) as people_fully_vaccinated, sum(total_vaccinations) as total_vaccinations, sum(people_vaccinated) as people_vaccinated
        #             from vaccineData v
        #             where v.country = 'India'
        #             group by country, date
        #             order by people_fully_vaccinated,total_Vaccinations,people_vaccinated ; """
        
        # self._q_data=self.get_data_mysql(self._con,query)
        
        # self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country','date', 'people_fully_vaccinated', 'total_vaccinations','people_vaccinated'])
        
        # X = self._vdf["date"]
        # Y1 =self._vdf["people_fully_vaccinated"]
        # Y2 =self._vdf["total_vaccinations"]
        # Y3 =self._vdf["people_vaccinated"]

        # plt.figure(figsize=(8,5))
        # plt.plot(X,Y1,linewidth=4,label="people fully vaccinated")
        # plt.plot(X,Y2,linewidth=3,label="total vaccinations")
        # plt.plot(X,Y3,linewidth=3,label="people vaccinated")
        # plt.xticks(rotation=90)
        # plt.legend()
        # plt.show()  
        # plt.savefig('People fully vaccinated,total vaccinations,people vaccinated - date wise trend .png')
  
        
        # Total Vaccination % of Population by Country
        query = """ select country, sum(total_vaccinations) as total_vaccinations
                    from vaccineData
                    group by country
                    order by total_vaccinations ; """
                    
        self._q_data=self.get_data_mysql(self._con,query)
        
        self._vdf = pd.DataFrame.from_records(self._q_data, columns =['country','total_vaccinations'])
        
        trace = go.Choropleth(
            locations = self._vdf['country'],
            locationmode='country names',
            z = self._vdf['total_vaccinations'],
            text = self._vdf['country'],
            autocolorscale =False,
            reversescale = True,
            colorscale = 'viridis',
            marker = dict(
                line = dict(
                    color = 'rgb(0,0,0)',
                    width = 0.5)
            ),
            colorbar = dict(
                title = 'Total vaccinations',
                tickprefix = '')
        )

        data = [trace]
        layout = go.Layout(
            title = 'Total vaccinations per country',
            geo = dict(
            showframe = True,
            showlakes = False,
            showcoastlines = True,
            projection = dict(
                type = 'natural earth'
            )
        )
    )
        fig = dict( data=data, layout=layout )
        iplot(fig)
        
    



