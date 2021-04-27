#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:20:45 2021

@author: venugopal
"""


import pandas as pd
import pandas as  pd
import pymongo
from pymongo import MongoClient
import deeksha_code
import vishnu_code
import srishti_code



if __name__=="__main__":
    
    df = pd.read_json('https://covid.ourworldindata.org/data/internal/megafile--vaccinations.json')
    client=MongoClient("mongodb+srv://dap:dap@dap.xbtpz.mongodb.net/Mydatabase?retryWrites=true&w=majority")
    db="dap_project"
    
    print("******************Executing Deekshas Code******************")
    # c1=deeksha_code.deeksha_code(client,db)
    # df1=pd.read_csv("data.csv")
    # df2=pd.read_excel("testing_eu.xlsx")
    # c1.main_1(df1,df2)
    # c1.main_2()
    
    
    print("******************Executing Srishtis Code******************")
    # s1 = srishti_code.srishti_code(client, db)
    # df = df.to_dict(orient="records")
    # s1.insert_into_mongo(df)
    # df = s1.get_data_mongo()
    # s1.main_1()
    # s1.main_2()
    # s1._df_list = [n for n in df]
    # s1._mongo_df = pd.DataFrame().from_dict(s1._df_list)
    print("******************Executing Vishnus Code******************")
    p1=vishnu_code.vishnu_code(client,db) 
    p1.main_one()
    p1.main_two()
    print("*"*20+"STARTING COMBINED ANALYSIS"+"*"*20)
    p1.main_three()