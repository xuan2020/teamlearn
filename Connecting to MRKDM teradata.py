# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:21:51 2019

@author: ZhoJi001
"""

#Alex Levin
import pyodbc
import teradata
udaExec = teradata.UdaExec (appName="HelloWorld", version="1.0",
        logConsole=False)
session = udaExec.connect(method="odbc", system="td-mrk-p-",
        username="MKTG_PRDFOCOPLETL", password="staples123");
for row in session.execute("select sales_visit_id, sales_visit_date, primary_site_id, visit_type_code, gross_sales_amt from PRD_MKTG_BIV.SALES_VISIT_SUMMARY sample 10"):
    print(row)
    
with udaExec.connect(method="odbc", system="td-mrk-p-",
        username="MKTG_PRDFOCOPLETL", password="staples123") as session: 
    for row in session.execute("select sales_visit_id, sales_visit_date, primary_site_id, visit_type_code, gross_sales_amt from PRD_MKTG_BIV.SALES_VISIT_SUMMARY sample 10"):
        print(row)
    
  session.execute(file="myqueries.sql")  
    
pyodbc.pooling = True

conn = pyodbc.connect('DSN=TDMKTG_PRF-PRD;uid=MKTG_PRDFOCOPLETL;pwd=staples123;',autocommit=True)
qry="select sales_visit_id, sales_visit_date, primary_site_id, visit_type_code, gross_sales_amt from PRD_MKTG_BIV.SALES_VISIT_SUMMARY sample 10;"
rows = conn.cursor()
qb1=rows.execute(qry)

for row in qb1:
    print(row)
    
