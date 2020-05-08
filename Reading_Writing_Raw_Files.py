# -*- coding: utf-8 -*-
"""
@author: KalPr001
"""


import numpy as np
import pandas as pd

sample = pd.read_csv("C:\PKal\SAS_MIGRATIONS_TEST\R\TEST_FILE_12M.CSV",encoding = "ISO-8859-1")

#To get rid off UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb8 in position 0 used endoding=

sample.count

for col in sample.columns:
    print(col)

# reading selected coulms
sample = pd.read_csv("C:\PKal\SAS_MIGRATIONS_TEST\R\TEST_FILE_12M.CSV",
                     usecols=["EMAIL_ADDRESS", "BUSINESS_CONSUMER_CODE", "OCP_SEG"],
                     na_values=['.']
                     ,encoding = "ISO-8859-1")
sample.count
max(sample.EMAIL_ADDRESS.apply(len))

#skipping rows
sample = pd.read_csv(r"C:\PKal\SAS_MIGRATIONS_TEST\R\TEST_FILE_12M.CSV",
                     header=0,
                     skiprows=range(1,1000),
                     nrows=1000,
                     na_values=['.']
                     ,encoding = "ISO-8859-1")
sample.dtypes
len(sample.EMAIL_ADDRESS)
max(sample.EMAIL_ADDRESS.apply(len))
sample.EMAIL_ADDRESS.apply(len).max()



sample.count

#Reading other delimited files

sample = pd.read_csv(r'C:\PKal\SAS_MIGRATIONS_TEST\R\out_file.csv', sep='|', index_col=False,encoding = "ISO-8859-1") 

#dataframe to csv
sample.to_csv(r'C:\PKal\SAS_MIGRATIONS_TEST\R\out_file.csv',
               columns=["EMAIL_ADDRESS", "BUSINESS_CONSUMER_CODE", "OCP_SEG"])

sample.to_csv(r'C:\PKal\SAS_MIGRATIONS_TEST\R\out_file.csv',
              columns=["EMAIL_ADDRESS", "BUSINESS_CONSUMER_CODE", "OCP_SEG"],
              sep="|",
              index= False
              )

#Reading Excel sheets

sample = pd.read_excel(r'C:\PKal\CampaignAnalytics\AnalysisRequests\Email\CA2247\Paper_SKU.xlsx', sheet_name=0, index=False)
sample.count

#Writing to Excel
sample.to_excel(r'C:\PKal\SAS_MIGRATIONS_TEST\R\Out_Paper_SKU.xlsx', sheet_name='test1',index=False)


#Writing to multiple sheets
writer = pd.ExcelWriter('C:\PKal\SAS_MIGRATIONS_TEST\R\Out_Paper_SKU.xlsx', engine = 'xlsxwriter')
sample.to_excel(writer, sheet_name='x1',index=False)
sample.to_excel(writer,sheet_name='x2')
writer.save()
writer.close()

sample.dtypes

#testing
s1=pd.DataFrame(list(sample.columns))
s1.to_excel(r'C:\PKal\SAS_MIGRATIONS_TEST\R\test_excel_output.xlsx', index=False)


        








