# -*- coding: utf-8 -*-
"""
Created on Sep 24

@author: cheryl

Programming Project: Recommendation
Examine item co-purchasing.

Assignment: working with data on purchases from retail stores. 
The task will be to create recommendations for someone who has just bought a product, 
based on which items are often bought together with the product.
"""
import os
import numpy as np
import pandas as pd
from astroid.__pkginfo__ import description



'''
Function fillPeopleProducts() with one parameter of type DataFrame, 
storing the purchasing data. The function should create and return a new data frame, 
which will summarize which products were bought by which customer. 
The new data frame must be indexed by the customer ids 
(presented in USER_ID column of the purchasing data) with column titles corresponding to product ids 
(PRODUCT_ID column of the purchasing data). 
'''
def fillPeopleProducts (purchaseDF):
    userProductArray = purchaseDF.iloc[:,:2].values
    userProductDict = {}
    for item in userProductArray:
        userProduct = userProductDict.get(item[0],[])
        userProduct.append(item[1])
        userProductDict[item[0]] = userProduct
    #print(userProductDict)
    
    userList = list(set(purchaseDF['USER_ID']))
    userList.sort()
    productList = list(set(purchaseDF['PRODUCT_ID']))
    productList.sort()
    outList = []
    innerList = []
    for user in userList:
        innerList = []
        for product in productList:
            if product in userProductDict.get(user):
                innerList.append(1)
            else:
                innerList.append(0)
        outList.append(innerList)
    frame = pd.DataFrame(outList, columns = productList,index = userList)    
    #pd.set_option('display.max_columns', 1000) 
    #pd.set_option('display.max_rows', 1000) 
    #pd.set_option('display.width', 1000)
    return frame 

'''
Function fillProductCoPurchase () with one parameter of type DataFrame, storing the purchasing data. 
The function should create a data frame representing the co-purchasing matrix. To do it, 
it must first call function fillPeopleProducts() to obtain the data frame with the summary of purchases
by customer, let’s call this data frame peopleProducts. Recall, that for each row, 
representing a customer and column representing a product, peopleProducts[i,j] stores 1 
if customer i bought product j, and 0 otherwise.
'''
def fillProductCoPurchase (purchaseDF):
    peopleProducts = fillPeopleProducts(purchaseDF)
    productList = list(set(purchaseDF['PRODUCT_ID']))
    productList.sort()
    #allZero = np.zeros((len(productList),len(productList)), dtype = int)
    copurchaseDF = pd.DataFrame(columns = productList, index = productList)
    pd.set_option('display.max_columns', 1000) 
    pd.set_option('display.max_rows', 1000) 
    pd.set_option('display.width', 1000)
    #print(copurchaseDF)

    for product1 in productList:
        for product2 in productList:
            if product1 != product2:
                vector1 = peopleProducts.loc[:,product1].values
                vector2 = peopleProducts.loc[:,product2].values
                copurchase = np.dot(vector1,vector2)
                copurchaseDF[product1][product2] = copurchase
            else:
                copurchaseDF[product1][product2] = 0 
    
    return (copurchaseDF,peopleProducts)

'''
Function findMostBought(), which will be passed the peopleProducts data frame as a parameter, 
and must return a list of items that have been purchased by more customers than any other item.
'''
def findMostBought(peopleProducts):
    '''
    productList = []
    boughtMost = 0
    productDict = {}
    for product in peopleProducts.columns:
        numberOfBought = peopleProducts[product].sum()
        products = productDict.get(numberOfBought,[])
        products.append(product)
        productDict[numberOfBought] = products
        if numberOfBought > boughtMost:
            boughtMost = numberOfBought
    '''        
    
    mostBoughtProducts = []
    maxNumberOfBought = peopleProducts.sum().max()
    for product in peopleProducts.columns:
        if peopleProducts[product].sum() == maxNumberOfBought:
            mostBoughtProducts.append(product)
     
    #print('mostBought:', mostBoughtProducts)   
    return mostBoughtProducts
#     return productDict.get(boughtMost)

'''
Function reformatProdData(), which will be passed the product data frame as a parameter. 
The product data contains a combination of the product name and category in the DESCRIPTION column. 
This function must separate the name from the category, leaving only the name in the DESCRIPTION column
 and creating a new CATEGORY column, with the category name. For example, 
 from the original product DESCRIPTION value Colorwave Rice Bowl (Dinnerware), 
 Colorwave Rice Bowl should be stored under DESCRIPTION and Dinnerware under CATEGORY.
'''
def reformatProdData(productDF):
    description = productDF.DESCRIPTION.str.split("(").str.get(0)
    category1 = productDF.DESCRIPTION.str.split("(").str.get(1)
    category = category1.str.split(")").str.get(0)
    productDF.DESCRIPTION = description
    productDF['CATEGORY'] = category
   # print(productDF)
    return 

'''the list of recommended product ids'''
def recommendedProductID (boughtProduct, copurchaseDF):
    
    mostBoughtProducts = []
    maxNumberOfBought = copurchaseDF[boughtProduct].max()
    
    # can find recommended products withut looping, using pandd=as functions
    
    for product in copurchaseDF.columns:
        if copurchaseDF[boughtProduct][product] == maxNumberOfBought:
            mostBoughtProducts.append(product)
#    print('mostBought:', mostBoughtProducts)

    return mostBoughtProducts, maxNumberOfBought


'''
Function printRecProducts(), which will be passed the product data frame and the list of 
recommended product ids. The function must produce a printout of all recommendations passed 
in via the second parameter, in alphabetical order by the category. 
Make sure to examine the sample interactions and implement the formatting requirements. 
The function should return no value.
'''

def printRecProducts(productDF, recommendProductsLists):
    #productDF = productDF.sort_values(by=['CATEGORY'])
    #print(productDF)
    #print(recommendProductsLists)   
    selected=productDF[ productDF['PRODUCT_ID'].isin (recommendProductsLists)  ]
    selected = selected[['CATEGORY', 'DESCRIPTION', 'PRICE']]
    selected.sort_values(by=['CATEGORY'], inplace=True)
    rowNum, columnNum = selected.values.shape
    selected.index = range(1, rowNum+1)
#    print(selected)

    currentCategory = ''
    printCategory = ''
    for i in range(1, rowNum+1):
        if printCategory != selected['CATEGORY'][i]:
            currentCategory = selected['CATEGORY'][i]
            printCategory = currentCategory
        else:
            currentCategory = ''
        maxLehgth = selected['CATEGORY'].str.len().max()    
        if pd.isna(selected['PRICE'][i]):
            
            print('IN', currentCategory.upper().ljust(maxLehgth), '--', selected['DESCRIPTION'][i])
        else:
            print('IN', currentCategory.upper().ljust(maxLehgth), '--', selected['DESCRIPTION'][i]+', $'+format(selected['PRICE'][i],'.2f'))
  
    return

'''Function main(), which will be called to start the program and works according to design.'''

def main ():
    cwd = os.getcwd()    
    folder = input("Please enter name of folder with product and purchase data files: (prod.csv and purchases.csv):")
 #   folder = 'pdata-tiny'
    folderpath = os.path.join(cwd,folder) 
    productPath = os.path.join(folderpath, 'prod.csv')
    productCSV = pd.read_csv(productPath)
    productDF = pd.DataFrame(productCSV)
    purchasePath = os.path.join(folderpath, 'purchases.csv')
    purchaseCSV = pd.read_csv(purchasePath)
    purchaseDF = pd.DataFrame(purchaseCSV)
    
    print('\nPreparing the co-purchasing matrix...\n')
    
    (copurchaseDF,peopleProducts) = fillProductCoPurchase (purchaseDF)
    findMostBought(peopleProducts)
    reformatProdData(productDF)
    #print(productDF)
        

    boolean = True
    while boolean:
        boughtProduct = input('Which product was bought? Enter product id or press enter to quit.')
        if boughtProduct == "":
            boolean = False
        else:
            mostBoughtProducts = findMostBought(peopleProducts)
            recommendProducts, maxNumberOfBought = recommendedProductID (boughtProduct, copurchaseDF)
            print('[Maximum co-purchasing score', maxNumberOfBought, ']')
            if maxNumberOfBought ==0 :
                print("Recommend with", boughtProduct.upper(), ":", mostBoughtProducts)
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('Suggest one of our most popular products:')
                printRecProducts(productDF, mostBoughtProducts)
            else:
                print("Recommend with", boughtProduct.upper(), ":", recommendProducts)
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')    
                print('People who bought it were most likely to buy:')
                printRecProducts(productDF, recommendProducts)
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~') 
        
    
    pd.set_option('display.max_columns', 1000) 
    pd.set_option('display.max_rows', 1000) 
    pd.set_option('display.width', 1000)
main()    

