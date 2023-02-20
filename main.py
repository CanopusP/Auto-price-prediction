import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

response = requests.get("https://www.autobonus.lt/avto/poisk/?search=1&cat=1&collapsrch=1&ussearch=&saveusersearch=1&ma=-1&mo=-1&bo=-1&fu=-1&ge=-1&p1=&p2=&y1=-1&y2=-1&doSearch=1")
page = BeautifulSoup(response.text, 'html.parser') 
count = (page.find("span", class_="result-count").text).replace("(","").replace(")","")

sample = int(input("Введите количество автомобилей в выборке в формате цифр кратных 20 (0,20,40 и т.д.): "))
dn = pd.DataFrame()
for i in range(0,sample, 20):   # int(count)+20
    resp = requests.get("https://www.autobonus.lt/avto/poisk/?cat=1&search=1&doSearch=1&collapsrch=1&cnt=561&ord=date&asc=desc&curr={}".format(i))
    soup = BeautifulSoup(resp.content, 'lxml')
    links = []
    c = 0
    for link in soup.find_all('a'):
            a = link.get('href')
            links.append(a)

    for k in set(links):
            if "https://www.autobonus.lt/avto/objavlenie" in str(k):
                car = requests.get(k)
                page = BeautifulSoup(car.text, 'html.parser')  
                content = page.find_all('div', class_='content-left')
                params = (content)[0].find_all('div', class_='param')
                parnumber = len(params)
                rows_dict = {}
                for j in range(0, parnumber -1): 
                    col1 = ((params)[j].find_all('div', class_='left'))[0].text
                    col2 = ((params)[j].find_all('div', class_='right'))[0].text
                    if "Цена" not in col1:
                        rows_dict[col1] = col2
                    else:
                        col1 = ((params)[j].find_all('div', class_='price-title'))[0].text
                        col2 = ((params)[j].find_all('div', class_='price'))[0].text 
                        rows_dict[col1] = col2

                s1 = pd.Series(rows_dict,index=rows_dict.keys(), name='car')
                dn = pd.concat([dn, pd.DataFrame(s1)], axis=1)
                c += 1

dataset = dn.transpose().reset_index()
dataset = dataset.drop(["index", "Категория объявления", "ID объявления"], axis = 1)

def cost(a):
    if a != "Договорная":
        a = a.replace("€","").replace(" ","")
        return int(a)
    else:
        return 0


def engine(column):
    if "cм³" in column:
        lst = column.split("cм³")
        s = lst[0].replace(" ","")
        return int(s)
    
def power(column):
    if "cм³" in column:
        lst = column.split("cм³")
        if "Л.С." in lst[1] and "kW" in lst[1]:
            lst2 = lst[1].split("Л.С.")
            s = lst2[0].replace(" ","").replace(",","")
            return int(s)
    elif "Л.С."in column and "kW" in column:
        lst = column.split("Л.С.")
        s = lst[0].replace(" ","")
        return int(s)

def tech(s):
    if s is np.nan:
        return 0
    else: 
        return 1

def km(a):
    if a == "nan" or a is np.nan:
        return 
    else:
        a = a.replace("км","").replace(" ","")
        return int(a)
    
def CO(a):
    if a == "nan" or a is np.nan:
        return
    else:
        a = a.replace("г/км","").replace(" ","").replace("~","")
        return int(a)
    
dataset["Цена"] = dataset["Цена"].apply(cost)

dataset["Двигатель"] = dataset["Двигатель"].astype(str)

dataset["Кубы"] = dataset["Двигатель"].apply(engine)
dataset["Кубы"] = dataset["Кубы"].fillna(dataset["Кубы"].mean())

dataset["Мощность"] = dataset["Двигатель"].apply(power)
dataset["Мощность"] = dataset["Мощность"].fillna(dataset["Мощность"].mean())

dataset = dataset.drop(["Двигатель", "Код декларации владельца", "VIN-номер"], axis = 1)   

dataset["Дата выпуска"] = dataset["Дата выпуска"].str[:4]
dataset["Дата выпуска"] = dataset["Дата выпуска"].astype(int)

dataset["Тип кузова"] = dataset["Тип кузова"].fillna("Другой")

dataset["Кол-во дверей"] = dataset["Кол-во дверей"].apply(lambda x: 4 if x == "4/5"  else 2)
dataset["Кол-во дверей"].value_counts()

dataset["Цвет"] = dataset["Цвет"].fillna("Другой")

dataset["Тип трансмиссии"] = dataset["Тип трансмиссии"].fillna("Другой")
dataset["Тип трансмиссии"].value_counts()

dataset["Руль"] = dataset["Руль"].apply(lambda x: 1 if x == "Правый" else 0)

dataset["Тех. осмотр действ. до"] = dataset["Тех. осмотр действ. до"].str[:4]
dataset["Тех. осмотр действ. до"] = dataset["Тех. осмотр действ. до"].fillna(0)
dataset["Тех. осмотр действ. до"] = dataset["Тех. осмотр действ. до"].astype(int)
dataset["Техническое состояние"] = dataset["Техническое состояние"].apply(tech)

dataset["Пробег"] = dataset["Пробег"].astype(str)
dataset["Пробег"] = dataset["Пробег"].apply(km)
dataset["Пробег"] = dataset["Пробег"].fillna(dataset["Пробег"].mean())

dataset["Выброс CO₂"] = dataset["Выброс CO₂"].astype(str)
dataset["Выброс CO₂"] = dataset["Выброс CO₂"].apply(CO)
dataset["Выброс CO₂"] = dataset["Выброс CO₂"].fillna(dataset["Выброс CO₂"].mean())

dataset["Обмен"] = dataset["Обмен"].apply(tech)

dataset["Лизинг"] = dataset["Лизинг"].apply(tech)

x = dataset[["Мощность", "Дата выпуска","Пробег","Кубы"]]
y = dataset["Цена"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lnr = LinearRegression()
lnr.fit(X_train, y_train)

Y_predicted = lnr.predict(X_test)

print("MSE LinearRegression: ", mean_squared_error(Y_predicted, y_test))
print("MAE LinearRegression: ", mean_absolute_error(Y_predicted, y_test))
print("R2 LinearRegression: ", r2_score(Y_predicted, y_test))
print("MAPE LinearRegression: ", mean_absolute_percentage_error(Y_predicted, y_test))

rfr = RandomForestRegressor(max_depth = 9)
rfr.fit(X_train, y_train)

Y_predicted = rfr.predict(X_test)

print("MSE RandomForestRegressor: ", mean_squared_error(Y_predicted, y_test))
print("MAE RandomForestRegressor: ", mean_absolute_error(Y_predicted, y_test))
print("R2 RandomForestRegressor: ", r2_score(Y_predicted, y_test))
print("MAPE RandomForestRegressor: ", mean_absolute_percentage_error(Y_predicted, y_test))
