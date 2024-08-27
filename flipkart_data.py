import requests
import urllib
from bs4 import BeautifulSoup
import pandas as pd

mobiles = {
    'name' : [],
    'specs' : [],
    'price' : []
}
page = 0
while(True):
    i = page
    if page==0:
        url = 'https://www.flipkart.com/search?q=mobiles&as=on&as-show=on&otracker=AS_Query_HistoryAutoSuggest_4_0_na_na_na&otracker1=AS_Query_HistoryAutoSuggest_4_0_na_na_na&as-pos=4&as-type=HISTORY&suggestionId=mobiles&requestId=f0208fd3-9995-40ba-9cca-5e8547aeb31f&p%5B%5D=facets.price_range.from%3D10000&p%5B%5D=facets.price_range.to%3DMax&p%5B%5D=facets.brand%255B%255D%3DSAMSUNG&p%5B%5D=facets.brand%255B%255D%3Drealme&p%5B%5D=facets.brand%255B%255D%3DPOCO&p%5B%5D=facets.brand%255B%255D%3DOPPO&p%5B%5D=facets.brand%255B%255D%3Dvivo&p%5B%5D=facets.brand%255B%255D%3DREDMI&p%5B%5D=facets.brand%255B%255D%3DNothing&p%5B%5D=facets.brand%255B%255D%3DMOTOROLA&p%5B%5D=facets.brand%255B%255D%3DMi&p%5B%5D=facets.brand%255B%255D%3DIQOO&p%5B%5D=facets.brand%255B%255D%3DGoogle&p%5B%5D=facets.brand%255B%255D%3DInfinix'
        page += 1

    else:
        print(i)
        url = 'https://www.flipkart.com/search?q=mobiles&as=on&as-show=on&otracker=AS_Query_HistoryAutoSuggest_4_0_na_na_na&otracker1=AS_Query_HistoryAutoSuggest_4_0_na_na_na&as-pos=4&as-type=HISTORY&suggestionId=mobiles&requestId=f0208fd3-9995-40ba-9cca-5e8547aeb31f&p%5B%5D=facets.price_range.from%3D10000&p%5B%5D=facets.price_range.to%3DMax&p%5B%5D=facets.brand%255B%255D%3DSAMSUNG&p%5B%5D=facets.brand%255B%255D%3Drealme&p%5B%5D=facets.brand%255B%255D%3DPOCO&p%5B%5D=facets.brand%255B%255D%3DOPPO&p%5B%5D=facets.brand%255B%255D%3Dvivo&p%5B%5D=facets.brand%255B%255D%3DREDMI&p%5B%5D=facets.brand%255B%255D%3DNothing&p%5B%5D=facets.brand%255B%255D%3DMOTOROLA&p%5B%5D=facets.brand%255B%255D%3DMi&p%5B%5D=facets.brand%255B%255D%3DIQOO&p%5B%5D=facets.brand%255B%255D%3DGoogle&p%5B%5D=facets.brand%255B%255D%3DInfinix&page='+str(i)
        page += 1

    source = requests.get(url).text
    soup = BeautifulSoup(source, 'html.parser')
    div = soup.find_all('div', class_='_3pLy-c row')
    if not len(div):
        break
    else:
        for j in div:
            mobiles['name'].append(j.find('div', class_="_4rR01T").text)
            specs = ''
            for i in j.find_all('li', class_='rgWa7D'):
                specs+= i.text+'#'
            mobiles['specs'].append(specs)
            try:
                mobiles['price'].append(j.find('div', class_='_30jeq3 _1_WHN1').text)
            except:
                mobiles['price'].append('na')

print('scraping completed')
flip_data = pd.DataFrame(mobiles, columns=mobiles.keys())
flip_data.to_csv('flip_data.csv', index=False)
print('data saved')


# print(len(prod_name))

# source = requests.get('https://www.flipkart.com/search?q=mobiles&as=on&as-show=on&otracker=AS_Query_TrendingAutoSuggest_1_0_na_na_na&otracker1=AS_Query_TrendingAutoSuggest_1_0_na_na_na&as-pos=1&as-type=TRENDING&suggestionId=mobiles&requestId=f4da3840-b1e3-46b1-874c-f757964e0ae0&p%5B%5D=facets.price_range.from%3D10000&p%5B%5D=facets.price_range.to%3DMax&p%5B%5D=facets.brand%255B%255D%3DGoogle&p%5B%5D=facets.brand%255B%255D%3DNothing&p%5B%5D=facets.brand%255B%255D%3DMOTOROLA&p%5B%5D=facets.brand%255B%255D%3DMi&p%5B%5D=facets.brand%255B%255D%3DIQOO&p%5B%5D=facets.brand%255B%255D%3DTecno&p%5B%5D=facets.brand%255B%255D%3DSAMSUNG&p%5B%5D=facets.brand%255B%255D%3Drealme&p%5B%5D=facets.brand%255B%255D%3DPOCO&p%5B%5D=facets.brand%255B%255D%3DAPPLE&p%5B%5D=facets.brand%255B%255D%3DInfinix&p%5B%5D=facets.brand%255B%255D%3DOPPO&p%5B%5D=facets.brand%255B%255D%3Dvivo&p%5B%5D=facets.brand%255B%255D%3DREDMI').text
#
# soup = BeautifulSoup(source, 'html.parser')




# div = soup.find_all('div', class_='_3pLy-c row')
# for j in div:
#     mobiles['name'].append(j.find('div', class_="_4rR01T").text)
#     specs = []
#     for i in j.find_all('li', class_='rgWa7D'):
#         specs.append(i.text)
#     mobiles['ram and storage'].append(specs[0])
#     mobiles['display'].append(specs[1])
#     mobiles['camera'].append(specs[2])
#     mobiles['battery'].append(specs[3])
#     mobiles['processor'].append(specs[4])
#     mobiles['warranty'].append(specs[5])
#     mobiles['price'].append(j.find('div', class_='_30jeq3 _1_WHN1').text)
# print(len(mobiles['processor']))