import requests
import csv
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from multiprocessing import Pool
import time
from selenium.webdriver.chrome.options import Options
import zipfile
from functools import partial

#JUST RETURNS THE SOUPIFIED CONTENT OF THAT URL BY USING LXML PARSER
def GET_SOUP(my_url):
	url = my_url
	response = requests.get(url)
	html = response.content
	soup = BeautifulSoup(html, features='lxml')
	return soup

#SCRAPE THE CONTENT OF THAT URL BY REPEATEDLY FINDING THE LOAD MORE BUTTON
def scrapeData(url):
    listOfComments=[]

    driver_options = Options()
    driver_options.add_argument("--no-sandbox")
    driver_options.add_argument("--start-maximized")

    driver = webdriver.Chrome('/usr/bin/chromedriver', options=driver_options)

    driver.get(url)

    count = 0
    while True:
        try:
            loadMoreButton = driver.find_element_by_id("load-more-trigger")
            time.sleep(0.1)
            loadMoreButton.click()
            count = 0
            time.sleep(0.1)
        except Exception as e:
            count += 1
            if count == 20:
                break

    soup_3=BeautifulSoup(driver.page_source, features="lxml")

    for item in soup_3.findAll('div', attrs={'class': 'review-container'}):
        try:
            headComment = item.find('div', attrs={'class': 'lister-item-content'}).find('a').text.replace('\n', '')
        except Exception as e:
            headComment = ""

        try:
            mainComment = item.find('div', attrs={'class': 'text show-more__control'}).text.replace('\n', '')
        except Exception as e:
            mainComment = ""

        try:
            commentRating = item.find('span', attrs={'class': 'rating-other-user-rating'}).find('span').text
        except Exception as e:
            commentRating = ""

        listOfComments.append([headComment, mainComment, commentRating])

    driver.quit()
    return [url, listOfComments]

#GET THE FINAL URL USED FOR SCRAPPING THE DATA FROM THE MAIN PAGE BY GETTING INTO EACH MOVIE
#THERE ARE SUCH HUNDRED MAIN PAGES CONTAINING HUNDRED MOVIES EACH
def getURLs(url):
    soup=GET_SOUP(url)
    table = soup.find('div', attrs={'class': 'lister-list'})

    listofMovieUrls=[]
    for row in table.findAll('div', attrs={'class': 'lister-item mode-detail'}):
        startTime = time.time()
        print("Work under progress! Please do not interfere with the system")

        try:
            rating = row.find('span', attrs={'class': 'ipl-rating-star__rating'}).text.replace('\n', '')
        except Exception as e:
            rating = ""

        try:
            title = row.find('h3', attrs={'class': 'lister-item-header'}).find('a').text.replace('\n', '')
        except Exception as e:
            title = ""

        try:
            url_2="https://www.imdb.com"+row.find('h3', attrs={'class': 'lister-item-header'}).find('a').get("href")
        except Exception as e:
            continue

        soup_2=GET_SOUP(url_2)

        try:
            url_3="https://www.imdb.com" + soup_2.find('div', attrs={'class': 'user-comments'}).findAll('a')[4].get("href")
        except Exception as e:
            continue

        listofMovieUrls.append({url_3: (title, rating)})
        driver_options = Options()
        driver_options.add_argument("--start-maximized")

    print("page completed")
    return listofMovieUrls


mainUrl = "https://www.imdb.com/list/ls057823854/"
url = "https://www.imdb.com/list/ls057823854/?sort=list_order,asc&st_dt=&mode=detail&page="

FinalList=[]
listOfUrls=[]
listOfUrls.append(mainUrl)

numPages = 1
#GETTING THOSE HUNDRED MAIN PAGES
for i in range(2, numPages+1):
	listOfUrls.append(url+str(i))

numCores = 4

parProcess = Pool(numCores)
result = parProcess.map(getURLs, listOfUrls)    # List of list of dicionaries
# LIST OF LIST OF DICTIONARY WHERE [DICTIONARY CONTAINS (FINAL URL --> MOVIE NAME & RATING OF THE MOVIE ITSELF)], [ONE LIST IS THOSE 100 MOVIE ON A SINGLE PAGE]
#[OUTER LIST IS BASICALLY FOR THOSE 100 MAIN PAGES]
FinalList=[]
finalDict = {}
for i in result:
    for j in i:
        for k in j:
            finalDict[k] = j[k]
            FinalList.append(k)

parProcesses = [Pool(numCores), Pool(numCores), Pool(numCores), Pool(numCores)]
numMovies = len(FinalList)
for i in range(0, 4):
    # parProcess2 = Pool(numCores)
    result2 = parProcesses[i].map(scrapeData, FinalList[(numMovies*i)//4:(numMovies*(i+1))//4])
    # movie = [url, listOfComments]
    for movie in result2:
        movieName = finalDict[movie[0]][0]
        movieRating = finalDict[movie[0]][1]
        print(movieName)
        outfile = open("./data/"+movieName.replace(' ', '_').replace('"', '_').replace('/', '_')+"_"+str(movieRating)+".tsv", "w")
        writer = csv.writer(outfile, delimiter='\t', lineterminator='\n')
        writer.writerow(['Comment Head','Comment Body', 'Comment Rating'])
        writer.writerows(movie[1])
