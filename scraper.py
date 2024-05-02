import time
import_time = time.time()
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

execution_time = time.time()

def scroll_down():
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

def store(title:str, description:str , page_url:str, image_url:str, item_number:int, page_number: int):
    """ Stores the given datapoints into a List """
    item_data = {
                "Title": title,
                "Image url": image_url,
                "Description": description,
                "Page url": page_url,
                "Item number": item_number,
                "Page number": page_number
                }
    data.append(item_data)

def store_error(link:str, item_number:int, page_number: int, page:str):
    """ Stores error data, given datapoints into a List """
    item_data = {
                "Page element link": link,
                "Item number": item_number,
                "Page number": page_number,
                "Page number": page
                }
    error_data.append(item_data)

website_url = "https://csmvs.in/all-collections/"

# Classes and Ids as per website
class_main = "collection-title"
img_id = "collections-main-image"
entire_class = "collection-details-data"
class_details = "collection-details-content"
xpath = f"//p[@class='{class_details}']/parent::*"
xpath_urls = "//div[@class='col post-item']"
xpath_href = ".//a"
page_class = "page-numbers"

data_loc = "Data_complete.csv"

page_counter = 1
article_counter = 0
error_counter = 0

# alternate_page = 
name = ""
height = ""
description = ""
collection = ""
object = ""
type = ""
material = ""
schools_culture_period = ""
technique = ""
date = ""
location = ""



page_urls = [f"https://csmvs.in/all-collections/page/{page_no}" for page_no in range(2,46)]
page_urls.insert(0, "https://csmvs.in/all-collections/")


visited = []
data = []
error_data = []

for page in page_urls:
    driver = webdriver.Chrome()
    driver.get(page)
    driver.maximize_window()
    # sleep(3)
    article_links = WebDriverWait(driver, 20).until(EC.presence_of_all_elements_located((By.XPATH, xpath_urls)), message=f"Could not all elemnts from {page_counter}")
    for link in article_links:
        try:
            href_article = WebDriverWait(link, 15).until(EC.presence_of_element_located((By.XPATH, xpath_href)), message= f"Could not find element - {article_counter}").get_attribute("href")
            driver.get(href_article)
            image_url = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, img_id))).get_attribute("src")
            paragraph = driver.find_element(By.XPATH, xpath).text
            title = paragraph.split("\n")[0]
            article_counter = article_counter + 1
            store(title, paragraph, href_article, image_url, article_counter, page_counter)
            driver.back()
        except:
            article_counter = article_counter + 1
            store_error(link, article_counter , page_counter , page)
            error_counter = error_counter +1

    
    
    print("Page no", page_counter)
    page_counter = page_counter + 1
driver.quit()

print(f"Scraping completed \nItems scraped - {article_counter} \nPages Scraped - {page_counter} \nFinal data count - {len(data)}\nError Count - {error_counter}")

pd_data = pd.DataFrame(data)
pd_data.to_csv(data_loc)
pd_data.to_parquet("data_parquet_complete.parquet")
pd_data.to_json("data_json_complete.json")

pd_error = pd.DataFrame(error_data)
pd_error.to_csv("Error_data_complete.csv")

end_time = time.time()


total_time = end_time - import_time
exec_time = end_time - execution_time 

print(f"Total time - {total_time} \nExecution time - {exec_time} \nImport time - {total_time-exec_time}")

