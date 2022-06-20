import numpy as np
from utils_for_collecting.predict import FacialExpressionModel
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import cv2
import os
import urllib.request
import time
from deep_translator import GoogleTranslator


path_utils = "utils_for_collecting//"
path_images = "GoogleCC0/"
facec = cv2.CascadeClassifier(path_utils+'haarcascade_frontalface_default.xml')
model = FacialExpressionModel(
    path_utils+"model.json", path_utils+"model_weights.h5")
languages = GoogleTranslator().get_supported_languages()
pic_counter = 1
src_set = set()


def google_scraper(emotion):
    global src_set
    global pic_counter
    PATH = path_utils+"chromedriver.exe"
    try:
        driver = webdriver.Chrome(PATH)
    except Exception as e:
        # https://chromedriver.chromium.org/downloads
        print("Please download the same version of Chromedriver as your browser: https://chromedriver.chromium.org/downloads")

    def scroll_down(driver):
        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
    url = "https://images.google.com/"
    driver.get(url)
    wait(driver, 10).until(EC.element_to_be_clickable(
        (By.XPATH, '//*[@id="L2AGLb"]/div'))).click()
    for l in languages:
        search_input = GoogleTranslator(
            source='en', target=l).translate(text=emotion+" human face")
        search = driver.find_element_by_name('q')
        search.send_keys(Keys.CONTROL, 'a')
        search.send_keys(search_input)
        search.send_keys(Keys.RETURN)
        time.sleep(1)
        try:
            wait(driver, 10).until(EC.element_to_be_clickable(
                (By.XPATH, '//*[@id="yDmH0d"]/div[3]/c-wiz/div[1]/div/div[1]/div[2]/div[2]/div'))).click()
            wait(driver, 10).until(EC.element_to_be_clickable(
                (By.XPATH, '//*[@id="yDmH0d"]/div[3]/c-wiz/div[2]/div[2]/c-wiz[1]/div/div/div[1]/div/div[5]/div/div[1]'))).click()
            wait(driver, 10).until(EC.element_to_be_clickable(
                (By.XPATH, '//*[@id="yDmH0d"]/div[3]/c-wiz/div[2]/div[2]/c-wiz[1]/div/div/div[3]/div/a[2]/div/span'))).click()
        except:
            print("Please change XPATHs search filters, usage rights and Creative Commons buttons. From time to time, the XPATHS may change as Google also updates the searchengine web interface, albeit minimally.")
        time.sleep(1)
        i = 1
        while True:
            try:
                src = wait(driver, 5).until(EC.element_to_be_clickable(
                    (By.XPATH, f'//*[@id="islrg"]/div[1]/div[{str(i)}]/a[1]/div[1]/img'))).get_attribute("src")
                i += 1
            except Exception as e:
                break
            imgpath = path_images+emotion+"/"+str(pic_counter)+".png"
            req = urllib.request.urlopen(src)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            fr = cv2.imdecode(arr, -1)
            gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            faces = facec.detectMultiScale(gray_fr, 1.3, 5)
            pred = None
            for (x, y, w, h) in faces:
                fc = gray_fr[y:y+h, x:x+w]
                roi = cv2.resize(fc, (48, 48))
                pred = model.predict_emotion(
                    roi[np.newaxis, :, :, np.newaxis])

            if pred is not None and pred.lower() == emotion and src not in src_set:
                pic_counter += 1
                cv2.imwrite(imgpath, roi)
                src_set.add(src)
            scroll_down(driver)
    driver.quit()


if __name__ == "__main__":
    EMOTIONS_LIST = ["angry", "disgust",
                     "fear", "happy",
                     "neutral", "sad",
                     "surprise"]

    for e in EMOTIONS_LIST:
        google_scraper(e)
