import unittest
import time
from flask import Flask
from selenium import webdriver
from selenium.webdriver.common.by import By

# from time.sleep import sleep
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

# from app import app

# for testing purpose, only use 1000 rows of the dataset

BASE_URL = "http://localhost:5000"

class AppTestCaseE2E(unittest.TestCase):
    def setUp(self):
        options = webdriver.ChromeOptions()
        # options.add_argument('--headless')
        options.add_argument('--no-sandbox') # Disables the sandbox for all process types that are normally sandboxed.
        options.add_argument('--disable-dev-shm-usage') # Overcomes limited resource problems.
        options.add_argument('--disable-gpu') # Applicable to windows os only
        # options.add_argument('--remote-debugging-port=9222')

        # web driver with remote for the test to run in the container
        # self.driver = webdriver.Remote(
        #     command_executor='http://chrome:4444/wd/hub',
        #     options=options)
        # self.driver.get("http://web:5000/")
        self.driver = webdriver.Chrome(executable_path="/Users/liujiaen/Documents/Codes/Machine-Learning-in-Production-Project/chromedriver-mac-arm64/chromedriver" ,options=options)
        self.driver.get(BASE_URL)
    
    def test_add_item_and_view_all(self):
        # Test adding a book

        self.driver.get(BASE_URL + "/add_book")

        self.driver.find_element(by= By.NAME , value="ProductId").send_keys("789")
        self.driver.find_element(by= By.NAME , value="UserId").send_keys("012")
        self.driver.find_element(by= By.NAME , value="title").send_keys("Test Book 2")
        self.driver.find_element(by= By.NAME , value="Score").send_keys("5")
        self.driver.find_element(by= By.NAME , value="Time").send_keys("2022-01-01")
        self.driver.find_element(by= By.NAME , value="authors").send_keys("Jane Doe")
        self.driver.find_element(by= By.NAME , value="categories").send_keys("Non-Fiction")

        self.driver.find_element(by= By.XPATH, value="/html/body/div[2]/form/button").click()

        # check if response of clicking the button is successful

        self.assertEqual(self.driver.current_url, BASE_URL + "/")

        # self.assertEqual(self.driver.current_url, BASE_URL)

        # Test viewing all books
        # back to home page
        self.driver.get(BASE_URL)
        self.driver.find_element(by=By.XPATH, value="/html/body/div[2]/a/button").click()
        # check if the added book is in the page
        self.assertIn("Test Book 2", self.driver.page_source)

        self.assertEqual(self.driver.current_url, BASE_URL + "/view_all")

    def test_reload_model_and_monitering(self):
        self.driver.get( BASE_URL + "/reload_model")
        # wait 15 seconds for the model to be reloaded
        time.sleep(5)
        # then check the monitoring page is loaded
        self.driver.get( BASE_URL + "/monitoring")
        self.assertEqual(self.driver.current_url, BASE_URL + "/monitoring")

    def test_end2end(self):
        self.test_add_item_and_view_all()

        # test user recommendation
        self.driver.get( BASE_URL + "/user_recommendation")

        # test user recommendation for user id AJAF1T6Q7XM94
        self.driver.find_element(by= By.NAME , value="user_id").send_keys("AJAF1T6Q7XM94")
        self.driver.find_element(by= By.XPATH, value="/html/body/div[2]/form/button").click()

        # check if the user recommendation page is loaded
        self.assertEqual(self.driver.current_url, BASE_URL + "/user_recommendation")

        self.test_reload_model_and_monitering()

        # test user recommendation
        self.driver.get( BASE_URL + "/user_recommendation")

    def tearDown(self):
        self.driver.quit()

if __name__ == '__main__':
    unittest.main()



