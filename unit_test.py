# add parent path to sys.path
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import unittest
from flask import Flask
from flask_testing import TestCase

from app import app

# for testing purpose, only use 1000 rows of the dataset

class AppTestCase(TestCase):
    def create_app(self):
        app.config['TESTING'] = True
        return app

    # Unit tests
    def test_index_get(self):
        response = self.client.get('/')
        # assert b"A3CJ4WX7P56EW" in response.data
        self.assert200(response)
        self.assert_template_used('index.html')

    def test_view_all(self):
        response = self.client.get('/view_all')
        # check if A3CJ4WX7P56EW is in the page
        self.assert200(response)
        self.assert_template_used('view_all.html')

    def test_user_recommendation_get(self):
        response = self.client.get('/user_recommendation')
        self.assert200(response)
        self.assert_template_used('user_recommendation.html')

if __name__ == '__main__':
    unittest.main()