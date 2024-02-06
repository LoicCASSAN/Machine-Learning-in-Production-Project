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

    # Integration tests
    def test_user_recommendation_post(self):
        response = self.client.post('/user_recommendation', data={'user_id': 'AJAF1T6Q7XM94'})
        self.assert200(response)
        self.assert_template_used('user_recommendation.html')

    def test_reload_model_and_monitering(self):
        response = self.client.get('/reload_model')
        # check the monitoring page is loaded

        self.assertEqual(response.location, '/')

    # End-to-end test
    def test_end_to_end(self):
        # Test adding a book
        response = self.client.post('/add_book', data={
            'ProductId': '789',
            'UserId': '012',
            'title': 'Test Book 2',
            'Score': '5',
            'Time': '2022-01-01',
            'authors': 'Jane Doe',
            'categories': 'Non-Fiction'
        })
        self.assertEqual(response.location, '/')

        # Test viewing all books
        response = self.client.get('/view_all')
        self.assert200(response)
        self.assert_template_used('view_all.html')

        # Test getting recommendations for a user
        response = self.client.post('/user_recommendation', data={'user_id': '012'})
        self.assert200(response)
        self.assert_template_used('user_recommendation.html')

        # Test reloading the model
        response = self.client.get('/reload_model')
        self.assertEqual(response.location, '/')
    
if __name__ == '__main__':
    unittest.main()