import numpy as np
import pandas as pd
import sqlalchemy as db
from bokeh.plotting import figure, output_file, show
import unittest
import math
import func_mapping

class Test(unittest.TestCase):
    def test_one(self):
        """Testing the code if it is running smoothly without errors"""
        """Initializing the engine and the metadata and connecting to a local database using sqlalchemy"""
        engine = db.create_engine("mysql+pymysql://root:root@localhost/assignment")
        metadata = db.MetaData()
        """Load the training data"""
        train_data = func_mapping.load_data('train',engine,metadata)
        if train_data.empty:
            raise Exception("Something went wrong while fetching train data")
        """Load the test data"""
        test_data = func_mapping.load_data('test',engine,metadata)
        if test_data.empty:
            raise Exception("Something went wrong while fetching test data")
        """Load the ideal data"""
        ideal_data = func_mapping.load_data('ideal',engine,metadata)
        if ideal_data.empty:
            raise Exception("Something went wrong while fetching ideal data")
        self.assertTrue(func_mapping.main(train_data, test_data, ideal_data))

    def test_two(self):
        """Testing the function to get the data from the database"""
        """Initializing the engine and the metadata and connecting to a local database using sqlalchemy"""
        engine = db.create_engine("mysql+pymysql://root:root@localhost/assignment")
        metadata = db.MetaData()
        """Load the training data"""
        train_data = func_mapping.load_data('train',engine,metadata)
        self.assertFalse(train_data.empty)

    def test_three(self):
        """Testing the method to find the least square error"""
        process = func_mapping.Calculations()
        a=pd.DataFrame({'l':[1,2,3,4,5]})
        b=pd.DataFrame({'l':[2,3,4,5,6]})
        d=process.find_lse(b,a)
        self.assertEqual(d.l,5)

    def test_four(self):
        """Testing the method to choose the ideal function"""
        process = func_mapping.Calculations()
        a=pd.DataFrame({'l':[1,2,3,4,5]})
        b=pd.DataFrame({'l':[2,3,4,5,6],'m':[1,2,3,4,5],'n':[1,2,5,4,3]})
        c=pd.DataFrame({'m':[1,2,3,4,5]})
        d=process.choose_ideal_funcs(a,b)

if __name__ == '__main__':
    unittest.main()