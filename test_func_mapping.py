import numpy as np
import pandas as pd
import sqlalchemy as db
from bokeh.plotting import figure, output_file, show
import unittest
import math
import func_mapping

class Test(unittest.TestCase):
    def test_one(self):
        """Testing the function to get the data from the database"""
        """Initializing the engine and the metadata and connecting to a local database using sqlalchemy"""
        engine = db.create_engine("mysql+pymysql://root:root@localhost/assignment")
        metadata = db.MetaData()
        process = func_mapping.Calculations()
        """Load the training data"""
        train_data = process.load_data('train',engine,metadata)
        """Checking if the data is fetched from the database"""
        self.assertFalse(train_data.empty)

    def test_two(self):
        """Testing the method to find the least square error"""
        process = func_mapping.Calculations()
        a=pd.DataFrame({'l':[1,2,3,4,5]})
        b=pd.DataFrame({'l':[2,3,4,5,6]})
        d=process.find_lse(b,a)
        """There are 2 lists with 5 elements with a difference of 1 between them."""
        """Sum of squares of each of them will be 5"""
        self.assertEqual(d.l,5)

    def test_three(self):
        """Testing the method to choose the ideal function"""
        process = func_mapping.Calculations()
        a=pd.DataFrame({'l':[1,2,3,4,5]})
        b=pd.DataFrame({'l':[2,3,4,5,6],'m':[1,2,3,4,5],'n':[1,2,5,4,3]})
        """The most ideal function will be the one with the same values"""
        c=pd.DataFrame({'m':[1,2,3,4,5]})
        d=process.choose_ideal_funcs(a,b)
        self.assertEqual(c.values.tolist(),d.values.tolist())

if __name__ == '__main__':
    unittest.main()