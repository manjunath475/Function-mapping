import numpy as np
import pandas as pd
import sqlalchemy as db
from bokeh.plotting import figure, output_file, show
import unittest
import math

class Calculations:
    """Class data contains all the methods required for finding the ideal functions to visualizaing the deviation between 
    ideal functions data and test data"""    
    def load_data(self, table_name,engine,metadata):
        """Method load_data takes the table name and engine data as input and outputs the whole table data from the 
        database"""
        table = db.Table(table_name, metadata, autoload_with = engine)
        query = db.select(table)
        get_data = pd.read_sql(query,engine)
        return get_data

    def find_lse(self,y, y_id):
        """Method find_lse returns the sum of the least square errors of the input functions"""
        return np.sum((y-y_id)**2)

    def choose_ideal_funcs(self,y_train, y_ideal):
        """Method choose_ideal_funcs takes training and ideal data's y values and returns the mapped ideal functions 
        corresponding to the training data functions"""
        ideal_functions = []
        ideal_funcs_di = dict()
        lse_li = []
        for i in range(len(y_train.columns)):
            """Finding the sum of least square errors of the functions to map the ideal functions"""
            for j in range(len(y_ideal.columns)):
                try:
                    lse = Calculations.find_lse(self, y_train.iloc[:,i], y_ideal.iloc[:,j])
                except:
                    raise Exception("Something went wrong while calculating least square errors")
                lse_li.append(lse)
            """Finding the minimum error function in the ideal data and mapping it to the training data function"""
            ind = lse_li.index(min(lse_li))
            ideal_function = y_ideal.iloc[:,ind]
            ideal_functions.append(ideal_function)
            ideal_funcs_di[ideal_function.name] = ideal_function 
            ideal_funcs = pd.DataFrame(ideal_funcs_di)
            lse_li.clear()
        return ideal_funcs
    
    def compute_mappings(self,ideal_funcs,y_train,test_data, x_ideal):
        """Method compute_mappings returns the valid points for the corresponding mapped ideal functions and function names
        of the same"""
        devs = dict()
        cols = list()
        output = pd.DataFrame(columns=['x','y','yy','col'])
        """Finding the column names of all the mapped ideal functions"""
        for c in ideal_funcs.columns:
            cols.append(c)
        deviation = pd.DataFrame(abs(y_train.values - ideal_funcs.values))
        ideal_funcs['x'] = x_ideal

        """Finding the deviations for all the mapped ideal functions"""
        for i in range(len(deviation.columns)):
            dev1 = np.max(deviation.iloc[:,i]) + math.sqrt(2)
            dev2 = np.max(deviation.iloc[:,i]) - math.sqrt(2)
            devs[cols[i]] = [dev2, dev1]

        """Merging the test_data and the mapped corresponding ideal functions"""
        ideal_funcs.set_index('x')
        test_data.set_index('x')
        merged_df = test_data.merge(ideal_funcs)
        
        """Finding the deviations, max_deviation, min_deviation and valid points for the mapped ideal functions"""
        for col in cols:
            merged_df[f"{col}_dev"] = abs(merged_df["y"]-merged_df[col])
            merged_df[f"{col}_dev_min"] = devs[col][0]
            merged_df[f"{col}_dev_max"] = devs[col][1]
            merged_df[f"{col}_valid"] = np.where((merged_df[f"{col}_dev"] >= merged_df[f"{col}_dev_min"]) & (merged_df[f"{col}_dev"] <= merged_df[f"{col}_dev_max"]),'True','False')
        return merged_df, cols
    
class Visualize(Calculations):
    def vis_data(self,output,col):
        """Method vis_data visualizes the deviation between the test data provided and mapped function of the 
        ideal data using bokeh"""
        y = output['y'].values.tolist()
        x_y = np.linspace(np.min(output['x']), np.max(output['x']),num=len(y))
        ymap = output[f'{col}'].values.tolist()
        x_ymap = np.linspace(np.min(output['x']), np.max(output['x']),num=len(ymap))

        """Plotting the deviation"""
        fig = figure(title=f'{col} Plot', x_axis_label='x', y_axis_label='y')
        fig.line(x_y, y, legend_label="test", line_color="blue", line_width=2)
        fig.line(x_ymap, ymap, legend_label=f'{col}', line_color="red", line_width=2)
        show(fig)
    
def main():
    """Initializing the object of the class data"""
    process = Visualize()

    """Initializing the engine and the metadata and connecting to a local database using sqlalchemy"""
    engine = db.create_engine("mysql+pymysql://root:root@localhost/assignment")
    metadata = db.MetaData()
    """Load the training data"""
    train_data = process.load_data('train',engine,metadata)
    if train_data.empty:
        raise Exception("Something went wrong while fetching train data")
    """Load the test data"""
    test_data = process.load_data('test',engine,metadata)
    if test_data.empty:
        raise Exception("Something went wrong while fetching test data")
    """Load the ideal data"""
    ideal_data = process.load_data('ideal',engine,metadata)
    if ideal_data.empty:
        raise Exception("Something went wrong while fetching ideal data")
    
    """training data"""
    x_train = train_data.iloc[:,1]
    y_train = train_data.iloc[:,2:]

    "Ideal data"
    x_ideal = ideal_data.iloc[:,1]
    y_ideal = ideal_data.iloc[:,2:]

    """Finding the ideal functions for the training data functions in the ideal data"""
    try:
        ideal_funcs = process.choose_ideal_funcs(y_train, y_ideal)
    except:
        raise Exception("Something went wrong while fetching ideal functions")

    """Get the output and the names of the mapped ideal function"""
    try: 
        merged_df, cols = process.compute_mappings(ideal_funcs, y_train, test_data, x_ideal)
    except:
        raise Exception("Something went wrong while mapping the ideal functions with the test data")

    """Plotting the deviation between the test data provided and mapped functions of the ideal data"""
    for col in cols:
        bool = merged_df[f'{col}_valid'].str.contains('True')
        output = merged_df[bool]
        try: 
            process.vis_data(output,col)
        except:
            raise Exception(f"Something went wrong while plotting the graph for deviation of the ideal function {col}")
    return True

if __name__ == '__main__':
    main()