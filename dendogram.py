    
    
from plotly import figure_factory as ff
import pandas as pd
import numpy as np

    
if __name__ == "__main__":

    features = [4, 0, 6, 3, 5]
    data = pd.read_csv("s10.csv").sample(frac=1).reset_index(drop=True)
    data = data.iloc[:,1:]
    data = data.iloc[:,features]
    print (data)    
    fig = ff.create_dendrogram(data)
    fig.update_layout(width=800, height=500)
    fig.show() 