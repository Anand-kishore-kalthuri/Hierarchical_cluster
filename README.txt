REQUIRED LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import sweetviz
from AutoClean import AutoClean
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering 
from sklearn import metrics
from clusteval import clusteval
import numpy as np
from sqlalchemy import create_engine
