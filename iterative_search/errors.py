# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from numeth.utils import means_and_errors, schedule_from_chains
sns.set()


from rich.progress import track

# %%
data = pd.read_csv("iterative_search/chain_iter_1.csv")
scheduler = schedule_from_chains(data)

print(scheduler)
analysis = means_and_errors(scheduler, data, 
                            random_variables=["m", "E"], 
                            estimators=[lambda x: np.mean(np.abs(x)), 
                                        lambda x: np.mean(np.abs(x)**2) - np.mean(np.abs(x))**2],
                            estimators_names=["mean", "ultravar"])
analysis.to_csv("analysis_m_err.csv")
print(analysis)



