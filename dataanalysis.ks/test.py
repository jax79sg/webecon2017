import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ll_data = [np.random.normal(0, std, 5) for std in range(1, 4)]
df=pd.DataFrame(ll_data)
print(df)
plt.boxplot(df.as_matrix(),vert=True,patch_artist=True)   # fill with color
plt.show()