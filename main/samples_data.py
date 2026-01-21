import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df= pd.read_csv("data/stitches_size.csv")

fig, ax = plt.subplots(1,1,figsize=(8, 6))

sns.scatterplot(data=df, x="stitches/cm (weft)", y="stitches/cm (warp)", hue="Sample size",  s=100, ax=ax)
ax.set_xlabel("Stitches/cm (weft)", fontsize=14)
ax.set_ylabel("Stitches/cm (warp)", fontsize=14)
ax.set_ylim(15,25)
ax.set_xlim(15,25)
ax.grid()

plt.show()
