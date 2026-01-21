import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
import numpy as np
def main():
    """
    df = pd.read_csv("data/oscillations.csv")
    sns.scatterplot(data=df, x="w", y="lambda", hue="Video")
    plt.grid()
    plt.xlabel("Pulsation (rad/s)", fontsize=14)
    plt.ylabel("Damping coefficient (1/s)",fontsize=14)
    plt.show()
    """
    
    #tridimensional plot, colormaps exploring the space of weftstretch/ warp stretch/ wave speed
    df= pd.read_csv("data/data_samples.csv")
    colonnes_voulues = ["stretch weft (nominal)", "stretch warp (nominal)", "wave speed (m/s)", "Kept"]
    df= df[colonnes_voulues]
    df["stretch weft (nominal)"] = pd.to_numeric(df["stretch weft (nominal)"], errors="coerce")
    df["stretch warp (nominal)"] = pd.to_numeric(df["stretch warp (nominal)"], errors="coerce")
    df["wave speed (m/s)"]       = pd.to_numeric(df["wave speed (m/s)"], errors="coerce")
    df = df.dropna()

    dfs =df[df["Kept"]==0.5]
    dfl =df[df["Kept"]==1.0]
    """
    #continuous label
    norm = mpl.colors.Normalize(
    vmin=dfs["wave speed (m/s)"].min(),
    vmax=dfs["wave speed (m/s)"].max()
)
    sm = mpl.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ax1=axes[0]
    sns.scatterplot(data=df, x="stretch weft (nominal)", y="stretch warp (nominal)", hue="wave speed (m/s)",legend=False, palette="viridis",ax=ax1) 
    ax1.set_xlabel("Stretch weft (nominal)", fontsize=14)
    ax1.set_ylabel("Stretch warp (nominal)", fontsize=14)
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label("Wave speed (m/s)", fontsize=12)
    ax1.set_title("Wave speed depending on warp and weft stretch (aggregated)")  
    ax1.grid()

    ax2=axes[1]
    sns.scatterplot(data=dfl, x="stretch weft (nominal)", y="stretch warp (nominal)", hue="wave speed (m/s)",legend=False, palette="viridis",ax=ax2)

    ax2.set_xlabel("Stretch weft (nominal)", fontsize=14)
    ax2.set_ylabel("Stretch warp (nominal)", fontsize=14)
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label("Wave speed (m/s)", fontsize=12)
    ax2.set_title("Wave speed depending on warp and weft stretch (large columns)")
    ax2.grid()
    plt.show()
    """
    #script to plot different graphs for each weft stretch
    wfss = {val: dfs[dfs["stretch weft (nominal)"] == val] for val in dfs["stretch weft (nominal)"].unique()}
    wfsl = {val: dfl[dfl["stretch weft (nominal)"] == val] for val in dfl["stretch weft (nominal)"].unique()}
    wfss_f = {k: v for k, v in wfss.items() if k > 1.7}
    wfsl_f = {k: v for k, v in wfsl.items() if k > 1.7}
    l= len (wfss_f)

    
    fig, axes = plt.subplots(1, l, figsize=(5.5, 3))

    if len(wfss) == 1:
        axes = [axes]  # Normalise si un seul plot
    
    for ax, (val, sdfs) in zip(axes, wfss_f.items()):
            
            sns.lineplot(data=sdfs, x="stretch warp (nominal)", y="wave speed (m/s)", color="blue", 
                     errorbar=("ci", 95),
                    err_style="bars",
                     ax=ax, label="Small cols", estimator="mean",marker="o",linestyle="")
        #sns.scatterplot(data=sdfs, x="stretch warp (nominal)", y="wave speed (m/s)", color="blue", ax=ax, label="narrow cols")
            sdfl = wfsl_f.get(val)
            if sdfl is not None:
                sns.lineplot(data=sdfl, x="stretch warp (nominal)",
                         errorbar=("ci", 95),
                         err_style="bars",
                         y="wave speed (m/s)", color="orange", ax=ax, label="Large cols", estimator="mean",marker="o",linestyle="")
            #sns.scatterplot(data=sdfl, x="stretch warp (nominal)", y="wave speed (m/s)", color="orange", ax=ax, label="wide cols")
            ax.set_title("Weft stretch =" + str(val), fontsize=10, fontweight="bold")
            ax.set_ylim(0,30)
            ax.set_xlabel("Warp stretch ", fontsize=14)
            ax.set_ylabel(" ", fontsize=14)
            
            ax.grid()
            ax.legend(loc="lower right")
    axes[0].set_ylabel("Wave speed (m/s)", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.show()
    
    
    return 
    
   
if __name__ == "__main__":
    main()