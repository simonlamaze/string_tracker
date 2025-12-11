import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
def main():

    df = pd.read_csv("data/oscillations.csv")
    sns.scatterplot(data=df, x="w", y="lambda", hue="Video")
    plt.grid()
    plt.xlabel("Pulsation (rad/s)", fontsize=14)
    plt.ylabel("Damping coefficient (1/s)",fontsize=14)
    plt.show()
    """
    colonnes_voulues = ["stretch weft (nominal)", "stretch warp (nominal)", "wave speed (m/s)", "Kept"]
    df= df[colonnes_voulues]
    df["stretch weft (nominal)"] = pd.to_numeric(df["stretch weft (nominal)"], errors="coerce")
    df["stretch warp (nominal)"] = pd.to_numeric(df["stretch warp (nominal)"], errors="coerce")
    df["wave speed (m/s)"]       = pd.to_numeric(df["wave speed (m/s)"], errors="coerce")
    df = df.dropna()
    dfs =df[df["Kept"]==0.5]
    dfl =df[df["Kept"]==1.0]
    wfss = {val: dfs[dfs["stretch weft (nominal)"] == val] for val in dfs["stretch weft (nominal)"].unique()}
    wfsl = {val: dfl[dfl["stretch weft (nominal)"] == val] for val in dfl["stretch weft (nominal)"].unique()}
    # all points togethers
    
    fig, axes = plt.subplots(1, len(wfss), figsize=(5 * len(wfss), 4))

    if len(wfss) == 1:
        axes = [axes]  # Normalise si un seul plot

    for ax, (val, sdfs) in zip(axes, wfss.items()):
        sns.lineplot(data=sdfs, x="stretch warp (nominal)", y="wave speed (m/s)", color="blue", 
                     errorbar=("ci", 95),
                    err_style="bars",
                     ax=ax, label="Small samples", estimator="mean",marker="o",linestyle="")
        #sns.scatterplot(data=sdfs, x="stretch warp (nominal)", y="wave speed (m/s)", color="blue", ax=ax, label="narrow cols")
        sdfl = wfsl.get(val)
        if sdfl is not None:
            sns.lineplot(data=sdfl, x="stretch warp (nominal)",
                         errorbar=("ci", 95),
                         err_style="bars",
                         y="wave speed (m/s)", color="orange", ax=ax, label="Large samples", estimator="mean",marker="o",linestyle="")
            #sns.scatterplot(data=sdfl, x="stretch warp (nominal)", y="wave speed (m/s)", color="orange", ax=ax, label="wide cols")
        ax.set_title("Weft stretch =" + str(val))
        ax.set_ylim(0,30)
        ax.grid()
        ax.legend(loc="lower right")
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.show()
    """
    return 
    
   
if __name__ == "__main__":
    main()