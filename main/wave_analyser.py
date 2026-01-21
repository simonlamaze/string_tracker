from turtle import pd
import cv2
from matplotlib import markers
from matplotlib.lines import Line2D
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from scipy import stats 
import matplotlib as mpl
import pandas as pds
#the score function for the gradient descent
#constants for the figures
fontsize_legend = 10
fontsize_axes = 14
fontsize_ticks = 10
linewidth_all = 3
markersize_all = 6

width_whole = 3.325 * 2
height_whole = 2.5
#######
def model(t,E, A,w,p,l):
    return E + A * np.exp(-l*t) * np.cos(w*t + p)

def truncate( list,threshold):
    """
    Truncate list values  until the first one reachingt the threshold. This was supposed to stop the list when the wave stopped, but it doesn't work before averaging. Maybe the opposite will work better ?
    """
    n = len(list)
    a=0
    while list[a]>threshold and a<n-1:
        a+=1
    return list[:a+1]
def average(list):
    a=0
    for i in range (len(list)):
        a+= list[i]
    return a/len(list)
def average_convert (list,N ,S):
    """
    Filters the velocity to get rid of noise, with N the number of points to average over( 5< N <20 overall). Also converts it in m/s, with S the size of a  in m
    Returns a list of the same size , of course N< len(list) ( can get quite high (0.2len) if we really want to smooth)
    """
    
    n = len(list)
    b=0
    flat_list=[]
    for i in range(n):
        if 0<=i<N:
            flat_list.append(average(list[:i+N])*S)
        elif N<=i<n-N:
          
            flat_list.append(average(list[i-N:i+N+1])*S)
        elif n-N<=i<n:
            
            flat_list.append(average(list[i-N:])*S)


    
    return flat_list
class WaveTrackingAnalyzer:
    """
    Analyze and visualize wave tracking data from JSON output.
    """
    #On every videos so far the scale was 0.0002 m/px

    
    def __init__(self, json_path):
        """
        Load tracking data from JSON file.
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.video_path = self.data['video_path']
        self.fps = self.data['fps']
        self.origin = self.data['origin']
        self.wave_front = self.data['wave_front']
        self.tracked_string= self.data['tracked_string']
        print(f"Loaded {len(self.wave_front)} tracking points")
        print(f"loaded {len(self.tracked_string)} points for the tracked string")
        print(f"Origin: ({self.origin['x']}, {self.origin['y']})")
        print(f"FPS: {self.fps}")
    
    def generate_plots(self, output_dir="outputs"):
        """
        Generate analysis plots:
        1. Wave front propagation (distance vs time)
        2. Wave front velocity (instantaneous and mean)
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        # extract tracking data
        frames = [p['frame'] for p in self.wave_front]
        times = [p['time']/4000 for p in self.wave_front]
        x_positions = [p['x'] for p in self.wave_front]
        y_positions = [p['y'] for p in self.wave_front]
        distances = [p['distance from origin'] for p in self.wave_front]
        
        #extract string tracket data
        frames_string = [p['frame'] for p in self.tracked_string]
        times_string = [p['time']/4000 for p in self.tracked_string]
        x_positions_string = [p['x'] for p in self.tracked_string]
        y_positions_string = [p['y'] for p in self.tracked_string]
        distances_string = [p['distance from origin'] for p in self.tracked_string]
        # calculate instantaneous velocities 
        velocities_px_per_s = []
        times_velocity = []
        
        for i in range(1, len(self.wave_front)):
            dt_seconds = times[i] - times[i-1]
            dd = distances[i] - distances[i-1]
            
            if dt_seconds > 0:
                v_px_s = dd / dt_seconds
                velocities_px_per_s.append(v_px_s)
                
            times_velocity.append(times[i])
        
        # style
        import seaborn as sns
        sns.set_style("whitegrid")
        sns.set_context("paper")
    
        
        
        df= pds.read_csv("data/data_samples.csv")
    
        colonnes_voulues = ["stretch weft (nominal)", "stretch warp (nominal)", "wave speed (m/s)", "Kept","Category "]
        df= df[colonnes_voulues]
        df["stretch weft (nominal)"] = pds.to_numeric(df["stretch weft (nominal)"], errors="coerce")
        df["stretch warp (nominal)"] = pds.to_numeric(df["stretch warp (nominal)"], errors="coerce")
        df["wave speed (m/s)"]       = pds.to_numeric(df["wave speed (m/s)"], errors="coerce")
        df = df.dropna()

        dfs =df[df["Kept"]==0.5]
        dfl =df[df["Kept"]==1.0]
        fig,axes= plt.subplots(1,3, figsize=(8, height_whole))
        
        #continuous label
        norm = mpl.colors.Normalize(
        vmin=dfs["wave speed (m/s)"].min(),
        vmax=dfs["wave speed (m/s)"].max())
        sm = mpl.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])

        
        ax0=axes[1]
        sns.scatterplot(data=df, x="stretch weft (nominal)", y="stretch warp (nominal)", hue="wave speed (m/s)",legend=False,style="Category ", markers=["o","s", "^"], palette="viridis",ax=ax0) 
        ax0.set_xlabel("Stretch weft", fontsize=14)
        ax0.set_ylabel("Stretch warp", fontsize=14)
        ax0.set_ylim(2,3.6)
        ax0.set_xlim(1,2)
        ax0.set_title("Small columns", fontsize=14, fontweight="bold", pad=5,loc="left")
        ax0.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        #ax0.set_title("Wave speed depending on warp and weft stretch (aggregated)")  
        
        
        ax1=axes[2]
        sns.scatterplot(data=dfl, x="stretch weft (nominal)", y="stretch warp (nominal)",legend=False, hue="wave speed (m/s)",style="Category ", markers=["o","s", "^"], palette="viridis",ax=ax1)

        ax1.set_xlabel("Stretch weft", fontsize=14)
        ax1.set_ylabel(" ", fontsize=14, fontweight="bold")
        ax1.set_ylim(2,3.6)
        ax1.set_xlim(1,2)
        ax1.set_title("Large columns", fontsize=14, fontweight="bold", pad=5, loc="left")
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        cbar = plt.colorbar(sm, ax=ax1)
        cbar.set_label("Wave speed (m/s)", fontsize=10)

        """
        
        #the legend for the categories
        categories = dfl["Category "].unique()
        markers = ["o", "s", "^", "D"]
        def Caté(cat):
            if cat == 1:
                return " A"
            elif cat ==2:
                return " B"
            elif cat == 3:
                return "C"
            else:
                return cat
            
        handles = [
        Line2D(
        [0], [0],
        marker=markers[i],
        markerfacecolor="none",   
        markeredgecolor="black",  
        color="black",
        linestyle="None",
        markersize=8,
        label=Caté(cat)
        )
        for i, cat in enumerate(categories)]

        ax1.legend(
        handles=handles,
        title="Category",
        fontsize=10,
        title_fontsize=10,
        
        frameon=True,
        loc="best")
        #ax1.set_title("Wave speed depending on warp and weft stretch (large columns)")


        
        # tracked string
        
        
        
        center= distances_string[0]
        print(len(times_string[1:]))
        
        distances_string=distances_string[1:]
        times_string=times_string[1:] # the first point is just the center, not the first real point
        print(len(distances_string))
        for i in range(len(distances_string)):
            distances_string[i]= distances_string[i]-center
        #on centre les positions autour de 0 pour mieux voir les oscillations
        # on ramène l'origine des temps à 0:
        min_time= min(times_string)
        for i in range(len(times_string)):
            times_string[i]= times_string[i]-min_time
        popt, pcov = curve_fit(model, times_string, distances_string, p0=[0, -5, -35000, 0, 0.01], maxfev=100000)
        E_opt,A_opt, w_opt, p_opt, l_opt = popt
        p_opt= p_opt%(2*np.pi)
        A2   = f"{A_opt:.2g}"
        w2   = f"{w_opt:.2g}"
        p2 = f"{p_opt:.2g}"
        l2   = f"{l_opt:.2g}"
        E2   = f"{E_opt:.2g}"
        print(f"Fitted parameters: A={A2}, w={w2}, p={p2}, l={l2}, E={E2}")
        tight_times= np.linspace(min(times_string), max(times_string), 1000)
        fitted_distances = model(np.array(tight_times), *popt)
        
        ax2.plot(tight_times, fitted_distances, '-', color='blue', linewidth=1.5, markersize=0, label = "model" )
        
        ax2.plot(times_string, distances_string,marker= 'o',  color='#ff7f0e', linewidth=0, markersize=2, label = "tracked string" )
        ax2.set_xlabel('Time (s)', fontsize=14)
        ax2.set_ylabel('Δx_string (px)', fontsize=14)
        ax2.legend()
        
        """
        """
        #that's the instantaneous velocity plot, very noisy
        correct_list=average_convert(velocities_px_per_s,50,0.0002)
        

        # we returned when the wave stopped by first averaging to get rid of the noise, and now we reaverage on the relevant measurements
        # If I had taken truncated list, I would have taken average measures containing irrelevant values
        if len(velocities_px_per_s) > 0:
            ax2.plot(times_velocity,correct_list, '-', color='#2ca02c', linewidth=1.5, alpha=0.7)
            
        ax2.set_xlabel('Time [s]', fontsize=11)
        ax2.set_ylabel('Velocity [m/s]', fontsize=11)
        ax2.set_title('Wave front velocity', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax2.set_ylim(20,30) # règle l'échelle pour une meilleure visualisation
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.legend(frameon=False, fontsize=10)
        """
        
        
        # 1. distance vs time (wave front propagation)
        
        
        
        ax2 = axes[0]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(times, distances)
        L=[]
        for i in range(len(times)):
            L.append((intercept+slope*times[i])*0.2)
        for i in range(len(distances)):
            distances[i]= distances[i]*0.2 # conversion en mm
        
        print(f"Mean wave velocity: {slope*0.0002} m/s")
        ax2.plot(times,L, '-', color='blue', linewidth=1.5, markersize=0, alpha=0.8, label='Linear fit' )
        ax2.legend(fontsize=10)
        ax2.plot(times, distances,marker='o', color='#ff7f0e', linewidth=0, markersize=1.5, alpha= 0.6)
        ax2.set_xlabel('Time [s]', fontsize=14)
        ax2.set_ylabel('Distance (mm)', fontsize=14)
        #ax2.set_title('Wave front propagation', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.legend()
        
        
        plt.tight_layout()
        plt.show()

        
        
        plot_path = f"{output_dir}/wave_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        
        sns.reset_orig()
        return
               
    
    def create_annotated_video(self, output_path="outputs/wave_tracking_video.mp4"):
        """
        Create annotated video showing:
        - Blue circle: origin point (wave source)
        - Red circle + vertical line: tracked wave front position
        """
        Path(output_path).parent.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(self.video_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        # create frame-to-point for more efficient access
        frame_to_point = {p['frame']: p for p in self.wave_front}

        # origin coordinates
        ox = int(self.origin['x'])
        oy = int(self.origin['y'])
                
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated = frame.copy()

            # draw origin point
            cv2.circle(annotated, (ox, oy), 3, (255, 0, 0), -1)
            
            # draw wave front if tracked up to this frame
            relevant_points = [p for p in self.wave_front if p['frame'] <= frame_idx]
            
            if len(relevant_points) > 0:
                
                current_point = relevant_points[-1]
                wx = int(current_point['x'])
                wy = int(current_point['y'])            
                
                cv2.line(annotated, (wx, 0), (wx, height), (0, 0, 255), 1, cv2.LINE_AA)
            
            out.write(annotated)
            
            frame_idx += 1
            
        
        cap.release()
        out.release()
        
    

def main():
    json_path = "results/wave_tracking.json"
    
    analyzer = WaveTrackingAnalyzer(json_path)
    
    # generate quantitative analysis plots
    analyzer.generate_plots(output_dir="outputs")

   
    

if __name__ == "__main__":
    main()



