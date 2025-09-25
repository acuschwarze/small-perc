"""
Voronoi Diagram Visualization of Network MSE Values
===================================================

This script creates a Voronoi diagram where each cell represents a network configuration
characterized by its size (number of nodes) and edge probability. The cells are colored
based on the Mean Squared Error (MSE) values for fitting the relative largest connected 
component after random or targeted node removal with the theoretical results obtained
from our combinatorial theory. MSE are displayed on a logarithmic scale.

Output: fig_voronoi.pdf - A Voronoi diagram with MSE-based coloring

Output figure is saved to 'repository root/figures/fig_voronoi.pdf'.

"""

# Import libraries
import os, sys
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.cm import ScalarMappable
from pathlib import Path
from typing import List, Tuple, Optional

# Add the parent directory to the path to import local libraries
REPO_ROOT = str(Path(__file__).parent.parent)
FIGURE_PATH = os.path.join(REPO_ROOT, 'figures')
FCACHE_PATH = os.path.join(REPO_ROOT, 'cache-figures')
CCACHE_PATH = os.path.join(REPO_ROOT, 'cache-combinatorics')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import from local libraries
from libs.utils import string_to_array
from libs.robustnessSimulations import fullDataTable

def polygon_area(vertices: List[Tuple[float, float]]) -> float:
    """
    Calculate the area of a polygon using the shoelace formula.
    
    Parameters:
    vertices: list of (x, y) tuples representing polygon vertices
    
    Returns:
    float: Area of the polygon
    """
    n = len(vertices)
    area = 0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]  # Wrap around to first vertex
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def add_colorbar_to_plot(mappable: ScalarMappable, use_log_scale: bool = False) -> Colorbar:
    """
    Add a colorbar to the current plot.
    
    Parameters:
    -----------
    mappable : matplotlib mappable
        The plot element to add colorbar for
    use_log_scale : bool
        Whether to format colorbar labels as powers of 10
        
    Returns:
    --------
    colorbar
        The created colorbar object
    """
    current_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar = fig.colorbar(mappable, cax=cax)
    
    if use_log_scale:
        tick_labels = colorbar.ax.get_yticklabels()
        new_labels = [label.get_text().replace('−', '-') for label in tick_labels]
        formatted_labels = [r'$10^{{{}}}$'.format(x) for x in new_labels]
        colorbar.ax.set_yticklabels(formatted_labels)
    
    plt.sca(current_axes)
    return colorbar

def compute_mse_data(targeted_removal=False, fname='real_networks_MSEs.csv', 
                     resource='real_networks_data.csv', recompute=False):

    resource_path = os.path.join(FCACHE_PATH, resource)
    if not os.path.exists(resource_path):
        data = fullDataTable(num_tries=100, max_size=100, recompute=recompute)
        data.to_csv(resource_path, sep=',', index=False, encoding='utf-8')

    fullData = pd.read_csv(os.path.join(FCACHE_PATH, resource))
    k = len(fullData)
    mse_array = np.zeros((k,4),dtype=object)
    
    for i in range(k):
        # retrieve n and p values
        n = fullData.iloc[i]['nodes']
        p = fullData.iloc[i]['edges'] / scipy.special.comb(n,2)

        # retrieve simulated and finite theory data
        print('fullData.iloc[i]', fullData.iloc[i])
        if not targeted_removal:
            sim = string_to_array(fullData.iloc[i]["real attack rLCC"], separator=" ")
            fin = string_to_array(fullData.iloc[i]["fin theory rand rLCC"], separator=" ")
        else:
            sim = string_to_array(fullData.iloc[i]["real rand rLCC"], separator=" ")
            fin = string_to_array(fullData.iloc[i]["fin theory attack rLCC"], separator=" ")

        # calculate mean square error
        mse = ((fin-sim)**2).mean()

        mse_array[i][0] = fullData.iloc[i]['network']
        mse_array[i][1] = n
        mse_array[i][2] = p
        mse_array[i][3] = mse

    df = pd.DataFrame(mse_array)
    # df.columns = ["network", "n", "p", "mse"]
    df.to_csv(os.path.join(FCACHE_PATH, fname))

    
class mse_data_bucket:

    def __init__(self, path: str) -> None:
        self.sourcefile: str = path
        self.data: Optional[pd.DataFrame] = None
        self.num_networks: int = 0
        self.network_sizes: np.ndarray = np.array([])
        self.edge_probabilities: np.ndarray = np.array([])
        self.mse_values: np.ndarray = np.array([])
        self.voronoi: Optional[Voronoi] = None

    def load(self) -> None:
        """Load data from CSV file and extract network properties."""
        fpath = os.path.join(FCACHE_PATH, self.sourcefile)
        if not os.path.exists(fpath):
            targeted_removal = ('targeted' in self.sourcefile)
            compute_mse_data(targeted_removal=targeted_removal, 
                             fname=self.sourcefile)

        self.data = pd.read_csv(fpath)
        self.num_networks = len(self.data)
        self.network_sizes = np.zeros(self.num_networks)
        self.edge_probabilities = np.zeros(self.num_networks)
        self.mse_values = np.zeros(self.num_networks)

        # Extract network properties from dataframe
        # Note: Columns are offset by 2 due to formatting in the CSV
        for idx in range(self.num_networks):
            self.network_sizes[idx] = self.data.iloc[idx]['1']
            self.edge_probabilities[idx] = self.data.iloc[idx]['2']
            self.mse_values[idx] = self.data.iloc[idx]['3']

    def filter(self, cutoff: float = 10) -> None:
        """
        Filter out MSE outliers above the cutoff threshold.
        
        Parameters:
        -----------
        cutoff : float
            Maximum MSE value to keep (outliers above this are filtered)
        """
        filtered_mse: List[float] = []
        filtered_sizes: List[float] = []
        filtered_probabilities: List[float] = []

        for i in range(len(self.mse_values)):
            if self.mse_values[i] < cutoff:
                filtered_mse.append(self.mse_values[i])
                filtered_sizes.append(self.network_sizes[i])
                filtered_probabilities.append(self.edge_probabilities[i])
            else:
                # Log outliers for debugging
                print(f"Outlier MSE: {self.mse_values[i]}, n={self.network_sizes[i]}, p={self.edge_probabilities[i]}")

        # Update arrays with filtered data
        self.mse_values = np.array(filtered_mse)
        self.network_sizes = np.array(filtered_sizes)
        self.edge_probabilities = np.array(filtered_probabilities)

    def transform_to_bounded_logarithmic(self, lower: float = -7, upper: float = 0.5) -> None:
        """
        Transform MSE values to logarithmic scale with bounds.
        
        Parameters:
        -----------
        lower : float
            Lower bound for log MSE values
        upper : float
            Upper bound for log MSE values
        """
        log_mse_values = np.log(self.mse_values)
        log_mse_values[log_mse_values > upper] = upper
        log_mse_values[log_mse_values < lower] = lower
        self.mse_values = log_mse_values

    def create_voronoi_diagram(self) -> None:
        """Create Voronoi diagram from network parameters."""
        # Prepare points for Voronoi diagram
        # Scale network sizes to [0,1] range to match probability scale
        cell_centers = np.column_stack((self.network_sizes / 100, self.edge_probabilities))

        # Add boundary ring points to ensure finite Voronoi regions
        boundary_cells = np.array([[np.cos(theta), np.sin(theta)] 
                                        for theta in np.linspace(0, 7)]) * 10
        boundary_cell_colors = np.array([1 for _ in np.linspace(0, 7)])

        # Combine data points with boundary points
        all_cell_centers = np.concatenate([cell_centers, boundary_cells])
        all_cell_colors = np.concatenate([self.mse_values, boundary_cell_colors])

        # Generate Voronoi diagram
        self.voronoi = Voronoi(all_cell_centers)

    def plot_voronoi_diagram(self, ax: Optional[Axes] = None) -> None:
        """
        Plot the Voronoi diagram colored by MSE values.
        
        Parameters:
        -----------
        ax : Optional[Axes]
            Matplotlib axes to plot on (uses current axes if None)
        """
        if ax is None:
            ax = plt.gca()

        # Setup color mapping for MSE values
        color_normalizer = plt.Normalize(vmin=-4, vmax=-0.3)
        colormap = cm.gnuplot2_r

        # Draw Voronoi diagram structure
        voronoi_plot_2d(self.voronoi, ax=ax, show_vertices=False, 
                        line_colors='grey', line_width=1, line_alpha=0.6, point_size=0)

        # Color Voronoi regions based on MSE values
        region_areas: List[float] = []

        for region_index, region in enumerate(self.voronoi.regions):
            # Skip infinite regions and empty regions
            if -1 not in region and len(region) > 0:
                # Get polygon vertices for this region
                polygon = [self.voronoi.vertices[j] for j in region]
                
                # Color the region based on its MSE value
                color = colormap(color_normalizer(self.mse_values[region_index]))
                ax.fill(*zip(*polygon), color=color)
                
                # Calculate and store region area
                area = polygon_area(polygon)
                region_areas.append(area)
                
                # Mark larger regions with their data points
                if area >= 0.006:
                    point_idx = np.where(self.voronoi.point_region == region_index)[0][0]
                    ax.plot([self.voronoi.points[point_idx, 0]], 
                            [self.voronoi.points[point_idx, 1]], 
                            color="grey", marker=".", markersize=5)

        # Print area statistics for analysis
        region_areas_array = np.array(region_areas)
        print(f"Area statistics - Mean: {np.mean(region_areas_array):.4f}, "
            f"Median: {np.median(region_areas_array):.4f}, "
            f"Max: {np.max(region_areas_array):.4f}, "
            f"Min: {np.min(region_areas_array):.4f}")

        # Add colorbar with logarithmic scale labels
        scalar_mappable = plt.cm.ScalarMappable(cmap="gnuplot2_r", norm=color_normalizer)
        scalar_mappable.set_array([])

        # Create colorbar with custom tick labels showing powers of 10
        colorbar = plt.colorbar(scalar_mappable, ax=axes[i]) #, label='MSE')
        tick_values = [-4, -3, -2, -1]
        colorbar.set_ticks(tick_values)
        tick_labels = [r'$10^{{{}}}$'.format(x) for x in tick_values]
        colorbar.set_ticklabels(tick_labels)

        # Set plot labels and title
        plt.sca(axes[i])
        plt.xlabel(r'Network Size $N$')
        if i == 0:
            plt.ylabel(r'Edge Probability $p$')
        plt.title('Voronoi Diagram Colored by MSE')
        # plt.title('MSE')

        # Set axis limits
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        # Format x-axis to show network sizes (0-100) instead of scaled values (0-1)
        plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        scale_factor = 100
        x_formatter = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * scale_factor))
        ax.xaxis.set_major_formatter(x_formatter)


def create_mse_histogram(data_buckets: List[mse_data_bucket], ax: Optional[Axes] = None) -> None:
    """
    Create histogram comparing MSE values for different removal strategies.
    
    Parameters:
    -----------
    data_buckets : List[mse_data_bucket]
        List of data bucket objects containing MSE values to plot
    ax : Optional[Axes]
        Matplotlib axes to plot on (uses current axes if None)
    """
    colors = ["tab:blue", "orange"]
    labels = ["random", "targeted"]
    
    if ax is None:
        ax = plt.gca()
    
    for i, data_bucket in enumerate(data_buckets):
        # Plot histogram
        ax.hist(data_bucket.mse_values, density=True, bins=10, alpha=0.65, 
                color=colors[i], label=labels[i])
    
    ax.legend(loc=2)
    ax.set_xlim([-7, -0.25])
    ax.set_ylim([0, 1])

    tick_values = [-7, -6, -5, -4, -3, -2, -1]
    ax.xaxis.set_ticks(tick_values)
    tick_labels = [r'$10^{{{}}}$'.format(x) for x in tick_values]
    ax.set_xticklabels(tick_labels)
    
    ax.set_xlabel(r'MSE')
    ax.set_ylabel(r'frequency')
    ax.set_title('Histogram of MSEs')

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')

if __name__ == "__main__":

    use_logarithmic: bool = True # toggle for logarithmic color map

    if True:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13,4), width_ratios=[10,10,8])
        plt.subplots_adjust(wspace=0.16, left=0.06, right=0.95)
    else:
        fig = plt.figure(figsize=(13,10))
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(212)
        axes = [ax1, ax2, ax3]
        #plt.subplots_adjust(wspace=0.1, left=0.06, right=0.97)

    # make voronoi plots
    # data_paths = ["MSEdata3D2.csv", "MSEdata3D2targeted.csv"]
    data_paths = ["real_network_mses_random.csv", "real_network_mses_targeted.csv"]
    filtered_data_sets: List[mse_data_bucket] = []

    for i, data_path in enumerate(data_paths):

        mse_data = mse_data_bucket(data_path)
        mse_data.load()
        mse_data.filter()
        if use_logarithmic:
            mse_data.transform_to_bounded_logarithmic()

        # Create Voronoi diagram
        mse_data.create_voronoi_diagram()

        mse_data.plot_voronoi_diagram(ax=axes[i])

        filtered_data_sets += [mse_data]

    create_mse_histogram(filtered_data_sets, ax=axes[2])

    # Add subplot labels
    subplot_labels: List[str] = ['(a)', '(b)', '(c)']
    for i in range(3):
        axes[i].text(0.965, 0.965, subplot_labels[i], transform=axes[i].transAxes, 
               fontsize=10, fontweight='normal', va='top', ha='right')

    # Save figure to PDF
    plt.savefig(os.path.join(FIGURE_PATH, "fig_voronoi.pdf"))