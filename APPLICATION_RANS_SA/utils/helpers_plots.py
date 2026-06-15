from utils.helpers_uq import *
import matplotlib.colors as mcolors

plt.rcParams.update({
    "font.size": 14,              # Tick and legend font size
    "axes.labelsize": 16,         # Axis label font
    "axes.titlesize": 18,         # Title font
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 3.,        # Frame border thickness
    "xtick.major.width": 3.,
    "ytick.major.width": 3.,
})


import matplotlib.ticker as ticker
from matplotlib.patches import Polygon as mplPolygon

def plot_mesh(mesh):
    coords = mesh.coordinates.dat.data_ro
    triangles = mesh.coordinates.cell_node_map().values
    triang = Triangulation(coords[:, 0], coords[:, 1], triangles)
    plt.figure(figsize=(12, 5.5))
    # Just plot mesh edges
    #plt.triplot(triang, color='black', linewidth=0.75)
    plt.triplot(triang, color='black', linewidth=0.25, alpha=1)
    #plt.xlabel("x",fontsize=18)
    #plt.ylabel("y",fontsize=18)
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=48) 
    plt.savefig("mesh.png", dpi=300,bbox_inches='tight')
    
def plot_field(field,fielddns,xes,yes,mesh,name,vmax,vmin,disc):
    cc=[]
    for i in range(len(xes)):
        cc.append([xes[i],yes[i]])
    # Firedrake Function 'u' has data arrays we need:
    # Coordinates of mesh vertices
    coords = mesh.coordinates.dat.data_ro
    cc = cc 
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    # Values of u at mesh vertices
    u_values = field.dat.data_ro
    u_valuesdns =fielddns.dat.data_ro
    triangles = mesh.coordinates.cell_node_map().values
    #Plot filled contours from Firedrake Function data
    pts = coords[triangles]                    # shape (n_cells, 3, 2)
    centroids = np.mean(pts, axis=1)          # shape (n_cells, 2)
    cx, cy = centroids[:, 0], centroids[:, 1]
    triang = Triangulation(x_coords, y_coords, triangles)
    # Plot filled contours from Firedrake Function data
    plt.figure(figsize=(16.5, 6))
    #plt.figure(figsize=(16.5, 3.5))
    base_cmap = plt.get_cmap("RdBu_r")
    # Extract only the red half (second half)
    red_half = mcolors.LinearSegmentedColormap.from_list("RdBu_r_red_half",base_cmap(np.linspace(0.48, 1, 256)))
    levels = np.linspace(vmin, vmax, disc)  # 30 contour levels
    filled = plt.tricontourf(x_coords,y_coords, u_values, levels=levels, cmap=red_half, extend='both')#RdBu_r
    #filled = plt.tricontourf(x_coords,y_coords, u_values, levels=levels, cmap="RdBu_r", extend='both')#RdBu_r
    #Overlay contour lines
    lines = plt.tricontour(x_coords,y_coords, u_valuesdns, levels=levels, colors="black", linewidths=0.75)
    hole_patch = mplPolygon(cc, facecolor='white', edgecolor='none', zorder=10)
    plt.gca().add_patch(hole_patch)
    cbar = plt.colorbar(filled)
    #cbar.set_label("u value", fontsize=18)      # Colorbar label font size
    tick_values = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(tick_values)
    cbar.ax.tick_params(labelsize=48)           # Colorbar tick label font size
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1g')) #1g'%.0f'
    #plt.xlabel("x",fontsize=18)
    #plt.ylabel("y",fontsize=18)
    plt.tight_layout()
    plt.xlim(1., 10) #full
    plt.ylim(0., 1)
    #plt.xlim(4., 6) #full
    #plt.ylim(0.45, 0.75)
    plt.tick_params(axis='both', which='major', labelsize=48) 
    plt.savefig(str(name)+".png", dpi=300,bbox_inches='tight')

