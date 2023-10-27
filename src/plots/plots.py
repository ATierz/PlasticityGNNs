import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.cm import ScalarMappable


def plot_graph_data(coord_x, coord_y, nodes_data, edge_index, title=''):
    # Create a graph
    G = nx.Graph()

    # Set nodes
    nodes = {i+1: (coord_x[i], coord_y[i]) for i in range(len(coord_x))}
    G.add_nodes_from(nodes)
    # set colours fro nodes
    # Define a colormap (e.g., 'viridis')
    cmap = plt.get_cmap('viridis')
    min_value = 0
    max_value = 2
    # Normalize the values to the range [0, 1]
    normalized_values = (np.array(nodes_data) - min_value) / (max_value - min_value)
    # Map the normalized values to colors
    colors_nodes = [cmap(val) for val in normalized_values]

    # Add edges
    edges = [(edge_index[0][i]+1, edge_index[1][i]+1) for i in range(len(edge_index[0]))]
    G.add_edges_from(edges)

    # Get node positions from the graph
    pos = nodes
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=colors_nodes, cmap=cmap, vmin=0, vmax=1)
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1, edge_color='gray')

    # Create a color bar legend for the heatmap
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])

    # Show the plot
    plt.axis('off')  # Turn off the axis
    plt.xlabel('x-coordiante')
    plt.ylabel('y-coordiante')
    plt.title(title)

    # Create a separate axes for the color bar
    cax = plt.axes([0.85, 0.1, 0.03, 0.8])  # Define the position and size of the color bar
    cbar = plt.colorbar(sm, cax=cax,)

    # Add labeled ticks to the color bar
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])  # Specify the tick values
    cbar.set_ticklabels(['0', '0.5', '1', '1.5', '2'])  # Specify the corresponding labels


    plt.show()



def plot_stress(coord_x, coord_y, stress_values, title=''):
    """
    Plot stress values at specified points.

    Parameters:
    - coordinates: 2D numpy array of shape (n, 2) containing x and y coordinates.
    - stress_values: 1D numpy array of length n, containing stress values.
    - points_to_plot: List of tuples specifying the (x, y) coordinates to plot.

    Returns:
    - None (displays the plot).
    """
    x = coord_x
    y = coord_y

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=stress_values, cmap='viridis', s=50, vmin=0, vmax=2, alpha=0.6)
    plt.colorbar(label='Stress')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title(f'Stress at Given Points iteration={title}')
    plt.grid()

    plt.show()



def plot_heatmap(coord_x, coord_y, stress_values, title=''):
    """
    Plot stress values as a heatmap with surface.

    Parameters:
    - coordinates: 2D numpy array of shape (n, 2) containing x and y coordinates.
    - stress_values: 1D numpy array of length n, containing stress values.
    - title: Title of the plot.

    Returns:
    - None (displays the plot).
    """
    x = coord_x
    y = coord_y

    plt.figure(figsize=(8, 6))

    # Create a filled contour plot (heat map) based on scatter points
    plt.tricontourf(x, y, stress_values, cmap='viridis', levels=100, vmin=0, vmax=2)

    # Overlay scatter plot on top of the heatmap
    plt.scatter(x, y, c=stress_values, cmap='viridis', s=50, vmin=0, vmax=2, alpha=0.6)

    plt.colorbar(label='Stress')

    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title(f'Stress Heatmap iteration={title}')
    plt.grid()

    plt.show()


def plot_heatmap_with_scatter(coord_x, coord_y, stress_values, title=''):
    """
    Plot stress values as a heatmap with surface and overlay scatter points with a darker line around the nodes.

    Parameters:
    - coordinates: 1D numpy arrays containing x and y coordinates.
    - stress_values: 1D numpy array of length n, containing stress values.
    - title: Title of the plot.

    Returns:
    - None (displays the plot).
    """
    x = coord_x
    y = coord_y

    plt.figure(figsize=(8, 6))

    # Create a filled contour plot (heat map) based on scatter points
    plt.tricontourf(x, y, stress_values, cmap='viridis', levels=100, vmin=0, vmax=2)

    # Overlay scatter plot on top of the heatmap with a darker line around nodes
    plt.scatter(x, y, c=stress_values, cmap='viridis', s=50, vmin=0, vmax=2, alpha=0.6, edgecolor='k', linewidth=1)

    plt.colorbar(label='Stress')

    # Set axis limits based on the coordinate data
    margin = 0.1
    x_min, x_max = min(x) - margin, max(x) + margin
    y_min, y_max = min(y) - margin, max(y) + margin
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title(f'Stress Heatmap with Scatter iteration={title}')

    # Set the aspect ratio to be equal
    plt.gca().set_aspect('equal', adjustable='box')

    plt.grid()



    plt.show()






