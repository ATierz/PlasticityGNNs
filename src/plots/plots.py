import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from PIL import Image


def plot_graph_data(coord_x, coord_y, nodes_data, edge_index, lim_values=(0, 2), lim_coord_x=(0, 1.), lim_coord_y=(0, 1.), title=None, save_path=None):
    """
    Create a graphical representation of a network graph with colored nodes and edges.

    Parameters:
    - coord_x (list): List of x-coordinates for the nodes.
    - coord_y (list): List of y-coordinates for the nodes.
    - nodes_data (list): List of data associated with each node, which is used for node color.
    - edge_index (tuple of lists): Tuple containing two lists representing the edges in the graph.
    - title (str): A title for the graph (optional).

    This function takes node coordinates, node data, and edge information to generate a graphical representation
    of a network graph. It assigns colors to the nodes based on the provided data and displays the graph with
    labeled axes, a color bar legend, and an optional title.

    The color of the nodes is determined by the 'nodes_data' parameter, and the colormap used is 'viridis.'
    The edges are drawn in gray, and the graph is displayed with a color bar to indicate the data values associated
    with the nodes.

    Example usage:
    plot_graph_data(coord_x, coord_y, nodes_data, edge_index, title='Graph Visualization')
    """

    # Creat fig subplot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a graph
    G = nx.Graph()

    # Set nodes coordinates
    nodes = {i+1: (coord_x[i], coord_y[i]) for i in range(len(coord_x))}
    G.add_nodes_from(nodes)

    # Add edges
    edges = [(edge_index[0][i]+1, edge_index[1][i]+1) for i in range(len(edge_index[0]))]
    G.add_edges_from(edges)

    # Define a colormap (e.g., 'viridis') for nodes
    cmap = plt.get_cmap('rainbow')

    # Normalize the values to the range [0, 1]
    normalized_values = (np.array(nodes_data) - lim_values[0]) / (lim_values[1] - lim_values[0])

    # Map the normalized values to colors
    colors_nodes = [cmap(val) for val in normalized_values]

    # Get node positions from the graph
    pos = nodes

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=70, node_color=colors_nodes, cmap=cmap, vmin=0, vmax=1)

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1, edge_color='black')

    # Create a color bar legend for the heatmap
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])

    # Show only x and y axes without the whole canvas
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Set axis
    ax.grid(True)
    ax.set_xlabel('x-coordinate')
    ax.set_ylabel('y-coordinate')
    ax.set_xlim(lim_coord_x[0], lim_coord_x[1])
    ax.set_ylim(lim_coord_y[0], lim_coord_y[1])

    plt.title(title)

    # Create a separate axes for the color bar
    cax = plt.axes([0.9, 0.15, 0.03, 0.7])  # Define the position and size of the color bar
    cbar = plt.colorbar(sm, cax=cax,)

    # Add labeled ticks to the color bar
    cbar.set_ticks([0., 0.25, 0.5, 0.75, 1])  # Specify the tick values
    cbar.set_ticklabels([num for num in np.linspace(lim_values[0], lim_values[1], 5)])  # Specify the corresponding labels

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()

    plt.show()


import plotly.graph_objects as go


def plot_graph_data_plotly(coord_x, coord_y, nodes_data, edge_index, lim_values=(0, 2), lim_coord=(1.25, 6), title=None, save_path=None):
    """
    Create a graphical representation of a network graph with colored nodes and edges using Plotly.

    Parameters:
    - coord_x (list): List of x-coordinates for the nodes.
    - coord_y (list): List of y-coordinates for the nodes.
    - nodes_data (list): List of data associated with each node, which is used for node color.
    - edge_index (tuple of lists): Tuple containing two lists representing the edges in the graph.
    - title (str): A title for the graph (optional).
    - save_path (str): Path to save the graph as an image (optional).

    This function takes node coordinates, node data, and edge information to generate a graphical representation
    of a network graph. It assigns colors to the nodes based on the provided data and displays the graph using Plotly.

    Example usage:
    plot_graph_data(coord_x, coord_y, nodes_data, edge_index, title='Graph Visualization')
    """

    # Create a graph
    G = nx.Graph()

    # Set nodes coordinates
    nodes = {i+1: (coord_x[i], coord_y[i]) for i in range(len(coord_x))}
    G.add_nodes_from(nodes)

    # Add edges
    edges = [(edge_index[0][i]+1, edge_index[1][i]+1) for i in range(len(edge_index[0]))]
    G.add_edges_from(edges)

    # Create an empty Plotly figure
    fig = go.Figure()

    # Normalize the values to the range [0, 1]
    normalized_values = (np.array(nodes_data) - lim_values[0]) / (lim_values[1] - lim_values[0])

    # Get node positions from the graph
    pos = nodes

    # Create a scatter plot for nodes
    for node, (x, y) in pos.items():
        node_text = f'({x:.2f}, {y:.2f})'  # Node coordinate label
        color_val = normalized_values[node - 1]  # Use the normalized value as color
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker=dict(color=color_val, size=10),
                                 text=[node_text], name=f'Node {node}'))

    # Create lines for edges
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(color='black', width=1)))

    # Configure axis and layout
    fig.update_layout(xaxis=dict(range=[0, lim_coord[0]], title_text='x-coordinate'),
                      yaxis=dict(range=[0, lim_coord[1]], title_text='y-coordinate'),
                      title_text=title, showlegend=True)

    if save_path is not None:
        fig.write_image(save_path)

    fig.show()


def make_gif(data, path, title):

    min_value, max_value = np.round(float(data[0].x[:, 3].min()), decimals=3), np.round(float(data[-1].x[:, 3].max())*1.1, decimals=3)
    min_coord_x, max_coord_x = min(float(data[0].x[:, 4].min()), float(data[-1].x[:, 4].min()))*1.1, max(float(data[0].x[:, 4].max()), float(data[-1].x[:, 4].max()))*1.1
    min_coord_y, max_coord_y = min(float(data[0].x[:, 5].min()), float(data[-1].x[:, 5].min())) * 1.1, max(float(data[0].x[:, 5].max()), float(data[-1].x[:, 5].max())) * 1.1

    for i, sample in enumerate(data):
        stress = sample.x[:, 3].tolist()
        coord_x = sample.x[:, 4].tolist()
        coord_y = sample.x[:, 5].tolist()
        steps = i

        plot_graph_data(coord_x, coord_y, stress, sample.edge_index.tolist(),
                        lim_values=(min_value, max_value), lim_coord_x=(min_coord_x, max_coord_x), lim_coord_y=(min_coord_y, max_coord_y), title=f'{title} iter={i}', save_path=path / f'iter={i}.png')

    images = []
    for i in range(steps):
        images.append(Image.open(path / f'iter={i}.png'))
    print(len(images))
    images[0].save(path / 'animation.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
