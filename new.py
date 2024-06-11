import streamlit as st
import pandas as pd
import lasio
import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib.colors import ListedColormap

# Disable deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Function to load LAS data
@st.cache_data
def load_las(file):
    las = lasio.read(file)
    df = las.df()
    df.dropna(inplace=True)
    return df

# Function to determine facies based on gamma-ray
def determine_facies(df, gamma_ray_col, baseline=75):
    if gamma_ray_col in df.columns:
        df['Facies'] = df[gamma_ray_col].apply(lambda x: 'Sand' if x < baseline else 'Shale')
    return df

st.title('Well Log Dashboard')

st.write("""
## `About`
:clap: This dashboard provides interactive visualizations of `well log data`, 
         including `facies plots`, `water saturation`, `porosity`, and `permeability`. Upload new data or use the example dataset provided.
""")

# Load example dataset
example_file = 'x1.las'
df = load_las(example_file)

# Sidebar for file upload and example data
st.sidebar.title("Options")

uploaded_file = st.sidebar.file_uploader("Upload a new LAS file", type=["las"])
if uploaded_file is not None:
    df = load_las(uploaded_file)

if st.sidebar.button("Use Example Dataset"):
    df = load_las(example_file)

# Dynamic column selection
st.sidebar.write("### Select Columns")
columns = df.columns.tolist()
gamma_ray_col = st.sidebar.selectbox("Select Gamma-Ray Column", columns)
cali_col = st.sidebar.selectbox("Select Caliper Column", columns)
resistivity_col = st.sidebar.selectbox("Select Resistivity Column", columns)
density_col = st.sidebar.selectbox("Select Density Column", columns)
acoustic_col = st.sidebar.selectbox("Select Acoustic Column", columns)
nphi_col = st.sidebar.selectbox("Select Neutron Column", columns)

# Add baseline for facies determination
baseline = st.sidebar.number_input("Set Gamma-Ray Baseline for Facies", value=75)

# Update DataFrame with facies based on selected gamma-ray column
df = determine_facies(df, gamma_ray_col, baseline)

# Function to plot individual logs
def plot_individual_logs(df, columns, depth_col='DEPT'):
    fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=(24, 20), sharey=True)
    
    for ax, col in zip(axes, columns):
        ax.plot(df[col], df.index)
        ax.set_xlabel(col, fontsize=12, color='black')
        ax.tick_params(axis='x', colors='black')
        ax.invert_yaxis()
    
    axes[0].set_ylabel('Depth', fontsize=12, color='black')
    axes[0].tick_params(axis='y', colors='black')
    plt.tight_layout()
    return fig

# Function to plot combination log
def plot_combination_log(df, gamma_ray_col, resistivity_col, density_col, nphi_col, baseline, depth_col='DEPT'):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(24, 20), sharey=True)
    
    # GR with facies color coding
    axes[0].plot(df[gamma_ray_col], df.index, color='black')
    axes[0].fill_betweenx(df.index, baseline, df[gamma_ray_col], where=(df[gamma_ray_col] < baseline), color='yellow', alpha=0.5, label='Sand')
    axes[0].fill_betweenx(df.index, baseline, df[gamma_ray_col], where=(df[gamma_ray_col] >= baseline), color='green', alpha=0.5, label='Shale')
    axes[0].axvline(x=baseline, color='red', linestyle='--', label='Baseline')
    axes[0].set_xlabel('Gamma-Ray', fontsize=12, color='black')
    axes[0].invert_yaxis()
    axes[0].legend()

    # Resistivity on a logarithmic scale
    axes[1].semilogx(df[resistivity_col], df.index, color='red')
    axes[1].set_xlabel('Resistivity', fontsize=12, color='black')

    # Overlaid RHOB and NPHI
    axes[2].plot(df[density_col], df.index, color='red', label='RHOB')
    axes[2].plot(df[nphi_col], df.index, color='blue', label='NPHI')
    axes[2].fill_betweenx(df.index, df[density_col], df[nphi_col], where=(df[density_col] > df[nphi_col]), facecolor='orange', alpha=0.3, label='RHOB > NPHI')
    axes[2].set_xlabel('RHOB/NPHI', fontsize=12, color='black')
    axes[2].legend()

    # Facies plot
    facies_colors = {'Sand': 'yellow', 'Shale': 'green'}
    facies_colormap = df['Facies'].map(facies_colors)
    cmap = ListedColormap(['yellow', 'green'])
    bounds = [0, 1, 2]
    norm = plt.Normalize(0, 2)
    facies_values = df['Facies'].apply(lambda x: 1 if x == 'Sand' else 2)

    axes[3].barh(df.index, np.ones(len(df)), color=facies_colormap, edgecolor='none')
    axes[3].set_xlabel('Facies', fontsize=12, color='black')
    axes[3].invert_yaxis()
    axes[3].set_xticks([])

    # Add color bar beside facies plot
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax, ticks=[1, 2])
    cbar.ax.set_yticklabels(['Sand', 'Shale'], rotation=90, va='center')
    cbar.ax.tick_params(labelsize=10, colors='black')
    cbar.set_label('Facies Type', fontsize=12, color='black')

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the main plot area to make space for the color bar
    return fig

# Individual log plots
individual_logs_columns = [nphi_col, density_col, gamma_ray_col, resistivity_col, cali_col, acoustic_col]
st.write("### `Individual Well Log Plots`")
fig_individual_logs = plot_individual_logs(df, individual_logs_columns)
st.pyplot(fig_individual_logs)

# Combination log plot
st.write("### Combination Log Plot")
fig_combination_log = plot_combination_log(df, gamma_ray_col, resistivity_col, density_col, nphi_col, baseline)
st.pyplot(fig_combination_log)

# Display the first few rows of the DataFrame to verify the facies column
st.write(df.head())
