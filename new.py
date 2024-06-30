import streamlit as st
import pandas as pd
import lasio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import seaborn as sns
from tabulate import tabulate
import warnings


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
# Disable deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Function to load LAS data
@st.cache_data
def load_las(file):
    las = lasio.read(file)
    df = las.df()
    df.reset_index(inplace=True)
    df.dropna(inplace=True)
    return df

# Function to determine facies based on gamma-ray
def determine_facies(df, gamma_ray_col, baseline=75):
    if gamma_ray_col in df.columns:
        df['Facies'] = df[gamma_ray_col].apply(lambda x: 'Sand' if x < baseline else 'Shale')
    return df

# Function to calculate porosity
def calculate_porosity(df, density_col, rho_qtz=2.65, rho_fl=1.05):
    df['Porosity'] = (rho_qtz - df[density_col]) / (rho_qtz - rho_fl)
    return df

# Function to perform petrophysical calculations
def perform_petrophysics(df, gamma_ray_col, resistivity_col, density_col):
    # Calculate water saturation (Sw) - placeholder example
    df['Sw'] = 1 - df[resistivity_col] / df[density_col]

    # Use existing permeability columns
    df['Perm'] = df['PERM_ATAGA']
    df['Eff_Perm'] = df['PERM_EFF_ATAGA']

    # Calculate volume of shale (Vsh) - placeholder example
    df['Vsh'] = df[gamma_ray_col] / 150  # Assuming a max GR of 150 for normalization

    return df

# Function to plot individual logs
def plot_individual_logs(df, columns, depth_col='DEPT'):
    fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=(24, 16), sharey=True)
    for ax, col in zip(axes, columns):
        ax.plot(df[col], df[depth_col])
        ax.set_xlabel(col, fontsize=12, color='black')
        ax.tick_params(axis='x', colors='black')
        ax.invert_yaxis()
    axes[0].set_ylabel('Depth', fontsize=12, color='black')
    axes[0].tick_params(axis='y', colors='black')
    plt.tight_layout()
    return fig

# Function to plot comprehensive well log and petrophysical data
def plot_comprehensive(df, depth_col='DEPT', gamma_ray_col='GR', resistivity_col='ILD', density_col='RHOB', nphi_col='NPHI'):
    fig, axes = plt.subplots(nrows=1, ncols=8, figsize=(28, 16), sharey=True)

    # GR with facies color coding
    axes[0].plot(df[gamma_ray_col], df[depth_col], color='black')
    axes[0].fill_betweenx(df[depth_col], 75, df[gamma_ray_col], where=(df[gamma_ray_col] < 75), color='yellow', alpha=0.5, label='Sand')
    axes[0].fill_betweenx(df[depth_col], 75, df[gamma_ray_col], where=(df[gamma_ray_col] >= 75), color='green', alpha=0.5, label='Shale')
    axes[0].axvline(x=75, color='red', linestyle='--', label='Baseline')
    axes[0].set_xlabel('Gamma-Ray', fontsize=12, color='black')
    axes[0].invert_yaxis()
    axes[0].legend()

    # Resistivity on a logarithmic scale
    axes[1].semilogx(df[resistivity_col], df[depth_col], color='red')
    axes[1].set_xlabel('Resistivity', fontsize=12, color='black')

    # Overlaid RHOB and NPHI with red color between them
    ax2 = axes[2].twiny()
    axes[2].plot(df[nphi_col], df[depth_col], color='blue', label='NPHI')
    ax2.plot(df[density_col], df[depth_col], color='red', label='RHOB')
    axes[2].fill_betweenx(df[depth_col], df[nphi_col], df[density_col], where=(df[nphi_col] < df[density_col]), color='red', alpha=0.3)
    axes[2].set_xlabel('NPHI', fontsize=12, color='black')
    ax2.set_xlabel('RHOB', fontsize=12, color='black')
    axes[2].legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Water Saturation (Sw)
    axes[3].plot(df['Sw'], df[depth_col], color='blue', linewidth=1)
    axes[3].fill_betweenx(df[depth_col], 0, df['Sw'], color='blue', alpha=0.3)
    axes[3].set_xlabel('Water Saturation (Sw)', fontsize=12, color='black')

    # Permeability (Perm and Eff_Perm)
    axes[4].plot(df['Perm'], df[depth_col], color='green', linewidth=1, label='Perm', alpha=0.6)
    axes[4].fill_betweenx(df[depth_col], 0, df['Perm'], color='green', alpha=0.3)
    axes[4].plot(df['Eff_Perm'], df[depth_col], color='red', linewidth=1, label='Eff_Perm', alpha=0.6)
    axes[4].fill_betweenx(df[depth_col], 0, df['Eff_Perm'], color='red', alpha=0.3)
    axes[4].set_xlabel('Permeability', fontsize=12, color='black')
    axes[4].legend()

    # Volume of Shale (Vsh)
    axes[5].plot(df['Vsh'], df[depth_col], color='purple', linewidth=1, alpha=0.6)
    axes[5].fill_betweenx(df[depth_col], 0, df['Vsh'], color='purple', alpha=0.3)
    axes[5].set_xlabel('Volume of Shale (Vsh)', fontsize=12, color='black')

    # Porosity
    axes[6].plot(df['Porosity'], df[depth_col], color='brown', linewidth=1)
    axes[6].fill_betweenx(df[depth_col], 0, df['Porosity'], color='blue', alpha=0.3)
    axes[6].set_xlabel('Porosity', fontsize=12, color='black')

    # Facies plot
    facies_colors = {'Sand': 'yellow', 'Shale': 'green'}
    facies_colormap = df['Facies'].map(facies_colors)
    cmap = ListedColormap(['yellow', 'green'])
    bounds = [0, 1, 2]
    norm = plt.Normalize(0, 2)
    facies_values = df['Facies'].apply(lambda x: 1 if x == 'Sand' else 2)

    axes[7].barh(df[depth_col], np.ones(len(df)), color=facies_colormap, edgecolor='none')
    axes[7].set_xlabel('Facies', fontsize=12, color='black')
    axes[7].invert_yaxis()
    axes[7].set_xticks([])

    # Add color bar beside facies plot
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax, ticks=[1, 2])
    cbar.ax.set_yticklabels(['Sand', 'Shale'], rotation=90, va='center')
    cbar.ax.tick_params(labelsize=10, colors='black')
    cbar.set_label('Facies Type', fontsize=12, color='black')

    axes[0].set_ylabel('Depth', fontsize=12, color='black')
    axes[0].tick_params(axis='y', colors='black')
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the main plot area to make space for the color bar
    return fig

# Function to plot violin plot for facies
def plot_violin_facies(df, gamma_ray_col):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x='Facies', y='Porosity', data=df, palette=['yellow', 'green'], ax=ax)
    ax.set_title('Lithology Analysis')
    return fig

# Function to generate and display the summary table
def display_summary_table(df, gamma_ray_col, resistivity_col):
    summary_data = {
        'Parameter': ['Gamma-Ray', 'Volume of Shale (Vsh)', 'Water Saturation (Sw)', 'Permeability (Perm)', 'Porosity'],
        'Value': [
            f"{df[gamma_ray_col].mean():.4f}",
            f"{df['Vsh'].mean():.4f}",
            f"{df['Sw'].mean():.4f}",
            f"{df['Perm'].mean():.4f}",
            f"{df['Porosity'].mean():.4f}"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_table = tabulate(summary_df, headers='keys', tablefmt='grid')
    st.write("### `Average Across the Reservoir`")
    st.text(summary_table)


# Main Streamlit app
def main():
    st.image("download.jpg", use_column_width=True)
    st.markdown("---")    
    st.write("""
    # `Well Log Analysis Dashboard`
    """)
    st.write("### `About`")
    st.write("""
              :clap: This Streamlit web app offers a powerful and intuitive platform for analyzing well log data:stuck_out_tongue_closed_eyes:. Designed for geologists, 
              petroleum engineers, and data scientists:blush:. This app simplifies the process of visualizing, interpreting, and managing well log data:wave:
    """)

    # Load example dataset
    example_file = 'x1.las'
    df = load_las(example_file)

    # Sidebar for file upload and example data
    st.sidebar.title("Sidebar Options :clap:")
    uploaded_file = st.sidebar.file_uploader(":smile: Upload a new LAS file", type=["las"])
    if uploaded_file is not None:
        df = load_las(uploaded_file)

    if st.sidebar.button("`Use Example Dataset`"):
        df = load_las(example_file)

    # Dynamic column selection for well logs
    st.sidebar.write("### Select Columns")
    columns = df.columns.tolist()
    gamma_ray_col = st.sidebar.selectbox("Select `Gamma-Ray` Column", columns, index=columns.index('GR'))
    cali_col = st.sidebar.selectbox("Select `Caliper` Column", columns, index=columns.index('CALI'))
    resistivity_col = st.sidebar.selectbox("Select `Resistivity` Column", columns, index=columns.index('ILD'))
    density_col = st.sidebar.selectbox("Select `Density` Column", columns, index=columns.index('RHOB'))
    acoustic_col = st.sidebar.selectbox("Select `Acoustic` Column", columns, index=columns.index('DT'))
    nphi_col = st.sidebar.selectbox("Select `Neutron` Column", columns, index=columns.index('NPHI'))

    # Add baseline for facies determination
    baseline = st.sidebar.number_input("`Set Gamma-Ray Baseline` for Facies` ", value=75)

    # Update DataFrame with facies based on selected gamma-ray column
    df = determine_facies(df, gamma_ray_col, baseline)

    # Calculate porosity
    df = calculate_porosity(df, density_col)

    # Perform and display petrophysical analysis
    df = perform_petrophysics(df, gamma_ray_col, resistivity_col, density_col)

    # Display the first few rows of the DataFrame to verify the facies column
    st.write("### `Dataset Head`:stuck_out_tongue_closed_eyes: ")
    st.write(df.head())
    st.markdown("---")

    # Display individual log plots
    st.write("### `Individual Well Log Plots`:open_mouth: ")
    fig_individual_logs = plot_individual_logs(df, [nphi_col, density_col, gamma_ray_col, resistivity_col, cali_col, acoustic_col])
    st.pyplot(fig_individual_logs)
    st.markdown("---")

    # Display comprehensive well log and petrophysical analysis plot
    st.write("### `Comprehensive Well Log and Petrophysical Analysis Plot`:blush: ")
    fig_comprehensive = plot_comprehensive(df)
    st.pyplot(fig_comprehensive)
    st.markdown("---")

    # Display violin plot for facies
    st.write("### `Lithology Analysis`:stuck_out_tongue_closed_eyes: ")
    fig_violin_facies = plot_violin_facies(df, gamma_ray_col)
    st.pyplot(fig_violin_facies)
    st.markdown("---")

    # Display summary table
    display_summary_table(df, gamma_ray_col, resistivity_col)
    st.markdown('---')
    
    st.write(" `New Data Summary Column`:wave: ")
    st.write(df.head())
    st.markdown('---')

    # Save button to save the figures and data
    if st.button(" `Generate Summary Table and Save Plots`:open_mouth:"):
        # Save plots as images
        fig_individual_logs.savefig("individual_logs.png")
        fig_comprehensive.savefig("comprehensive_plot.png")
        fig_violin_facies.savefig("violin_plot.png")
        st.write("Plots saved successfully.")

        # Save DataFrame as CSV
        df.to_csv("well_log_data.csv", index=False)
        st.write(" `Data saved successfully`.")


    st.write("`Built with ❤️ by Elijah & Joshua`")
if __name__ == "__main__":
    main()
