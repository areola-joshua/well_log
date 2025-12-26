import streamlit as st
import pandas as pd
import lasio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        
        /* Dark theme for about section */
        .stApp {
            background-color: #0E1117;
        }
        
        .about-section {
            background-color: #1E1E1E;
            padding: 25px;
            border-radius: 15px;
            border-left: 5px solid #FF4B4B;
            margin-bottom: 20px;
            color: #FFFFFF;
        }
        
        .about-section h4 {
            color: #FF4B4B;
        }
        
        .about-section li {
            color: #FFFFFF;
        }
        
        /* Card styling */
        .metric-card {
            background-color: #262730;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #3A3A4A;
            color: #FFFFFF;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #1E1E1E;
        }
        
        /* Table styling */
        .dataframe {
            background-color: #262730 !important;
            color: #FFFFFF !important;
        }
        
        </style>
        """
st.set_page_config(page_title="Professional Well Log Analysis", page_icon="⛏️", layout="wide")
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Disable deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Function to load LAS data - FIXED with io.StringIO to resolve "well issue"
@st.cache_data
def load_las(file):
    try:
        if isinstance(file, str):
            las = lasio.read(file)
        else:
            # Read the bytes and decode to string for lasio
            bytes_data = file.read().decode("utf-8")
            las = lasio.read(io.StringIO(bytes_data))
        
        df = las.df()
        df.reset_index(inplace=True)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading LAS file: {e}")
        return None

# Function to determine facies based on gamma-ray with baseline
def determine_facies(df, gamma_ray_col, baseline=75):
    if gamma_ray_col in df.columns:
        df['Facies'] = df[gamma_ray_col].apply(lambda x: 'Sand' if x < baseline else 'Shale')
        # Store baseline for reference
        df['GR_Baseline'] = baseline
    return df

# Function to calculate porosity from density log
def calculate_porosity(df, density_col, rho_qtz=2.65, rho_fl=1.05):
    df['Porosity_Density'] = (rho_qtz - df[density_col]) / (rho_qtz - rho_fl)
    df['Porosity_Density'] = df['Porosity_Density'].clip(0, 0.4)  # Limit to realistic values
    return df

# Function to calculate volume of shale using Larionov method
def calculate_vsh(df, gr_col, gr_min=None, gr_max=None):
    if gr_min is None:
        gr_min = df[gr_col].min()
    if gr_max is None:
        gr_max = df[gr_col].max()
    
    df['Vsh_Larionov'] = (df[gr_col] - gr_min) / (gr_max - gr_min)
    df['Vsh_Larionov'] = df['Vsh_Larionov'].clip(0, 1)
    return df, gr_min, gr_max

# Function to calculate water saturation using Archie's equation
def calculate_sw_archie(df, rt_col, por_col, rw=0.1, a=1.0, m=2.0, n=2.0):
    # Avoid division by zero
    por_valid = df[por_col].clip(0.01, 0.4)
    rt_valid = df[rt_col].clip(0.1, 1000)
    
    df['Sw_Archie'] = ((a * rw) / (rt_valid * (por_valid ** m))) ** (1/n)
    df['Sw_Archie'] = df['Sw_Archie'].clip(0, 1)
    return df

# Function to calculate permeability using Timur equation
def calculate_permeability(df, por_col, sw_col):
    por_valid = df[por_col].clip(0.01, 0.4)
    sw_valid = df[sw_col].clip(0.1, 1)
    
    df['Perm_Timur'] = 0.136 * (por_valid ** 4.4) / (sw_valid ** 2)
    df['Perm_Timur'] = df['Perm_Timur'].clip(0.01, 10000)
    return df

# Function to perform comprehensive petrophysical calculations
def perform_petrophysics(df, gamma_ray_col, resistivity_col, density_col, nphi_col, baseline=75):
    # Calculate Vsh using Larionov method
    df, gr_min, gr_max = calculate_vsh(df, gamma_ray_col)
    
    # Calculate porosity from density
    df = calculate_porosity(df, density_col)
    
    # Calculate effective porosity (density-neutron average)
    df['Porosity_Effective'] = (df['Porosity_Density'] + df[nphi_col]) / 2
    df['Porosity_Effective'] = df['Porosity_Effective'].clip(0, 0.4)
    
    # Calculate water saturation using Archie's equation
    df = calculate_sw_archie(df, resistivity_col, 'Porosity_Effective', rw=0.1)
    
    # Calculate permeability using Timur equation
    df = calculate_permeability(df, 'Porosity_Effective', 'Sw_Archie')
    
    # Calculate hydrocarbon saturation
    df['Sh'] = 1 - df['Sw_Archie']
    df['Sh'] = df['Sh'].clip(0, 1)
    
    # Calculate net-to-gross based on baseline
    df['NTG'] = df[gamma_ray_col].apply(lambda x: 1 if x < baseline else 0)
    
    # Calculate pay flag (good reservoir rock)
    df['Pay_Flag'] = ((df['Facies'] == 'Sand') & 
                      (df['Porosity_Effective'] > 0.1) & 
                      (df['Sw_Archie'] < 0.6)).astype(int)
    
    return df, gr_min, gr_max

# Function to filter dataframe by depth
def filter_by_depth(df, depth_col, start_depth, end_depth):
    if start_depth is not None and end_depth is not None:
        filtered_df = df[(df[depth_col] >= start_depth) & (df[depth_col] <= end_depth)].copy()
        return filtered_df
    return df.copy()

# --- REPLACED WITH YOUR CORRECTED LOG PLOT CODE ---
def plot_well_logs(df, depth_col='DEPT', gamma_ray_col='GR', cali_col='CALI', 
                   resistivity_col='ILD', density_col='RHOB', nphi_col='NPHI', sonic_col='DT',
                   start_depth=None, end_depth=None, figsize=(18, 14)):
    """
    Track 1: GR + Caliper | Track 2: Resistivity | Track 3: RHOB + NPHI | Track 4: DT
    """
    df_filtered = df[(df[depth_col] >= start_depth) & (df[depth_col] <= end_depth)].copy()
    
    # Unit correction for NPHI (converts % to decimal)
    if df_filtered[nphi_col].max() > 1.0:
        df_filtered['NPHI_dec'] = df_filtered[nphi_col] / 100.0
    else:
        df_filtered['NPHI_dec'] = df_filtered[nphi_col]

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=figsize, sharey=True)
    
    # Track 1: Gamma Ray + Caliper
    ax1 = axes[0]
    baseline = 75
    ax1.plot(df_filtered[gamma_ray_col], df_filtered[depth_col], color='black', linewidth=0.8, label='GR')
    ax1.fill_betweenx(df_filtered[depth_col], baseline, df_filtered[gamma_ray_col], 
                      where=(df_filtered[gamma_ray_col] >= baseline), color='green', alpha=0.5)
    ax1.fill_betweenx(df_filtered[depth_col], df_filtered[gamma_ray_col], baseline, 
                      where=(df_filtered[gamma_ray_col] < baseline), color='yellow', alpha=0.5)
    ax1.set_xlabel('GR (API)', color='black', fontweight='bold')
    ax1.set_xlim(0, 150)
    
    ax1c = ax1.twiny()
    ax1c.plot(df_filtered[cali_col], df_filtered[depth_col], color='blue', linewidth=0.8, linestyle='--', label='CALI')
    ax1c.set_xlabel('Caliper (in)', color='blue', fontweight='bold')
    ax1c.set_xlim(6, 16)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)

    # Track 2: Resistivity (Log Scale)
    ax2 = axes[1]
    ax2.semilogx(df_filtered[resistivity_col], df_filtered[depth_col], color='red', linewidth=1)
    ax2.set_xlabel('Resistivity (Ω.m)', color='red', fontweight='bold')
    ax2.set_xlim(0.2, 2000)
    ax2.grid(True, which='both', alpha=0.3)

    # Track 3: RHOB + NPHI with Crossover
    ax3 = axes[2]
    ax3.plot(df_filtered['NPHI_dec'], df_filtered[depth_col], color='blue', linewidth=1, label='NPHI')
    ax3.set_xlim(0.45, -0.15)
    ax3b = ax3.twiny()
    ax3b.plot(df_filtered[density_col], df_filtered[depth_col], color='red', linewidth=1, label='RHOB')
    ax3b.set_xlim(1.95, 2.95)
    
    # Crossover Shading (Yellow for potential Gas/Sand)
    rho_norm = (2.71 - df_filtered[density_col]) / (2.71 - 1.0)
    ax3.fill_betweenx(df_filtered[depth_col], df_filtered['NPHI_dec'], rho_norm, 
                      where=(rho_norm > df_filtered['NPHI_dec']), color='yellow', alpha=0.4)
    ax3.set_xlabel('NPHI (v/v)', color='blue', fontweight='bold')
    ax3b.set_xlabel('RHOB (g/cc)', color='red', fontweight='bold')

    # Track 4: Sonic (DT)
    ax4 = axes[3]
    ax4.plot(df_filtered[sonic_col], df_filtered[depth_col], color='purple', linewidth=0.8)
    ax4.set_xlabel('DT (us/ft)', color='purple', fontweight='bold')
    ax4.set_xlim(140, 40)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# --- REPLACED WITH YOUR CORRECTED PETROPHYSICS CODE ---
def plot_petrophysics_with_facies(df, depth_col='DEPT', gr_col='GR', 
                                  por_col='PHIE_ATAGA', vsh_col='Vsh', 
                                  sw_col='Sw', facies_col='Facies', 
                                  start_depth=None, end_depth=None):
    
    # Filter depth range
    df_f = df[(df[depth_col] >= start_depth) & (df[depth_col] <= end_depth)].copy()
    
    # --- COLUMN SELECTION ---
    # Prioritize PHIE_ATAGA, then PHID_ATAGA
    if 'PHIE_ATAGA' in df_f.columns:
        actual_por = 'PHIE_ATAGA'
    elif 'PHID_ATAGA' in df_f.columns:
        actual_por = 'PHID_ATAGA'
    else:
        actual_por = por_col

    actual_vsh = 'Vsh_Larionov' if 'Vsh_Larionov' in df_f.columns else vsh_col
    actual_sw = 'Sw_Archie' if 'Sw_Archie' in df_f.columns else sw_col

    fig, axes = plt.subplots(1, 5, figsize=(16, 12), sharey=True)
    
    # Track 1: Gamma Ray (Green for Shale, Yellow for Sand)
    baseline = 75
    axes[0].plot(df_f[gr_col], df_f[depth_col], color='black', lw=0.8)
    axes[0].fill_betweenx(df_f[depth_col], baseline, df_f[gr_col], 
                          where=(df_f[gr_col] >= baseline), color='green', alpha=0.5)
    axes[0].fill_betweenx(df_f[depth_col], df_f[gr_col], baseline, 
                          where=(df_f[gr_col] < baseline), color='yellow', alpha=0.5)
    axes[0].axvline(x=baseline, color='red', linestyle='--', lw=1)
    axes[0].set_xlabel('GR (API)', fontweight='bold')
    axes[0].set_xlim(0, 150)
    axes[0].invert_yaxis()

    # Track 2: Porosity (FIXED: Shading to the left)
    por_plot_df = df_f[[depth_col, actual_por]].dropna()
    axes[1].plot(por_plot_df[actual_por], por_plot_df[depth_col], color='blue', lw=1.2)
    
    # FIXED: Fill from the curve to the LEFT (0.5) to avoid the solid block on the right
    axes[1].fill_betweenx(por_plot_df[depth_col], por_plot_df[actual_por], 0, 
                          color='blue', alpha=0.3)
    
    axes[1].set_xlabel('Porosity', color='blue', fontweight='bold') # Simplified label
    axes[1].set_xlim(0, 0.5) # 0.5 on left, 0 on right
    axes[1].grid(True, alpha=0.3)

    # Track 3: Vsh
    axes[2].plot(df_f[actual_vsh], df_f[depth_col], color='brown', lw=1)
    axes[2].fill_betweenx(df_f[depth_col], 0, df_f[actual_vsh], color='brown', alpha=0.3)
    axes[2].set_xlabel('Vsh (v/v)', color='brown', fontweight='bold')
    axes[2].set_xlim(0, 1)

    # Track 4: Sw
    sw_data = df_f[actual_sw].clip(0, 1)
    axes[3].plot(sw_data, df_f[depth_col], color='red', lw=1)
    axes[3].fill_betweenx(df_f[depth_col], sw_data, 1, color='green', alpha=0.4)
    axes[3].fill_betweenx(df_f[depth_col], 0, sw_data, color='blue', alpha=0.1)
    axes[3].set_xlim(1, 0)
    axes[3].set_xlabel('Sw (v/v)', color='red', fontweight='bold')

    # Track 5: Facies
    ax5 = axes[4]
    f_colors = {'Sand': 'yellow', 'Shale': 'green'}
    if facies_col in df_f.columns:
        for facies, color in f_colors.items():
            mask = df_f[facies_col] == facies
            ax5.fill_betweenx(df_f[depth_col], 0, 1, where=mask, color=color, alpha=0.8)
    ax5.set_xlabel('Facies', fontweight='bold')
    ax5.set_xticks([])

    plt.tight_layout()
    return fig

# --- REPLACED WITH YOUR CORRECTED CROSSPLOT CODE ---
def plot_professional_crossplots(df, gamma_ray_col='GR', resistivity_col='ILD', 
                               density_col='RHOB', nphi_col='NPHI', figsize=(14, 6)):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Crossplot 1: GR vs Porosity (NPHI) colored by Facies
    sand_mask = df['Facies'] == 'Sand'
    shale_mask = df['Facies'] == 'Shale'
    
    # Plot sand points
    ax1.scatter(df.loc[sand_mask, nphi_col], df.loc[sand_mask, gamma_ray_col], 
                c='yellow', s=50, edgecolors='black', linewidth=0.5, 
                alpha=0.7, label='Sand', marker='o')
    
    # Plot shale points
    ax1.scatter(df.loc[shale_mask, nphi_col], df.loc[shale_mask, gamma_ray_col], 
                c='green', s=50, edgecolors='black', linewidth=0.5, 
                alpha=0.7, label='Shale', marker='s')
    
    # Add GR baseline
    baseline = df['GR_Baseline'].iloc[0] if 'GR_Baseline' in df.columns else 75
    ax1.axhline(y=baseline, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'GR Baseline ({baseline} API)')
    
    ax1.set_xlabel('Neutron Porosity - NPHI (v/v)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Gamma Ray - GR (API)', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(df[nphi_col].min() - 0.05, df[nphi_col].max() + 0.05)
    ax1.set_ylim(0, 150)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_title('GR vs NPHI Colored by Facies', fontsize=11, fontweight='bold')
    
    # Crossplot 2: Density vs Neutron colored by GR
    sc2 = ax2.scatter(df[nphi_col], df[density_col], c=df[gamma_ray_col], 
                      cmap='viridis', s=50, edgecolors='black', linewidth=0.5, 
                      alpha=0.7, vmin=0, vmax=150)
    
    ax2.set_xlabel('Neutron Porosity - NPHI (v/v)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Bulk Density - RHOB (g/cc)', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(df[nphi_col].min() - 0.05, df[nphi_col].max() + 0.05)
    ax2.set_ylim(df[density_col].min() - 0.1, df[density_col].max() + 0.1)
    
    # Add color bar
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label('Gamma Ray (API)', fontsize=9)
    
    # Add sand line (typical density-neutron crossover for sandstone)
    x_line = np.linspace(df[nphi_col].min(), df[nphi_col].max(), 100)
    y_line = 2.65 - 0.8 * x_line  # Approximate sandstone trend
    ax2.plot(x_line, y_line, 'r--', linewidth=1.5, alpha=0.7, label='Sandstone Trend')
    
    ax2.legend(fontsize=9, loc='upper right')
    ax2.set_title('Density-Neutron Crossplot Colored by GR', fontsize=11, fontweight='bold')
    
    plt.suptitle('Lithology and Petrophysical Crossplots', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig

# Function to generate comprehensive petrophysical summary table
def display_petrophysical_summary(df, gamma_ray_col, resistivity_col, start_depth, end_depth):
    
    # Filter data by depth for summary
    df_filtered = filter_by_depth(df, 'DEPT', start_depth, end_depth)
    
    # Calculate reservoir averages for sand zones only
    sand_mask = df_filtered['Facies'] == 'Sand'
    sand_df = df_filtered[sand_mask]
    pay_mask = df_filtered['Pay_Flag'] == 1
    pay_df = df_filtered[pay_mask]
    
    # Calculate statistics
    if len(sand_df) > 0:
        sand_gr_avg = sand_df[gamma_ray_col].mean()
        sand_rt_avg = sand_df[resistivity_col].mean()
        sand_phi_avg = sand_df['Porosity_Effective'].mean()
        sand_sw_avg = sand_df['Sw_Archie'].mean()
        sand_sh_avg = sand_df['Sh'].mean()
        sand_perm_med = sand_df['Perm_Timur'].median()
        sand_vsh_avg = sand_df['Vsh_Larionov'].mean()
    else:
        sand_gr_avg = sand_rt_avg = sand_phi_avg = sand_sw_avg = sand_sh_avg = sand_perm_med = sand_vsh_avg = 0
    
    if len(pay_df) > 0:
        pay_thickness = pay_df['DEPT'].max() - pay_df['DEPT'].min()
    else:
        pay_thickness = 0
    
    summary_data = {
        'Parameter': [
            'Gamma Ray (API)',
            'Resistivity (Ω.m)',
            'Effective Porosity (v/v)',
            'Water Saturation (v/v)',
            'Hydrocarbon Saturation (v/v)',
            'Permeability (mD)',
            'Volume of Shale (v/v)',
            'Net-to-Gross Ratio',
            'Reservoir Thickness (m)',
            'Pay Thickness (m)'
        ],
        'Whole Interval': [
            f"{df_filtered[gamma_ray_col].mean():.1f}",
            f"{df_filtered[resistivity_col].mean():.1f}",
            f"{df_filtered['Porosity_Effective'].mean():.3f}",
            f"{df_filtered['Sw_Archie'].mean():.3f}",
            f"{df_filtered['Sh'].mean():.3f}",
            f"{df_filtered['Perm_Timur'].median():.2f}",
            f"{df_filtered['Vsh_Larionov'].mean():.3f}",
            f"{df_filtered['NTG'].mean():.3f}",
            f"{(df_filtered['DEPT'].max() - df_filtered['DEPT'].min()):.1f}",
            f"{pay_thickness:.1f}"
        ],
        'Sand Zones Only': [
            f"{sand_gr_avg:.1f}",
            f"{sand_rt_avg:.1f}",
            f"{sand_phi_avg:.3f}",
            f"{sand_sw_avg:.3f}",
            f"{sand_sh_avg:.3f}",
            f"{sand_perm_med:.2f}",
            f"{sand_vsh_avg:.3f}",
            "1.000",
            f"{(sand_df['DEPT'].max() - sand_df['DEPT'].min()):.1f}" if len(sand_df) > 0 else "0.0",
            f"{pay_thickness:.1f}"
        ],
        'Units': [
            "API",
            "Ω.m",
            "v/v",
            "v/v",
            "v/v",
            "mD",
            "v/v",
            "ratio",
            "meters",
            "meters"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Display with Streamlit
    st.write("### 📊 Petrophysical Summary Report")
    st.write(f"**Depth Interval:** {start_depth:.1f}m - {end_depth:.1f}m")
    
    # Create a styled table
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Display interpretation statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sand Zones", f"{len(sand_df)}", 
                  f"{len(sand_df)/len(df_filtered)*100:.1f}%")
    
    with col2:
        st.metric("Pay Zones", f"{len(pay_df)}", 
                  f"{len(pay_df)/len(df_filtered)*100:.1f}%")
    
    with col3:
        avg_porosity = df_filtered['Porosity_Effective'].mean()
        st.metric("Avg Porosity", f"{avg_porosity:.3f}")
    
    with col4:
        avg_perm = df_filtered['Perm_Timur'].median()
        st.metric("Avg Permeability", f"{avg_perm:.2f} mD")
    
    # Display cutoffs
    st.markdown("---")
    st.write("**Interpretation Cutoffs:**")
    
    cutoff_col1, cutoff_col2, cutoff_col3 = st.columns(3)
    
    with cutoff_col1:
        st.write("• **GR:** < 75 API for sand")
        st.write("• **Rt:** > 10 Ω.m for HC")
    
    with cutoff_col2:
        st.write("• **Porosity:** > 0.10 for reservoir")
        st.write("• **Sw:** < 0.60 for pay")
    
    with cutoff_col3:
        st.write("• **Vsh:** < 0.40 for clean sand")
        st.write("• **NTG:** > 0.50 for economic")
    
    return summary_df

# Function to generate PDF report
def generate_pdf_report(df, summary_df, fig_logs, fig_petro, fig_crossplot, 
                        start_depth, end_depth, filename="well_petrophysical_report.pdf"):
    
    # Create a BytesIO buffer for the PDF
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                            rightMargin=72, leftMargin=72, 
                            topMargin=72, bottomMargin=72)
    
    # Get styles
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = styles["Title"]
    title_style.alignment = 1  # Center alignment
    story.append(Paragraph("<b>WELL PETROPHYSICAL ANALYSIS REPORT</b>", title_style))
    story.append(Spacer(1, 12))
    
    # Report metadata
    meta_style = styles["Normal"]
    meta_style.alignment = 1
    story.append(Paragraph(f"<b>Depth Interval:</b> {start_depth:.1f}m - {end_depth:.1f}m", meta_style))
    story.append(Paragraph(f"<b>Date Generated:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", meta_style))
    story.append(Spacer(1, 20))
    
    # Summary section
    story.append(Paragraph("<b>PETROPHYSICAL SUMMARY</b>", styles["Heading2"]))
    story.append(Spacer(1, 12))
    
    # Create summary table
    summary_data = [summary_df.columns.tolist()] + summary_df.values.tolist()
    summary_table = Table(summary_data)
    
    # Style the table
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ])
    
    summary_table.setStyle(table_style)
    story.append(summary_table)
    story.append(Spacer(1, 30))
    
    # Add statistics
    sand_df = df[df['Facies'] == 'Sand']
    pay_df = df[df['Pay_Flag'] == 1]
    
    stats_text = f"""
    <b>Reservoir Statistics:</b><br/>
    • Total Data Points: {len(df):,}<br/>
    • Sand Zones: {len(sand_df):,} ({len(sand_df)/len(df)*100:.1f}%)<br/>
    • Pay Zones: {len(pay_df):,} ({len(pay_df)/len(df)*100:.1f}%)<br/>
    • Average Porosity: {df['Porosity_Effective'].mean():.3f}<br/>
    • Average Water Saturation: {df['Sw_Archie'].mean():.3f}<br/>
    • Median Permeability: {df['Perm_Timur'].median():.2f} mD<br/>
    """
    
    story.append(Paragraph(stats_text, styles["Normal"]))
    story.append(Spacer(1, 20))
    
    # Interpretation notes
    story.append(Paragraph("<b>INTERPRETATION NOTES</b>", styles["Heading2"]))
    story.append(Spacer(1, 12))
    
    notes_text = """
    1. Facies determination based on Gamma Ray cutoff (Sand: GR < 75 API, Shale: GR ≥ 75 API)<br/>
    2. Water saturation calculated using Archie's equation with Rw = 0.1 Ω.m<br/>
    3. Permeability estimated using Timur equation<br/>
    4. Pay zones defined as: Sand facies, Φ > 0.10, and Sw < 0.60<br/>
    5. Net-to-Gross calculated based on sand percentage in interval<br/>
    """
    
    story.append(Paragraph(notes_text, styles["Normal"]))
    
    doc.build(story)
    buffer.seek(0)
    
    return buffer

# Main Streamlit app
def main():
    # Initialize session state for start/end depth
    if 'start_depth' not in st.session_state:
        st.session_state.start_depth = None
    if 'end_depth' not in st.session_state:
        st.session_state.end_depth = None
    
    # Header with logo and title
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        try:
            st.image("download.jpg", use_container_width=True)
        except:
            st.info("Logo placeholder (download.jpg)")
    
    st.markdown("---")
    
    # Title and description with dark theme
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>⛏️ Well Log Analysis Dashboard</h1>", unsafe_allow_html=True)
    
    # About section with dark theme
    st.markdown("""
    <div class='about-section'>
    <h4>About this Application</h4>
    <p>This professional-grade application provides comprehensive well log analysis capabilities for geoscientists and petrophysicists. 
    It includes advanced petrophysical calculations, customizable visualization tools, and reservoir characterization features.</p>
    
    <strong>Key Features:</strong>
    <ul>
        <li>📈 Multi-track well log visualization with depth control</li>
        <li>🧪 Advanced petrophysical calculations (Archie, Timur, Larionov)</li>
        <li>🎨 Professional crossplots and facies analysis</li>
        <li>📊 Comprehensive summary reports with industry cutoffs</li>
        <li>💾 Data export capabilities including PDF reports</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Load default dataset or state
    df = None
    example_file = 'x1.las'
    
    # Sidebar configuration
    st.sidebar.markdown("<h2 style='color: #1E3A8A;'>⚙️ Analysis Parameters</h2>", unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("📤 Upload LAS File", type=["las"])
    
    if uploaded_file is not None:
        df = load_las(uploaded_file)
        if df is not None:
            st.sidebar.success("File uploaded successfully!")
    elif st.sidebar.button("📊 Use Example Dataset"):
        try:
            df = load_las(example_file)
            st.sidebar.info("Example dataset loaded")
        except:
            st.sidebar.error("Example file x1.las not found.")
    
    # If no data loaded yet, attempt to load example automatically
    if df is None:
        try:
            df = load_las(example_file)
        except:
            pass

    if df is not None:
        # Depth range selection
        st.sidebar.markdown("### 📏 Depth Range Selection")
        depth_min = float(df['DEPT'].min())
        depth_max = float(df['DEPT'].max())
        
        # Initialize session state values if not set
        if st.session_state.start_depth is None or st.session_state.start_depth < depth_min:
            st.session_state.start_depth = depth_min
        if st.session_state.end_depth is None or st.session_state.end_depth > depth_max:
            st.session_state.end_depth = depth_max
        
        start_depth = st.sidebar.number_input("Start Depth (m)", 
                                             value=st.session_state.start_depth, 
                                             min_value=depth_min, 
                                             max_value=depth_max, 
                                             step=10.0,
                                             key="start_depth_input")
        
        end_depth = st.sidebar.number_input("End Depth (m)", 
                                            value=st.session_state.end_depth, 
                                            min_value=depth_min, 
                                            max_value=depth_max, 
                                            step=10.0,
                                            key="end_depth_input")
        
        # Update session state
        st.session_state.start_depth = start_depth
        st.session_state.end_depth = end_depth
        
        # Column selection
        st.sidebar.markdown("### 🎯 Log Curve Selection")
        columns = df.columns.tolist()
        
        # Try to find default columns
        default_gr = 'GR' if 'GR' in columns else columns[0]
        default_cali = 'CALI' if 'CALI' in columns else columns[1] if len(columns) > 1 else columns[0]
        default_rt = 'ILD' if 'ILD' in columns else columns[2] if len(columns) > 2 else columns[0]
        default_rhob = 'RHOB' if 'RHOB' in columns else columns[3] if len(columns) > 3 else columns[0]
        default_nphi = 'NPHI' if 'NPHI' in columns else columns[4] if len(columns) > 4 else columns[0]
        default_sonic = 'DT' if 'DT' in columns else columns[5] if len(columns) > 5 else columns[0]
        
        gamma_ray_col = st.sidebar.selectbox("Gamma-Ray Column", columns, index=columns.index(default_gr))
        cali_col = st.sidebar.selectbox("Caliper Column", columns, index=columns.index(default_cali))
        resistivity_col = st.sidebar.selectbox("Resistivity Column", columns, index=columns.index(default_rt))
        density_col = st.sidebar.selectbox("Density Column", columns, index=columns.index(default_rhob))
        nphi_col = st.sidebar.selectbox("Neutron Porosity Column", columns, index=columns.index(default_nphi))
        sonic_col = st.sidebar.selectbox("Sonic (DT)", columns, index=columns.index(default_sonic))
        
        # Petrophysical parameters
        st.sidebar.markdown("### 🧪 Petrophysical Parameters")
        gr_baseline = st.sidebar.slider("GR Baseline for Facies (API)", 50, 100, 75)
        
        # Filtering and Calculations
        df_filtered = filter_by_depth(df, 'DEPT', start_depth, end_depth)
        df_filtered = determine_facies(df_filtered, gamma_ray_col, gr_baseline)
        df_filtered, gr_min, gr_max = perform_petrophysics(df_filtered, gamma_ray_col, 
                                                           resistivity_col, density_col, 
                                                           nphi_col, gr_baseline)
        
        # Main content area
        st.markdown("---")
        st.markdown("<h2 style='color: #1E3A8A;'>📋 Data Preview</h2>", unsafe_allow_html=True)
        
        # Display data statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Depth Interval", f"{end_depth - start_depth:.1f} m")
        with col2:
            st.metric("Data Points", f"{len(df_filtered):,}")
        with col3:
            sand_count = (df_filtered['Facies'] == 'Sand').sum()
            st.metric("Sand Zones", f"{sand_count}")
        with col4:
            st.metric("Shale Zones", f"{len(df_filtered) - sand_count}")
        
        st.dataframe(df_filtered.head(10), use_container_width=True)
        
        # Section 1: Professional Well Log Plot
        st.markdown("---")
        st.markdown("<h2 style='color: #1E3A8A;'>📈 Well Log Display</h2>", unsafe_allow_html=True)
        st.write(f"**Depth Range:** {start_depth:.1f}m - {end_depth:.1f}m")
        
        fig_logs = plot_well_logs(df_filtered, 'DEPT', gamma_ray_col, cali_col, 
                                  resistivity_col, density_col, nphi_col, sonic_col,
                                  start_depth, end_depth)
        st.pyplot(fig_logs)
        
        # Section 2: Petrophysical Analysis with Facies
        st.markdown("---")
        st.markdown("<h2 style='color: #1E3A8A;'>🧪 Petrophysical Analysis Plot</h2>", unsafe_allow_html=True)
        
        fig_petro = plot_petrophysics_with_facies(df_filtered, 'DEPT', gamma_ray_col, 
                                                  'Porosity_Effective', 'Vsh_Larionov', 
                                                  'Sw_Archie', 'Facies', 
                                                  start_depth, end_depth)
        st.pyplot(fig_petro)
        
        # Section 3: Professional Crossplots
        st.markdown("---")
        st.markdown("<h2 style='color: #1E3A8A;'>📊 Petrophysical Crossplots</h2>", unsafe_allow_html=True)
        
        fig_crossplot = plot_professional_crossplots(df_filtered, gamma_ray_col, 
                                                 resistivity_col, density_col, nphi_col)
        st.pyplot(fig_crossplot)
        
        # Section 4: Comprehensive Summary
        st.markdown("---")
        summary_df = display_petrophysical_summary(df_filtered, gamma_ray_col, 
                                                   resistivity_col, start_depth, end_depth)
        
        # Section 5: Export Options
        st.markdown("---")
        st.markdown("<h2 style='color: #1E3A8A;'>💾 Export Results</h2>", unsafe_allow_html=True)
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            if st.button("💾 Save All Plots"):
                fig_logs.savefig("professional_well_logs.png", dpi=300, bbox_inches='tight')
                fig_petro.savefig("petrophysical_analysis.png", dpi=300, bbox_inches='tight')
                fig_crossplot.savefig("crossplots.png", dpi=300, bbox_inches='tight')
                st.success("Plots saved as high-resolution PNG files!")
                st.balloons()
        
        with export_col2:
            if st.button("📊 Export Processed Data"):
                df_filtered.to_csv("processed_well_data.csv", index=False)
                st.success("Data exported to CSV file!")
                st.balloons()
        
        with export_col3:
            if st.button("📄 Generate PDF Report"):
                pdf_buffer = generate_pdf_report(df_filtered, summary_df, fig_logs, 
                                                fig_petro, fig_crossplot, 
                                                start_depth, end_depth)
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_buffer,
                    file_name="well_petrophysical_report.pdf",
                    mime="application/pdf"
                )
                st.success("PDF report generated!")
                st.balloons()
    
    else:
        st.warning("Please upload a LAS file or load the example dataset to begin analysis.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit | Professional Well Log Analysis v2.0</p>
    <p>© 2024 Elijah & Joshua | Petroleum Data Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()