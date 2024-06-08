import streamlit as st
import pandas as pd
import lasio
import plotly.graph_objs as go

# Load your LAS data
las = lasio.read("TMB-01_plan.las")
df = las.df()

# Streamlit app
st.title(':clap: Well Log Dashboard')

# Get the column names from the dataframe
log_columns = df.columns.tolist()

# Dropdown to select log type
log_type = st.selectbox(
    'Select Log Type',
    log_columns  # Use actual column names from the LAS file
)

# Slider to filter by depth
depth_min, depth_max = st.slider(
    'Select Depth Range',
    float(df.index.min()), float(df.index.max()), (float(df.index.min()), float(df.index.max()))
)

# Filter the data by depth
filtered_df = df[(df.index >= depth_min) & (df.index <= depth_max)]

# Plotting the selected log
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=filtered_df[log_type],
    y=filtered_df.index,
    mode='lines',
    name=log_type
))

# Specify the figure size using width and height
fig.update_layout(
    title=f'{log_type} Log',
    xaxis=dict(
        title=log_type,
        type='log'
    ),
    yaxis=dict(
        title='Depth',
        autorange='reversed'
    ),
    height=1000,  # Set the height of the figure
    width=400   # Set the width of the figure
)

st.plotly_chart(fig)