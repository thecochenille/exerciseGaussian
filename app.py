#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import random
import streamlit as st








def create_radius_range(df,radius):
    ''' Adding and subtracting a radius value into the first column of the dataframe
    Input: dataframe with at least mone column
    Output: dataframe with column 'min' and 'max' added
    '''
    df['max'] = np.add(df['x'],radius)
    df['min'] = np.subtract(df['x'],radius)
    return df
    
    
def count_values_in_range(df):
    #creating my list of range from the df
    ranges = [(l, r) for l, r in zip(df['min'], df['max'])]
    
    #extracting x col as a list
    coordinates = df['x'].tolist()
    
    count_range = lambda l, r: sum(l <= x <= r for x in coordinates) - 1 # this function will count the elements in my list of coordinates within a given range
    ranges_count_dict = [{'range': f'({l}, {r})', 'count': count_range(l, r)} for l, r in ranges]
    ranges_df = pd.DataFrame(ranges_count_dict)
    
    ranges_df['x'] = coordinates
    ranges_df = ranges_df.drop('range', axis=1)
    return ranges_df


def bootstrap(df, n_sample, n_reps):
    min_n_sample = 2
    max_n_sample = len(df) - 10 # I set up this maximum n_sample (from looking at the plot distribution) because otherwise the boostrap doesn't make sense anymore as it samples almost all the values and will give the same mean or at least it becomes skewed.
    
    if not (min_n_sample <= n_sample <= max_n_sample):
        raise ValueError(f"the number of sample is not between {min_n_sample} and {max_n_sample}")
    bootstrap_replicates = [sum(random.sample(df['count'].tolist(), n_sample)) / n_sample for _ in range(n_reps)]

    return bootstrap_replicates
    
    
def SW_bootstrap(df):

    n_sample_range = range(2,100)
    n_reps = 1000
    p_vals = [stats.shapiro(bootstrap(df, x, n_reps))[1]
              for x in n_sample_range]
    p_vals_df = pd.DataFrame({'n_sample': n_sample_range, 'p_value': p_vals})
    p_vals_df['log p_value'] = -np.log10(p_vals_df['p_value'])
        
    
    return p_vals_df


def create_histogram(data):
    plt.figure(figsize=(5, 3))
    plt.hist(data, bins=40)
    plt.title("Histogram")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    return plt

def create_plot(data):
    plt.figure(figsize=(5, 3))
    plt.plot(final_pval['log p_value'],final_pval['n_sample'])
    plt.axvline(x=-np.log10(0.05), color='r', linestyle='--', label=f'Threshold: {-np.log10(0.05)}')

    plt.title("Shapiro-Wilk p-value by sample size")
    plt.xlabel("Sample size")
    plt.ylabel("-log10(p-value)")
    return plt




# Streamlit app
st.title("Let's identify the minimum sample size for this exercice")

# Add a sidebar for user input
st.sidebar.header("Written by Isabelle Vea")

st.sidebar.write(" In this app, you can visualize how the number of sample size defined in the exercice affects the resampling (boostrap) distribution. You can explore sample size and radius (even if the exercice is set at radius 5 (default)).")

st.sidebar.write(" 1. Set a low sample number such as 2 and look at the histogram, does it look normal?")
st.sidebar.write(" 2. Set a large sample number such as 50 or 100 and look at the histogram, does it look normal now?")





st.sidebar.header("Parameters")
st.write("Let's set the random seed for bootstraping")

random_seed = st.number_input(label="Random seed", value=42)
sample_size = st.sidebar.number_input("Sample Size", min_value=2, max_value=700, value=2)
radius = st.sidebar.number_input("Radius", min_value=2, max_value=100, value=5)


st.write("In this exercice, we have a one-dimensional dataset, we calculated the number of neighbors using a radius from these coordinates and sampled n coordinates, get the count of neighbors, average and repeated this for 1000 replicates.")

st.write("Question: What is the minimum sample size of coordinate sampling necessary so that the boostrapped average count has a Gaussion distribution?")


st.write("Let's load the dataset")

# Generate data based on user input
data_load_state = st.text('Loading data...')

data = pd.read_csv("dataset.csv", header=None)
data.rename(columns={0: 'x'}, inplace=True)

data_load_state.text('Loading data...done!')



data_range = create_radius_range(data,radius)
counts = count_values_in_range(data)


random_seed=random_seed
bootstrap_list =bootstrap(counts, sample_size, 1000)

st.subheader(f"Explore resampling distribution")

st.write(f"This is the bootstrap distribution for sample size: {sample_size} and radius: {radius} ")

histogram = create_histogram(bootstrap_list)
st.pyplot(histogram)

random_seed=random_seed
final_pval = SW_bootstrap(counts)



st.subheader(f"Identify minimum sample size")
st.write(f"To identify the minimum sample size where for the average count to have a resampled normal distribution, I tested the normality of all resampling made in a sample size range between 2 and 100 (to make it faster as we know that above this number the distribution is Gaussian.)")


lineplot = create_plot(final_pval)
st.pyplot(lineplot)


minsample1=final_pval[final_pval['p_value'] > 0.05].sort_values(by='n_sample').head(1)

minsample2=final_pval[final_pval['p_value'] > 0.01].sort_values(by='n_sample').head(1)

st.write(f"The minimum sample size to have a boostrap with a normal distribution for p-value = 0.05:")
st.write(minsample1)


st.write(f"The minimum sample size to have a boostrap with a normal distribution for p-value = 0.01:")
st.write(minsample2)


