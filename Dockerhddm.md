# Dataset


```python
%matplotlib inline

# scitnific computing and plotting
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

# HDDM related packages
import pymc as pm
import hddm
import kabuki
import arviz as az
print("The current HDDM version is: ", hddm.__version__)
print("The current kabuki version is: ", kabuki.__version__)
print("The current PyMC version is: ", pm.__version__)
print("The current ArviZ version is: ", az.__version__)
```

    The current HDDM version is:  1.0.1RC
    The current kabuki version is:  0.6.5RC4
    The current PyMC version is:  2.3.8
    The current ArviZ version is:  0.15.1


## Data Manipulation and Data Clearning


```python
data = hddm.load_csv('/home/jovyan/work/NbackData.csv', skiprows=1)
```


```python
data_rename = data.rename(columns={
    "Stim.RT": "rt",      
    "Subject": "subj_idx", 
    "Stim.ACC": "response" 
})
```


```python
data_select_col = data_rename[['subj_idx', 'rt', 'response', 'BlockType','StimType','TargetType']]
```


```python
data_filtered = data_select_col[(data_rename['rt'] > 150)].dropna()
```


```python
dataset = pd.DataFrame()  # Initialize an empty DataFrame
dataset=data_filtered[['subj_idx', 'rt', 'response', 'BlockType','StimType','TargetType']]
dataset['rt'] = data_filtered['rt'] / 1000  # Convert 'rt' to seconds
dataset['Stim_cog_emo'] = np.where(data_filtered['StimType'].isin(["NegFace", "PosFace"]), "Affective", "Cognitive")

# Print the new dataset
print(dataset)
```

           subj_idx     rt  response BlockType StimType TargetType Stim_cog_emo
    2         11684  1.340       1.0    0-Back  NegFace    nonlure    Affective
    3         11684  1.835       1.0    0-Back  NegFace    nonlure    Affective
    4         11684  1.544       1.0    0-Back  NegFace     target    Affective
    5         11684  1.167       1.0    0-Back  NegFace       lure    Affective
    6         11684  0.999       1.0    0-Back  NegFace    nonlure    Affective
    ...         ...    ...       ...       ...      ...        ...          ...
    46344     12484  0.636       1.0    0-Back  NegFace     target    Affective
    46345     12484  0.438       1.0    0-Back  NegFace       lure    Affective
    46346     12484  0.338       1.0    0-Back  NegFace    nonlure    Affective
    46347     12484  0.393       1.0    0-Back  NegFace       lure    Affective
    46348     12484  0.716       1.0    0-Back  NegFace    nonlure    Affective
    
    [34888 rows x 7 columns]



```python
print("The number of trials: ", dataset.shape[0])
print("The number of variables: ", dataset.shape[1])  # This should now work
print("The number of participants: ", dataset['subj_idx'].unique().shape[0])  # Ensure subj_idx is in dataset

```

    The number of trials:  34888
    The number of variables:  7
    The number of participants:  259



```python

data = hddm.utils.flip_errors(dataset)

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='RT', ylabel='count', title='RT distributions')
for i, subj_data in data.groupby('subj_idx'):
    subj_data.rt.hist(bins=20, histtype='step', ax=ax)
```


    
![png](output_9_0.png)
    


# Modeling

## Prior Check


```python
hddm.model_config.model_config["ddm"]
```




    {'doc': 'Basic DDM. Meant for use with the LAN extension. \nNote that the boundaries here are coded as -a, and a in line with all other models meant for the LAN extension. \nTo compare model fits between standard HDDM and HDDMnn when using the DDM model, multiply the boundary (a) parameter by 2. \nWe recommend using standard HDDM if you are interested in the basic DDM, but you might want to use this for testing.',
     'params': ['v', 'a', 'z', 't'],
     'params_trans': [0, 0, 1, 0],
     'params_std_upper': [1.5, 1.0, None, 1.0],
     'param_bounds': [[-3.0, 0.3, 0.1, 0.001], [3.0, 2.5, 0.9, 2.0]],
     'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
     'params_default': [0.0, 1.0, 0.5, 0.001],
     'hddm_include': ['v', 'a', 't', 'z'],
     'choices': [-1, 1],
     'slice_widths': {'v': 1.5,
      'v_std': 1,
      'a': 1,
      'a_std': 1,
      'z': 0.1,
      'z_trans': 0.2,
      't': 0.01,
      't_std': 0.15}}



## Model for individual with BlockType * Stim_cog_emo conditions


```python
models = []
# Fit 5 HDDM models
for i in range(5):
    m = hddm.HDDM(dataset, 
                  include=['a', 'v', 't','z'], 
                  depends_on={'v': ['BlockType', 'Stim_cog_emo'],
                              'a': ['BlockType', 'Stim_cog_emo'],
                              't': ['BlockType', 'Stim_cog_emo'],
                              'z': ['BlockType', 'Stim_cog_emo']},
                  informative=True,
                  is_group_model=True,
                  )
    
    # Find starting values and sample from the posterior
    m.find_starting_values()
    m.sample(10000, burn=4000)
    
    # Append the fitted model to the list
    models.append(m)

# Perform Gelman-Rubin diagnostic
gelman_rubin_results = hddm.analyze.gelman_rubin(models)

```

    No model attribute --> setting up standard HDDM
    Set model to ddm


    /opt/conda/lib/python3.9/site-packages/scipy/optimize/_optimize.py:2309: RuntimeWarning: invalid value encountered in double_scalars
      tmp2 = (x - v) * (fx - fw)


     [-----------------100%-----------------] 10001 of 10000 complete in 16212.2 sechddm sampling elpased time:  16217.849 s
    No model attribute --> setting up standard HDDM
    Set model to ddm


    /opt/conda/lib/python3.9/site-packages/scipy/optimize/_optimize.py:2309: RuntimeWarning: invalid value encountered in double_scalars
      tmp2 = (x - v) * (fx - fw)


     [-----------------100%-----------------] 10001 of 10000 complete in 13017.9 sechddm sampling elpased time:  13023.783 s
    No model attribute --> setting up standard HDDM
    Set model to ddm


    /opt/conda/lib/python3.9/site-packages/scipy/optimize/_optimize.py:2309: RuntimeWarning: invalid value encountered in double_scalars
      tmp2 = (x - v) * (fx - fw)


     [-----------------100%-----------------] 10001 of 10000 complete in 12962.8 sechddm sampling elpased time:  12968.495 s
    No model attribute --> setting up standard HDDM
    Set model to ddm


    /opt/conda/lib/python3.9/site-packages/scipy/optimize/_optimize.py:2309: RuntimeWarning: invalid value encountered in double_scalars
      tmp2 = (x - v) * (fx - fw)


     [-----------------100%-----------------] 10001 of 10000 complete in 12945.2 sechddm sampling elpased time:  12950.974 s
    No model attribute --> setting up standard HDDM
    Set model to ddm


    /opt/conda/lib/python3.9/site-packages/scipy/optimize/_optimize.py:2309: RuntimeWarning: invalid value encountered in double_scalars
      tmp2 = (x - v) * (fx - fw)


     [-----------------100%-----------------] 10001 of 10000 complete in 14930.3 sechddm sampling elpased time:  14936.114 s


## Result


```python
# @title Print R_hat for convergence check
# Convert the Gelman-Rubin results to a DataFrame
gelman_rubin_results = pd.DataFrame.from_dict(gelman_rubin_results, orient='index', columns=['Gelman-Rubin'])

# Reset the index so the 'subject_id' becomes a column
gelman_rubin_results.reset_index(inplace=True)

# Rename the index column to 'subject_id'
gelman_rubin_results.rename(columns={'index': 'subject_id'}, inplace=True)

# Save the DataFrame to a CSV
gelman_rubin_results.to_csv('gelman_rubin_results.csv', index=False)
```


```python
# @title Print Model Summery
for i, model in enumerate(models):
    print(f"Model {i+1} statistics:")
    print(model.gen_stats())
    
model_stats_summary= pd.DataFrame(model.gen_stats())
model_stats_summary.reset_index(inplace=True)

# Rename the index column to 'subject_id'
model_stats_summary.rename(columns={'index': 'subject_id'}, inplace=True)

# Save the DataFrame to a CSV file
model_stats_summary.to_csv('model_stats_summary.csv', index=False) 
```

    Model 1 statistics:
                                        mean       std      2.5q       25q   
    a(0-Back.Affective)             1.781864  0.022344  1.737911  1.767027  \
    a(0-Back.Cognitive)             1.826426  0.022773  1.781723  1.810951   
    a(2-Back.Affective)             1.776499  0.021315  1.734625  1.762223   
    a(2-Back.Cognitive)             1.795977  0.021695  1.753145  1.781228   
    a_std                           0.248541  0.011336  0.227285  0.240862   
    ...                                  ...       ...       ...       ...   
    z_subj(2-Back.Cognitive).12480  0.468095   0.01454  0.439349  0.458967   
    z_subj(2-Back.Cognitive).12481  0.466743  0.014547  0.437383  0.458029   
    z_subj(2-Back.Cognitive).12482  0.466965  0.014397  0.436995  0.458359   
    z_subj(2-Back.Cognitive).12483  0.466702  0.014485  0.437851  0.457947   
    z_subj(2-Back.Cognitive).12484  0.461627  0.014894  0.429142  0.453272   
    
                                         50q       75q     97.5q    mc err  
    a(0-Back.Affective)              1.78142  1.796817  1.826159  0.000786  
    a(0-Back.Cognitive)             1.826399  1.841409  1.872049  0.000775  
    a(2-Back.Affective)             1.776444  1.791256   1.81877  0.000661  
    a(2-Back.Cognitive)             1.795973  1.810319  1.839107  0.000765  
    a_std                           0.248142  0.255648  0.271863  0.000542  
    ...                                  ...       ...       ...       ...  
    z_subj(2-Back.Cognitive).12480  0.467824  0.476903  0.498422  0.000469  
    z_subj(2-Back.Cognitive).12481  0.466716  0.475493  0.496498  0.000469  
    z_subj(2-Back.Cognitive).12482   0.46712   0.47584  0.495219  0.000436  
    z_subj(2-Back.Cognitive).12483  0.466686   0.47565  0.495741  0.000474  
    z_subj(2-Back.Cognitive).12484  0.462768  0.471004   0.48899  0.000547  
    
    [4164 rows x 8 columns]
    Model 2 statistics:
                                        mean       std      2.5q       25q   
    a(0-Back.Affective)             1.782006  0.023183  1.736944  1.766305  \
    a(0-Back.Cognitive)             1.829251  0.023003  1.784613  1.813459   
    a(2-Back.Affective)             1.777174  0.021602  1.734561  1.762784   
    a(2-Back.Cognitive)             1.796508  0.020958  1.755354  1.782132   
    a_std                           0.249285  0.011271   0.22828  0.241414   
    ...                                  ...       ...       ...       ...   
    z_subj(2-Back.Cognitive).12480  0.468727  0.016335  0.438915  0.458606   
    z_subj(2-Back.Cognitive).12481  0.467007  0.015987  0.435878  0.457396   
    z_subj(2-Back.Cognitive).12482  0.467467  0.015594  0.437722  0.457718   
    z_subj(2-Back.Cognitive).12483  0.466672  0.015948  0.435456  0.457096   
    z_subj(2-Back.Cognitive).12484  0.461409  0.016015  0.427135  0.452396   
    
                                         50q       75q     97.5q    mc err  
    a(0-Back.Affective)             1.781857  1.797853  1.827344  0.000854  
    a(0-Back.Cognitive)             1.829108  1.844878  1.874602  0.000854  
    a(2-Back.Affective)             1.776928  1.791293  1.820708  0.000669  
    a(2-Back.Cognitive)             1.796474   1.81032  1.838553  0.000619  
    a_std                           0.248993  0.256597  0.271943  0.000537  
    ...                                  ...       ...       ...       ...  
    z_subj(2-Back.Cognitive).12480  0.467548  0.477812  0.505305  0.000691  
    z_subj(2-Back.Cognitive).12481  0.466224  0.476393  0.500757  0.000639  
    z_subj(2-Back.Cognitive).12482  0.466813   0.47638  0.500885  0.000641  
    z_subj(2-Back.Cognitive).12483  0.465737  0.475698   0.50153  0.000617  
    z_subj(2-Back.Cognitive).12484  0.462048  0.471173  0.492116  0.000655  
    
    [4164 rows x 8 columns]
    Model 3 statistics:
                                        mean       std      2.5q       25q   
    a(0-Back.Affective)             1.781055  0.022627  1.735413  1.765887  \
    a(0-Back.Cognitive)             1.825693  0.023197  1.780637  1.810374   
    a(2-Back.Affective)             1.775022  0.021661  1.733473  1.759878   
    a(2-Back.Cognitive)             1.795315  0.020803   1.75481  1.781058   
    a_std                             0.2483  0.011291  0.226394  0.240442   
    ...                                  ...       ...       ...       ...   
    z_subj(2-Back.Cognitive).12480  0.469094  0.014868  0.439741  0.460212   
    z_subj(2-Back.Cognitive).12481  0.468103  0.014815  0.439204  0.459535   
    z_subj(2-Back.Cognitive).12482  0.468263  0.015089  0.438296  0.459553   
    z_subj(2-Back.Cognitive).12483  0.467866  0.014638  0.439651  0.459183   
    z_subj(2-Back.Cognitive).12484  0.462706  0.015051  0.428745  0.454746   
    
                                         50q       75q     97.5q    mc err  
    a(0-Back.Affective)             1.781464  1.796182  1.825539  0.000764  
    a(0-Back.Cognitive)             1.825525  1.840551  1.873472   0.00084  
    a(2-Back.Affective)             1.774977  1.789755  1.818008  0.000609  
    a(2-Back.Cognitive)             1.795088  1.808844  1.837033  0.000584  
    a_std                           0.248309  0.255741  0.270735  0.000514  
    ...                                  ...       ...       ...       ...  
    z_subj(2-Back.Cognitive).12480  0.468084  0.477459  0.502344  0.000558  
    z_subj(2-Back.Cognitive).12481   0.46741  0.476076  0.500141  0.000529  
    z_subj(2-Back.Cognitive).12482  0.467681   0.47676  0.500814  0.000535  
    z_subj(2-Back.Cognitive).12483  0.467153  0.475868  0.498768  0.000512  
    z_subj(2-Back.Cognitive).12484  0.463662  0.471727  0.490148  0.000642  
    
    [4164 rows x 8 columns]
    Model 4 statistics:
                                        mean       std      2.5q       25q   
    a(0-Back.Affective)             1.782355  0.022767   1.73812  1.766704  \
    a(0-Back.Cognitive)              1.82801  0.023051  1.781208  1.813093   
    a(2-Back.Affective)             1.776998  0.021363  1.735771  1.762572   
    a(2-Back.Cognitive)             1.796238  0.021457  1.754878  1.781517   
    a_std                           0.249903   0.01095  0.229285   0.24232   
    ...                                  ...       ...       ...       ...   
    z_subj(2-Back.Cognitive).12480   0.46858  0.017344   0.43607  0.457437   
    z_subj(2-Back.Cognitive).12481  0.467341  0.017493  0.435938  0.455944   
    z_subj(2-Back.Cognitive).12482  0.467533   0.01732   0.43539  0.456291   
    z_subj(2-Back.Cognitive).12483  0.466625  0.016816  0.434568  0.455995   
    z_subj(2-Back.Cognitive).12484  0.459662  0.016979  0.425184  0.449311   
    
                                         50q       75q     97.5q    mc err  
    a(0-Back.Affective)             1.782779  1.797829   1.82597  0.000722  
    a(0-Back.Cognitive)             1.828237  1.843359  1.873099  0.000841  
    a(2-Back.Affective)             1.776778  1.791602  1.818899  0.000617  
    a(2-Back.Cognitive)              1.79635  1.811149  1.837495  0.000639  
    a_std                           0.249548  0.257178  0.272116  0.000457  
    ...                                  ...       ...       ...       ...  
    z_subj(2-Back.Cognitive).12480  0.467873  0.479007   0.50499  0.000576  
    z_subj(2-Back.Cognitive).12481   0.46659  0.477436  0.506034  0.000565  
    z_subj(2-Back.Cognitive).12482   0.46687  0.478275  0.504432  0.000529  
    z_subj(2-Back.Cognitive).12483  0.466044  0.476778  0.501708  0.000514  
    z_subj(2-Back.Cognitive).12484  0.459776  0.470454  0.492986  0.000496  
    
    [4164 rows x 8 columns]
    Model 5 statistics:
                                        mean       std      2.5q       25q   
    a(0-Back.Affective)             1.781784  0.022648  1.737657  1.766428  \
    a(0-Back.Cognitive)             1.827839  0.023888  1.781292  1.811682   
    a(2-Back.Affective)             1.775864  0.021752  1.732559  1.761385   
    a(2-Back.Cognitive)             1.797441  0.021217  1.756718  1.782906   
    a_std                           0.249653  0.011585  0.228181  0.241695   
    ...                                  ...       ...       ...       ...   
    z_subj(2-Back.Cognitive).12480  0.467923   0.01649  0.439928  0.457197   
    z_subj(2-Back.Cognitive).12481  0.466613   0.01645  0.436325  0.455903   
    z_subj(2-Back.Cognitive).12482  0.466395  0.016267  0.436029  0.456182   
    z_subj(2-Back.Cognitive).12483  0.465764  0.016426  0.436028  0.455305   
    z_subj(2-Back.Cognitive).12484  0.460024  0.015979  0.426375  0.450761   
    
                                         50q       75q     97.5q    mc err  
    a(0-Back.Affective)             1.781616  1.796857  1.825764  0.000764  
    a(0-Back.Cognitive)             1.827678  1.844096  1.875354  0.000956  
    a(2-Back.Affective)             1.776008  1.790512   1.81869  0.000628  
    a(2-Back.Cognitive)             1.796964  1.811758  1.839475  0.000598  
    a_std                           0.249064  0.257376   0.27409  0.000537  
    ...                                  ...       ...       ...       ...  
    z_subj(2-Back.Cognitive).12480  0.466169  0.476949  0.506093  0.000692  
    z_subj(2-Back.Cognitive).12481  0.465304  0.476315  0.502983  0.000644  
    z_subj(2-Back.Cognitive).12482  0.464912  0.475477  0.501541  0.000608  
    z_subj(2-Back.Cognitive).12483  0.464341  0.475181  0.502142  0.000572  
    z_subj(2-Back.Cognitive).12484  0.459766   0.46924  0.492573  0.000527  
    
    [4164 rows x 8 columns]



```python
# @title Merge Dataset
merged_models = pd.merge(model_stats_summary, gelman_rubin_results, on='subject_id', how='inner')  # Use 'inner' for only matching rows

# Save the merged DataFrame to a CSV file
merged_models.to_csv('merged_models.csv', index=False)
print(merged_models)
```

                              subject_id      mean       std      2.5q       25q   
    0                a(0-Back.Affective)  1.781784  0.022648  1.737657  1.766428  \
    1                a(0-Back.Cognitive)  1.827839  0.023888  1.781292  1.811682   
    2                a(2-Back.Affective)  1.775864  0.021752  1.732559  1.761385   
    3                a(2-Back.Cognitive)  1.797441  0.021217  1.756718  1.782906   
    4                              a_std  0.249653  0.011585  0.228181  0.241695   
    ...                              ...       ...       ...       ...       ...   
    3119  t_subj(2-Back.Cognitive).12481  0.334402  0.035259  0.259956  0.311477   
    3120  t_subj(2-Back.Cognitive).12482  0.320944  0.032633  0.251125   0.29991   
    3121  t_subj(2-Back.Cognitive).12483  0.315376  0.030851  0.249179  0.295918   
    3122  t_subj(2-Back.Cognitive).12484  0.269148  0.032726   0.20757  0.248076   
    3123                           z_std  0.056869  0.023631  0.020065  0.039027   
    
               50q       75q     97.5q    mc err  Gelman-Rubin  
    0     1.781616  1.796857  1.825764  0.000764      1.000181  
    1     1.827678  1.844096  1.875354  0.000956      1.002104  
    2     1.776008  1.790512   1.81869  0.000628      1.000923  
    3     1.796964  1.811758  1.839475  0.000598      1.000723  
    4     0.249064  0.257376   0.27409  0.000537      1.002188  
    ...        ...       ...       ...       ...           ...  
    3119  0.336815  0.359618  0.395799  0.000822      1.000308  
    3120  0.323261  0.344378  0.377316  0.000746      1.000047  
    3121  0.318204  0.337305  0.367193  0.000563      1.000168  
    3122  0.267672  0.287419  0.341912  0.000456      1.000142  
    3123  0.052718  0.074315  0.105436  0.002311      1.032339  
    
    [3124 rows x 10 columns]



```python
filtered_data = merged_models[
    merged_models['subject_id'].isin([
        'a(0-Back.Affective)', 
        'a(0-Back.Cognitive)', 
        'a(2-Back.Affective)', 
        'a(2-Back.Cognitive)',
        'v(0-Back.Affective)',
        'v(0-Back.Cognitive)',
        'v(2-Back.Affective)', 
        'v(2-Back.Cognitive)',
        't(0-Back.Affective)', 
        't(0-Back.Cognitive)', 
        't(2-Back.Affective)', 
        't(2-Back.Cognitive)',
        'z(0-Back.Affective)',
        'z(0-Back.Cognitive)',
        'z(2-Back.Affective)', 
        'z(2-Back.Cognitive)',
    ])]
print(filtered_data)
```

                   subject_id      mean       std      2.5q       25q       50q   
    0     a(0-Back.Affective)  1.781784  0.022648  1.737657  1.766428  1.781616  \
    1     a(0-Back.Cognitive)  1.827839  0.023888  1.781292  1.811682  1.827678   
    2     a(2-Back.Affective)  1.775864  0.021752  1.732559  1.761385  1.776008   
    3     a(2-Back.Cognitive)  1.797441  0.021217  1.756718  1.782906  1.796964   
    1041  v(0-Back.Affective)  1.708849  0.055056  1.603041  1.672002  1.708405   
    1042  v(0-Back.Cognitive)  1.913756  0.055841  1.806287  1.874761  1.913222   
    1043  v(2-Back.Affective)  1.281662  0.053605  1.177186  1.245258  1.281697   
    1044  v(2-Back.Cognitive)  1.289031  0.053785  1.182983  1.252846  1.289183   
    2082  t(0-Back.Affective)  0.322684  0.006343  0.310026  0.318439  0.322744   
    2083  t(0-Back.Cognitive)  0.319041  0.006588  0.306074  0.314609  0.319017   
    2084  t(2-Back.Affective)  0.366622  0.007028  0.352604  0.361925  0.366589   
    2085  t(2-Back.Cognitive)  0.380898  0.007018  0.366855  0.376201   0.38097   
    
               75q     97.5q    mc err  Gelman-Rubin  
    0     1.796857  1.825764  0.000764      1.000181  
    1     1.844096  1.875354  0.000956      1.002104  
    2     1.790512   1.81869  0.000628      1.000923  
    3     1.811758  1.839475  0.000598      1.000723  
    1041  1.746029  1.816038   0.00167      1.001507  
    1042  1.951455  2.026222  0.001855      1.003493  
    1043  1.317676  1.386356  0.001577      1.000459  
    1044  1.325667  1.393067  0.001635      1.000708  
    2082  0.327044  0.334693  0.000207      1.001150  
    2083  0.323625  0.331694  0.000215      1.005052  
    2084  0.371561  0.380034  0.000216      1.000516  
    2085  0.385551  0.394905  0.000245      1.001058  



```python
# Create a bar plot for the mean values
plt.figure(figsize=(10, 6))
plt.bar(filtered_data['subject_id'], filtered_data['mean'], yerr=filtered_data['std'], capsize=5)
plt.xticks(rotation=45)
plt.xlabel('Condition')
plt.ylabel('Mean')
plt.title('Mean Values Across Conditions with Standard Deviation')
plt.tight_layout()
plt.show()
```


    
![png](output_20_0.png)
    


# Posterior Visualization


```python
#Visual convergence check

for i, model in enumerate(models):
    print(f"Plotting posteriors for Model {i+1}...")
    model.plot_posteriors(['a', 't', 'v'])
    plt.title(f"Model {i+1} Posteriors for 'a', 't', 'v','z'")
    plt.show()
```

    Plotting posteriors for Model 1...
    Plotting a(0-Back.Affective)
    Plotting a(0-Back.Cognitive)
    Plotting a(2-Back.Affective)
    Plotting a(2-Back.Cognitive)
    Plotting v(0-Back.Affective)
    Plotting v(0-Back.Cognitive)
    Plotting v(2-Back.Affective)
    Plotting v(2-Back.Cognitive)
    Plotting t(0-Back.Affective)
    Plotting t(0-Back.Cognitive)
    Plotting t(2-Back.Affective)
    Plotting t(2-Back.Cognitive)



    
![png](output_22_1.png)
    



    
![png](output_22_2.png)
    



    
![png](output_22_3.png)
    



    
![png](output_22_4.png)
    



    
![png](output_22_5.png)
    



    
![png](output_22_6.png)
    



    
![png](output_22_7.png)
    



    
![png](output_22_8.png)
    



    
![png](output_22_9.png)
    



    
![png](output_22_10.png)
    



    
![png](output_22_11.png)
    



    
![png](output_22_12.png)
    


    Plotting posteriors for Model 2...
    Plotting a(0-Back.Affective)
    Plotting a(0-Back.Cognitive)
    Plotting a(2-Back.Affective)
    Plotting a(2-Back.Cognitive)
    Plotting v(0-Back.Affective)
    Plotting v(0-Back.Cognitive)
    Plotting v(2-Back.Affective)
    Plotting v(2-Back.Cognitive)
    Plotting t(0-Back.Affective)
    Plotting t(0-Back.Cognitive)
    Plotting t(2-Back.Affective)
    Plotting t(2-Back.Cognitive)



    
![png](output_22_14.png)
    



    
![png](output_22_15.png)
    



    
![png](output_22_16.png)
    



    
![png](output_22_17.png)
    



    
![png](output_22_18.png)
    



    
![png](output_22_19.png)
    



    
![png](output_22_20.png)
    



    
![png](output_22_21.png)
    



    
![png](output_22_22.png)
    



    
![png](output_22_23.png)
    



    
![png](output_22_24.png)
    



    
![png](output_22_25.png)
    


    Plotting posteriors for Model 3...
    Plotting a(0-Back.Affective)
    Plotting a(0-Back.Cognitive)
    Plotting a(2-Back.Affective)
    Plotting a(2-Back.Cognitive)
    Plotting v(0-Back.Affective)
    Plotting v(0-Back.Cognitive)
    Plotting v(2-Back.Affective)
    Plotting v(2-Back.Cognitive)
    Plotting t(0-Back.Affective)
    Plotting t(0-Back.Cognitive)
    Plotting t(2-Back.Affective)
    Plotting t(2-Back.Cognitive)



    
![png](output_22_27.png)
    



    
![png](output_22_28.png)
    



    
![png](output_22_29.png)
    



    
![png](output_22_30.png)
    



    
![png](output_22_31.png)
    



    
![png](output_22_32.png)
    



    
![png](output_22_33.png)
    



    
![png](output_22_34.png)
    



    
![png](output_22_35.png)
    



    
![png](output_22_36.png)
    



    
![png](output_22_37.png)
    



    
![png](output_22_38.png)
    


    Plotting posteriors for Model 4...
    Plotting a(0-Back.Affective)
    Plotting a(0-Back.Cognitive)
    Plotting a(2-Back.Affective)
    Plotting a(2-Back.Cognitive)
    Plotting v(0-Back.Affective)
    Plotting v(0-Back.Cognitive)
    Plotting v(2-Back.Affective)
    Plotting v(2-Back.Cognitive)
    Plotting t(0-Back.Affective)
    Plotting t(0-Back.Cognitive)
    Plotting t(2-Back.Affective)
    Plotting t(2-Back.Cognitive)



    
![png](output_22_40.png)
    



    
![png](output_22_41.png)
    



    
![png](output_22_42.png)
    



    
![png](output_22_43.png)
    



    
![png](output_22_44.png)
    



    
![png](output_22_45.png)
    



    
![png](output_22_46.png)
    



    
![png](output_22_47.png)
    



    
![png](output_22_48.png)
    



    
![png](output_22_49.png)
    



    
![png](output_22_50.png)
    



    
![png](output_22_51.png)
    


    Plotting posteriors for Model 5...
    Plotting a(0-Back.Affective)
    Plotting a(0-Back.Cognitive)
    Plotting a(2-Back.Affective)
    Plotting a(2-Back.Cognitive)
    Plotting v(0-Back.Affective)
    Plotting v(0-Back.Cognitive)
    Plotting v(2-Back.Affective)
    Plotting v(2-Back.Cognitive)
    Plotting t(0-Back.Affective)
    Plotting t(0-Back.Cognitive)
    Plotting t(2-Back.Affective)
    Plotting t(2-Back.Cognitive)



    
![png](output_22_53.png)
    



    
![png](output_22_54.png)
    



    
![png](output_22_55.png)
    



    
![png](output_22_56.png)
    



    
![png](output_22_57.png)
    



    
![png](output_22_58.png)
    



    
![png](output_22_59.png)
    



    
![png](output_22_60.png)
    



    
![png](output_22_61.png)
    



    
![png](output_22_62.png)
    



    
![png](output_22_63.png)
    



    
![png](output_22_64.png)
    


the parameters of the DDM map onto different cognitive processes: speed-accuracy settings (boundary separation a), response bias (starting point z), information processing speed (drift rate d), and non-decision time (Ter). These parameters are sometimes called “free parameters,” in the sense that they can take on different values (“freely”) – and just like the knobs on a stereo, changing each parameter affects DDM behavior.


```python
v_0Back_Aff, v_0Back_Cog = model.nodes_db.node[['v(0-Back.Affective)', 'v(0-Back.Cognitive)']]
v_2Back_Aff, v_2Back_Cog = model.nodes_db.node[['v(2-Back.Affective)', 'v(2-Back.Cognitive)']]

# Plot the posterior distributions for each condition
hddm.analyze.plot_posterior_nodes([v_0Back_Cog, v_0Back_Aff, v_2Back_Cog, v_2Back_Aff])

# Add labels and title to the plot
plt.xlabel('Drift-rate')
plt.ylabel('Posterior Probability')
plt.title('Posterior of Drift-rate Group Means (0-Back and 2-Back, Cognitive vs. Affective)')

print("P(v.0-Back.Affective > v.0-Back.Cognitive) = ", (v_0Back_Aff.trace() > v_0Back_Cog.trace()).mean())
print("P(v.0-Back.Cognitive > v.0-Back.Affective) = ", (v_0Back_Cog.trace() > v_0Back_Aff.trace()).mean())

print("P(v.2-Back.Affective > v.2-Back.Cognitive) = ", (v_2Back_Aff.trace() > v_2Back_Cog.trace()).mean())
print("P(v.2-Back.Cognitive > v.2-Back.Affective) = ", (v_2Back_Cog.trace() > v_2Back_Aff.trace()).mean())

print("P(v.0-Back.Affective > v.2-Back.Affective) = ", (v_0Back_Aff.trace() > v_2Back_Aff.trace()).mean())
print("P(v.0-Back.Cognitive > v.2-Back.Cognitive) = ", (v_0Back_Cog.trace() > v_2Back_Cog.trace()).mean())

print("P(v.2-Back.Affective > v.0-Back.Affective) = ", (v_2Back_Aff.trace() > v_0Back_Aff.trace()).mean())
print("P(v.2-Back.Cognitive > v.0-Back.Cognitive) = ", (v_2Back_Cog.trace() > v_0Back_Cog.trace()).mean())

```

    P(v.0-Back.Affective > v.0-Back.Cognitive) =  0.004333333333333333
    P(v.0-Back.Cognitive > v.0-Back.Affective) =  0.9956666666666667
    P(v.2-Back.Affective > v.2-Back.Cognitive) =  0.4578333333333333
    P(v.2-Back.Cognitive > v.2-Back.Affective) =  0.5421666666666667
    P(v.0-Back.Affective > v.2-Back.Affective) =  1.0
    P(v.0-Back.Cognitive > v.2-Back.Cognitive) =  1.0
    P(v.2-Back.Affective > v.0-Back.Affective) =  0.0
    P(v.2-Back.Cognitive > v.0-Back.Cognitive) =  0.0



    
![png](output_24_1.png)
    


By convention, the lower boundary is assigned a value of 0 on the y-axis and distance to the upper boundary is defined by a parameter representing boundary separation (a). Larger values of a mean that the decision-making process must travel further (up or down) to reach a boundary. The effect of larger a is thus that decision-making will be slower and more cautious: slower because more evidence is required before a distant boundary is reached and a response is triggered, and higher accuracy because it will be rare for the decision process to “mistakenly” cross the wrong boundary (Lerche et al., 2020). Boundary separation is in arbitrary units, but is often assumed to range from about 0.5–2. It is often assumed that the degree of boundary separation is at least partly under conscious control, depending on whether there is an emphasis on speed (low a) or accuracy (high a).


```python
a_0Back_Aff, a_0Back_Cog = model.nodes_db.node[['a(0-Back.Affective)', 'a(0-Back.Cognitive)']]
a_2Back_Aff, a_2Back_Cog = model.nodes_db.node[['a(2-Back.Affective)', 'a(2-Back.Cognitive)']]

# Plot the posterior distributions for each condition
hddm.analyze.plot_posterior_nodes([a_0Back_Cog, a_0Back_Aff, a_2Back_Cog, a_2Back_Aff])

# Add labels and title to the plot
plt.xlabel('Bounday')
plt.ylabel('Posterior Probability')
plt.title('Bounday Group Means (0-Back and 2-Back, Cognitive vs. Affective)')

print("P(a.0-Back.Affective > a.0-Back.Cognitive) = ", (a_0Back_Aff.trace() > a_0Back_Cog.trace()).mean())
print("P(a.0-Back.Cognitive > a.0-Back.Affective) = ", (a_0Back_Cog.trace() > a_0Back_Aff.trace()).mean())

print("P(a.2-Back.Affective > a.2-Back.Cognitive) = ", (a_2Back_Aff.trace() > a_2Back_Cog.trace()).mean())
print("P(a.2-Back.Cognitive > a.2-Back.Affective) = ", (a_2Back_Cog.trace() > a_2Back_Aff.trace()).mean())

print("P(a.0-Back.Affective > a.2-Back.Affective) = ", (a_0Back_Aff.trace() > a_2Back_Aff.trace()).mean())
print("P(a.0-Back.Cognitive > a.2-Back.Cognitive) = ", (a_0Back_Cog.trace() > a_2Back_Cog.trace()).mean())

print("P(a.2-Back.Affective > a.0-Back.Affective) = ", (a_2Back_Aff.trace() > a_0Back_Aff.trace()).mean())
print("P(a.2-Back.Cognitive > a.0-Back.Cognitive) = ", (a_2Back_Cog.trace() > a_0Back_Cog.trace()).mean())
```

    P(a.0-Back.Affective > a.0-Back.Cognitive) =  0.0725
    P(a.0-Back.Cognitive > a.0-Back.Affective) =  0.9275
    P(a.2-Back.Affective > a.2-Back.Cognitive) =  0.2385
    P(a.2-Back.Cognitive > a.2-Back.Affective) =  0.7615
    P(a.0-Back.Affective > a.2-Back.Affective) =  0.575
    P(a.0-Back.Cognitive > a.2-Back.Cognitive) =  0.8396666666666667
    P(a.2-Back.Affective > a.0-Back.Affective) =  0.425
    P(a.2-Back.Cognitive > a.0-Back.Cognitive) =  0.16033333333333333



    
![png](output_26_1.png)
    



```python
t_0Back_Aff, t_0Back_Cog = model.nodes_db.node[['t(0-Back.Affective)', 't(0-Back.Cognitive)']]
t_2Back_Aff, t_2Back_Cog = model.nodes_db.node[['t(2-Back.Affective)', 't(2-Back.Cognitive)']]

# Plot the posterior distributions for each condition
hddm.analyze.plot_posterior_nodes([t_0Back_Cog, t_0Back_Aff, t_2Back_Cog, t_2Back_Aff])

# Add labels and title to the plot
plt.xlabel('None-Decisional Time')
plt.ylabel('Posterior Probability')
plt.title('None-Decisional Time Group Means (0-Back and 2-Back, Cognitive vs. Affective)')

print("P(t.0-Back.Affective > t.0-Back.Cognitive) = ", (t_0Back_Aff.trace() > t_0Back_Cog.trace()).mean())
print("P(t.0-Back.Cognitive > t.0-Back.Affective) = ", (t_0Back_Cog.trace() > t_0Back_Aff.trace()).mean())

print("P(t.2-Back.Affective > t.2-Back.Cognitive) = ", (t_2Back_Aff.trace() > t_2Back_Cog.trace()).mean())
print("P(t.2-Back.Cognitive > t.2-Back.Affective) = ", (t_2Back_Cog.trace() > t_2Back_Aff.trace()).mean())

print("P(t.0-Back.Affective > t.2-Back.Affective) = ", (t_0Back_Aff.trace() > t_2Back_Aff.trace()).mean())
print("P(t.0-Back.Cognitive > t.2-Back.Cognitive) = ", (t_0Back_Cog.trace() > t_2Back_Cog.trace()).mean())

print("P(t.2-Back.Affective > t.0-Back.Affective) = ", (t_2Back_Aff.trace() > t_0Back_Aff.trace()).mean())
print("P(t.2-Back.Cognitive > t.0-Back.Cognitive) = ", (t_2Back_Cog.trace() > t_0Back_Cog.trace()).mean())

```

    P(t.0-Back.Affective > t.0-Back.Cognitive) =  0.6578333333333334
    P(t.0-Back.Cognitive > t.0-Back.Affective) =  0.3421666666666667
    P(t.2-Back.Affective > t.2-Back.Cognitive) =  0.07866666666666666
    P(t.2-Back.Cognitive > t.2-Back.Affective) =  0.9213333333333333
    P(t.0-Back.Affective > t.2-Back.Affective) =  0.0
    P(t.0-Back.Cognitive > t.2-Back.Cognitive) =  0.0
    P(t.2-Back.Affective > t.0-Back.Affective) =  1.0
    P(t.2-Back.Cognitive > t.0-Back.Cognitive) =  1.0



    
![png](output_27_1.png)
    



```python
z_0Back_Aff, z_0Back_Cog = model.nodes_db.node[['z(0-Back.Affective)', 'z(0-Back.Cognitive)']]
z_2Back_Aff, z_2Back_Cog = model.nodes_db.node[['z(2-Back.Affective)', 'z(2-Back.Cognitive)']]

# Plot the posterior distributions for each condition
hddm.analyze.plot_posterior_nodes([z_0Back_Aff, z_0Back_Cog, z_2Back_Aff, z_2Back_Cog])

# Add labels and title to the plot
plt.xlabel('Decision Boundary (z)')
plt.ylabel('Posterior Probability')
plt.title('Decision Boundary Group Means (0-Back and 2-Back, Cognitive vs. Affective)')
plt.show()

# Calculate probabilities for the 0-Back conditions
print("P(z.0-Back.Affective > z.0-Back.Cognitive) = ", (z_0Back_Aff.trace() > z_0Back_Cog.trace()).mean())
print("P(z.0-Back.Cognitive > z.0-Back.Affective) = ", (z_0Back_Cog.trace() > z_0Back_Aff.trace()).mean())

# Calculate probabilities for the 2-Back conditions
print("P(z.2-Back.Affective > z.2-Back.Cognitive) = ", (z_2Back_Aff.trace() > z_2Back_Cog.trace()).mean())
print("P(z.2-Back.Cognitive > z.2-Back.Affective) = ", (z_2Back_Cog.trace() > z_2Back_Aff.trace()).mean())

print("P(z.0-Back.Affective > z.2-Back.Affective) = ", (z_0Back_Aff.trace() > z_2Back_Aff.trace()).mean())
print("P(z.0-Back.Cognitive > z.2-Back.Cognitive) = ", (z_0Back_Cog.trace() > z_2Back_Cog.trace()).mean())

print("P(z.2-Back.Affective > z.0-Back.Affective) = ", (z_2Back_Aff.trace() > z_0Back_Aff.trace()).mean())
print("P(z.2-Back.Cognitive > z.0-Back.Cognitive) = ", (z_2Back_Cog.trace() > z_0Back_Cog.trace()).mean())
```


    
![png](output_28_0.png)
    


    P(z.0-Back.Affective > z.0-Back.Cognitive) =  1.0
    P(z.0-Back.Cognitive > z.0-Back.Affective) =  0.0
    P(z.2-Back.Affective > z.2-Back.Cognitive) =  0.8946666666666667
    P(z.2-Back.Cognitive > z.2-Back.Affective) =  0.10533333333333333
    P(z.0-Back.Affective > z.2-Back.Affective) =  1.0
    P(z.0-Back.Cognitive > z.2-Back.Cognitive) =  0.31783333333333336
    P(z.2-Back.Affective > z.0-Back.Affective) =  0.0
    P(z.2-Back.Cognitive > z.0-Back.Cognitive) =  0.6821666666666667



```python

```
