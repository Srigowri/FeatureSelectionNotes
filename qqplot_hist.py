#checking for gaussian distribution of a feature

```
Variable transformation to check for gaussian distribution
1. Logorithm 
2. Exponential
3. Square root
4. Reciprocal
5. Box-cox
```


import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

def diagnostic_plots(df, variable):
    # function to plot a histogram and a Q-Q plot
    # side by side, for a certain variable
    
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist()

    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=pylab)

    plt.show()
    
diagnostic_plots(data, 'Height')

data['Height_log'] = np.log(data.Height)
diagnostic_plots(data, 'Height_log')

data['Height_reci'] = 1/(data.Height)
diagnostic_plots(data, 'Height_reci')

data['Height_exp'] = data.Height**(1/1.2)
diagnostic_plots(data, 'Height_exp')

data['Height_root'] = data.Height**(1/2)
diagnostic_plots(data, 'Height_root')

data['Height_boxcox'] = stats.boxcox(data.Height)
diagnostic_plots(data, 'Height_boxcox')


