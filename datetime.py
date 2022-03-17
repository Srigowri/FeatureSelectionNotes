import pandas as pd

#consider that data has a date column for date of birth
data['dob'] = pd.to_datetime(data['dob'])
data['dob_day'] = data['dob'].dt.day
data['dob_month'] = data['dob'].dt.month
data['dob_dayofweek'] = data['dob'].dt.dayofweek
data['dob_year'] = data['dob'].dt.year
data['dob_quarter'] = data['dob'].dt.quarter
data['dob_weekday_name'] = data['dob'].dt.weekday_name
data['dob_is_weekend'] = np.where(data['weekday_name].isin(['Saturday', 'Sunday']),1,0)
                                       
#date difference
data['graduation'] - data['dob']
                                       
#to be added in future
#date time second formatting, differences etc.                                       
                                       
                                
                                

