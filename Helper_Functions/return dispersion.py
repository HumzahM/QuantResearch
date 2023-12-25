import wrds
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

query = f"""
    SELECT date, SUM(NUMTRD) AS "total daily trades" 
    FROM crsp.dsf
    WHERE date BETWEEN '1925-01-01' AND '2022-12-30'
    GROUP BY date
    ORDER BY date;
    """

db = wrds.Connection(wrds_username='humzahm')
result = db.raw_sql(query, date_cols=['date'])
result = result[result['total daily trades'].notna()]
result['log total daily trades'] = np.log(result['total daily trades'])
plt.plot(result['date'], result['log total daily trades'])
plt.show()