import pandas as pd
data= {
     'Products Name' : ['Face Wash', 'Soap', 'Gel'],
     'Category' : ['A', 'B', 'A'],
     'Price' : [200, 190, 250]
     }
df = pd.DataFrame(data,index=["i", "ii", "iii"])
print(df)
print()
df['Discounted Price'] = df['Price'] * 0.90
print(df)
