# ChatGPTAI_TEST
---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.6
  nbformat: 4
  nbformat_minor: 2
---

::: {.cell .markdown}
# Test of chatgptAI and its capabilities

I am utilizing a dataset that I found online which has information on
different loans and many variables associated with each loan
:::

::: {.cell .code execution_count="44"}
``` python
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
llm = OpenAI(api_token="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")
pandas_ai = PandasAI(llm)
df = pd.read_csv('colum.csv')
df
```

::: {.output .stream .stderr}
    /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/pandasai/__init__.py:145: UserWarning: `PandasAI` (class) is deprecated since v1.0 and will be removed in a future release. Please use `SmartDataframe` instead.
      warnings.warn(
:::

::: {.output .execute_result execution_count="44"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>member_id</th>
      <th>CLI</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>...</th>
      <th>il_util</th>
      <th>open_rv_12m</th>
      <th>open_rv_24m</th>
      <th>max_bal_bc</th>
      <th>all_util</th>
      <th>total_rev_hi_lim</th>
      <th>inq_fi</th>
      <th>total_cu_tl</th>
      <th>inq_last_12m</th>
      <th>Unnamed: 69</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1077501</td>
      <td>1296599</td>
      <td>5000</td>
      <td>36 months</td>
      <td>10.65</td>
      <td>162.87</td>
      <td>B</td>
      <td>B2</td>
      <td>NaN</td>
      <td>10+ years</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1077430</td>
      <td>1314167</td>
      <td>2500</td>
      <td>60 months</td>
      <td>15.27</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>Ryder</td>
      <td>&lt; 1 year</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1077175</td>
      <td>1313524</td>
      <td>2400</td>
      <td>36 months</td>
      <td>15.96</td>
      <td>84.33</td>
      <td>C</td>
      <td>C5</td>
      <td>NaN</td>
      <td>10+ years</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1076863</td>
      <td>1277178</td>
      <td>10000</td>
      <td>36 months</td>
      <td>13.49</td>
      <td>339.31</td>
      <td>C</td>
      <td>C1</td>
      <td>AIR RESOURCES BOARD</td>
      <td>10+ years</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1075358</td>
      <td>1311748</td>
      <td>3000</td>
      <td>60 months</td>
      <td>12.69</td>
      <td>67.79</td>
      <td>B</td>
      <td>B5</td>
      <td>University Medical Group</td>
      <td>1 year</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1068575</td>
      <td>1303001</td>
      <td>15300</td>
      <td>60 months</td>
      <td>22.06</td>
      <td>423.10</td>
      <td>F</td>
      <td>F4</td>
      <td>OSSI</td>
      <td>6 years</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>96</th>
      <td>1049528</td>
      <td>1280928</td>
      <td>12800</td>
      <td>60 months</td>
      <td>11.71</td>
      <td>282.86</td>
      <td>B</td>
      <td>B3</td>
      <td>NCS Technologies</td>
      <td>4 years</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>97</th>
      <td>1068542</td>
      <td>1303143</td>
      <td>17500</td>
      <td>60 months</td>
      <td>17.27</td>
      <td>437.47</td>
      <td>D</td>
      <td>D3</td>
      <td>Travelers Insurance</td>
      <td>7 years</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>98</th>
      <td>1068350</td>
      <td>1302971</td>
      <td>3500</td>
      <td>36 months</td>
      <td>6.03</td>
      <td>106.53</td>
      <td>A</td>
      <td>A1</td>
      <td>J&amp;J Steel Inc</td>
      <td>10+ years</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1067874</td>
      <td>1302235</td>
      <td>6000</td>
      <td>60 months</td>
      <td>12.69</td>
      <td>135.57</td>
      <td>B</td>
      <td>B5</td>
      <td>Anadarko Petroleum Corporation</td>
      <td>&lt; 1 year</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 70 columns</p>
</div>
```
:::
:::

::: {.cell .markdown}
# Data cleanup and analysis

I am now using GPT to cleanup the data and create a new column where it
attemts to predict the likelihood of credit default with the data in the
dataset
:::

::: {.cell .code execution_count="8"}
``` python
instruction = "create an index to represent the likelihood of credit default in the dataset and add a column to the dataset that represents this index for each ID and remove all empty columns"
new_df = pandas_ai.run(df, instruction)
path_new_df = "/Users/ateskursunoglu/Desktop/new_df.csv"
new_df.to_csv(path_new_df)
```
:::

::: {.cell .markdown}
# Visualizations

After the data cleanup and creation of the credit default likelihood
index, I am now using GPT to visualize correlations between the new
index and other values in order to understand the data and their
relationships better.
:::

::: {.cell .code execution_count="10"}
``` python
pandas_ai.run(path_new_df, "create a heatmap that shows the 5 closest correlated variables to the credit default likelihood")
```

::: {.output .stream .stderr}
    <string>:20: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
    <string>:20: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
:::

::: {.output .display_data}
![](vertopal_8e93063348f44fe69a2c6359f32303fd/e1bca63c94b2206a3bae60ce528e7cfd267f445c.png)
:::
:::

::: {.cell .code execution_count="12"}
``` python
pandas_ai.run(path_new_df, "create a heatmap that shows the 15 closest correlated variables to the credit default likelihood and set the figure size to a large size in order for the data to be readable")
```

::: {.output .stream .stderr}
    <string>:20: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
    <string>:20: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
:::

::: {.output .display_data}
![](vertopal_8e93063348f44fe69a2c6359f32303fd/af7847f32a20803c63074f29b99c46ce8653f2d5.png)
:::
:::

::: {.cell .markdown}
Now that I have got some idea of the corellations, I am taking a deeper
dive into the variables relationships.
:::

::: {.cell .code execution_count="16"}
``` python
pandas_ai.run(path_new_df, "use some form of visualization to show the correlation between the default likelihood and oustanding principal amount of the loan")
```

::: {.output .display_data}
![](vertopal_8e93063348f44fe69a2c6359f32303fd/13eec752415b30d67c918b39fb85954207c8d00f.png)
:::
:::

::: {.cell .markdown}
This correlation did not seen to lead anywhere so I am now looking at
alternate variables.
:::

::: {.cell .code execution_count="18"}
``` python
pandas_ai.run(path_new_df, "use some form of visualization to show the correlation between the default likelihood and the recoveries variable")
```

::: {.output .display_data}
![](vertopal_8e93063348f44fe69a2c6359f32303fd/4abcd59e97d76ace8a996a67e513a0515acd01e4.png)
:::
:::

::: {.cell .code execution_count="19"}
``` python
pandas_ai.run(path_new_df, "how many ID's have a recoveries value of 0")
```

::: {.output .execute_result execution_count="19"}
    85
:::
:::

::: {.cell .code execution_count="26"}
``` python
pandas_ai.run(path_new_df, "remove all ID's with a recoveries value of 0 and then visualize the relationship between the remaining recoveries and default likelihood")
```

::: {.output .stream .stderr}
    <string>:21: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    <string>:21: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
:::

::: {.output .display_data}
![](vertopal_8e93063348f44fe69a2c6359f32303fd/81b580675c812ecad858e4734e55a7dfe81702d7.png)
:::
:::

::: {.cell .code execution_count="22"}
``` python
pandas_ai.run(path_new_df, "how many ID's have a default likelihood of 1")
```

::: {.output .execute_result execution_count="22"}
    18
:::
:::

::: {.cell .code execution_count="24"}
``` python
pandas_ai.run(path_new_df, "remove all ID's with a recoveries value of not 0 and then visualize the relationship between the remaining recoveries and default likelihood")
```

::: {.output .stream .stderr}
    <string>:21: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    <string>:21: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
:::

::: {.output .display_data}
![](vertopal_8e93063348f44fe69a2c6359f32303fd/3ad5c33b27c0e661712a1bf486e7a43b991ced7c.png)
:::
:::

::: {.cell .code execution_count="30"}
``` python
pandas_ai.run(path_new_df, "show the amount of ID's with a recoveries value of 0 that have a default likelihood of 0 vs the amount of ID's with a recoveries value of 0 that have a default likelihood of 1 using a bar graph")
```

::: {.output .display_data}
![](vertopal_8e93063348f44fe69a2c6359f32303fd/3e3b9eb4b6b1e418e936f1e0f260b4b23e68fe26.png)
:::
:::

::: {.cell .code execution_count="29"}
``` python
pandas_ai.run(path_new_df, "would it make sense to assume that if an ID has a recoveries value of 0, they are highly likely to default on their credit")
```

::: {.output .execute_result execution_count="29"}
    'The percentage of IDs with recoveries = 0 that default on their credit is 3.5294117647058822%.'
:::
:::

::: {.cell .markdown}
This correlation also seems to have not led to any significant
conclusions.
:::

::: {.cell .markdown}
# Final Conclusions

I was not able to conclude much but the process that will be carried out
with the real data will be similar. A continuous attempt to create
correlations in data and make connections accordingly.

The difficulty with this dataset was that there was no historic data on
any credits that have defaulted. It was simply the credit lines that
were open. This made it difficult to analyse.

In future implimentations I think it would be a good idea to utilize
machine ChatGPT API to create more datapoints and then running machine
learning models such as Random Forest and Decision Tree to make
connections and then use ChatGPT AI again to dive deeper into the
conclusions drawn from the machine learning models.

I think the combination of both machine learning and ChatGPT AI would be
very powerful for meaningful conclusions on the real data.
:::
