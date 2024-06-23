import pandas as pd

file_path = 'data/results.xlsx'
xls = pd.ExcelFile(file_path)

sheets = {}
for sheet_name in xls.sheet_names:
    if sheet_name != "Values of y":
        sheets[sheet_name] = xls.parse(sheet_name)

# Convert the non-numeric columns to the index to facilitate numeric operations
for sheet_name, df in sheets.items():
    df.set_index(df.columns[0], inplace=True)

def process_sheet(df):
    # Adding a small value to each cell
    df += 0.00001
    # Calculate the mean of each column
    col_means = df.mean(axis=0)
    # Normalize each cell by the column mean
    norm_df = df.div(col_means, axis=1)
    # Calculate the mean of each row in the normalized dataframe
    row_means = norm_df.mean(axis=1)
    
    return col_means, norm_df, row_means

# Process each sheet
results = {}
for sheet_name, df in sheets.items():
    col_means, norm_df, row_means = process_sheet(df)
    results[sheet_name] = {
        'col_means': col_means,
        'norm_df': norm_df,
        'row_means': row_means
    }

# Extract the dataframes for further use
means_dfs = {k: v['col_means'] for k, v in results.items()}
norm_dfs = {k: v['norm_df'] for k, v in results.items()}
row_means_dfs = {k: v['row_means'] for k, v in results.items()}

# Create a new dataframe to hold the row means from all sheets
row_means_combined = pd.DataFrame()

# Collect row means from each sheet into the combined dataframe
for sheet_name, row_means in row_means_dfs.items():
    row_means_combined[sheet_name] = row_means.round(3)

# Save the combined row means to a new sheet in the Excel file
with pd.ExcelWriter('data/norm_results.xlsx') as writer:
    for sheet_name, df in sheets.items():
        df.to_excel(writer, sheet_name=f'{sheet_name}_original')
        means_dfs[sheet_name].to_excel(writer, sheet_name=f'{sheet_name}_col_means')
        norm_dfs[sheet_name].to_excel(writer, sheet_name=f'{sheet_name}_norm')
        row_means_dfs[sheet_name].to_excel(writer, sheet_name=f'{sheet_name}_row_means')
    row_means_combined.to_excel(writer, sheet_name='Combined_Row_Means')