import pandas as pd
import os


"""
CSV Aggregation and Data Merging Script

INPUT:
- Two PAM50 summary CSV files: ST and Visium gene expression data
- ST_Image_CSV_paths.csv: Contains image, coordinate, and gene expression file paths

OUTPUT:
- aggregated_pam50_summary.csv: Combined ST + Visium data with source labels
- merged_cleaned.csv: Final dataset with paths and PAM50 classifications matched by sample ID

PROCESS:
1. Aggregates two PAM50 CSV files vertically, adding 'source' column (ST/Visium)
2. Normalizes sample IDs from both datasets (removes extensions, '_count' suffix, lowercase)
3. Merges path data with PAM50 classifications based on matching sample IDs
4. Outputs clean dataset with image paths and corresponding quality/subtype annotations
"""


# First, aggregate the two CSV files
def aggregate_csvs(file1_path, file2_path, output_path, add_source_column=True):
    """
    Aggregate two CSV files by stacking them vertically
    
    Args:
        file1_path: Path to first CSV file
        file2_path: Path to second CSV file  
        output_path: Path for the output aggregated CSV
        add_source_column: Whether to add a column indicating source file
    
    Returns:
        pandas.DataFrame: The aggregated DataFrame
    """
    
    print("Reading first CSV file...")
    df1 = pd.read_csv(file1_path)
    
    print("Reading second CSV file...")
    df2 = pd.read_csv(file2_path)
    
    print(f"File 1 shape: {df1.shape}")
    print(f"File 2 shape: {df2.shape}")
    
    # Verify columns match
    if list(df1.columns) != list(df2.columns):
        print("Warning: Column names don't match exactly")
        print(f"File 1 columns: {list(df1.columns)}")
        print(f"File 2 columns: {list(df2.columns)}")
        
        # Use intersection of columns
        common_columns = list(set(df1.columns) & set(df2.columns))
        df1 = df1[common_columns]
        df2 = df2[common_columns]
        print(f"Using common columns: {common_columns}")
    
    # Add source column if requested
    if add_source_column:
        df1['source'] = 'ST'
        df2['source'] = 'Visium'
    
    # Concatenate the DataFrames
    print("Aggregating DataFrames...")
    aggregated_df = pd.concat([df1, df2], ignore_index=True)
    
    # Save the result
    print(f"Saving aggregated data to {output_path}...")
    aggregated_df.to_csv(output_path, index=False)
    
    print(f"Aggregation complete!")
    print(f"Total rows: {len(aggregated_df)}")
    print(f"File 1 rows: {len(df1)}")
    print(f"File 2 rows: {len(df2)}")
    print(f"Output shape: {aggregated_df.shape}")
    
    return aggregated_df

# Aggregate the CSV files first
if __name__ == "__main__":
    # Your local file paths (since you're inside SSH)
    file1 = "/local1/ofir/shalevle/STImage_1K4M/ST/gene_exp/pam50_pseudobulk_enhanced_summary.csv"
    file2 = "/local1/ofir/shalevle/STImage_1K4M/Visium/gene_exp/pam50_pseudobulk_enhanced_summary1.csv"
    
    # Output file
    output_file = "/local1/ofir/shalevle/STImage_1K4M/aggregated_pam50_summary.csv"
    
    # Run the aggregation
    try:
        aggregated_df = aggregate_csvs(file1, file2, output_file)
        print("\nFirst few rows of aggregated data:")
        print(aggregated_df.head())
        print("\nColumn info:")
        print(aggregated_df.info())
        
    except Exception as e:
        print(f"Error: {e}")

    # Now continue with the rest of your code
    print("\n" + "="*50)
    print("Starting data merging process...")
    print("="*50)

    # Load data
    df_paths = pd.read_csv('/local1/ofir/shalevle/STImage_1K4M/ST_Image_CSV_paths.csv')
    df_sum = pd.read_csv('/local1/ofir/shalevle/STImage_1K4M/aggregated_pam50_summary.csv')

    # Extract and normalize sample_id
    df_paths['sample_id'] = df_paths['image_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0].strip().lower())

    # Normalize patient_id and remove '_count'
    df_sum['patient_id_clean'] = df_sum['patient_id'].astype(str).str.replace('_count', '', regex=False).str.strip().str.lower()

    # Print side-by-side for debugging
    print("\n🔍 Comparison (first 5 rows):")
    print(pd.DataFrame({
        'sample_id': df_paths['sample_id'].head(5),
        'patient_id_clean': df_sum['patient_id_clean'].head(5)
    }))

    # Show exact mismatches
    sample_ids = set(df_paths['sample_id'])
    patient_ids = set(df_sum['patient_id_clean'])
    intersection = sample_ids & patient_ids
    print(f"\n✅ Matches found: {len(intersection)}")
    print(f"❌ Missing examples: {list(sample_ids - intersection)[:5]}")

    # Keep only needed columns
    df_sum_small = df_sum[['patient_id_clean', 'final_quality', 'top_subtype']]

    # Merge
    merged = pd.merge(
        df_paths,
        df_sum_small,
        left_on='sample_id',
        right_on='patient_id_clean',
        how='left'
    )

    # Drop unwanted columns
    merged = merged[['image_path', 'coord_path', 'gene_exp_path', 'final_quality', 'top_subtype']]

    # Save
    merged.to_csv('/local1/ofir/shalevle/STImage_1K4M/merged_cleaned.csv', index=False)
    
    print(f"\n✅ Final merged file saved with {len(merged)} rows")
    print("Process complete!")

# Quick one-liner version for just aggregation
def quick_aggregate():
    """
    Quick aggregation function - just run this
    """
    file1 = "/local1/ofir/shalevle/STImage_1K4M/ST/gene_exp/pam50_pseudobulk_enhanced_summary.csv"
    file2 = "/local1/ofir/shalevle/STImage_1K4M/Visium/gene_exp/pam50_pseudobulk_enhanced_summary1.csv"
    
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    df1['source'] = 'ST'
    df2['source'] = 'Visium'
    
    result = pd.concat([df1, df2], ignore_index=True)
    result.to_csv('/local1/ofir/shalevle/STImage_1K4M/aggregated_pam50_summary.csv', index=False)
    
    print(f"Done! Combined {len(df1)} + {len(df2)} = {len(result)} rows")
    return result