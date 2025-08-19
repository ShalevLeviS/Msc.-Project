#!/usr/bin/env python3
# -------------------------------------------------------------------
#  ENHANCED PAM50 SUBTYPE CALLER - PSEUDO-BULK IMPLEMENTATION
#  With improved confidence assessment, QC, normalization, and validation
# -------------------------------------------------------------------
import os, pandas as pd
import numpy as np
from collections import Counter
from rpy2.robjects import r, pandas2ri, default_converter, StrVector
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr


"""
Enhanced PAM50 Subtype Classifier - Pseudo-Bulk Implementation

Purpose: Classifies breast cancer spatial transcriptomics data into PAM50 molecular subtypes 
using pseudo-bulk aggregation with comprehensive quality control and validation

Inputs:
- Gene expression CSV files (genes × tiles per patient)
- PAM50 reference centroids from genefu R package
- Quality control parameters (min counts, genes, tiles)

Key Features:
- Enhanced tile-level QC (count thresholds, gene detection, outlier removal)
- Pseudo-bulk aggregation (sum across all tiles per patient)  
- PAM50-optimized normalization (CPM → log2 → median centering → robust scaling)
- Multi-metric confidence assessment (correlation, margin, separation)
- Biological validation against marker genes (ESR1, PGR, ERBB2, etc.)
- Comprehensive quality reporting and recommendations

Process:
1. Load patient tile data → QC filtering → outlier removal
2. Aggregate tiles into pseudo-bulk profile per patient
3. Normalize using PAM50-specific pipeline
4. Calculate correlations with PAM50 centroids
5. Assign subtype + confidence level + biological validation
6. Generate quality reports and recommendations

Output: 
- Individual patient results with confidence metrics
- Master summary with subtype distribution and quality analysis
- Quality report with recommendations for borderline cases
"""


# 0. Setup R environment
os.environ["R_LIBS_USER"] = os.path.expanduser(
    "~/R/x86_64-redhat-linux-gnu-library/4.5")
r('.libPaths(Sys.getenv("R_LIBS_USER"))')
r('options(warn=-1)')

print("🔍 STEP 1: Verify PAM50 object setup")
print("=" * 60)

try:
    r('suppressPackageStartupMessages(library(genefu))')
    r('data(pam50)')  # loads pam50 and possibly pam50.robust
    
    # Create pam50.robust if it doesn't exist (common issue)
    r('''
    if (!exists("pam50.robust")) {
        pam50.robust <- pam50
        cat("Created pam50.robust from pam50\\n")
    }
    ''')
    
    # Check what objects exist
    r_objects = list(r('ls()'))
    print(f"📋 Available objects: {r_objects}")
    
    # Get the legal probe IDs (ONLY these will work)
    legal_probes = list(r('rownames(pam50$centroids)'))
    print(f"✅ Legal probe IDs from pam50$centroids: {len(legal_probes)} genes")
    print(f"    First 10: {legal_probes[:10]}")
    
    genefu = importr("genefu", suppress_messages=True)
    
except Exception as e:
    print(f"❌ FATAL: PAM50 loading failed: {e}")
    raise

print(f"\n🔍 STEP 2: ENHANCED PSEUDO-BULK IMPLEMENTATION")
print("=" * 60)

BASE = "/local1/ofir/shalevle/STImage_1K4M/ST/gene_exp"
conv = default_converter + pandas2ri.converter

# Enhanced Quality Control Functions
def enhanced_tile_qc(expr_df, min_total_counts=500, min_genes_detected=1000):
    """Enhanced quality control at tile level"""
    
    print(f"    📊 Original tiles: {expr_df.shape[1]}")
    
    # Calculate tile metrics
    tile_total_counts = expr_df.sum(axis=0)
    tile_genes_detected = (expr_df > 0).sum(axis=0)
    tile_pam50_coverage = expr_df.loc[expr_df.index.isin(legal_probes)].sum(axis=0)
    
    # Quality filters
    high_count_tiles = tile_total_counts >= min_total_counts
    high_gene_tiles = tile_genes_detected >= min_genes_detected
    has_pam50_tiles = tile_pam50_coverage > 0
    
    # Combine filters
    quality_tiles = high_count_tiles & high_gene_tiles & has_pam50_tiles
    
    print(f"    ✅ High count tiles (≥{min_total_counts}): {high_count_tiles.sum()}")
    print(f"    ✅ High gene tiles (≥{min_genes_detected}): {high_gene_tiles.sum()}")
    print(f"    ✅ PAM50 expressing tiles: {has_pam50_tiles.sum()}")
    print(f"    🎯 Quality tiles retained: {quality_tiles.sum()}")
    
    if quality_tiles.sum() < 10:
        print(f"    ⚠️  WARNING: Very few quality tiles ({quality_tiles.sum()})")
    
    return expr_df.loc[:, quality_tiles], quality_tiles.sum()

def filter_outlier_tiles(expr_df, pam50_genes, outlier_threshold=3.0):
    """Remove tiles with aberrant PAM50 expression patterns"""
    
    # Focus on PAM50 genes only
    pam50_expr = expr_df.loc[expr_df.index.isin(pam50_genes)]
    
    # Calculate tile-wise PAM50 scores
    tile_pam50_totals = pam50_expr.sum(axis=0)
    
    # Remove extreme outliers (very high or very low PAM50 expression)
    q1, q3 = tile_pam50_totals.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - outlier_threshold * iqr
    upper_bound = q3 + outlier_threshold * iqr
    
    normal_tiles = (tile_pam50_totals >= lower_bound) & (tile_pam50_totals <= upper_bound)
    
    print(f"    📊 Outlier filtering: {normal_tiles.sum()}/{len(normal_tiles)} tiles retained")
    print(f"    📊 PAM50 total range: {tile_pam50_totals.min():.1f} - {tile_pam50_totals.max():.1f}")
    
    return expr_df.loc[:, normal_tiles], normal_tiles.sum()

def enhanced_pam50_normalization(patient_expr_series, legal_probes_set):
    """Enhanced normalization following PAM50 best practices"""
    
    # Step 1: Filter to PAM50 genes before any normalization
    pam50_expr = patient_expr_series[patient_expr_series.index.isin(legal_probes_set)]
    
    print(f"    🎯 PAM50 genes available: {len(pam50_expr)} / 50")
    
    # Step 2: CPM normalization (counts per million)
    total_counts = pam50_expr.sum()
    cpm = (pam50_expr / total_counts) * 1e6
    
    # Step 3: Log2 transformation with pseudocount
    log_cpm = np.log2(cpm + 1)
    
    # Step 4: PAM50-specific median centering
    median_expr = log_cpm.median()
    centered_expr = log_cpm - median_expr
    
    # Step 5: Robust scaling (optional - use if high variance)
    mad = np.median(np.abs(centered_expr - centered_expr.median()))
    if mad > 0:
        scaled_expr = centered_expr / (1.4826 * mad)
    else:
        scaled_expr = centered_expr
    
    normalization_stats = {
        'total_counts': total_counts,
        'median_log_cpm': median_expr,
        'mad_scaled': mad,
        'expr_range': (scaled_expr.min(), scaled_expr.max()),
        'n_genes_normalized': len(scaled_expr)
    }
    
    print(f"    📊 Normalization stats:")
    print(f"       Total PAM50 counts: {total_counts:.0f}")
    print(f"       Median log2(CPM+1): {median_expr:.3f}")
    print(f"       Expression range: {scaled_expr.min():.2f} to {scaled_expr.max():.2f}")
    
    return scaled_expr, normalization_stats

def classify_confidence(corr_score, margin, separation, rel_strength, n_genes):
    """Enhanced confidence classification"""
    
    # Quality factors
    gene_quality = 1.0 if n_genes >= 45 else 0.9 if n_genes >= 40 else 0.8
    
    # Base confidence score (0-1)
    base_score = (
        0.4 * min(corr_score / 0.6, 1.0) +      # Correlation strength (cap at 0.6)
        0.3 * min(margin / 0.15, 1.0) +         # Margin strength (cap at 0.15)  
        0.2 * min(separation, 1.0) +            # Separation quality
        0.1 * min(rel_strength / 0.3, 1.0)      # Relative strength
    ) * gene_quality
    
    # Confidence levels with adjusted thresholds
    if base_score >= 0.75 and corr_score >= 0.45 and margin >= 0.12:
        return "HIGH", base_score
    elif base_score >= 0.55 and corr_score >= 0.25 and margin >= 0.05:
        return "MODERATE", base_score  
    elif base_score >= 0.35 and corr_score >= 0.15:
        return "LOW", base_score
    else:
        return "VERY_LOW", base_score

def biological_validation(patient_expr, subtype_call, correlations_dict):
    """Validate PAM50 calls against known biological patterns"""
    
    # Key marker genes for validation (if available in data)
    marker_genes = {
        'ESR1': 'LumA/LumB',     # Estrogen receptor
        'PGR': 'LumA/LumB',      # Progesterone receptor  
        'ERBB2': 'Her2',         # HER2 receptor
        'MKI67': 'proliferation', # Ki67 proliferation marker
        'CCNE1': 'LumB/Basal',   # Cyclin E1
        'FOXC1': 'Basal',        # Basal marker
        'KRT5': 'Basal',         # Basal cytokeratin
        'KRT14': 'Basal',        # Basal cytokeratin
    }
    
    validation_results = {}
    warnings = []
    
    # Check marker gene expression patterns
    for gene, expected_pattern in marker_genes.items():
        if gene in patient_expr.index:
            expr_level = patient_expr[gene]
            validation_results[f'{gene}_expr'] = expr_level
            
            # Validate against subtype call
            if subtype_call in ['LumA', 'LumB'] and gene in ['ESR1', 'PGR']:
                if expr_level < -1.0:  # Low expression
                    warnings.append(f"Low {gene} expression ({expr_level:.2f}) for {subtype_call} subtype")
            
            elif subtype_call == 'Her2' and gene == 'ERBB2':
                if expr_level < 0.0:  # Low HER2 expression
                    warnings.append(f"Low ERBB2 expression ({expr_level:.2f}) for Her2 subtype")
            
            elif subtype_call == 'Basal' and gene in ['ESR1', 'PGR']:
                if expr_level > 0.5:  # High hormone receptor expression
                    warnings.append(f"High {gene} expression ({expr_level:.2f}) for Basal subtype")
    
    # Check correlation patterns
    if subtype_call == 'LumA':
        if correlations_dict.get('LumB', 0) > correlations_dict.get('LumA', 0):
            warnings.append("LumA call but higher LumB correlation")
    
    elif subtype_call == 'Her2':
        if max(correlations_dict.get('LumA', 0), correlations_dict.get('LumB', 0)) > correlations_dict.get('Her2', 0):
            warnings.append("Her2 call but higher Luminal correlation")
    
    # Check for Normal-like patterns (should be rare in cancer samples)
    if subtype_call == 'Normal':
        normal_corr = correlations_dict.get('Normal', 0)
        if normal_corr < 0.3:
            warnings.append("Weak Normal subtype call - may indicate technical issues")
    
    # Overall validation score
    validation_score = 1.0 - (len(warnings) * 0.2)  # Reduce score for each warning
    validation_score = max(0.0, min(1.0, validation_score))
    
    return {
        'validation_score': validation_score,
        'warnings': warnings,
        'marker_expressions': validation_results,
        'n_warnings': len(warnings)
    }

def final_quality_assessment(confidence_level, validation_score, n_genes, correlation_score):
    """Combine all quality metrics for final assessment"""
    
    if (confidence_level == 'HIGH' and validation_score >= 0.8 and 
        n_genes >= 45 and correlation_score >= 0.5):
        return 'EXCELLENT'
    elif (confidence_level in ['HIGH', 'MODERATE'] and validation_score >= 0.6 and 
          n_genes >= 40 and correlation_score >= 0.3):
        return 'GOOD'
    elif (confidence_level != 'VERY_LOW' and validation_score >= 0.4 and 
          n_genes >= 35):
        return 'ACCEPTABLE'
    else:
        return 'QUESTIONABLE'

# Initialize master summary list
master_summary_data = []

for csv in sorted(f for f in os.listdir(BASE) if f.endswith(".csv")):
    print(f"\n📁 Processing Patient: {csv}")
    
    try:
        # STEP 1: Load patient's tile data
        df = pd.read_csv(os.path.join(BASE, csv))
        if "Unnamed: 0" in df.columns:
            df = df.set_index("Unnamed: 0")
        
        # Transpose: genes as rows, tiles as columns
        expr = df.T  # Shape: (genes, tiles_for_this_patient)
        expr.index = expr.index.str.upper()           # uppercase gene names
        expr = expr[~expr.index.duplicated()]         # remove duplicates
        
        print(f"    📊 Original data: {expr.shape[0]} genes × {expr.shape[1]} tiles")
        
        # STEP 2: ENHANCED QUALITY CONTROL
        print("🔧 Enhanced Quality Control")
        
        # Basic QC
        expr_qc, n_quality_tiles = enhanced_tile_qc(expr, min_total_counts=300, min_genes_detected=800)
        
        if n_quality_tiles < 5:
            print(f"    ❌ Too few quality tiles ({n_quality_tiles}) → Patient skipped")
            continue
        
        # Outlier filtering
        expr_final, n_final_tiles = filter_outlier_tiles(expr_qc, legal_probes, outlier_threshold=2.5)
        
        if n_final_tiles < 5:
            print(f"    ❌ Too few tiles after outlier removal ({n_final_tiles}) → Patient skipped")
            continue
        
        print(f"    ✅ Final tile count: {n_final_tiles}")
        
        # STEP 3: PSEUDO-BULK AGGREGATION
        print("🔬 PSEUDO-BULK: Aggregating all tiles into single patient profile")
        
        patient_pseudo_bulk = expr_final.sum(axis=1)  # Sum across tiles (axis=1)
        print(f"    ✅ Pseudo-bulk profile created: {len(patient_pseudo_bulk)} genes")
        
        # STEP 4: ENHANCED NORMALIZATION
        print("🔧 Enhanced PAM50-Specific Normalization")
        
        expr_normalized, norm_stats = enhanced_pam50_normalization(
            patient_pseudo_bulk, 
            set(legal_probes)
        )
        
        # Update available_genes to match normalized data
        available_genes = list(expr_normalized.index)
        print(f"    ✅ Genes ready for PAM50 classification: {len(available_genes)}")
        
        if len(available_genes) < 35:
            print(f"    ❌ Too few PAM50 genes ({len(available_genes)}/50) → skipped")
            continue
        
        # STEP 5: Get PAM50 centroids for available genes
        print("📐 Preparing PAM50 centroids")
        
        r.assign('available_genes_r', StrVector(available_genes))
        r('centroids_subset <- pam50$centroids[available_genes_r, , drop=FALSE]')
        
        # Convert centroids to pandas DataFrame
        centroids_values = []
        centroid_names = list(r('colnames(centroids_subset)'))
        
        for i, subtype in enumerate(centroid_names):
            subtype_values = list(r(f'centroids_subset[, {i+1}]'))
            centroids_values.append(subtype_values)
        
        centroids_df = pd.DataFrame(
            dict(zip(centroid_names, centroids_values)),
            index=available_genes
        )
        
        print(f"    ✅ Centroids prepared for {len(centroid_names)} subtypes")
        
        # STEP 6: PAM50 CLASSIFICATION
        print("🎯 PAM50 Classification of pseudo-bulk profile")
        
        # Calculate correlations with each PAM50 centroid
        correlations = {}
        for subtype_name in centroids_df.columns:
            centroid_vals = pd.Series(centroids_df[subtype_name].values, index=available_genes)
            corr = expr_normalized.corr(centroid_vals, method='spearman')
            correlations[subtype_name] = corr if not pd.isna(corr) else 0.0
        
        # Assign subtype with highest correlation
        patient_subtype = max(correlations, key=correlations.get)
        patient_correlation = correlations[patient_subtype]
        
        # STEP 7: ENHANCED CONFIDENCE ASSESSMENT
        print("🔍 Enhanced Confidence Assessment")
        
        # Calculate enhanced confidence metrics
        sorted_subtypes = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        top_corr = sorted_subtypes[0][1]
        second_corr = sorted_subtypes[1][1]
        third_corr = sorted_subtypes[2][1] if len(sorted_subtypes) > 2 else 0
        
        confidence_margin = top_corr - second_corr
        separation_score = (top_corr - second_corr) / (second_corr - third_corr + 0.001)
        relative_strength = top_corr / (sum(abs(corr) for corr in correlations.values()) + 0.001)
        
        confidence_level, confidence_score = classify_confidence(
            patient_correlation, confidence_margin, separation_score, 
            relative_strength, len(available_genes)
        )
        
        print(f"    🏥 PATIENT SUBTYPE: {patient_subtype}")
        print(f"    📈 Correlation: {patient_correlation:.3f}")
        print(f"    🎯 Confidence Level: {confidence_level} (score: {confidence_score:.3f})")
        print(f"    📊 Separation Score: {separation_score:.3f}")
        
        # Flag special cases
        if confidence_margin < 0.05:
            print(f"    ⚠️  BORDERLINE: Very close call between {sorted_subtypes[0][0]} and {sorted_subtypes[1][0]}")
        
        if patient_correlation < 0.2:
            print(f"    ⚠️  WEAK SIGNAL: Low correlation with all centroids")
        
        # STEP 8: BIOLOGICAL VALIDATION
        print("🧬 Biological Validation & Sanity Checks")
        
        validation = biological_validation(expr_normalized, patient_subtype, correlations)
        
        print(f"    🧬 Validation Score: {validation['validation_score']:.3f}")
        print(f"    ⚠️  Warnings: {validation['n_warnings']}")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"       • {warning}")
        
        # STEP 9: FINAL QUALITY ASSESSMENT
        final_quality = final_quality_assessment(
            confidence_level, validation['validation_score'], 
            len(available_genes), patient_correlation
        )
        
        print(f"    🏆 Final Quality: {final_quality}")
        
        if final_quality == 'QUESTIONABLE':
            print(f"    ⚠️  RECOMMENDATION: Consider manual review or additional validation")
        
        # STEP 10: Save results
        print("💾 Saving results")
        
        # Create enhanced results
        enhanced_results = {
            'patient_id': csv.replace('.csv', ''),
            'pam50_subtype': patient_subtype,
            'correlation_score': patient_correlation,
            'confidence_margin': confidence_margin,
            'confidence_level': confidence_level,
            'confidence_score': confidence_score,
            'separation_score': separation_score,
            'relative_strength': relative_strength,
            'gene_coverage': len(available_genes) / 50.0,
            'n_tiles_aggregated': n_final_tiles,
            'n_pam50_genes_used': len(available_genes),
            'total_counts_aggregated': norm_stats['total_counts'],
            'validation_score': validation['validation_score'],
            'n_biological_warnings': validation['n_warnings'],
            'biological_warnings': '; '.join(validation['warnings']) if validation['warnings'] else 'None',
            'final_quality': final_quality,
            'top_subtype': sorted_subtypes[0][0],
            'second_subtype': sorted_subtypes[1][0],
            'top_correlation': sorted_subtypes[0][1],
            'second_correlation': sorted_subtypes[1][1],
        }
        
        # Add individual subtype correlations
        for subtype, corr in correlations.items():
            enhanced_results[f'corr_{subtype.lower()}'] = corr
        
        # Save individual patient results
        results_df = pd.DataFrame([enhanced_results])
        output_file = csv.replace('.csv', '_pam50_enhanced.csv')
        output_path = os.path.join(BASE, output_file)
        results_df.to_csv(output_path, index=False)
        print(f"    💾 Results saved to: {output_file}")
        
        # Add to master summary
        master_summary_data.append(enhanced_results)
        
        print(f"    ✅ Patient {csv.replace('.csv', '')} completed successfully")
        
    except Exception as e:
        print(f"    ❌ Classification failed for {csv}: {e}")
        continue

# ENHANCED SUMMARY AND QUALITY REPORTING
print(f"\n📊 ENHANCED MASTER SUMMARY & QUALITY ANALYSIS")
print("=" * 70)

if master_summary_data:
    master_summary_df = pd.DataFrame(master_summary_data)
    
    # Enhanced statistics
    total_patients = len(master_summary_data)
    print(f"✅ Total patients processed: {total_patients}")
    
    # Subtype distribution with percentages
    subtype_counts = Counter(master_summary_df['pam50_subtype'])
    print(f"\n📊 PAM50 Subtype Distribution:")
    for subtype in ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']:
        count = subtype_counts.get(subtype, 0)
        percentage = (count / total_patients) * 100
        print(f"    {subtype:8s}: {count:3d} patients ({percentage:5.1f}%)")
    
    # Confidence level distribution
    confidence_counts = Counter(master_summary_df['confidence_level'])
    print(f"\n🎯 Confidence Level Distribution:")
    for conf_level in ['HIGH', 'MODERATE', 'LOW', 'VERY_LOW']:
        count = confidence_counts.get(conf_level, 0)
        percentage = (count / total_patients) * 100
        print(f"    {conf_level:12s}: {count:3d} patients ({percentage:5.1f}%)")
    
    # Quality assessment distribution
    quality_counts = Counter(master_summary_df['final_quality'])
    print(f"\n🏆 Final Quality Distribution:")
    for quality in ['EXCELLENT', 'GOOD', 'ACCEPTABLE', 'QUESTIONABLE']:
        count = quality_counts.get(quality, 0)
        percentage = (count / total_patients) * 100
        print(f"    {quality:12s}: {count:3d} patients ({percentage:5.1f}%)")
    
    # Detailed quality metrics
    print(f"\n📈 QUALITY METRICS:")
    
    # Correlation scores
    mean_correlation = master_summary_df['correlation_score'].mean()
    median_correlation = master_summary_df['correlation_score'].median()
    print(f"    Correlation scores - Mean: {mean_correlation:.3f}, Median: {median_correlation:.3f}")
    
    # Confidence margins
    mean_margin = master_summary_df['confidence_margin'].mean()
    median_margin = master_summary_df['confidence_margin'].median()
    print(f"    Confidence margins - Mean: {mean_margin:.3f}, Median: {median_margin:.3f}")
    
    # Gene coverage
    mean_genes = master_summary_df['n_pam50_genes_used'].mean()
    min_genes = master_summary_df['n_pam50_genes_used'].min()
    max_genes = master_summary_df['n_pam50_genes_used'].max()
    print(f"    PAM50 gene coverage - Mean: {mean_genes:.1f}, Range: {min_genes}-{max_genes}")
    
    # Tiles per patient
    mean_tiles = master_summary_df['n_tiles_aggregated'].mean()
    median_tiles = master_summary_df['n_tiles_aggregated'].median()
    print(f"    Tiles per patient - Mean: {mean_tiles:.0f}, Median: {median_tiles:.0f}")
    
    # Biological validation
    mean_validation = master_summary_df['validation_score'].mean()
    high_validation = (master_summary_df['validation_score'] >= 0.8).sum()
    print(f"    Biological validation - Mean score: {mean_validation:.3f}")
    print(f"    High validation scores (≥0.8): {high_validation}/{total_patients} ({100*high_validation/total_patients:.1f}%)")
    
    # Flag problematic cases
    print(f"\n⚠️  QUALITY FLAGS:")
    
    low_correlation = (master_summary_df['correlation_score'] < 0.2).sum()
    if low_correlation > 0:
        print(f"    {low_correlation} patients with very low correlation scores (<0.2)")
    
    low_margin = (master_summary_df['confidence_margin'] < 0.05).sum()
    if low_margin > 0:
        print(f"    {low_margin} patients with borderline calls (margin <0.05)")
    
    few_genes = (master_summary_df['n_pam50_genes_used'] < 40).sum()
    if few_genes > 0:
        print(f"    {few_genes} patients with limited PAM50 gene coverage (<40 genes)")
    
    few_tiles = (master_summary_df['n_tiles_aggregated'] < 50).sum()
    if few_tiles > 0:
        print(f"    {few_tiles} patients with few tiles (<50 tiles)")
    
    # Subtype-specific quality analysis
    print(f"\n📊 SUBTYPE-SPECIFIC QUALITY:")
    for subtype in ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']:
        subtype_data = master_summary_df[master_summary_df['pam50_subtype'] == subtype]
        if len(subtype_data) > 0:
            mean_corr = subtype_data['correlation_score'].mean()
            mean_margin = subtype_data['confidence_margin'].mean()
            high_conf = (subtype_data['confidence_level'] == 'HIGH').sum()
            print(f"    {subtype:8s}: Mean corr={mean_corr:.3f}, Mean margin={mean_margin:.3f}, High conf={high_conf}/{len(subtype_data)}")
    
    # Save enhanced summary
    enhanced_summary_path = os.path.join(BASE, 'pam50_pseudobulk_enhanced_summary.csv')
    master_summary_df.to_csv(enhanced_summary_path, index=False)
    print(f"\n💾 Enhanced summary saved to: pam50_pseudobulk_enhanced_summary.csv")
    
    # Generate quality report
    quality_report_path = os.path.join(BASE, 'pam50_quality_report.txt')
    with open(quality_report_path, 'w') as f:
        f.write("PAM50 PSEUDO-BULK CLASSIFICATION QUALITY REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total patients processed: {total_patients}\n\n")
        
        f.write("SUBTYPE DISTRIBUTION:\n")
        for subtype, count in subtype_counts.items():
            f.write(f"  {subtype}: {count} ({100*count/total_patients:.1f}%)\n")
        
        f.write("\nCONFIDENCE DISTRIBUTION:\n")
        for conf, count in confidence_counts.items():
            f.write(f"  {conf}: {count} ({100*count/total_patients:.1f}%)\n")
        
        f.write(f"\nQUALITY METRICS:\n")
        f.write(f"  Mean correlation: {mean_correlation:.3f}\n")
        f.write(f"  Mean confidence margin: {mean_margin:.3f}\n")
        f.write(f"  Mean PAM50 genes: {mean_genes:.1f}\n")
        
        if low_correlation + low_margin + few_genes + few_tiles > 0:
            f.write(f"\nQUALITY CONCERNS:\n")
            if low_correlation > 0:
                f.write(f"  {low_correlation} patients with low correlation\n")
            if low_margin > 0:
                f.write(f"  {low_margin} patients with borderline calls\n")
            if few_genes > 0:
                f.write(f"  {few_genes} patients with limited gene coverage\n")
            if few_tiles > 0:
                f.write(f"  {few_tiles} patients with few tiles\n")
    
    print(f"📄 Quality report saved to: pam50_quality_report.txt")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    high_quality_rate = (confidence_counts.get('HIGH', 0) + confidence_counts.get('MODERATE', 0)) / total_patients * 100
    
    if high_quality_rate >= 70:
        print(f"    ✅ Good overall quality ({high_quality_rate:.1f}% high/moderate confidence)")
    elif high_quality_rate >= 50:
        print(f"    ⚠️  Moderate quality ({high_quality_rate:.1f}% high/moderate confidence)")
        print(f"    💡 Consider reviewing low-confidence cases manually")
    else:
        print(f"    ❌ Low overall quality ({high_quality_rate:.1f}% high/moderate confidence)")
        print(f"    💡 Review normalization and QC parameters")
        print(f"    💡 Consider additional filtering or alternative methods")
    
    if few_genes > total_patients * 0.2:
        print(f"    💡 Many patients have limited PAM50 gene coverage - check gene annotations")
    
    if mean_tiles < 100:
        print(f"    💡 Low tile counts - consider less stringent QC or different aggregation")

else:
    print(f"⚠️  No patients were successfully processed")

print(f"\n🎉 ENHANCED PAM50 CLASSIFICATION COMPLETE!")
print("=" * 70)
print("📋 ENHANCEMENTS IMPLEMENTED:")
print("✅ Enhanced quality control with tile filtering")
print("✅ Improved confidence assessment with multiple metrics")
print("✅ PAM50-optimized normalization pipeline")
print("✅ Biological validation and sanity checks")
print("✅ Comprehensive quality reporting and recommendations")
print("✅ Individual patient quality scores")
print("✅ Outlier detection and removal")
print("✅ Enhanced error handling and warnings")
print("\n📊 Expected: Higher quality, more confident PAM50 classifications!")
print("📊 Better identification of borderline and questionable cases!")