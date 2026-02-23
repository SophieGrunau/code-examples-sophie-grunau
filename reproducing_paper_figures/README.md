```markdown
# Reproducing Published Research: PCA/SVM Geological Analysis

**Author:** Lea Sophie Grunau

Adaptation of the PCA/SVM analysis from O'Sullivan et al. (2020) to classify detrital apatite 
grains by lithology using trace element geochemistry. Originally adapted to help a geology 
colleague replicate the paper's figures with his own sample dataset.

## Background

O'Sullivan et al. (2020) developed an SVM-based classification method for apatite provenance 
analysis using Sr/Y vs. ΣLREE geochemistry. This script reproduces Figures 3b and 3c from 
the paper and applies the trained model to new samples.

The original code is from the paper's supplementary materials. Adaptations are marked with 
`#SOPHIE:` comments throughout.

## Contents

- `S3_PCA_SVM_method_code.R` — main analysis script
- `data/` — input data files (OSullivan_plot_3c_data.txt, new_samples.txt)
- `output/` — generated figures and tables
- `O'Sullivan et al 2020 [...].pdf` — original paper

## Usage

```r
# In RStudio: open the script and run
# Working directory is set automatically to the script location
Rscript S3_PCA_SVM_method_code.R
```

## Configuration

To use your own samples, update `new_samples.txt` in the `data/` folder and edit these 
lines at the top of the script:

- `sample_group_name` — names for your sample groups
- `lithology_clas_name` — label for lithology categories

## Output

- `Plot3b_new_data_density.pdf` — Sr/Y vs ΣLREE biplot with new samples and density contours
- `Plot3c_SVM_model.pdf` — SVM classification background
- `Plot3c_SVM_model_new_data_by_group.pdf` — new samples coloured by group
- `Plot3c_SVM_model_new_data_by_group_age.pdf` — new samples coloured by age
- `Plot3c_SVM_model_misclassification_table.txt` — SVM model accuracy table
- `Plot3c_SVM_model_new_data_table.txt` — predicted lithology for each grain

## Reference

O'Sullivan et al. (2020). *The trace element composition of apatite and its application 
to detrital provenance studies.* https://doi.pangaea.de/10.1594/PANGAEA.906570
```