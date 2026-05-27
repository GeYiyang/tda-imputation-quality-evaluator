# TDAssure User Guide

This guide provides a practical introduction to the TDAssure dashboard and its main functionalities.

The purpose of this document is to help users:
- launch the application
- upload datasets
- configure pipeline parameters
- interpret the main outputs

This guide intentionally focuses on practical usage rather than the full mathematical and methodological foundations of the framework.

For detailed methodological explanations, theoretical background, and algorithmic design, please refer to the associated research paper.

---

## 1. Launching the Application

After downloading and extracting the project folder, users can launch the dashboard by double-clicking:

`Start_App.bat`

During the first launch, the application may take several minutes to:
- create a virtual environment
- install required dependencies
- initialize the local Shiny server

Once initialization is complete, the dashboard should automatically open in the default web browser.

If the browser does not open automatically, users can manually access:

`http://127.0.0.1:8000`

### Important Notes

- The first launch is usually slower than subsequent launches.
- The application may temporarily consume high CPU resources during initialization and pipeline execution.

---

## 2. Uploading Datasets

TDAssure supports up to three dataset uploads:

1. Raw dataset (optional)
2. Complete case dataset
3. Imputed dataset

All datasets should currently be provided in CSV format.

### Raw Dataset (Optional)

The raw dataset may contain missing values and can be uploaded for exploratory purposes.

When provided, the dashboard can display:
- dataset previews
- missing value summaries
- missing value visualizations

Uploading the raw dataset is optional and is not required for running the main TDA comparison pipeline.

### Complete-Case Dataset

The complete-case dataset refers to the subset of the original raw dataset that contains no missing values.

It is used as the reference dataset for comparison because all selected records are fully observed.

### Imputed Dataset

The imputed dataset refers to the part of the original dataset that contained missing values and has subsequently been completed using an imputation method.

It is compared against the complete-case dataset to assess whether the imputation preserves the overall data structure.

### Data Requirements

- Complete-case and imputed datasets should contain the same feature structure.
- Column names should ideally be consistent across datasets.
- Mixed variable types are allowed.
- Numeric features are automatically selected for Mapper construction and topological analysis.

### Upload Procedure

Datasets can be uploaded using the upload panels located on the left side of the dashboard:

- Upload Raw CSV (optional)
- Upload Complete Case CSV
- Upload Imputed CSV

After upload, the dashboard can generates:
- dataset previews
- missing value summaries
- missing value visualizations

### Important Notes

- Large datasets may require substantial memory and CPU resources.
- Extremely high-dimensional datasets may significantly increase runtime.

---

## 3. Understanding the Dashboard Layout

The TDAssure dashboard is organized into several main sections.

### Data Upload Panel

Located on the left side of the dashboard.

It also contains the main pipeline parameter settings and execution controls.

### Dataset Preview Section

This section displays:
- dataset previews
- missing value summaries
- missing value visualizations

### Mapper Visualization Section

This section displays Mapper-based network visualizations generated from the uploaded datasets.

The Mapper graph provides a simplified structural representation of the dataset topology.

### Pipeline Monitoring Section

This section provides:
- real-time execution logs
- pipeline progress updates
- parameter-combination status monitoring

Large parameter combinations or high permutation counts may require substantial runtime.

### Results Section

This section displays:
- significance summaries
- p-value distributions
- pipeline result tables

These outputs are generated automatically after successful pipeline execution.

---

## 4. Configuring Pipeline Parameters

TDAssure allows users to customize several pipeline parameters before running the analysis.

TDA parameter configuration can be highly sensitive, and certain parameter combinations may substantially affect both runtime and analytical behavior.

To keep this guide practical and accessible, only the most essential parameter descriptions and recommended starting configurations are provided here.

Users interested in the methodological and mathematical foundations of parameter selection are encouraged to refer to the associated research paper.

Different parameter settings may substantially affect:
- runtime
- Mapper graph structure
- sensitivity of the topological comparison

For first-time users, it is recommended to begin with small parameter combinations and low permutation counts.

### Resolution

The resolution parameter controls how finely the data space is partitioned during Mapper construction.

Higher resolution values generally produce:
- more Mapper nodes
- more detailed graph structures

Lower resolution values generally produce:
- simpler Mapper graphs
- faster computation
- lower structural sensitivity

For most initial analyses, resolution values can typically start from `5`, while values above `50` are generally not recommended.

### Gain

The gain parameter controls the amount of overlap between neighboring Mapper regions.

Higher gain values generally produce:
- greater overlap between regions
- more connected Mapper graphs

Lower gain values generally produce:
- less overlap
- more separated graph structures

For most initial analyses, gain values between `0.1` and `0.4` are generally recommended.

### Parameter Lists

Both the Resolution and Gain settings support multiple values separated by commas.

For example:

`Resolution: 5,8,10`

`Gain: 0.1,0.2,0.3`

When multiple values are provided, TDAssure automatically evaluates all parameter combinations sequentially.

This functionality allows users to explore how different Mapper settings may affect the resulting topological analysis.

Due to the sensitivity of TDA parameter configurations, exploratory analyses often require testing multiple parameter combinations in order to better reveal underlying data structure patterns.

### Base Clusters per Bin

This parameter controls the maximum number of local clusters generated within each Mapper region.

Higher values generally produce:
- more detailed local graph structures
- increased graph complexity

Lower values generally produce:
- simpler Mapper structures
- faster computation
- more stable initial analyses

Very high cluster settings may dramatically increase computational cost without necessarily improving interpretability.

For most initial analyses, values between `1` and `3` are generally recommended.

For more structurally complex datasets, exploratory settings between `5` and `20` may also be considered.

### Permutations

The permutation parameter controls how many repeated random comparisons are performed during the statistical evaluation stage.

Higher permutation counts generally provide:
- more stable statistical estimation
- more reliable p-value estimation
- increased computational cost

Lower permutation counts generally provide:
- faster runtime
- quicker exploratory analyses
- less stable statistical estimates

For initial testing, values between `50` and `99` are generally recommended.

For more stable statistical evaluation, values between `199` and `999` may also be considered, although computational cost may increase substantially.

### Parallel Jobs

This parameter controls how many CPU processes are used simultaneously during pipeline execution.

It can be configured according to the number of available CPU cores and directly affects computational speed.

If users are unfamiliar with their CPU specifications, values between `2` and `4` are generally recommended.

For high-performance CPUs, values between `8` and `16` — or even higher — may also be considered.

---

## 5. Running the Pipeline

After configuring the desired parameter settings, users can start the analysis pipeline by clicking:

`Render Mapper` or `Run Pipeline`

The dashboard will automatically:
- build Mapper graph
- execute all selected parameter combinations
- update runtime logs
- display pipeline progress
- generate Mapper comparisons and statistical outputs

Depending on dataset size and parameter complexity, runtime may vary substantially.

Users may stop the pipeline using the stop button if required.

---

## 6. Understanding Pipeline Outputs

### Mapper Visualization

The Mapper graph provides a visual representation of the structural organization of the dataset.

In general, greater visual similarity between the complete case and imputed Mapper graphs may suggest that the imputation process has better preserved the original data structure.

However, Mapper visualizations are primarily intended as an intuitive and exploratory reference rather than a definitive quantitative evaluation.

The final assessment of structural similarity should primarily rely on the quantitative pipeline outputs, including the statistical comparison results and p-value analyses.

### Pipeline Result Table

The pipeline result table summarizes the quantitative comparison results generated under different parameter combinations.

Each row represents one complete pipeline evaluation using a specific set of Mapper parameters.

The table include:
- Mapper parameter settings
- topological comparison statistics
- p-values
- significance indicators
- runtime information

In general:
- lower p-values may suggest larger structural differences between the complete-case and imputed datasets
- higher p-values may suggest greater structural similarity

However, p-values should be interpreted cautiously and in combination with:
- Mapper visualizations
- parameter settings
- overall analytical context

The pipeline outputs are intended to support exploratory structural evaluation rather than provide a single absolute measure of imputation quality.

### p-value Distribution

The p-value distribution plot provides an overview of the statistical comparison results across all tested parameter combinations.

A concentration of lower p-values may suggest broader structural differences between the complete-case and imputed datasets across multiple Mapper settings.

Conversely, higher p-values across many parameter combinations may suggest greater structural consistency.

As with all exploratory TDA analyses, interpretation should consider:
- parameter sensitivity
- Mapper visualization patterns
- overall analytical context

---

## 7. Troubleshooting

### The application appears slow or unresponsive

Large datasets and extensive parameter combinations may substantially increase runtime.

This is expected behavior for computationally intensive TDA workflows.

For initial testing, users are encouraged to:
- reduce permutation counts
- test fewer parameter combinations
- use smaller datasets

### High CPU usage

High CPU utilization during pipeline execution is expected, particularly when using multiple parallel jobs.
