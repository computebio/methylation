The source code contains the following configuration switches and routines:

1. The config.DATASET is the configuration switch to select the dataset to analyze. Switch value of 1 is GSE63384 (35 lung cancer cell samples, 35 normal lung cell samples). Switch value of 2 is GSE40032 (64 endometrial cancer cell samples, 23 normal endometrial cell samples). Switch value of 3 is GSE17648 (22 colorectal cancer cell samples, 22 normal colorectal cell samples). Switch value of 4 is GSE73003 (20 hepatocellular cancer samples, 20 normal hepatocellular cell samples). 

2. The routines called in the Main.m program are:

LoadData(): This routine loads the DNA methylation beta value vectors and outputs original sequence and sequence after WHT. 

AnalysisWHTStepValues(): This routine performs WHT transform domain analysis on DNA methylation beta value vectors, computes mean values of three steps, and visualizes three-step results.

ClassifiersWHTDomain(): This routine performs machine learning classifications of cancer and normal cells according to WHT transform-domain vectors, and outputs the classification accuracy results.

OriginalSeqClassification(): This routine performs machine learning classifications by the original sequence and outputs the classification accuracy results.

3. The following switches control to run which data analysis routine:

config.RUN_TRANSFORM_ANALYSIS: Switch to run WHT transform domain analysis routine or not (0 - do not run the analysis; 1 - Run the analysis).

config.RUN_WHT_CLASSIFICATION: Switch to run WHT cell classification routine or not (0 - do not run the analysis; 1 - Run the analysis).

config.RUN_ORIGINAL_SEQ_CLASSIFICATION: Switch to run original sequence classification routine or not (0 - do not run the analysis; 1 - Run the analysis).

