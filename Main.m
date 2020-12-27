%------------------------------------------
% Cell Classification Matlab Script.
%------------------------------------------

% Environment setup.
clear all;
close all;
clc;
rng(10); % Random seed generation.

% -----------------------------------------
% Set up the configurations.
% -----------------------------------------

% Switch to which dataset to analyze
% 1 - GSE63384,	35 lung cancer cell samples, 35 normal lung cell samples
% 2 - GSE40032,	64 endometrial cancer cell samples, 23 normal endometrial cell samples
% 3 - GSE17648,	22 colorectal cancer cell samples, 22 normal colorectal cell sample
% 4 - GSE73003,	20 hepatocellular cancer samples, 20 hepatocellular normal samples
config.DATASET = 1;

% Switch to run transform domain analysis or not.
% 0 - Do not run transform domain analysis
% 1 - Run transform domain analysis
config.RUN_TRANSFORM_ANALYSIS = 1;

% Switch to run WHT cell classification or not.
% 0 - Do not run WHT cell classification
% 1 - Run WHT cell classification
config.RUN_WHT_CLASSIFICATION = 0;

% Switch to run original sequence classification or not.
% 0 - Do not run original sequence classification
% 1 - Run original sequence classification
config.RUN_ORIGINAL_SEQ_CLASSIFICATION = 1;

% -----------------------------------------
% Load data and perform transform domain analysis.
% -----------------------------------------

% Load DNA methylation datasets and display data on command line: T - tumor cells, O - normal cells.
[NumCancerCells, NumNormalCells, cell_samples, SampleSize, data_loaded, Y, species] = ...
    LoadData(config);
    
% Perform transform domain analysis of the three-step values.
[normal_cell_avr_step, tumor_cell_avr_step] = ...
    AnalysisWHTStepValues(config, NumCancerCells, NumNormalCells, data_loaded);

    
% ----------------------------------------------------------------
% Two major routines to perform classification computations:
% ClassifiersWHTDomain() - WHT truncated sequence based classification.
% OriginalSeqClassification() - original sequence based classification
% ----------------------------------------------------------------


if config.RUN_WHT_CLASSIFICATION == 1
    
    % ----------------------------------------
    % Classification with WHT truncated sequence.
    % The machine learning classifiers are evaluated.
    
    [Accuracy_SVM, Accuracy_kNN, Accuracy_DT, ...
        Accuracy_Boosting, Accuracy_Bagging, ...
        Accuracy_Subspace, Accuracy_NN, time_record] = ...
        ClassifiersWHTDomain(config, cell_samples, SampleSize, ...
        NumCancerCells, NumNormalCells, data_loaded, Y, species);
    
end

if config.RUN_ORIGINAL_SEQ_CLASSIFICATION == 1
    
    % ----------------------------------------
    % Classification with original sequence. 
    % The machine learning classifiers are evaluated.
    
    [Accuracy_SVM_OriginalSeq, Accuracy_kNN_OriginalSeq, ...
    Accuracy_DT_OriginalSeq, Accuracy_Boosting_OriginalSeq, ...
    Accuracy_Bagging_OriginalSeq, Accuracy_Subspace_OriginalSeq, ...
    Accuracy_NN_OriginalSeq, time_record_2] = ...
        OriginalSeqClassification(config, cell_samples, SampleSize, ...
        NumCancerCells, NumNormalCells, data_loaded, Y, species)
    
end






