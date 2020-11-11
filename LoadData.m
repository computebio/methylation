function [NumCancerCells, NumNormalCells, cell_samples, SampleSize, data_loaded, Y] = ...
    LoadData(config)
 
switch config.DATASET
    case 1
        M = importdata('GSE63384.mat');
        species=importdata('species35to35.mat');
        NumCancerCells = 35;
        NumNormalCells = 35;
    case 2
        M = importdata('GSE40032.mat');
        species=importdata('species64to23.mat');
        NumCancerCells = 64;
        NumNormalCells = 23;
    case 3
        M = importdata('GSE17648.mat');
        species=importdata('species22to22.mat');
        NumCancerCells = 22;
        NumNormalCells = 22;
    case 4
        M = importdata('GSE73003.mat');
        species=importdata('species.mat');
        NumCancerCells = 20;
        NumNormalCells = 20;
end

Y = species;
tabulate(Y);

cell_samples = M(:,2:end);
cell_samples = fwht(cell_samples);
cell_samples = abs(cell_samples);
cell_samples = cell_samples';

SampleSize = size(M,2);

data_loaded = M;

    