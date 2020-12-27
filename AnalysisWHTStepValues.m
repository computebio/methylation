

function [normal_cell_avr_step, tumor_cell_avr_step] = AnalysisWHTStepValues(config, NumCancerCells, NumNormalCells, data_loaded)
% Performs WHT transformation on the DNA methylation beta value vectors, computes mean values of the three steps, and visualzes three-step results.


NumSteps = 3;

if config.RUN_TRANSFORM_ANALYSIS == 1  
    
    cancer_cell = data_loaded(:,2:(NumCancerCells + 1));
    temp_vector1 = fwht(cancer_cell);
    % Extract from the 2nd element to remove measurement bias in the 1st element.
    wht_domain_vector1 = abs(temp_vector1(2:101,:)); 
    
    cancer_free_cell = data_loaded(:,(NumCancerCells + 2):(NumCancerCells + NumNormalCells + 1));
    temp_vector2 = fwht(cancer_free_cell);
    wht_domain_vector2 = abs(temp_vector2(2:101,:));
    
    for n = 1:NumCancerCells
        for i = 1:NumSteps
            if i==1
                tumor_cell_step(i,n)=mean(wht_domain_vector1(1:3,n:n));
            elseif i==2
                tumor_cell_step(i,n)=mean(wht_domain_vector1(4:7,n:n));
            else
                tumor_cell_step(i,n)=mean(wht_domain_vector1(8:31,n:n));
            end
        end
    end
    tumor_cell_avr_step = mean(tumor_cell_step,2);
    
    
    for n = 1:NumNormalCells
        for i=1:NumSteps
            if i==1
                normal_cell_step(i,n)=mean(wht_domain_vector2(1:3,n:n));
            elseif i==2
                normal_cell_step(i,n)=mean(wht_domain_vector2(4:7,n:n));
            else
                normal_cell_step(i,n)=mean(wht_domain_vector2(8:31,n:n));
            end
        end
    end
    
    normal_cell_avr_step = mean(normal_cell_step,2);
    
    
elseif config.RUN_TRANSFORM_ANALYSIS == 0
    normal_cell_avr_step = 0;
    tumor_cell_avr_step = 0;
end

PLOT_FIGURES = 1;
if (PLOT_FIGURES == 1)
    
    figure(1);
    bar(mean(wht_domain_vector1(1:60,:),2));
    xlabel('WHT transform-domain vector index');
    ylabel('WHT transform-domain vector value');
    
    figure(2);
    bar(mean(wht_domain_vector2(1:60,:),2));
    xlabel('WHT transform-domain vector index');
    ylabel('WHT transform-domain vector value');
    
end



% step_values.normal_cell_avr_step = normal_cell_avr_step;
% step_values.tumor_avr_step = tumor_cell_avr_step;
% A = data_loaded(:,2:(NumCancerCells + 1));
% temp_vector3 = fwht(A);
% wht_domain_vector3 = abs(temp_vector3(2:101,:));
% B = data_loaded(:,(NumCancerCells + 2):(NumCancerCells + NumNormalCells));
% temp_vector4 = fwht(B);
% wht_domain_vector4 = abs(temp_vector4(2:101,:));

