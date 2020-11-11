

function step_values = AnalysisWHTStepValues(config, NumCancerCells, NumNormalCells, data_loaded)
% ----------------------------------------------------
% Description: 
% Input(s):
% Output(s):
% ----------------------------------------------------


if config.RUN_TRANSFORM_ANALYSIS == 1
    
    
    cancer = data_loaded(:,2:(NumCancerCells + 1));
    y = fwht(cancer);
    y1=abs(y(2:101,:));
    % figure(1)
    %  plot(y1)
    
    cancer_free = data_loaded(:,(NumCancerCells + 2):(NumCancerCells + NumNormalCells + 1));
    y = fwht(cancer_free);
    y2=abs(y(2:101,:));
    %  figure(2)
    %  plot(y2)
    
    
    PLOT_FIGURES = 0;
    if (PLOT_FIGURES == 1)
        
        figure(1);
        histogram(mean(data_loaded(:,2:(NumCancerCells + 1)),2),50);
        xlabel('Beta value');
        ylabel('Number of beta values in the defined range');
        title('GSE40032, cancer cells');
        ylim([1,3000]);
        
        
        figure(2);
        bar(mean(y1(1:50,:),2));
        xlabel('WHT transform-domain vector index');
        ylabel('WHT transform-domain vector value');
        title('GSE40032, cancer cells');
        ylim([1,0.05]);
        
        
        figure(3);
        bar(mean(y2(1:50,:),2));
        xlabel('WHT transform-domain vector index');
        ylabel('WHT transform-domain vector value');
        title('GSE40032, normal cells');
        ylim([1,0.05]);
        
        
    end
    
    
    A=data(:,2:(NumCancerCells + 1));
    y = fwht(A);
    y1=abs(y(2:101,:));
    for n=1:NumCancerCells
        for i=1:3
            if i==1
                tumor_step(i,n)=mean(y1(1:3,n:n));
            elseif i==2
                tumor_step(i,n)=mean(y1(4:7,n:n));
            else
                tumor_step(i,n)=mean(y1(8:31,n:n));
            end
        end
    end
    tumor_avr_step=mean(tumor_step,2)
    
    
    B = data_loaded(:,(NumCancerCells + 2):(NumCancerCells + NumNormalCells));
    y=fwht(B);
    y2=abs(y(2:101,:));
    for n=1:NumNormalCells
        for i=1:3
            if i==1
                non_tumor_step(i,n)=mean(y2(1:3,n:n));
            elseif i==2
                non_tumor_step(i,n)=mean(y2(4:7,n:n));
            else
                non_tumor_step(i,n)=mean(y2(8:31,n:n));
            end
        end
    end

    nontumor_avr_step=mean(non_tumor_step,2)
    
    % non_tumor_step=non_tumor_step';
    
    
elseif config.RUN_TRANSFORM_ANALYSIS == 0
    tumor_avr_step = 0;
    nontumor_avr_step = 0;
    
end

step_values.tumor_avr_step = tumor_avr_step;
step_values.nontumor_avr_step = nontumor_avr_step;


% figure(1);
% plot(nontumor_avr_step);


