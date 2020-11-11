    

function [Accuracy_SVM_OriginalSeq, Accuracy_kNN_OriginalSeq, ...
    Accuracy_NB_OriginalSeq, Accuracy_DT_OriginalSeq, ...
    Accuracy_AdaBoost_OriginalSeq, Accuracy_LogitBoost_OriginalSeq, ...
    Accuracy_Bagging_OriginalSeq, Accuracy_Subspace_OriginalSeq, ...
    Accuracy_LinearModel_OriginalSeq, Accuracy_NN_OriginalSeq, ...
    Accuracy_LSTM_OriginalSeq, time_record_2] = ...
        OriginalSeqClassification(config, cell_samples, SampleSize, ...
        data_loaded, Y)


% Orignal Sequence Classification.

X_M = M(:,2:(NumCancerCells + NumNormalCells + 1)).';

[NumCases,lengthVector] = size(X_M);

% VariedFeatureLength = [100:1e3:lengthVector];

% VariedFeatureLength = [20:20:500];

% The whole sequence.
VariedFeatureLength = lengthVector;

for n_iter = 1:length(VariedFeatureLength)
    n_iter
    
    varied_feature_length = VariedFeatureLength(n_iter);
    varied_feature_length
    
    X = X_M(:,1:varied_feature_length);
    
    Y = species;
    
    % ----------------------------------
    % SVM.
    
    tic;
    
    SVMModel = fitcsvm(X,Y,'Standardize',true,'KernelFunction','RBF',...
        'KernelScale','auto');
    svm_resub_loss = resubLoss(SVMModel);
    
    
    CVSVMModel_Test_Sample = fitcsvm(X,Y,'Holdout',0.2,'ClassNames',{'T','O'},...
        'Standardize',true);
    CompactSVMModel = CVSVMModel_Test_Sample.Trained{1}; % Extract the trained, compact classifier
    testInds = test(CVSVMModel_Test_Sample.Partition);   % Extract the test indices
    XTest = X(testInds,:);
    YTest = Y(testInds,:);
    svm_holdout_loss = loss(CompactSVMModel,XTest,YTest);
    
    
    SVMModel = fitcsvm(X,Y,'Standardize',true,'ClassNames',{'T','O'});
    CVSVMModel = crossval(SVMModel,'Kfold', 3);
    svm_kfold_loss = kfoldLoss(CVSVMModel);
    
    
    SVMModel = fitcsvm(X,Y,'Standardize',true,'ClassNames',{'T','O'});
    NumSample = size(X,1);
    CVSVMModel = crossval(SVMModel,'Kfold', NumSample);
    svm_leaveout_loss = kfoldLoss(CVSVMModel);
    
    
    Accuracy_SVM_OriginalSeq(n_iter,:) = [1-svm_holdout_loss;...
        1-svm_kfold_loss;...
        1-svm_leaveout_loss]
    
    
    time_record_2(n_iter,1) = toc;
    tic;
    
    % ----------------------------------
    % kNN.
    
    kNNModel = fitcknn(X,Y);
    CV_kNNModel = crossval(kNNModel, 'Holdout', 0.3);
    knn_holdout_loss = kfoldLoss(CV_kNNModel);
    
    CV_kNNModel = crossval(kNNModel, 'Kfold', 3);
    knn_kfold_loss = kfoldLoss(CV_kNNModel);
    
    CV_kNNModel = crossval(kNNModel, 'Leaveout', 'on');
    knn_leaveout_loss = kfoldLoss(CV_kNNModel);
    
    Accuracy_kNN_OriginalSeq(n_iter,:) = [1-knn_holdout_loss;...
        1-knn_kfold_loss;...
        1-knn_leaveout_loss]
    
    time_record_2(n_iter,2) = toc;
    tic;
    
    
    % ----------------------------------
    % Naive Bayes Model.
    
    % Naive Bayes cannot be run for this vector length.
    
    NBNoClassification = 1;
    
    if NBNoClassification == 1

        Accuracy_NB_OriginalSeq(n_iter,:) = [NaN; NaN; NaN];
        
    
    else
        
        NB_Model = fitcnb(X,Y);
        CV_NB_Model = crossval(NB_Model,'Holdout',0.3);
        nb_holdout_loss = kfoldLoss(CV_NB_Model);
        
        CV_NB_Model = crossval(NB_Model,'Kfold', 3);
        nb_kfold_loss = kfoldLoss(CV_NB_Model);
        
        CV_NB_Model = crossval(NB_Model,'Leaveout', 'on');
        nb_leaveout_loss = kfoldLoss(CV_NB_Model);
        
        Accuracy_NB_OriginalSeq(n_iter,:) = [1-nb_holdout_loss;...
            1-nb_kfold_loss;...
            1-nb_leaveout_loss];
        
    end
    
    time_record_2(n_iter,3) = toc;
    tic;
    
    
    % ----------------------------------
    % Decision Tree.
    
    DT_Model = fitctree(X,Y);
    CV_DT_Model = crossval(DT_Model,'Holdout',0.3);
    dt_holdout_loss = kfoldLoss(CV_DT_Model);
    
    CV_DT_Model = crossval(DT_Model,'Kfold', 3);
    dt_kfold_loss = kfoldLoss(CV_DT_Model);
    
    CV_DT_Model = crossval(DT_Model,'Leaveout', 'on');
    dt_leaveout_loss = kfoldLoss(CV_DT_Model);
    
    Accuracy_DT_OriginalSeq(n_iter,:) = [1-dt_holdout_loss;...
        1-dt_kfold_loss;...
        1-dt_leaveout_loss]
        x
    time_record_2(n_iter,4) = toc;
    tic;
    
    
    % ----------------------------------
    % Ensemble classifier of AdaBoost.
    
    % AdaBoost cannot be run for the long vector length.
    
    AdaBoostNoClassification = 1;
    
    if AdaBoostNoClassification == 1
        
        Accuracy_AdaBoost_OriginalSeq(n_iter,:) = [NaN; NaN; NaN];
        
    else
        
        Ensemble_Model = fitensemble(X,Y,'AdaBoostM1',50,'Tree');
        CV_Ensemble_Model = crossval(Ensemble_Model,...
            'Holdout',0.3);
        en_holdout_loss = kfoldLoss(CV_Ensemble_Model);
        
        CV_Ensemble_Model = crossval(Ensemble_Model,...
            'Kfold', 3);
        en_kfold_loss = kfoldLoss(CV_Ensemble_Model);
        
        CV_Ensemble_Model = crossval(Ensemble_Model,...
            'Leaveout', 'on');
        en_leaveout_loss = kfoldLoss(CV_Ensemble_Model);
        
        Accuracy_AdaBoost_OriginalSeq(n_iter,:) = [1-en_holdout_loss;...
            1-en_kfold_loss;...
            1-en_leaveout_loss];
    end
    
    time_record_2(n_iter,5) = toc;
    tic;
    
    
    % ----------------------------------
    % Ensemble classifier of LogitBoost.
    
    LogitBoostNoClassification = 0;
    
    if LogitBoostNoClassification == 1
        
        Accuracy_LogitBoost_OriginalSeq(n_iter,:) = [NaN; NaN; NaN];
        
    else
        
        Ensemble_Model = fitensemble(X,Y,'LogitBoost',50,'Tree');
        CV_Ensemble_Model = crossval(Ensemble_Model,...
            'Holdout',0.3);
        en_holdout_loss = kfoldLoss(CV_Ensemble_Model);
        
        CV_Ensemble_Model = crossval(Ensemble_Model,...
            'Kfold', 3);
        en_kfold_loss = kfoldLoss(CV_Ensemble_Model);
        
        CV_Ensemble_Model = crossval(Ensemble_Model,...
            'Leaveout', 'on');
        en_leaveout_loss = kfoldLoss(CV_Ensemble_Model);
        
        Accuracy_LogitBoost_OriginalSeq(n_iter,:) = [1-en_holdout_loss;...
            1-en_kfold_loss;...
            1-en_leaveout_loss]
        
    end
    
    time_record_2(n_iter,6) = toc;
    tic;
    
    
    % ----------------------------------
    % Ensemble classifier of Bagging.
    
    Ensemble_Model = fitensemble(X,Y,'Bag',50,'Tree',...
        'type','classification');
    
    CV_Ensemble_Model = crossval(Ensemble_Model,...
        'Holdout',0.3);
    bg_holdout_loss = kfoldLoss(CV_Ensemble_Model);
    
    CV_Ensemble_Model = crossval(Ensemble_Model,...
        'Kfold', 3);
    bg_kfold_loss = kfoldLoss(CV_Ensemble_Model);
    
    CV_Ensemble_Model = crossval(Ensemble_Model,...
        'Leaveout', 'on');
    bg_leaveout_loss = kfoldLoss(CV_Ensemble_Model);
    
    Accuracy_Bagging_OriginalSeq(n_iter,:) = [1-bg_holdout_loss;...
        1-bg_kfold_loss;...
        1-bg_leaveout_loss]
    
    time_record_2(n_iter,7) = toc;
    tic;
    
    
    % ----------------------------------
    % Ensemble classifier of Subspace.
    % Random Subspace requires kNN learners.
    % Other ensemble method cannot adopt kNN learners.
    
    Ensemble_Model = fitensemble(X,Y,'Subspace',50,'KNN',...
        'type','classification');
    
    CV_Ensemble_Model = crossval(Ensemble_Model,...
        'Holdout',0.3);
    bg_holdout_loss = kfoldLoss(CV_Ensemble_Model);
    
    CV_Ensemble_Model = crossval(Ensemble_Model,...
        'Kfold', 3);
    bg_kfold_loss = kfoldLoss(CV_Ensemble_Model);
    
    CV_Ensemble_Model = crossval(Ensemble_Model,...
        'Leaveout', 'on');
    bg_leaveout_loss = kfoldLoss(CV_Ensemble_Model);
    
    Accuracy_Subspace_OriginalSeq(n_iter,:) = [1-bg_holdout_loss;...
        1-bg_kfold_loss;...
        1-bg_leaveout_loss]
    
    time_record_2(n_iter,8) = toc;
    tic;
    
    
    % ----------------------------------
    % Logistic Linear Model.
    % Tested: loss too high, not applicable.
    
    LinearModelNoClassification = 1;
    
    if LinearModelNoClassification == 1
        
       Accuracy_LinearModel_OriginalSeq(n_iter,:) = [NaN; NaN; NaN]; 
        
    else
        
        Logistic_LinearModel = fitclinear(X,Y,...
            'Learner','logistic','Holdout',0.3);
        logistic_linear_holdout_loss =...
            kfoldLoss(Logistic_LinearModel);
        
        Logistic_LinearModel = fitclinear(X,Y,...
            'Learner','logistic','Kfold', 3);
        logistic_linear_kfold_loss =...
            kfoldLoss(Logistic_LinearModel);
        
        Logistic_LinearModel = fitclinear(X,Y,...
            'Learner','logistic','Leaveout', 'on');
        logistic_linear_leaveout_loss =...
            kfoldLoss(Logistic_LinearModel);
        
        Accuracy_LinearModel_OriginalSeq(n_iter,:) = [1-logistic_linear_holdout_loss;...
            1-logistic_linear_kfold_loss;...
            1-logistic_linear_leaveout_loss];
        
    end
    
    time_record_2(n_iter,9) = toc;
    tic;
    
    
    % save all_data.mat;
    
    
    % -----------------------------
    
    NN_LSTM_No_Classifcation = 0;
    
    if NN_LSTM_No_Classifcation == 1
        
        Accuracy_NN_OriginalSeq(n_iter, iter_num_neuron) = NaN;
        
    else
        
        x_source = X.';
        
        target1 = repmat([1,0]',1,NumCancerCells);
        target2 = repmat([0,1]',1,NumNormalCells);
        
        target = [target1, target2];
        
        tic;
        
        % NumNeuronVector = [10:10:100];
        
        NumNeuronVector = 100;
        
        for iter_num_neuron = 1:length(NumNeuronVector)
            
            num_neuron = NumNeuronVector(iter_num_neuron);
            
            net = patternnet(num_neuron);
            
            [net,tr] = train(net,x_source,target);
            
            outputs = net(x_source);
            errors = gsubtract(target,outputs);
            
            NN_result_temp = 1 - perform(net,target,outputs);
            
            Accuracy_NN_OriginalSeq(n_iter, iter_num_neuron) = ...
                NN_result_temp
            
        end
        
    end
    
    
    time_record_2(n_iter,10) = toc;
    tic;
    
    % ------------------------------
    % LSTM
    
        
    if NN_LSTM_No_Classifcation == 1

        Accuracy_LSTM_OriginalSeq(n_iter,iter_num_neuron) = ...
            NaN;
         
    else
        
        Y_LSTM = categorical(Y);
        
        
        [NumCases,lengthVector] = size(X);
        
        % Set the length of feature space to the
        % DNA methylation vector.
        for n2 = 1:NumCases
            X_LSTM{n2} = X(n2,:)';
        end
        
        % NumLSTMNeuronVector = [10:10:100];
        NumLSTMNeuronVector = 100;
        
        for iter_num_neuron = 1:length(NumLSTMNeuronVector)
            
            num_lstm = NumLSTMNeuronVector(iter_num_neuron);
            
            inputSize = lengthVector;
            outputSize = num_lstm;
            
            outputMode = 'last';
            numClasses = 2;
            
            layers = [ ...
                sequenceInputLayer(inputSize)
                lstmLayer(outputSize,'OutputMode',outputMode)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer]
            
            maxEpochs = 300;
            miniBatchSize = 20;
            options = trainingOptions('sgdm', ...
                'MaxEpochs',maxEpochs, ...
                'MiniBatchSize',miniBatchSize);
            
            clear net;
            
            net = trainNetwork(X_LSTM,Y_LSTM,layers,options);
            
            % For complete data training, must run on HPC.
            % net = trainNetwork(X_PriorWHT_Complete,Y_LSTM,layers,options);
            
            clear YPred;
            
            miniBatchSize = 20;
            YPred = classify(net,X_LSTM, ...
                'MiniBatchSize',miniBatchSize);
            
            accuracy = sum(YPred == Y_LSTM)./numel(Y_LSTM);
            
            Accuracy_LSTM_OriginalSeq(n_iter,iter_num_neuron) = ...
                accuracy
            
        end
        
    end
    
    time_record_2(n_iter,11) = toc;
    
    % save all_data.mat;
    
end


save long_seq_result.mat Accuracy_SVM_OriginalSeq Accuracy_kNN_OriginalSeq ...
    Accuracy_DT_OriginalSeq  Accuracy_LogitBoost_OriginalSeq Accuracy_Bagging_OriginalSeq ...
    Accuracy_Subspace_OriginalSeq Accuracy_NN_OriginalSeq ...
    Accuracy_LSTM_OriginalSeq time_record_2;




