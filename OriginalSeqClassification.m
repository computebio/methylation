    

function [Accuracy_SVM_OriginalSeq, Accuracy_kNN_OriginalSeq, ...
    Accuracy_DT_OriginalSeq, Accuracy_Boosting_OriginalSeq, ...
    Accuracy_Bagging_OriginalSeq, Accuracy_Subspace_OriginalSeq, ...
    Accuracy_NN_OriginalSeq, time_record_2] = ...
        OriginalSeqClassification(config, cell_samples, SampleSize, ...
        NumCancerCells, NumNormalCells, data_loaded, Y, species)

% Performs machine learning classifications by the original sequence. 
% In each classifier, the kfold loss is loss at k = 3, and the leaveout loss
% is the loss at k = sample size.
% For example, for SVM, svm_kfold_loss is loss at k = 3, and
% svm_leaveout_loss is the loss at k = sample size.



X_M = data_loaded(:,2:(NumCancerCells + NumNormalCells + 1)).';

[NumCases,lengthVector] = size(X_M);

% The whole sequence is the input sequence of all classifiers.
VariedFeatureLength = lengthVector;

for n_iter = length(VariedFeatureLength)
    
    varied_feature_length = VariedFeatureLength(n_iter);
    
    X = X_M(:,1:varied_feature_length);
    
    Y = species;
    
    % ----------------------------------
    % Support Vector Machine.
    
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
    
    
    Accuracy_SVM_OriginalSeq(n_iter,:) = [1-svm_leaveout_loss;...
        1-svm_kfold_loss]
    
    
    time_record_2(n_iter,1) = toc;
    tic;
    
    % ----------------------------------
    % k-Nearest Neighbors.
    
    kNNModel = fitcknn(X,Y);
    CV_kNNModel = crossval(kNNModel, 'Holdout', 0.3);
    knn_holdout_loss = kfoldLoss(CV_kNNModel);
    
    CV_kNNModel = crossval(kNNModel, 'Kfold', 3);
    knn_kfold_loss = kfoldLoss(CV_kNNModel);
    
    CV_kNNModel = crossval(kNNModel, 'Leaveout', 'on');
    knn_leaveout_loss = kfoldLoss(CV_kNNModel);
    
    Accuracy_kNN_OriginalSeq(n_iter,:) = [1-knn_leaveout_loss;...
        1-knn_kfold_loss]
    
    time_record_2(n_iter,2) = toc;
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
    
    Accuracy_DT_OriginalSeq(n_iter,:) = [1-dt_leaveout_loss;...
        1-dt_kfold_loss]

    time_record_2(n_iter,4) = toc;
    tic;
    
    
    % ----------------------------------
    % Ensemble Method of Boosting.
    
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
    
    Accuracy_Boosting_OriginalSeq(n_iter,:) = [1-en_leaveout_loss;...
        1-en_kfold_loss]
    
    time_record_2(n_iter,6) = toc;
    tic;
    
    
    % ----------------------------------
    % Ensemble Method of Bagging.
    
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
    
    Accuracy_Bagging_OriginalSeq(n_iter,:) = [1-bg_leaveout_loss;...
        1-bg_kfold_loss]
    
    time_record_2(n_iter,7) = toc;
    tic;
    
    
    % ----------------------------------
    % Ensemble Method of Subspace  
    
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
    
    Accuracy_Subspace_OriginalSeq(n_iter,:) = [1-bg_leaveout_loss;...
        1-bg_kfold_loss]
    
    time_record_2(n_iter,8) = toc;
    tic;
    
        
    % -----------------------------
    % Feedforward Neural Network.
    
    x_source = X.';
    
    target1 = repmat([1,0]',1,NumCancerCells);
    target2 = repmat([0,1]',1,NumNormalCells);
    
    target = [target1, target2];
    
    tic;
    
    NumNeuronVector = 100;
    
    for iter_num_neuron = 1:length(NumNeuronVector)
        
        num_neuron = NumNeuronVector(iter_num_neuron);
        
        net = patternnet(num_neuron);
        
        [net,tr] = train(net,x_source,target);
        
        outputs = net(x_source);
        errors = gsubtract(target,outputs);
        
        NN_result_temp = 1 - perform(net,target,outputs);
        
        Accuracy_NN_OriginalSeq(n_iter, iter_num_neuron) = ...
            NN_result_temp;
        
    end
    
    
    time_record_2(n_iter,10) = toc;
    tic;
    
    
end



