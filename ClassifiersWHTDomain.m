
function [Accuracy_SVM, Accuracy_kNN, Accuracy_DT, ...
        Accuracy_Boosting, Accuracy_Bagging, ...
        Accuracy_Subspace, Accuracy_NN, time_record] = ...
        ClassifiersWHTDomain(config, cell_samples, SampleSize, ...
        NumCancerCells, NumNormalCells, data_loaded, Y, species)

% Performs machine learning classifications of cancer and normal cells 
% according to WHT transform-domain vectors.
% In each classifier, the kfold loss is loss at k = 3, and the leaveout loss
% is the loss at k = sample size.
% For example, for SVM, svm_kfold_loss is loss at k = 3, and
% svm_leaveout_loss is the loss at k = sample size.


% The classification accuracy with WHT is the averaged result of 
% classification accuracies at three WHT feature space values equaling 83, 89, and 95
n_vector = [83, 89, 95];

for n_iter = 1:length(n_vector)
    
    n = n_vector(n_iter);
    
    data = cell_samples(:,2:n);
    
    X = data;
    
    NumSample = SampleSize - 1;
    for k = 1:NumSample
        X_PriorWHT_Complete{k,1} = data_loaded(:,k+1).';
    end
    
    
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
    
    
    Accuracy_SVM(n_iter,:) = [1-svm_leaveout_loss;...
        1-svm_kfold_loss]
    
    
    time_record(n_iter,1) = toc;
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
    
    Accuracy_kNN(n_iter,:) = [1-knn_leaveout_loss;...
        1-knn_kfold_loss]
    
    time_record(n_iter,2) = toc;
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
    
    Accuracy_DT(n_iter,:) = [1-dt_leaveout_loss;...
        1-dt_kfold_loss]
    
    time_record(n_iter,4) = toc;
    tic;
    
    
    % ----------------------------------
    % Ensemble classifier of LogitBoost.
    
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
    
    Accuracy_LogitBoost(n_iter,:) = [1-en_leaveout_loss;...
        1-en_kfold_loss]
    
    time_record(n_iter,6) = toc;
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
    
    Accuracy_Bagging(n_iter,:) = [1-bg_leaveout_loss;...
        1-bg_kfold_loss]
    
    time_record(n_iter,7) = toc;
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
    
    Accuracy_Subspace(n_iter,:) = [1-bg_leaveout_loss;...
        1-bg_kfold_loss]
    
    time_record(n_iter,8) = toc;
    tic;
    
    
    % -----------------------------
    
    x_source = data.';
    
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
        
        Accuracy_NN(n_iter, iter_num_neuron) = NN_result_temp;
        
    end
    
    time_record(n_iter,10) = toc;
    tic;
        
end
