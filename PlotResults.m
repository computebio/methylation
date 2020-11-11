

% Plotting the results.

load all_data_saved1.mat;

    
figure(1);
% n_vector_idx = 2:5:100;
% x_axis = n_vector(n_vector_idx);

x_axis = n_vector;
plot(x_axis,Accuracy_DT(:,2),'s-k');
hold on;
plot(x_axis,Accuracy_SVM(:,2),'*-k');
hold on;
plot(x_axis,Accuracy_Subspace(:,2),'^-k');
hold on;
plot(x_axis,Accuracy_kNN(:,2),'<-k');
hold on;
plot(x_axis,Accuracy_Bagging(:,2),'o-k');

xlabel('Length of feature space vector');
ylabel('Classification accuracy');
% title('GSE40032');
ylim([0.5,1]);
legend('Decision Tree',...
    'SVM',...
    'Subspace',...
    'kNN',...
    'Bagging');






% NB and AdaBoost cannot be run for whole seq.
% plot(x_axis,Accuracy_NB(:,2),'+-k');
% hold on;
% plot(x_axis,Accuracy_AdaBoost(:,2),'*-k');
% hold on;

% plot(x_axis,Accuracy_LogitBoost(:,2),'s-k');
% hold on;

% Performance issues: Feedfoward, LSTM for short seq.
% LinearModel for both.

% plot(x_axis,Accuracy_LinearModel(:,2),'*-k');
% hold on;




% -----------------------------------------

% xlabel('Length of feature space vector');
% ylabel('Classification accuracy');
% % title('GSE73003');
% ylim([0.5,1]);
% legend('SVM, Holdout Method',...
%     'SVM, k-fold Cross Validation',...
%     'kNN, Holdout Method',...
%     'kNN, k-fold Cross Validation',...
%     'NB, Holdout Method',...
%     'NB, k-fold Cross Validation');



% % FIGURE 1; plot feature space length from 32 to 102.
% 
% figure(1);
% 
% % X=[32;37;42;47;52;57;62;67;72;77;82;87;92;97;102;107;112;117;122;127;132;137;142;147;152;157;162;167;172;177;182;187;192;197];
% % X2=[32;37;42;47;52;57;62;67;72;77;82;87;92;97;102].';
% 
% X2 = [3:3:100];
% plot(X2,RATE(1:27,1),'+-k')
% hold on;
% plot(X2,RATE(1:27,2),'v-k')
% hold on;
% plot(X2,RATE(1:27,3),'o-k')
% hold on
% plot(X2,RATE(1:27,4),'>-k')
% hold on;
% plot(X2,RATE(1:27,5),'s-b')
% hold on;
% plot(X2,RATE(1:27,8),'*-r')
% hold on;
% xlabel('Length of feature space vector after WHT');
% ylabel('Classification accuracy');
% legend('SVM','kNN','Naive Beyas','Decision tree','Adaboost','Bagging');
% ylim([0.8 1]);
% % NOTE: Adaboost are the same to LogitBoost_result, Subspace_KNN_result, Bagging_result;
% 
% 
% 
% % FIGURE 2: plot feature space vector with raw data.
% 
% figure(2);
% 
% NN_result_M_mean = mean(NN_result_M);
% LSTM_result_M_mean = mean(LSTM_result_M);
% 
% 
% NN_LSTM_Comparison = [NN_result_M_mean; LSTM_result_M_mean].';
% 
% NN_LSTM_Comparison_2 = NN_LSTM_Comparison(1:2:end,:);
% x_vector_bar = [100:40:500];
% 
% bar(NN_LSTM_Comparison_2);
% xlabel('Length of feature space vector');
% ylabel('Classification accuracy');
% legend('Feedforward network','LSTM-based RNN network');
% set(gca,'xticklabel',{[100:40:500]});
% ylim([0.7 1]);
% 
% 
% % FIGURE 3: plot feature space vector with raw data.
% 
% % length of feature vector = 300, 11th
% % NumNeuronVector = [5:5:100];
% % VariedFeatureLength = [100:20:500];
% 
% figure(3);
% 
% NN_result_M_300 = NN_result_M(2:2:end,11);
% LSTM_result_M_300 = LSTM_result_M(2:2:end,11);
% NN_LSTM_Comparison_300 = [NN_result_M_300 LSTM_result_M_300];
% x_vector_bar_2 = [10:10:100];
% handle = bar(NN_LSTM_Comparison_300);
% handle(1).FaceColor = 'k';
% handle(2).FaceColor = 'b';
% 
% xlabel('Number of neurons in the second layer');
% ylabel('Classification accuracy');
% legend('Feedforward network','LSTM-based RNN network');
% set(gca,'xticklabel',{x_vector_bar_2});
% ylim([0.8 1]);
% 
% 
% 
% 
% NumNeuronVector = [50:50:600];
% 
% VariedFeatureLength = [100:100:1000];
% 
% NN_result_M(iter_num_neuron,feature_length_iter)
% 
% LSTM_result_M(iter_num_neuron, feature_length_iter)
% 
% 
% 
% 
% 
% 
% figure(3);
% 
% X=[32;37;42;47;52;57;62;67;72;77;82;87;92;97;102;107;112;117;122;127;132;137;142;147;152;157;162;167;172;177;182;187;192;197];
% plot(X,RATE_NN(:,1),'v-b')
% hold on;
% plot(X,RATE_NN(:,3),'+-b')
% hold on;
% plot(X,RATE_LSTM(:,1),'*-b')
% hold on;
% plot(X,RATE_LSTM(:,3),'s-b')
% hold on;
% xlabel('Length of feature space vector');
% ylabel('Classification accuracy');
% legend('Feedforward network with 50 Neurons',...
%     'LSTM network with 200 Neurons',...
%     'LSTM network with 50 Neurons',...
%     'LSTM network with 200 Neurons');
% 
% 
% 
% % NOTE: NN_result fluctuating.
% % Plot bar graph to compare with LSTM.
% % Plot averaging result of NN and LSTM.
% 
% RATE_NN_Mean = mean(RATE_NN);
% RATE_LSTM_Mean = mean(RATE_LSTM);
% 
% 
% figure(2);
% X=[32;37;42;47;52;57;62;67;72;77;82;87;92;97;102;107;112;117;122;127;132;137;142;147;152;157;162;167;172;177;182;187;192;197];
% plot(X,NN_result(1,:),'*-b')
% hold on;
% plot(X,NN_result(3,:),'^-b')
% hold on;
% plot(X,NN_result(5,:),'+-b')
% hold on;
% % plot(X,NN_result(7,:),'<-b')
% % hold on;
% % plot(X,NN_result(9,:),'s-b')
% % hold on;
% plot(X,RATE(:,5),'*-r')
% hold on;
% xlabel('Length of feature space vector');
% ylabel('Classification accuracy');
% legend('Neural Network with 10 Neurons',...
%     'Neural Network with 20 Neurons',...
%     'Neural Network with 30 Neurons',...
%     'Ensemble method - Adaboost');




% ---------------------------------
% Backup Code.


%     figure(1);
%     X=[5:5:100]';
%     plot(X,Accuracy_SVM(:,1),'*-k');
%     hold on;
%     plot(X,Accuracy_SVM(:,2),'o-r');
%     hold on;
%     plot(X,Accuracy_kNN(:,1),'v-k');
%     hold on;
%     plot(X,Accuracy_kNN(:,2),'+-r');
%     hold on;
%     plot(X,Accuracy_NB(:,1),'<-k');
%     hold on;
%     plot(X,Accuracy_NB(:,2),'s-r');
%     hold on;
%     xlabel('Length of feature space vector');
%     ylabel('Classification accuracy');
%     % title('GSE73003');
%     ylim([0.5,1]);
%     legend('SVM, Holdout Method',...
%         'SVM, k-fold Cross Validation',...
%         'kNN, Holdout Method',...
%         'kNN, k-fold Cross Validation',...
%         'NB, Holdout Method',...
%         'NB, k-fold Cross Validation');
%
%
%     figure(2);
%     X=[5:5:100]';
%     plot(X,Accuracy_DT(:,1),'*-k');
%     hold on;
%     plot(X,Accuracy_DT(:,2),'o-r');
%     hold on;
%     plot(X,Accuracy_Boosting(:,1),'v-k');
%     hold on;
%     plot(X,Accuracy_Boosting(:,2),'+-r');
%     hold on;
%     xlabel('Length of feature space vector');
%     ylabel('Classification accuracy');
%     title('GSE73003');
%     ylim([0.5,1]);
%     legend('Decision Tree, Holdout Method',...
%         'Decision Tree, k-fold Cross Validation',...
%         'AdaBoost, Holdout Method',...
%         'AdaBoost, k-fold Cross Validation');
%
%
%
%     for n_iter = 1:length(n_vector)
%
%         n = n_vector(n_iter);
%
%         data = cell_samples(:,2:n);
%
%         X = data;
%
%         NumSample = SampleSize - 1;
%         for k = 1:NumSample
%             X_PriorWHT_Complete{k,1} = M(:,k+1).';
%         end
%
%
%         % Mdl = fitensemble(data,species,'AdaBoostM1',50,'Tree','Holdout',0.5);
%         % L1 = kfoldLoss(Mdl,'Mode','Cumulative');
%         % AdaBoostM1_result =1-L1(end)
%
%
%         time_record(5) = toc;
%
%
%         rng default; % For reproducibility
%         Mdl = fitensemble(data,species,'LogitBoost',50,'Tree','Holdout',0.5);
%         % L = resubLoss(Mdl,'Mode','Cumulative');
%         % LogitBoost_result =1-L(end)
%         L2 = kfoldLoss(Mdl,'Mode','Cumulative');
%         LogitBoost_result =1-L2(end)
%
%         Mdl = fitensemble(data,species,'RobustBoost',50,'Tree','Holdout',0.5);
%         % L = resubLoss(Mdl,'Mode','Cumulative');
%         % RobustBoost_result =1-L(end)
%         L3 = kfoldLoss(Mdl,'Mode','Cumulative');
%         RobustBoost_result =1-L3(end)
%
%         Mdl = fitensemble(data,species,'Subspace',50,'Discriminant','Holdout',0.5);
%         % L = resubLoss(Mdl,'Mode','Cumulative');
%         % Subspace_Discr_result =1-L(end)
%         L6 = kfoldLoss(Mdl,'Mode','Cumulative');
%         Subspace_Discr_result =1-L6(end)
%
%
% %         Mdl = fitensemble(data,species,'Subspace',10,'KNN');
% %         L7 = resubLoss(Mdl);
% %         Subspace_KNN_result =1-L7(end)
%
%
%         Mdl = fitensemble(data,species,'Bag',50,'Tree','Type','classification','Holdout',0.5);
%         L8 = kfoldLoss(Mdl,'Mode','Cumulative');
%         Bagging_result =1-L8(end)
%
%
%         % Random forest: Bootstrap-aggregated (bagged) decision trees
%         Mdl = TreeBagger(50,data,species,'OOBPrediction','On',...
%             'Method','classification');
%         TreeBagger_Loss = oobError(Mdl);
%         % L = resubLoss(Mdl);
%         random_forest =1-TreeBagger_Loss(end)
%
%
%         % RATE(n_iter,1)=svm;
%         RATE(n_iter,2)=knn;
%         RATE(n_iter,3)=beyas;
%         RATE(n_iter,4)=tree;
%         RATE(n_iter,5)=AdaBoostM1_result;
%         RATE(n_iter,6)=LogitBoost_result;
%         RATE(n_iter,7)=RobustBoost_result;
%         RATE(n_iter,8)=Subspace_Discr_result;
%         % RATE(n_iter,9)=Subspace_KNN_result;
%         RATE(n_iter,9)=Bagging_result;
%         RATE(n_iter,10)=random_forest;
%
%
%         x_source = data.';
%
%         target1 = repmat([1,0]',1,NumCancerCells);
%         target2 = repmat([0,1]',1,NumNormalCells);
%
%         target = [target1, target2];
%
%         tic;
%
%         % NumNeuronVector = [10:10:100];
%
%         NumNeuronVector = 100;
%
%         for iter_num_neuron = 1:length(NumNeuronVector)
%
%             num_neuron = NumNeuronVector(iter_num_neuron);
%
%             net = patternnet(num_neuron);
%
%             [net,tr] = train(net,x_source,target);
%
%             outputs = net(x_source);
%             errors = gsubtract(target,outputs);
%
%             NN_result_temp = 1 - perform(net,target,outputs);
%
%             NN_result(iter_num_neuron, n_iter) = NN_result_temp;
%
%         end
%
%         time_record(6) = toc;
%         tic;
%
%         % ------------------------------
%         % LSTM
%
%         %         for k = 1:size(Y)
%         %             if Y{k} == 'T'
%         %                 Y_LSTM(k,1) = 1;
%         %             elseif Y{k} == 'O'
%         %                 Y_LSTM(k,1) = 2;
%         %             end
%         %         end
%
%         Y_LSTM = categorical(Y);
%
%
%         [NumCases,lengthVector] = size(X);
%
%         % Set the length of feature space to the
%         % DNA methylation vector.
%         for n2 = 1:NumCases
%             X_LSTM{n2} = X(n2,:)';
%         end
%
%
%         % NumLSTMNeuronVector = [10:10:100];
%         NumLSTMNeuronVector = 100;
%
%         for iter_num_neuron = 1:length(NumLSTMNeuronVector)
%
%             num_lstm = NumLSTMNeuronVector(iter_num_neuron);
%
%             inputSize = lengthVector;
%             outputSize = num_lstm;
%
%             outputMode = 'last';
%             numClasses = 2;
%
%             layers = [ ...
%                 sequenceInputLayer(inputSize)
%                 lstmLayer(outputSize,'OutputMode',outputMode)
%                 fullyConnectedLayer(numClasses)
%                 softmaxLayer
%                 classificationLayer]
%
%             maxEpochs = 300;
%             miniBatchSize = 20;
%             options = trainingOptions('sgdm', ...
%                 'MaxEpochs',maxEpochs, ...
%                 'MiniBatchSize',miniBatchSize);
%
%             clear net;
%
%             net = trainNetwork(X_LSTM,Y_LSTM,layers,options);
%
%             % For complete data training, must run on HPC.
%             % net = trainNetwork(X_PriorWHT_Complete,Y_LSTM,layers,options);
%
%             clear YPred;
%
%             miniBatchSize = 20;
%             YPred = classify(net,X_LSTM, ...
%                 'MiniBatchSize',miniBatchSize);
%
%             accuracy = sum(YPred == Y_LSTM)./numel(Y_LSTM);
%
%             LSTM_result(iter_num_neuron, n_iter) = ...
%                 accuracy;
%
%         end
%
%         time_record(7) = toc;
%
%         % ------------------------------
%         % RNN
%
%         % NumNeuronVector = [10:5:20];
%         %
%         % for iter_num_neuron = 1:length(NumNeuronVector)
%         %
%         % num_neuron = NumNeuronVector(iter_num_neuron);
%         %
%         % % [X,T] = simpleseries_dataset;
%         % net_rnn = layrecnet(1:2,num_neuron);
%         %
%         % [Xs,Xi,Ai,Ts] = preparets(net_rnn,x_source,target);
%         % net_rnn = train(net_rnn,Xs,Ts,Xi,Ai);
%         % view(net)
%         % Y = net_rnn(Xs,Xi,Ai);
%         % perf_rnn = perform(net_rnn,Y,Ts);
%         %
%         % RNN_result(iter_num_neuron, (n-27)/5) = 1 - perf_rnn;
%         %
%         % end
%
%
%
%     end
%
%
% %     for n = 32:5:200
% %         RATE_NN((n-27)/5,:) = NN_result(:, (n-27)/5);
% %         RATE_LSTM((n-27)/5,:) = LSTM_result(:, (n-27)/5);
% %     end
%
%
%     RATE_NN = NN_result.';
%     RATE_LSTM = LSTM_result.';
%
%
%     save all_data.mat;
%
%
% end
%
%
