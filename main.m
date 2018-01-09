originaldata = importdata('lenses.txt');
[data, label] = readData(originaldata);
[train_data,train_label,test_data,test_label] = split(data,label);
%--------------------------------------------------------------------
%traning and testing the data
%prediction for traning and testing data
 [ prior, age, specpres, astig, tear ] = NaiveBayesTrain( train_data, train_label );
 [train_acc, predicted_label ] = NaiveBayesPredict( train_data, train_label, prior,age, specpres, astig, tear );
 [test_acc, test_predict] = NaiveBayesPredict(test_data, test_label, prior,age, specpres, astig, tear);
 cm = confusionMatrix(test_label, test_predict);
 %--------------------------------------------------------------------
%report the result
disp('With encoding');
disp(['Training accuracy: ' num2str(train_acc) '%']);
disp(['Test accuracy: ' num2str(test_acc) '%']);
disp('Confusion matrix');
cm