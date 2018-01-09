original_data = importdata('iris.data');
[data, label] = readData(original_data);
[train_data,train_label,test_data,test_label] = split(data,label);
%--------------------------------------------------------------------
%traning and testing the data
%prediction for traning and testing data
[ prior, sl, sw, pl, pw ] = NaiveBayesTrain( train_data, train_label );
[train_acc, train_predict ] = NaiveBayesPredict( train_data, train_label, prior, sl, sw, pl, pw ) ;
[test_acc, test_predict] = NaiveBayesPredict(test_data, test_label, prior, sl, sw, pl, pw);
cm = confusionMatrix(test_label, test_predict);
%--------------------------------------------------------------------
%report the result
disp('With encoding');
disp(['Training accuracy: ' num2str(train_acc) '%']);
disp(['Test accuracy: ' num2str(test_acc) '%']);
disp('Confusion matrix');
cm