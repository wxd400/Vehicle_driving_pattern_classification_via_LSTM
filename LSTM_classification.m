%   LSTM classification

RandomOrder=randperm(size(Xdata,1));
Xdata = Xdata(RandomOrder);
Ydata = Ydata(RandomOrder);

XTrain=[];
YTrain=[];
XTest=[];
YTest=[];

numTrainSteps = floor(0.6*numel(Xdata));
numValidSteps = floor(0.8*numel(Xdata));

XTrain = Xdata(1:numTrainSteps,1);
YTrain = Ydata(1:numTrainSteps,1);
XValid = Xdata(numTrainSteps+1:numValidSteps,1);
YValid = Ydata(numTrainSteps+1:numValidSteps,1);
XTest = Xdata(numValidSteps+1:end,1);
YTest = Ydata(numValidSteps+1:end,1);

Grid_research_accuracy=[];

inputSize = 15;
numHiddenUnits = 25;
numClasses = 2;

numObservationsTrain = numel(XTrain);
for i=1:numObservationsTrain
    sequence = XTrain{i};
    sequenceLengthsTrain(i) = size(sequence,2);
end

[sequenceLengthsTrain,idx] = sort(sequenceLengthsTrain);
XTrain = XTrain(idx);
YTrain = YTrain(idx);

numObservationsValid = numel(XValid);
for i=1:numObservationsValid
    sequence = XValid{i};
    sequenceLengthsValid(i) = size(sequence,2);
end
[sequenceLengthsValid,idx] = sort(sequenceLengthsValid);
XValid = XValid(idx);
YValid = YValid(idx);


numObservationsTest = numel(XTest);
for i=1:numObservationsTest
    sequence = XTest{i};
    sequenceLengthsTest(i) = size(sequence,2);
end
[sequenceLengthsTest,idx] = sort(sequenceLengthsTest);
XTest = XTest(idx);
YTest = YTest(idx);



layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% maxEpochs = 100;
miniBatchSize = 10;

%     'MaxEpochs',maxEpochs, ...
%     'ValidationData',{XValid,YValid}, ...
%     'ValidationFrequency',16,...

for i_epoch=1:1:700

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.0005,...
    'MaxEpochs',1, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
     'Verbose',1);%, ...
     %'Plots','training-progress');
 
if i_epoch==1
    [net,info_train ] = trainNetwork(XTrain,YTrain,layers,options);
else
    [net,info_train ] = trainNetwork(XTrain,YTrain,net.Layers,options);
end

CurveLoss(i_epoch,1)=info_train.TrainingLoss(1,15);
CurveAccuracy(i_epoch,1)=info_train.TrainingAccuracy(1,15);

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'InitialLearnRate',1.0e-300,...
    'MaxEpochs',1, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
     'Verbose',1);%, ...
     %'Plots','training-progress');
 
[net,info_valid ] = trainNetwork(XValid,YValid,net.Layers,options);
CurveLoss(i_epoch,2)=info_valid.TrainingLoss(1,5);
CurveAccuracy(i_epoch,2)=info_valid.TrainingAccuracy(1,5);

[net,info_test ] = trainNetwork(XTest,YTest,net.Layers,options);
CurveLoss(i_epoch,3)=info_test.TrainingLoss(1,5);
CurveAccuracy(i_epoch,3)=info_test.TrainingAccuracy(1,5);

end


trac_1=smooth(CurveAccuracy(:,1),20);
trac_2=smooth(CurveAccuracy(:,2),20);
trac_3=smooth(CurveAccuracy(:,3),20);
plot(trac_1,'DisplayName','trac_1');hold on;plot(trac_2,'DisplayName','trac_2');plot(trac_3,'DisplayName','trac_3');hold off;
trls_1=smooth(CurveLoss(:,1),20);
trls_2=smooth(CurveLoss(:,2),20);
trls_3=smooth(CurveLoss(:,3),20);
plot(trls_1,'DisplayName','trls_1');hold on;plot(trls_2,'DisplayName','trls_2');plot(trls_3,'DisplayName','trls_3');hold off;
