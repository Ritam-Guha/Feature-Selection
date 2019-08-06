function [performance]=mlpClassifier(trainTemp,trainLabel,test,testLabel,agent,paramValue)
    hiddenLayerSize = paramValue;  
    net = patternnet(hiddenLayerSize);
    net.trainParam.showWindow = false;
    numAgents=size(agent,1);
    performance=zeros(1,numAgents);
     for loop1=1:numAgents        
        if (sum(agent(loop1,:)==1)==0)
            performance(1,loop1)=0;
        else
            [row,col]=size(trainLabel);
            target=trainLabel(1:row,1:col);     
            input=trainTemp(1:row,agent(loop1,:)==1);

            inputs = input';
            targets = target';

            % Setup Division of Data for Training, Validation, Testing        
            net.divideParam.trainRatio = 85/100;
            net.divideParam.valRatio = 15/100;
            net.divideParam.testRatio = 0/100;

            % Train the Network           
            [net,~] = train(net,inputs,targets);

            % Test the Network
            [row,col]=size(testLabel);
            target=testLabel(1:row,1:col);

            %for normal selection
            input=test(1:row,agent(loop1,:)==1);

            inputs = input';
            targets = target';
            outputs = net(inputs);

            %outputs
            [differ,~] = confusion(targets,outputs);
            performance(1,loop1)=(1-differ)*100;      
%             fprintf('The number of features  : %d\n', sum(agent(loop1,:)==1));
%             fprintf('Percentage Correct Classification   : %f%%\n', 100*performance(1,loop1));        
        end
     end
end