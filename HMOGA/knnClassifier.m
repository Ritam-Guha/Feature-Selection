function [performance]=knnClassifier(train,trainLabel,test,testLabel,agent,paramValue)
    numAgents=size(agent,1);
    performance=zeros(1,numAgents);
    for loop1=1:numAgents
        if(sum(agent(loop1,:)==1)~=0)
            classifyTest=test(:,agent(loop1,:)==1);
            classifyTrain=train(:,agent(loop1,:)==1);        
            trainSize=size(trainLabel,1);
            label=zeros(1,trainSize);
            for loop2=1:trainSize
                label(1,loop2)=find(trainLabel(loop2,:),1);           
            end
            knnModel=fitcknn(classifyTrain,label,'NumNeighbors',paramValue,'Standardize',1);
            [label,~] = predict(knnModel,classifyTest);     
            testSize=size(testLabel,1);
            lab=zeros(testSize,1);
            for loop2=1:testSize                 
                lab(loop2,1)=find(testLabel(loop2,:),1);           
            end
            differ = sum(lab ~= label)/testSize; 
            performance(1,loop1)=(1-differ)*100;        
        else
            performance(1,loop1)=0;      
        end
    end
end