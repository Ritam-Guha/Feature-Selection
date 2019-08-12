function [finalPopulation,finalAccuracy]=crossValidate(agent,classifierType,paramValue,fold)
    
    global train trainLabel
    
    rng('default');
    numAgents=size(agent,1);
    rows=size(train,1);
    selection=createDivision(fold);    
    finalAccuracy=zeros(1,numAgents);    
    finalPopulation=agent;
    for loop1=1:numAgents
        data=train(:,agent(loop1,:)==1);
        accuracy=zeros(1,fold);
        if (size(data,2)==0)
            finalAccuracy(1,loop1) = 0;
            return;
        end
        for loop2=1:fold
            trainTestFormat=zeros(rows,1);
            for loop3=1:rows
                if selection(1,loop3)==loop2
                    trainTestFormat(loop3,1)=1;
                end
            end
            crossTrain=train(trainTestFormat(:)==0,:);
            crossTrainLabel=trainLabel(trainTestFormat(:)==0,:);
            crossTest=train(trainTestFormat(:)==1,:);
            crossTestLabel=trainLabel(trainTestFormat(:)==1,:);
            accuracy(1,loop2) = classify(crossTrain,crossTrainLabel,crossTest,crossTestLabel,agent(loop1,:),classifierType,paramValue);              
        end
        finalAccuracy(1,loop1) = mean(accuracy);
    end    
    [~,index]=sort(finalAccuracy,'descend');
    finalAccuracy=finalAccuracy(index);
    finalPopulation=finalPopulation(index,:);
end

function [selection] = createDivision(fold)
    global trainLabel 
    rows = size(trainLabel,1);
    sizeTraining=size(trainLabel,1);
    labels=zeros(1,sizeTraining);
    for loop=1:sizeTraining
        labels(1,loop)=find(trainLabel(loop,:),1);
    end
    maxLabelNum = max(labels);
    selection=zeros(1,rows);
    for loop1=1:maxLabelNum
        count1=sum(labels(:)==loop1);        
        samplesPerFold=int16(floor((count1/fold)));
        for loop2=1:fold
            count=0;
            for loop3=1:rows
                if(labels(loop3)==loop1 && selection(loop3)==0)
                    selection(loop3)=loop2;
                    count=count+1;
                end
                if(count==samplesPerFold)
                    break;
                end
            end
        end
        loop2=1;
        for loop3=1:rows
            if(selection(loop3)==0 && labels(loop3)==loop1)
                selection(loop3)=loop2;
                loop2=loop2+1;
            end
        end
    end
end
