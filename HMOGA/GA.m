function[memory] = GA(train,trainLabel,test,testLabel,numAgents,numIteration,classifierType,paramValue)    
    
    numFeatures=size(test,2);
    %initialize your variable here
    methodName='GA';
    minFeaturePercentage=30;
    maxFeaturePercentage=80;
    mcross=int16(5);
    fold=3;
    
   
    population=dataCreate(numAgents,numFeatures,minFeaturePercentage,maxFeaturePercentage);

    memory.population=zeros(2*numAgents,numFeatures);
    memory.accuracy=zeros(1,2*numAgents);

    tic
    for iterNo=1:numIteration        
        [population,accuracy]=crossValidate(train,trainLabel,population,classifierType,paramValue,fold);                      
        limit = randi(mcross-2,1)+2;
        for loop1=1:limit                  
            accuracyCS(1:numAgents)=accuracy(1:numAgents);
            for loop2= 2:numAgents
                accuracyCS(loop2)=accuracyCS(loop2)+accuracyCS(loop2-1);
            end
            maxcs=accuracyCS(numAgents);
            for loop2= 1:numAgents
                accuracyCS(loop2)=accuracyCS(loop2)/maxcs;
            end            
            firstParentId=find(accuracyCS>rand(1),1,'first');
            secondParentId=find(accuracyCS>rand(1),1,'first');             
            probCross=rand(1);
            probMutation=rand(1);
            [population,accuracy]=crossover(train,trainLabel,population,firstParentId,secondParentId,probCross,probMutation,accuracy,classifierType,paramValue,fold);                                        
        end

        memory=updateMemory(memory,population,accuracy);                     
    end
    time=toc;
    [population,accuracy]=crossValidate(train,trainLabel,population,classifierType,paramValue,fold);
    memory=updateMemory(memory,population,accuracy);
    [memory.finalPopulation,memory.finalAccuracy]=populationRank(train,trainLabel,test,testLabel,population,classifierType,paramValue);
    displayMemory(memory);    
    toc
   
end

function [memory]=updateMemory(memory,population,accuracy)
    numFeatures=size(population,2);
    numAgents=2*size(population,1);
    weight1=100;
    weight2=1.0/numFeatures;
    temp1=(accuracy*weight1)'+((numFeatures-sum(population,2))*weight2);
    temp2=(memory.accuracy*weight1)'+((numFeatures-sum(memory.population,2))*weight2);
    temp1=[temp2;temp1];
    memory.accuracy=[memory.accuracy accuracy];
    memory.population=[memory.population;population];
    [~,index]=sort(temp1,'descend');
    memory.accuracy=memory.accuracy(index);
    memory.population=memory.population(index,:);

    memory.accuracy=memory.accuracy(1:numAgents);
    memory.population=memory.population(1:numAgents,:);
end

function []=displayMemory(memory)
    numAgents=size(memory.finalAccuracy,2);
    disp('Memory - ');
    for loop=1:numAgents
        fprintf('numFeatures - %d\tAccuracy - %f\n',sum(memory.finalPopulation(loop,:)),memory.finalAccuracy(loop));
    end
end
