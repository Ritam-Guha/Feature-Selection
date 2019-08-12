function[population,accuracy] = mainPSO(datasetName,numAgents,numIteration,numRuns,classifierType,paramValue)
    warning off;
    global test fold
    
    numFeatures=size(test,2);   
    %initialize your variable here
    methodName='BPSO';
    currentAccuracy=zeros(1,numAgents);
    personalBestAccuracy=zeros(1,numAgents);
    globalBestPopulation=zeros(1,numFeatures);
    globalAccuracy=0;
    velocity=zeros(numAgents,numFeatures);
    minFeaturePercentage=30;
    maxFeaturePercentage=80;
    fold=3;
    
    for runNo=1:numRuns
        currentPopulation=dataCreate(numAgents,numFeatures,minFeaturePercentage,maxFeaturePercentage);        
        personalBestPopulation=dataCreate(numAgents,numFeatures,minFeaturePercentage,maxFeaturePercentage);        
        
        for iterNo=1:numIteration
            [velocity]=updateVelocity(velocity,currentPopulation,personalBestPopulation,globalBestPopulation);
            [currentPopulation,currentAccuracy]=updatePositions(velocity,currentPopulation,currentAccuracy,classifierType,paramValue,fold);
            [personalBestPopulation,personalBestAccuracy,globalBestPopulation,globalAccuracy]=updateBest(currentPopulation,currentAccuracy,personalBestPopulation,personalBestAccuracy,globalBestPopulation,globalAccuracy);
        end       
           
        [population,accuracy]=populationRank(currentPopulation,classifierType,paramValue);
    end
end

function [velocity]=updateVelocity(velocity,currentPopulation,personalBestPopulation,globalBestPopulation)
    rng('shuffle');
    [numAgents,numFeatures]=size(velocity);
    for loop1=1:numAgents
        for loop2=1:numFeatures
            velocity(loop1,loop2)=velocity(loop1,loop2)+(rand(1)*(personalBestPopulation(loop1,loop2)-currentPopulation(loop1,loop2)))+(rand(1)*(globalBestPopulation(1,loop2)-currentPopulation(loop1,loop2)));
        end
    end
end

function [currentPopulation,currentAccuracy]=updatePositions(velocity,currentPopulation,currentAccuracy,classifierType,paramValue,fold)
    rng('shuffle');
    [numAgents,numFeatures]=size(velocity);
    for loop1=1:numAgents
        for loop2=1:numFeatures
            temp=1/(1+exp(velocity(loop1,loop2)));
            if(rand(1)<temp)
                currentPopulation(loop1,loop2)=1;
            else
                currentPopulation(loop1,loop2)=0;
            end
        end
        [currentPopulation(loop1,:),currentAccuracy(1,loop1)]=crossValidate(currentPopulation(loop1,:),classifierType,paramValue,fold);
    end
end

function [personalBestPopulation,personalBestAccuracy,globalBestPopulation,globalAccuracy]=updateBest(currentPopulation,currentAccuracy,personalBestPopulation,personalBestAccuracy,globalBestPopulation,globalAccuracy)
    [numAgents,~]=size(currentPopulation);
    for loop1=1:numAgents
        if(currentAccuracy(1,loop1)>personalBestAccuracy(1,loop1))
            personalBestPopulation(loop1,:)=currentPopulation(loop1,:);
            personalBestAccuracy(1,loop1)=currentAccuracy(1,loop1);
        end
        if(globalAccuracy<currentAccuracy(1,loop1))
            globalBestPopulation(1,:)=currentPopulation(loop1,:);
            globalAccuracy=currentAccuracy(1,loop1);
        end
    end
end
