function[] = main(datasetName,numAgents,numIteration,numRuns,classifierType,paramValue)
    warning off;
    global train trainLabel test testLabel fold
    filePath = strcat('Data/',datasetName,'/',datasetName,'_');
    data = importdata(strcat(filePath,'data.mat'));
    train = data.train;
    trainLabel = data.trainLabel;       
    test = data.test;    
    testLabel = data.testLabel;  
    
    numFeatures=size(train,2);   
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
        mkdir(['Results/' datasetName '/'],['Run_' int2str(runNo)]);       
        currentPopulation=dataCreate(numAgents,numFeatures,minFeaturePercentage,maxFeaturePercentage);        
        personalBestPopulation=dataCreate(numAgents,numFeatures,minFeaturePercentage,maxFeaturePercentage);        
        
        memory.population=zeros(2*numAgents,numFeatures);
        memory.accuracy=zeros(1,2*numAgents);
        memory.finalPopulation=zeros(0,0);
        memory.finalAccuracy=zeros(0,0);
        
        tic
        for iterNo=1:numIteration
            mkdir(['Results/' datasetName '/Run_' int2str(runNo)],['Iteration_' int2str(iterNo)]);            
            [velocity]=updateVelocity(velocity,currentPopulation,personalBestPopulation,globalBestPopulation);
            [currentPopulation,currentAccuracy]=updatePositions(velocity,currentPopulation,currentAccuracy,classifierType,paramValue,fold);
            [personalBestPopulation,personalBestAccuracy,globalBestPopulation,globalAccuracy]=updateBest(currentPopulation,currentAccuracy,personalBestPopulation,personalBestAccuracy,globalBestPopulation,globalAccuracy);
            memory=updateMemory(memory,currentPopulation,currentAccuracy);
            displayMemory(memory);
            
            saveFileName = strcat('Results/',datasetName,'/Run_',int2str(runNo),'/Iteration_',int2str(iterNo),'/',datasetName,'_result_',methodName,'_pop_',int2str(numAgents),'_iter_',int2str(numIteration),'_',classifierType,'_',int2str(paramValue),'.mat');
            save(saveFileName,'memory');
        end
        time=toc;
        
        memory=updateMemory(memory,currentPopulation,currentAccuracy);     
        [memory.finalPopulation,memory.finalAccuracy]=populationRank(currentPopulation,classifierType,paramValue);
        displayMemory(memory);
        mkdir(['Results/' datasetName '/Run_' int2str(runNo)],'Final');
        saveFileName = strcat('Results/',datasetName,'/Run_',int2str(runNo),'/Final/',datasetName,'_result_',methodName,'_pop_',int2str(numAgents),'_iter_',int2str(numIteration),'_',classifierType,'_',int2str(paramValue),'.mat');
        save(saveFileName,'memory','time');
        toc
    end
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

function []=displayMemory(memory)
    
    numAgents=size(memory.accuracy,2);    
    disp('Intermediate Memory - ');
    for loop=1:numAgents/2
        fprintf('numFeatures - %d\tAccuracy - %f\n',sum(memory.population(loop,:)),memory.accuracy(loop));
    end
    numAgents=size(memory.finalAccuracy,2);
    if (numAgents > 0)
        disp('Final Memory - ');
        for loop=1:numAgents
            fprintf('finalNumFeatures - %d\tfinalAccuracy - %f\n',sum(memory.finalPopulation(loop,:)),memory.finalAccuracy(loop));
        end
    end    
end
