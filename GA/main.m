function[] = main(datasetName,numAgents,numIteration,numRuns,classifierType,paramValue)
    warning off;
    global train trainLabel test testLabel fold
    filePath = strcat('Data/',datasetName,'/',datasetName,'_');
    data = importdata(strcat(filePath,'data.mat'));
    train = data.train;
    trainLabel = data.trainLabel;       
    test = data.test;    
    testLabel = data.testLabel;  
    
    numFeatures=size(test,2);
    %initialize your variable here
    methodName='GA';
    minFeaturePercentage=30;
    maxFeaturePercentage=80;
    mcross=int16(5);
    fold=3;
    
    for runNo=1:numRuns
        mkdir(['Results/' datasetName '/'],['Run_' int2str(runNo)]);       
        population=dataCreate(numAgents,numFeatures,minFeaturePercentage,maxFeaturePercentage);
        
        memory.population=zeros(2*numAgents,numFeatures);
        memory.accuracy=zeros(1,2*numAgents);
        memory.finalPopulation=zeros(0,0);
        memory.finalAccuracy=zeros(0,0);
        
        tic
        for iterNo=1:numIteration
            mkdir(['Results/' datasetName '/Run_' int2str(runNo)],['Iteration_' int2str(iterNo)]);
            [population,accuracy]=crossValidate(population,classifierType,paramValue,fold);                      
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
                [population,accuracy]=crossover(population,firstParentId,secondParentId,probCross,probMutation,accuracy,classifierType,paramValue,fold);
                fprintf('\n');
            end
            
            memory=updateMemory(memory,population,accuracy);   
            displayMemory(memory);
            saveFileName = strcat('Results/',datasetName,'/Run_',int2str(runNo),'/Iteration_',int2str(iterNo),'/',datasetName,'_result_',methodName,'_pop_',int2str(numAgents),'_iter_',int2str(numIteration),'_',classifierType,'_',int2str(paramValue),'.mat');
            save(saveFileName,'memory');
        end
        time=toc;
        [population,accuracy]=crossValidate(population,classifierType,paramValue,fold);
        memory=updateMemory(memory,population,accuracy);
        [memory.finalPopulation,memory.finalAccuracy]=populationRank(population,classifierType,paramValue);
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

