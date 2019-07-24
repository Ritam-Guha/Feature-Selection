%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  DGA source codes version 1.0                                     %
%                                                                   %
%  Developed in MATLAB R2016a                                       %
%                                                                   %
%   Main Paper: Guha, Ritam, Manosij Ghosh, Souvik Kapri,           % 
%                 Sushant Shaw, Shyok Mutsuddi, Vikrant Bhateja,    %
%                 and Ram Sarkar.                                   %
%                 "Deluge based Genetic Algorithm for               %
%                 feature selection."                               %
%                 Evolutionary Intelligence (2019): 1-11.           %
%                                                                   %                                         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[] = main(datasetName,numAgents,numIteration,numRuns,classifierType,paramValue)    
    
    global train trainLabel test testLabel 
    rng('shuffle');
    filePath = strcat('Data/',datasetName,'/',datasetName,'_');
    train = importdata(strcat(filePath,'train.mat'));
    train = train.input;
    trainLabel = importdata(strcat(filePath,'train_label.mat'));   
    trainLabel = trainLabel.input1;
    test = importdata(strcat(filePath,'test.mat'));
    test = test.test;
    testLabel = importdata(strcat(filePath,'test_label.mat'));     
    testLabel = testLabel.test1;
    
    numFeatures=size(test,2);
    %initialize your variable here
    methodName='DGA';
    minFeaturePercentage=30;
    maxFeaturePercentage=80;
    mcross=int16(5);
    fold=3;
    
    for runNo=1:numRuns
        mkdir(['Results/' datasetName '/'],['Run_' int2str(runNo)]);       
        population=dataCreate(numAgents,numFeatures,minFeaturePercentage,maxFeaturePercentage);
        [population,accuracy]=crossValidate(population,classifierType,paramValue,fold);
        
        memory.population=zeros(2*numAgents,numFeatures);
        memory.accuracy=zeros(1,2*numAgents);
        
        tic
        for iterNo=1:numIteration
            fprintf('itearion no-------%d\n',iterNo);
            mkdir(['Results/' datasetName '/Run_' int2str(runNo)],['Iteration_' int2str(iterNo)]);
                                  
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
                [population,accuracy]=crossoverGDA(population,firstParentId,secondParentId,probCross,accuracy,classifierType,paramValue,fold);                                  
            end
            population=GDA(population,classifierType,paramValue,fold);
            [population,accuracy]=crossValidate(population,classifierType,paramValue,fold);
            memory=updateMemory(memory,population,accuracy);             
            saveFileName = strcat('Results/',datasetName,'/Run_',int2str(runNo),'/Iteration_',int2str(iterNo),'/',datasetName,'_result_',methodName,'_pop_',int2str(numAgents),'_iter_',int2str(numIteration),'_',classifierType,'_',int2str(paramValue),'.mat');
            save(saveFileName,'memory');
        end
        time=toc;
        [population,accuracy]=crossValidate(population,classifierType,paramValue,fold);
        memory=updateMemory(memory,population,accuracy);
        [memory.finalPopulation,memory.finalAccuracy]=populationRank(train,trainLabel,test,testLabel,population,classifierType,paramValue);
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
    numAgents=size(memory.finalAccuracy,2);
    disp('Memory - ');
    for loop=1:numAgents
        fprintf('numFeatures - %d\tAccuracy - %f\n',sum(memory.finalPopulation(loop,:)),memory.finalAccuracy(loop));
    end
end
