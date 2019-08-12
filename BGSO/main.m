%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  BGSO source codes version 1.0                                    %
%                                                                   %
%  Developed in MATLAB R2016a                                       %
%                                                                   %
%   Main Paper:                                                     %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[] = main(datasetName,numAgents,numIteration,numRuns,classifierType,paramValue)
    
    global train trainLabel test testLabel fold
    filePath = strcat('Data/',datasetName,'/',datasetName,'_');
    data = importdata(strcat(filePath,'data.mat'));
    train = data.train;
    trainLabel = data.trainLabel;       
    test = data.test;    
    testLabel = data.testLabel;     
    
    numFeatures=size(test,2);
    %initialize your variable here
    methodName='BGSO';
    fold=3;
    
    for runNo=1:numRuns
        mkdir(['Results/' datasetName '/'],['Run_' int2str(runNo)]);       
        
        population=zeros(1,numFeatures);
        memory.finalPopulation=zeros(0,0);
        memory.finalAccuracy=zeros(0,0);
        
        tic;
        [popGA,accGA]=mainGA(datasetName,numAgents,numIteration,1,classifierType,paramValue);
        [popPSO,accPSO]=mainPSO(datasetName,numAgents,numIteration,1,classifierType,paramValue);
        
        for loop=1:numAgents
            population(1,:)=double(population(1,:)) + double(popPSO(loop,:)).*(accPSO(1,loop));
            population(1,:)=double(population(1,:)) + double(popGA(loop,:)).*(accGA(1,loop));
        end
        
        population = population >= mean(population);
        accuracy=crossValidate(population,classifierType,paramValue,fold);
           
        for loop=1:numFeatures
            temp=population;
            temp(1,loop)=~population(1,loop);
            tempAccuracy=crossValidate(temp,classifierType,paramValue,fold);
            if(tempAccuracy>accuracy)
              population=temp;
              accuracy=tempAccuracy;
           end
        end
        
        time=toc;
        [memory.finalPopulation,memory.finalAccuracy]=populationRank(population,classifierType,paramValue);
        displayMemory(memory);
        mkdir(['Results/' datasetName '/Run_' int2str(runNo)],'Final');
        saveFileName = strcat('Results/',datasetName,'/Run_',int2str(runNo),'/Final/',datasetName,'_result_',methodName,'_pop_',int2str(numAgents),'_iter_',int2str(numIteration),'_',classifierType,'_',int2str(paramValue),'.mat');
        save(saveFileName,'memory','time');        
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
    if (numAgents > 0)
        fprintf('\nFinal Memory - \n');
        for loop=1:numAgents
            fprintf('finalNumFeatures - %d\tfinalAccuracy - %f\n',sum(memory.finalPopulation(loop,:)),memory.finalAccuracy(loop));
        end
    end    
end
