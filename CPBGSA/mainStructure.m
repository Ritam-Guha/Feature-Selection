%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  CPBGSA source codes version 1.0                                    %
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
    methodName='CPBGSA';
    minFeaturePercentage=30;
    maxFeaturePercentage=80;
    fold=3;
    clusterPercent=40;
    numClusters=int16((clusterPercent/100)*numAgents);
    
    for runNo=1:numRuns
        histogramPop = zeros(numClusters,numFeatures);
        mkdir(['Results/' datasetName '/'],['Run_' int2str(runNo)]);       
        population=dataCreate(numAgents,numFeatures,minFeaturePercentage,maxFeaturePercentage);
                
        tic
        [index, ~, ~] = clusterWithHammingDist(population,minFeaturePercentage,maxFeaturePercentage,numClusters,classifierType,paramValue,fold);
        acc1=zeros(1,numAgents);
        acc2=zeros(1,numClusters);
        for loop1=1:numAgents
            [~,acc1(1,loop1)]=crossValidate(population(loop1,:),classifierType,paramValue,fold);
            for loop2=1:numFeatures
                if(population(loop1,loop2)==1)
                    histogramPop(index(loop1,1),loop2)=histogramPop(index(loop1,1),loop2)+acc1(1,loop1);
                end
            end        
        end             
            
        for loop=1:numClusters
            cutoff = mean(histogramPop(loop,:)); 
            histogramPop(loop,:)=histogramPop(loop,:)>cutoff; 
            [~,acc2(1,loop)]=crossValidate(population(loop1,:),classifierType,paramValue,fold);
        end
        
        time=toc;
        [memory.finalPopulation,memory.finalAccuracy]=mainBGSA(datasetName,numClusters,numIteration,1,classifierType,paramValue,histogramPop);                        
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
    
    numAgents=size(memory.accuracy,2);    
    fprintf('\nIntermediate Memory - \n');
    for loop=1:numAgents/2
        fprintf('numFeatures - %d\tAccuracy - %f\n',sum(memory.population(loop,:)),memory.accuracy(loop));
    end
    numAgents=size(memory.finalAccuracy,2);
    if (numAgents > 0)
        fprintf('\nFinal Memory - \n');
        for loop=1:numAgents
            fprintf('finalNumFeatures - %d\tfinalAccuracy - %f\n',sum(memory.finalPopulation(loop,:)),memory.finalAccuracy(loop));
        end
    end    
end
