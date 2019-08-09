%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  ____ source codes version 1.0                                    %
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
    methodName='';
    minFeaturePercentage=30;
    maxFeaturePercentage=80;
    fold=3;
    methods = {'GA','WFACOFS','BPSO'};
    methodWisePopulation = zeros(numAgents,numFeatures,size(methods,2));
    methodWiseAccuracy = zeros(1,numAgents,size(methods,2));
    
    
    
    for runNo=1:numRuns
        mkdir(['Results/' datasetName '/'],['Run_' int2str(runNo)]); 
        
        for methodNo=1:size(methods,2)
            path=strcat('Data/Metaheuristic Results/',methods{1,methodNo},'/',datasetName,'/','Run_',int2str(runNo),'/Final/',datasetName,'_result_',methods{1,methodNo},'_pop_',int2str(numAgents),'_iter_',int2str(numIteration),'_',classifierType,'_',int2str(paramValue),'.mat');
%             disp(path);
            tempMemory=importdata(path);
            tempMemory=tempMemory.memory;
            methodWisePopulation(:,:,methodNo)=tempMemory.finalPopulation;
            methodWiseAccuracy(:,:,methodNo)=tempMemory.finalAccuracy;
        end
            
        memory.population=zeros(2*numAgents,numFeatures);
        memory.accuracy=zeros(1,2*numAgents);
        memory.finalPopulation=zeros(0,0);
        memory.finalAccuracy=zeros(0,0);
        iterNo = numIteration;
        
        tic        
        population = zeros(numAgents,numFeatures);        
                  
        for agentNo=1:numAgents
            intersectSelect = [];
            intersectDiscard = [];
            combine=zeros(1,numFeatures);
            for featureNo=1:numFeatures
                state=1;
                for methodNo=1:size(methods,2)                    
                    combine(1,featureNo)=combine(1,featureNo)+(methodWisePopulation(agentNo,featureNo,methodNo)*methodWiseAccuracy(1,agentNo,methodNo));
                    if(methodWisePopulation(agentNo,featureNo,methodNo)==0)
                        state=0;
                        break;
                    end
                    if(state==1)
                        intersectSelect=[intersectSelect,featureNo];
                    end
                end
                state=0;
                for methodNo=1:size(methods,2)
                    if(methodWisePopulation(agentNo,featureNo,methodNo)==1)
                        state=1;
                        break;
                    end
                    if(state==0)
                        intersectDiscard=[intersectDiscard,featureNo];
                    end
                end                
            end
            
            population(agentNo,intersectSelect(1,:))=1;
            averageWeight = mean(combine(1,:));
            population(agentNo,combine>=averageWeight)=1;
            [population,accuracy]=crossValidate(population,classifierType,paramValue,fold);
            memory=updateMemory(memory,population,accuracy);
        end
        
        time=toc;
        [population,accuracy]=crossValidate(population,classifierType,paramValue,fold);
        memory=updateMemory(memory,population,accuracy);
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
