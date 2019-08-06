%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  HMOGA source codes version 1.0                                   %
%                                                                   %
%  Developed in MATLAB R2016a                                       %
%                                                                   %
%   Main Paper: Ghosh, Manosij, Ritam Guha, Riktim Mondal,          %
%                 Pawan Kumar Singh, Ram Sarkar, and Mita Nasipuri. %
%                 "Feature selection using histogram-based          %
%                 multi-objective GA for handwritten Devanagari     %
%                 numeral recognition."                             %
%                 In Intelligent Engineering Informatics,           %
%                 pp. 471-479. Springer, Singapore, 2018.           %                                                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[] = main(datasetName,numAgents,numIteration,numRuns,classifierType,paramValue)
    warning off;
    global train trainLabel test testLabel trainTotal trainLabelTotal testTotal testLabelTotal fold
    filePath = strcat('Data/',datasetName,'/',datasetName,'_');
    data = importdata(strcat(filePath,'data.mat'));
    trainTotal = data.train;
    trainLabelTotal = data.trainLabel;       
    testTotal = data.test;    
    testLabelTotal = data.testLabel;  
        
    %initialize your variable here
    methodName='WFACOFS';    
    
    for runNo=1:numRuns
        mkdir(['Results/' datasetName '/'],['Run_' int2str(runNo)]);       
        
        histogram=zeros(1,size(trainTotal,2));
        numTrainRows=size(trainTotal,1);
        numTestRows=size(testTotal,1);
        count1=int16(numTrainRows*.75);
        count2=int16(numTestRows*.75);
        numGARun=5;
        fold=3;
        
        tic
        for GANo=1:numGARun
            fprintf('GA running for the time - %d\n',GANo);
            temp1=rand(1,numTrainRows);
            [~,temp1]=sort(temp1);
            temp2=rand(1,numTestRows);
            [~,temp2]=sort(temp2);
            train=trainTotal(temp1(1:count1),:);
            trainLabel=trainLabelTotal(temp1(1:count1),:);
            test=testTotal(temp2(1:count2),:);
            testLabel=testLabelTotal(temp2(1:count2),:);            
            [memoryGA]=GA(numAgents,numIteration,classifierType,paramValue);
            temp1 = memoryGA.finalPopulation;
            temp2 = memoryGA.finalAccuracy;                        
            disp('------------------------------------------------------');            
            for loop=1:numAgents
                histogram=histogram + temp1(loop,:).*temp2(1,loop);
            end
        end        
        cutoff=rms(histogram);            
        features=histogram(1,:)>cutoff;
        time=toc;        
        [memory.finalPopulation,memory.finalAccuracy]=populationRank(features,classifierType,paramValue);
        displayMemory(memory);
        mkdir(['Results/' datasetName '/Run_' int2str(runNo)],'Final');
        saveFileName = strcat('Results/',datasetName,'/Run_',int2str(runNo),'/Final/',datasetName,'_result_',methodName,'_pop_',int2str(numAgents),'_iter_',int2str(numIteration),'_',classifierType,'_',int2str(paramValue),'.mat');
        save(saveFileName,'memory','time');
        toc
    end
end

function []=displayMemory(memory)        
    numAgents=size(memory.finalAccuracy,2);
    if (numAgents > 0)
        disp('Final Memory - ');
        for loop=1:numAgents
            fprintf('finalNumFeatures - %d\tfinalAccuracy - %f\n',sum(memory.finalPopulation(loop,:)),memory.finalAccuracy(loop));
        end
    end    
end
