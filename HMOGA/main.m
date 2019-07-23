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
%                 pp. 471-479. Springer, Singapore, 2018.           %                                        %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[] = main(datasetName,numAgents,numIteration,numRuns,classifierType,paramValue)
    filePath = strcat('Data/',datasetName,'/',datasetName,'_');
    train = importdata(strcat(filePath,'train.mat'));
    train = train.input;
    trainLabel = importdata(strcat(filePath,'train_label.mat'));   
    trainLabel = trainLabel.input1;
    test = importdata(strcat(filePath,'test.mat'));
    test = test.test;
    testLabel = importdata(strcat(filePath,'test_label.mat'));     
    testLabel = testLabel.test1;
        
    %initialize your variable here
    methodName='HMOGA';    
    
    for runNo=1:numRuns
        mkdir(['Results/' datasetName '/'],['Run_' int2str(runNo)]);       
        
        histogram=zeros(1,size(train,2));
        numTrainRows=size(train,1);
        numTestRows=size(test,1);
        count1=int16(numTrainRows*.75);
        count2=int16(numTestRows*.75);
        numGARun=5;
        
        tic
        for GANo=1:numGARun
            fprintf('GA running for the time - %d\n',GANo);
            temp1=rand(1,numTrainRows);
            [~,temp1]=sort(temp1);
            temp2=rand(1,numTestRows);
            [~,temp2]=sort(temp2);
            [memoryGA]=GA(train(temp1(1:count1),:),trainLabel(temp1(1:count1),:),test(temp2(1:count2),:),testLabel(temp2(1:count2),:),numAgents,numIteration,classifierType,paramValue);
            temp1 = memoryGA.finalPopulation;
            temp2 = memoryGA.finalAccuracy;                        
            disp('------------------------------------------------------');            
            for loop=1:numAgents
                temp1(loop,:)=temp1(loop,:).*temp2(1,loop);
            end
            histogram=histogram+sum(temp1,1);
        end
        cutoff=rms(histogram);        
        features=histogram(1,:)>cutoff;
        time=toc;        
        [memory.finalPopulation,memory.finalAccuracy]=populationRank(train,trainLabel,test,testLabel,features,classifierType,paramValue);
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
        fprintf('numFeatures - %d\tAccuracy - %f\numGARun',sum(memory.finalPopulation(loop,:)),memory.finalAccuracy(loop));
    end
end
