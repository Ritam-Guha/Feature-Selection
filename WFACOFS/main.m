%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WFACOFS source codes version 1.0                                 %
%                                                                   %
%  Developed in MATLAB R2016a                                       %
%                                                                   %
%   Main Paper: Ghosh, Manosij, Ritam Guha, Ram Sarkar,             %
%                 and Ajith Abraham.                                %
%                 "A wrapper-filter feature selection technique     %
%                 based on ant colony optimization."                %
%                 Neural Computing and Applications: 1-19.          %                                                
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[] = main(datasetName,numAgents,numIteration,numRuns,classifierType,paramValue)
    warning off;
    global train trainLabel test testLabel fold
    
    filePath = strcat('Data/',datasetName,'/',datasetName,'_');
    data = importdata(strcat(filePath,'data.mat'));
    train = data.train;
    trainLabel = data.trainLabel;       
    test = data.test;    
    testLabel = data.testLabel;         
    simMatrix=load(strcat('Data/simMatrix/simMatrix_',datasetName,'.mat'));
    simMatrix=simMatrix.matrix;
    
    methodName='WFACOFS';
    alpha=1;
    beta=1;
    phi=0.8;
    decay=0.15;
    numFeatures=size(train,2);
    mean=numFeatures*.75;
    prevEntry=1;
    minFeaturePercentage=20;
    maxFeaturePercentage=50;
    fold=3;
    
    for runNo=1:numRuns
        mkdir(['Results/' datasetName '/'],['Run_' int2str(runNo)]);
        pheromone=zeros(1,numFeatures);
        population=dataCreate(numAgents,numFeatures,minFeaturePercentage,maxFeaturePercentage);

        memory.population=zeros(2*numAgents,numFeatures);
        memory.accuracy=zeros(1,2*numAgents);        
        memory.finalPopulation=zeros(0,0);
        memory.finalAccuracy=zeros(0,0);
        
        average=zeros(1,numIteration);
        pheromone=pheromone+0.1; 
        tic
        for iterNo=1:numIteration
            mkdir(['Results/' datasetName '/Run_' int2str(runNo)],['Iteration_' int2str(iterNo)]);
            [population,accuracy]=crossValidate(population,classifierType,paramValue,fold);                
            pheromone=generatePheromone(population,accuracy,pheromone,phi,decay);
            average(1,iterNo)=sum(accuracy)/numAgents;
            for agentNo=1:numAgents
                order=rand(1,numFeatures);
                [~,order]=sort(order);
                feature=int16(normrnd(mean,numFeatures*0.1));
                feature=max(feature,5);
                feature=min(feature,numFeatures);
%                 fprintf('features to be added - %d\n',feature);        
                count=0;
                temp=zeros(1,numFeatures);
                for loop=1:numFeatures
                    if count>feature
                        break;
                    end            
                    if count ~= 0
                        value=featureProbability(temp,simMatrix,pheromone,prevEntry,order(loop),alpha,beta);
                    else               
                        value=1;
                    end            
                    if rand<value                    
                        temp(1,order(loop))=1;
                        count = count+1;
                        prevEntry=order(loop);            
                    end
                end
                population(agentNo,:)=temp;
            end
            memory=updateMemory(memory,population,accuracy);
            displayMemory(memory);
            mean=sum(memory.accuracy*sum(memory.population,2))/sum(memory.accuracy);                  
            mean=mean+(mean*0.1); 
%             fprintf('Mean-%f\n',mean);
            saveFileName = strcat('Results/',datasetName,'/Run_',int2str(runNo),'/Iteration_',int2str(iterNo),'/',datasetName,'_result_',methodName,'_pop_',int2str(numAgents),'_iter_',int2str(numIteration),'_',classifierType,'_',int2str(paramValue),'.mat');
            save(saveFileName,'memory');
        end
        time=toc;
        [population,accuracy]=crossValidate(population,classifierType,paramValue,fold);
        memory=updateMemory(memory,population,accuracy);
        [memory.finalPopulation,memory.finalAccuracy]=populationRank(population,classifierType,paramValue);
        displayMemory(memory);
        mkdir(['Results/' datasetName '/Run_' int2str(runNo)],['Final']);
        saveFileName = strcat('Results/',datasetName,'/Run_',int2str(runNo),'/Final/',datasetName,'_result_',methodName,'_pop_',int2str(numAgents),'_iter_',int2str(numIteration),'_',classifierType,'_',int2str(paramValue),'.mat');
        save(saveFileName,'memory','time');
        toc
    end
end

function [pheromone]=generatePheromone(population,accuracy,pheromone,phi,decay)
    numAgents=size(population,1);
    numFeatures=size(population,2);
    pheromone=pheromone*(1-decay);
    pheromoneIncrement=zeros(1,numFeatures);
    for loop=1:numAgents
        if loop==1            
            temp=2*((phi*accuracy(1,loop))+(1-phi)*((numFeatures-sum(population(loop,:)==1))/numFeatures));
        else            
            temp=(phi*accuracy(1,loop))+(1-phi)*((numFeatures-sum(population(loop,:)==1))/numFeatures);
        end
        pheromoneIncrement(1,(population(loop,:)==1))=pheromoneIncrement(1,(population(loop,:)==1))+temp;
    end
    pheromone=pheromone+pheromoneIncrement;
    pheromone=pheromone/max(pheromone);
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

function [value]=featureProbability(ant,simMatrix,pheromone,prev,pos,alpha,beta)
    index=find(ant(1,:)==0);
    total=0;
    for loop=1:size(index,2)    
        if (isempty(pheromone(1,index(1,loop))) || isempty(pheromone(1,index(1,loop))))
            disp(index);
            disp(loop);
            stop;
        end
        total=total+((pheromone(1,index(1,loop))^alpha)*(simMatrix(index(1,loop),prev)^beta));  
    end
    value=(pheromone(1,pos)^alpha)*(simMatrix(pos,prev)^beta);
    total=total/size(index,2);
    value=(value)/(total); 
    value=value/1.75;
    value=min(value,1);
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
