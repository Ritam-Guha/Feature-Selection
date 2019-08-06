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
    methodName='BGSA';
    minFeaturePercentage=30;
    maxFeaturePercentage=80;
    fold=3;
    velocities = zeros(numAgents,numFeatures);
    
    for runNo=1:numRuns
        mkdir(['Results/' datasetName '/'],['Run_' int2str(runNo)]);       
        population=dataCreate(numAgents,numFeatures,minFeaturePercentage,maxFeaturePercentage);

        memory.population=zeros(2*numAgents,numFeatures);
        memory.accuracy=zeros(1,2*numAgents);
        
        tic
        for iterNo=1:numIteration
            mkdir(['Results/' datasetName '/Run_' int2str(runNo)],['Iteration_' int2str(iterNo)]);
            [population,accuracy]=crossValidate(population,classifierType,paramValue,fold);          
            
            mass=massCalculation(accuracy);                
            [velocities]=updateVelocities(mass,velocities,population,iterNo,numIteration);
            [population]=updatePosition(population,velocities);
            [population,accuracy]=populationRank(population,classifierType,paramValue);
            
            memory=updateMemory(memory,population,accuracy);                     
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
    end
end

function [dist]=distance(vec1,vec2)
    dist=0;
    for loop=1:size(vec1,2)
        dist=dist+((vec1(loop)-vec2(loop))^2);
    end
end

function [mass]=massCalculation(accuracy)
    worst=min(accuracy);
    best=max(accuracy);
    mass=accuracy;
    mass=mass-worst;
    mass=mass/(best-worst);
    total=sum(mass(1,:));
    mass=mass/total;
end

function [velocities]=updateVelocities(mass,velocities,population,count,iter)
    rng('shuffle');
    [rows,cols]=size(velocities);
    force=zeros(rows,cols);
    k=int16(rows+((1-rows)*(count-1))/(iter-1));      
    g=exp(-20*(count/iter));    
    for loop1=1:rows
        for loop2=1:cols
            for loop3=1:k
                if loop1~=loop3
                    temp=g*((mass(loop1)*mass(loop3))/(distance(population(loop1,:),population(loop3,:)+1)))*(population(loop3,loop2)-population(loop1,loop2));
                    force(loop1,loop2)=force(loop1,loop2)+(rand*temp);
                end
            end
        end
    end
    for loop1=1:rows
        force(loop1,:)=force(loop1,:)/mass(loop1);
    end

    for loop1=1:rows
        for loop3=1:cols
            velocities(loop1,loop3)=(rand*velocities(loop1,loop3))+force(loop1,loop3);
            velocities(loop1,loop3)=min(velocities(loop1,loop3),6);
        end
    end
end

function [population]=updatePosition(population,velocities)
    [rows,cols]=size(population);
    for loop1=1:rows
        for loop2=1:cols        
            if (rand < tanh(velocities(loop1,loop2)))
                population(loop1,loop2)=1-population(loop1,loop2);
            end        
        end
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
    for loop1=1:numAgents
        fprintf('numFeatures - %d\tAccuracy - %f\n',sum(memory.finalPopulation(loop1,:)),memory.finalAccuracy(loop1));
    end
end
