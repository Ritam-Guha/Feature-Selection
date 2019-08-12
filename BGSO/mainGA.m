function[population,accuracy] = mainGA(datasetName,numAgents,numIteration,numRuns,classifierType,paramValue)
    warning off;
    global test fold 
    
    numFeatures=size(test,2);
    %initialize your variable here
    methodName='GA';
    minFeaturePercentage=30;
    maxFeaturePercentage=80;
    mcross=int16(5);
    fold=3;
    
    for runNo=1:numRuns           
        population=dataCreate(numAgents,numFeatures,minFeaturePercentage,maxFeaturePercentage);              
        for iterNo=1:numIteration
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
        end

        [population,accuracy]=populationRank(population,classifierType,paramValue);        
    end
end

