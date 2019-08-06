function [agents] = dataCreate(numAgents,numFeatures,minPercentage,maxPercentage)
    rng('shuffle');
    minPercentage = minPercentage/100;
    maxPercentage = maxPercentage/100;
    min=int16(numFeatures*minPercentage);
    max=int16(numFeatures*maxPercentage);    
    
    agents=zeros(numAgents,numFeatures);
    for loop1=1:numAgents
        curFeatures=(min) + int16(abs(rand*(max-min)));
        temp=rand(1,numFeatures);
        [~,temp]=sort(temp);
        for loop2=1:curFeatures
            agents(loop1,temp(1,loop2))=1;
        end      
    end
end