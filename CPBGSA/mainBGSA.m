function[population,accuracy] = mainBGSA(datasetName,numAgents,numIteration,numRuns,classifierType,paramValue,population)    
    global test fold     
    numFeatures=size(test,2);
    %initialize your variable here
    methodName='BGSA';
    fold=3;
    velocities = zeros(numAgents,numFeatures);
    
    for runNo=1:numRuns   
        for iterNo=1:numIteration        
            [population,accuracy]=crossValidate(population,classifierType,paramValue,fold);                      
            mass=massCalculation(accuracy);                
            [velocities]=updateVelocities(mass,velocities,population,iterNo,numIteration);
            [population]=updatePosition(population,velocities);
        end
            [population,accuracy]=populationRank(population,classifierType,paramValue);   
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

