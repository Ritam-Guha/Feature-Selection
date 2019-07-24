function [finalPopulation] = GDA(population,classifierType,paramValue,fold)

    global train 
    rng('shuffle');
    popSize=size(population,1);
    numFeatures=size(train,2);
    finalPopulation=zeros(size(population));
    finalAccuracy=zeros(1,popSize);    
    rainSpeed=0.005;    
    alpha=0.5;
    initDecrement=0.05;
    maxDecIteration=10;
    [primaryPopulation,primaryAccuracy]=crossValidate(population,classifierType,paramValue,fold);
    
    for loop1=1:popSize
        curQuality=1;
        waterLevel= ((alpha*primaryAccuracy(1,loop1)+ (1-alpha)*(sum(primaryPopulation(loop1,:)==0)/numFeatures))/100)-initDecrement;
%         fprintf('water level=%f\n',waterLevel);
        iter=0;
        newFeatureSet=primaryPopulation(loop1,:);
        newSetAccuracy=0;
        peak=newFeatureSet;
        peakAccuracy=newSetAccuracy;
        while (iter <= maxDecIteration && waterLevel<=curQuality)
            newFeatureSet=mutate(newFeatureSet,rand(1));
            [newFeatureSet,newSetAccuracy]=crossValidate(newFeatureSet,classifierType,paramValue,fold);
%             disp(primaryPopulation(loop1,:));
%             disp(newFeatureSet);
%             fprintf('fin-%f init-%f\n',newSetAccuracy,primaryAccuracy(1,loop1));
            prevQuality=curQuality;
            curQuality=(alpha*newSetAccuracy + (1-alpha)*(sum(newFeatureSet==0)/numFeatures))/100;
%             fprintf('cur qual=%f prev qual=%f\n',curQuality,prevQuality);
            
            if(curQuality<prevQuality)
                iter=iter+1;
%                 fprintf('\nQuality decreased for iteration - %d\n',iter);
%                 fprintf('water level=%f cur qual=%f\n',waterLevel,curQuality);
            end      
            
            if (curQuality >= waterLevel)
                waterLevel = waterLevel+rainSpeed;
                iter=0;
%                 fprintf('\nNew Water Level - %f\n',waterLevel);
                if (newSetAccuracy>peakAccuracy)
                   peak=newFeatureSet;
                   peakAccuracy=newSetAccuracy;
                end
            end
            
                 
        end
        finalPopulation(loop1,:)=peak;
        finalAccuracy(1,loop1)=peakAccuracy;
    end

    fprintf('\n--final result of GD--\n');
    for loop1=1:popSize
        fprintf('numFeatures-%d acc-%f increase-%f\n',sum(finalPopulation(loop1,:)),finalAccuracy(1,loop1),finalAccuracy(1,loop1)-primaryAccuracy(1,loop1));
    end

end
function [population] = mutate(population,probM)
    numFeatures=size(population,2);
    for i=1:numFeatures
        if(rand<=probM)
            population(1,i)=1-population(1,i);
        end
    end
end


