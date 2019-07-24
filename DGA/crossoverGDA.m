function [population,accuracy]=crossoverGDA(population,firstParentId,secondParentId,probCross,accuracy,classifierType,paramValue,fold)

    rng('shuffle');
    [row,col]=size(population);
    child(1,1:col)=population(firstParentId,:);
    child(2,1:col)=population(secondParentId,:);

    for loop=1:col
        if(rand(1)<=probCross)
            temp=child(1,loop);
            child(1,loop)=child(2,loop);
            child(2,loop)=temp;
        end
    end

    [child(1,:),accuracyFirstChild]=crossValidate(child(1,:),classifierType,paramValue,fold);
    [child(2,:),accuracySecondChild]=crossValidate(child(2,:),classifierType,paramValue,fold);
    for loop = 1:row
        if(chromosomeComparator(population(loop,1:col),accuracy(1,loop),child(1,1:col),accuracyFirstChild)<0)
            fprintf('Replaced chromosome at %d in crossover with %d\n',loop,firstParentId);
            population(loop,1:col)=child(1,1:col);
            accuracy(1,loop)=accuracyFirstChild;          
            break;
        end
    end
    for loop = 1:row
        if(chromosomeComparator(population(loop,1:col),accuracy(1,loop),child(2,1:col),accuracySecondChild)<0)
            fprintf('Replaced chromosome at %d in crossover with %d\n',loop,secondParentId);
            population(loop,1:col)=child(2,1:col);
            accuracy(1,loop)=accuracySecondChild;
            break;
        end
    end
end


function [val]=chromosomeComparator(origPopulation,origAccuracy,child,childAccuracy)
    [~,col]=size(origPopulation);
    count1=(sum(origPopulation(1:col)==0));
    count2=(sum(child(1:col)==0));    
    if count1==col
            val=-1;
    elseif count2==col
            val=1;
    elseif ((abs(origAccuracy-childAccuracy) > .01) || (count1==count2))
        if origAccuracy>childAccuracy
            val=1;
        else
            val=-1;
        end
    elseif ((origAccuracy>=childAccuracy) && (count1>=count2))
        val=1;
    elseif ((origAccuracy<=childAccuracy) && (count1<=count2))
        val=-1;
    else
        w1=1;w2=4;
        count1=count1/col;
        count2=count2/col;
        val=((w1*count1)+(w2*origAccuracy))-((w1*count2)+(w2*childAccuracy));
        if val>0
            val=1;
        elseif val<0
            val=-1;
        end
    end
end