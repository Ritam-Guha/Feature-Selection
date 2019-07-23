function [population,accuracy]=crossover(train,trainLabel,population,firstParentId,secondParentId,probCross,probMutation,accuracy,classifierType,paramValue,fold)
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
    
    mutCount=int16((col*5)/100);
    for loop=1:mutCount
        point=int16(rand(1)*(col-1))+1;
        if(rand(1)<=probMutation)
            child(1,point)=1-child(1,point);
        end
        point=int16(rand(1)*(col-1))+1;
        if(rand(1)<=probMutation)
            child(2,point)=1-child(2,point);
        end
    end

    [child(1,:),accuracyFirstChild]=crossValidate(train,trainLabel,child(1,:),classifierType,paramValue,fold);
    [child(2,:),accuracySecondChild]=crossValidate(train,trainLabel,child(2,:),classifierType,paramValue,fold);
    for loop = 1:row
        if(chromosomecomparator(population(loop,1:col),accuracy(loop),child(1,1:col),accuracyFirstChild)<0)           
            population(loop,1:col)=child(1,1:col);
            accuracy(loop)=accuracyFirstChild;          
            break;
        end
    end
    for loop = 1:row
        if(chromosomecomparator(population(loop,1:col),accuracy(loop),child(2,1:col),accuracySecondChild)<0)            
            population(loop,1:col)=child(2,1:col);
            accuracy(loop)=accuracySecondChild;
            break;
        end
    end
end


function [val]=chromosomecomparator(origPopulation,origAccuracy,child,childAccuracy)
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