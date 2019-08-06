function [performance]=svmClassifier(train,trainLabel,test,testLabel,agent,paramValue)
    numAgents=size(agent,1);
    performance=zeros(1,numAgents);
    for loop1=1:numAgents
        if(sum(agent(loop1,:)==1)~=0)
            test=test(:,agent(loop1,:)==1);
            train=train(:,agent(loop1,:)==1);

            trainSize=size(trainLabel,1);
            label=zeros(1,trainSize);
            for loop2=1:trainSize
                label(1,loop2)=find(trainLabel(loop2,:),1);
            end

            if max(label)==2
                svmModel=fitcsvm(train,label,'KernelFunction','rbf','Standardize',true,'ClassNames',[1 2]);
            else
                class=zeros(1,max(label));
                for loop3=1:max(label)
                    class(loop3)=loop3;
                end
                temp = templateSVM('Standardize',1,'KernelFunction',paramValue,'Solver','SMO','KernelScale','auto');
                svmModel = fitcecoc(train,label,'Learners',temp,'ClassNames',class,'Coding','onevsall');
            end
            [label,~] = predict(svmModel,test);
            testSize=size(testLabel,1);
            lab=zeros(testSize,1);
            for loop3=1:testSize
                lab(loop3,1)=find(testLabel(loop3,:),1);       
            end
            differ = sum(lab ~= label)/testSize;
            performance(1,loop1)=(1-differ)*100;
%             fprintf('Number of features - %d\n',sum(agent(loop1,:)==1));
%             fprintf('The correct classification is %f\n',performance(1,loop1));
        else
            performance(1,loop1)=0;
        end
    end
end