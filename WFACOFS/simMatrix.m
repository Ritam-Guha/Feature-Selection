function [] = simMatrix()
%     datasetNames={'BreastCancer','BreastEW','CongressEW','Exactly','Exactly2','HeartEW','Ionosphere','KrVsKpEW','Lymphography','M-of-n','PenglungEW','Sonar','SpectEW','Tic-tac-toe','Vote','WaveformEW','Wine','Zoo','Arrhythmia','Madelon'};
    datasetNames={'Sensor_activity_recognition'};
    for loop=1:size(datasetNames,2)
        simMatrixMulti(datasetNames{loop});
    end
end

function []=simMatrixMulti(strIn)
data=importdata(strcat('Data/',strIn,'/',strIn,'_data.mat'));
train=data.train;

[~,numFeatures]=size(train);
matrix=zeros(numFeatures);
 add=0;
 count=0;
for loop1=1:numFeatures
    for loop2=1:numFeatures
        if loop1~=loop2 && matrix(loop1,loop2)==0
            temp=sum(train(:,loop1).*train(:,loop2))/(sqrt(sum(train(:,loop1).*train(:,loop1)))*sqrt(sum(train(:,loop2).*train(:,loop2))));            
                temp=1/temp;
                if(isnan(temp))                    
                    temp=1;
                end
                 if(~isinf(temp))
                     add=add+temp;
                     count=count+1;
                 end
            
            matrix(loop1,loop2)=temp;
            matrix(loop2,loop1)=temp;
        end
    end
end
  avg=add/count;
  startCol=2;
  for loop1=1:numFeatures
      for loop2=startCol:numFeatures
          if(isinf(matrix(loop1,loop2)))
              	matrix(loop1,loop2)=avg;
		matrix(loop2,loop1)=avg;
          end
      end
	startCol=startCol+1;
  end
save(strcat('Data/simMatrix/simMatrix_',strIn,'.mat'),'matrix');
end