function [] = simMatrix()
    str={'BreastCancer','BreastEW','CongressEW','Exactly','Exactly2','HeartEW','Ionosphere','KrVsKpEW','Lymphography','M-of-n','PenglungEW','Sonar','SpectEW','Tic-tac-toe','Vote','WaveformEW','Wine','Zoo','Arrhythmia','Madelon'};
    for i=1:size(str,2)
        simMatrixMulti(str{i});
    end
end
function []=simMatrixMulti(strIn)
x = importdata(strcat('Data/',strIn,'/',strIn,'_train.mat'));
x=x.input;

[~,num]=size(x);
%fprintf('Number of cols:%d\n',num);
matrix=zeros(num);
 add=0;
 num1=0;
for i=1:num
    for j=1:num
        if i~=j && matrix(i,j)==0
            temp=sum(x(:,i).*x(:,j))/(sqrt(sum(x(:,i).*x(:,i)))*sqrt(sum(x(:,j).*x(:,j))));
            
                temp=1/temp;
                if(isnan(temp))
                    fprintf('NaN present\n');
                    temp=1;
                end
                 if(~isinf(temp))
                     add=add+temp;
                     num1=num1+1;
                 end
            
            matrix(i,j)=temp;
            matrix(j,i)=temp;
        end
    end
end
  avg=add/num1;
  start_col=2;
  for i=1:num
      for j=start_col:num
          if(isinf(matrix(i,j)))
              	matrix(i,j)=avg;
		matrix(j,i)=avg;
          end
      end
	start_col=start_col+1;
  end
save(strcat('Data/simMatrix/simMatrix_',strIn,'.mat'),'matrix');
end