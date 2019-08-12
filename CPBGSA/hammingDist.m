function dist = hammingDist(sample, clusterCenter)
    dist = 0;
    for loop = 1:size(sample, 2)
        if(sample(loop) ~= clusterCenter(loop))
            dist = dist + 1;
        end
    end
end