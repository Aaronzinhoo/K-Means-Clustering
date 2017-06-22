function [labels, centroids] = KMeans(data,K)
num = 96*144;
%initialize the centroids to some value
samples = randi(num,1,K); 
centroids = data(samples,:);
centroids;
TOL = .01;
old_e = 1000000000000.0;
x_norm = sum(data.^2,2);
sizes = zeros(K);
it = 1;
while 1
    old_centroids = centroids;
    centroid_norm = sum(centroids.^2,2);
    Dist = x_norm+centroid_norm';
    Dist = Dist + -2*(data*centroids');
    % now have the indices and get the sums
    [min_dists,labels] = min(Dist,[],2);
    centroids(:,:) = 0;
    sizes(:) = 0;
    for i=1:K
        assignment = labels==i;
        sizes(i) = sum(assignment,1);
        centroids(i,:) = sum(data(assignment,:),1);
        if sizes(i) > 0
            centroids(i,:) = centroids(i,:)/sizes(i);
        else 
            centroids(i,:) = NaN(1,3);
        end
    end
    e = sum(min_dists);
    if abs(e-old_e)/old_e <= TOL %relative accuracy of the error? or normalize?
        break;
    end
    old_e = e;
    it = it+1;
end    
end