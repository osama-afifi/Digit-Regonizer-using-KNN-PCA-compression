function  pred  = KNN( sample , X , y,  K, classes )

featureSize = size(X,2);
n = size(X,1);
tempVec = zeros(n,2);
pred = 0;

for i = 1:n;
    cost = 0;
    for j = 1:featureSize;
        cost = cost + abs(sample(j)- X(i,j));
    end;
    tempVec(i,1) = cost;
    tempVec(i,2) = i;
end;

sortedVec = sortrows(tempVec,1);

KN = zeros(n);
KN = sortedVec(1:K,2);

votes = zeros(classes,1);

for i = 1:K;
    votes(y(KN(i)),1) = votes(y(KN(i)),1)+1;
end;

[~,pred] = max(votes);


end

