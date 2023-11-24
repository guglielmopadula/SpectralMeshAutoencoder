myedges=load("edges.mat");
myedges=myedges.edges;
s=myedges(1,:);
t=myedges(2,:);
s=s+1;
t=t+1;
G=graph(s,t);
G=simplify(G);
L=laplacian(G);
[v,d]=eigs(L,512,'smallestabs');
save("v.mat","v");