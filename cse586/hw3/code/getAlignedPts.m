function [ptsA, pars] = getAlignedPts( pts0, pts1 );

% aligning pts1 to pts0 (with a similarity transform)

% form the vectors

x0 = pts0(:,1);
y0 = pts0(:,2);

x1 = pts1(:,1);
y1 = pts1(:,2);

len = length(x0);

% set up the linear system to be solved

% compute all the necessary terms

x0x1 = dot( x0, x1 );
x1one = dot( x1, ones( size( x1 ) ) );
y0y1 = dot( y0, y1 );
y1one = dot( y1, ones( size( y1 ) ) );
x1x1 = dot( x1, x1 );
y1y1 = dot( y1, y1 );
x0y1 = dot( x0, y1 );
y0x1 = dot( y0, x1 );
x0one = dot( x0, ones( size( x0 ) ) );
y0one = dot( y0, ones( size( y0 ) ) );
oneone = dot( ones( size( x1 ) ), ones( size( x1 ) ) );

% now set up the 4x4 system of equations in the form
%
% A*[a,b,tx,ty] = c
%

A = [x1x1+y1y1 0 x1one y1one;...
     0 x1x1+y1y1 -y1one x1one;...
     x1one -y1one oneone 0;...
     y1one x1one 0 oneone];

c = [x0x1+y0y1; y0x1-x0y1; x0one; y0one];

ptsA = [];

res = A\c;

a = res(1);
b = res(2);

% get the resulting scale and angle paramters

pars.s = sqrt(a^2+b^2);
pars.phi = acos(a/pars.s);

% get the translational part

pars.t1 = res(3);
pars.t2 = res(4);

% apply the transformation to obtain the aligned point set

AE = [a -b; b a];

ptsA = zeros(size(pts1));

for iI=1:size( pts1, 1 )

  cPt = AE*[pts1(iI,1); pts1(iI,2)] + [pars.t1; pars.t2];
  
  ptsA(iI,1) = cPt(1);
  ptsA(iI,2) = cPt(2);
  
end
