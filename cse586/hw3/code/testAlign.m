clear all
close all

% load two cases and align one to the other

pts0 = readPoints('dat/107_0764.pts');
pts1 = readPoints('dat/107_0766.pts');

% first plot the unaligned case
figure

drawFaceParts( -pts0, 'r-');
drawFaceParts( -pts1, 'k-');
axis off
axis equal

% now align it
% align 1 to 0

[ptsA,pars] = getAlignedPts( pts0, pts1 );

figure
drawFaceParts( -pts0, 'r-');
drawFaceParts( -ptsA, 'k-');
axis off
axis equal