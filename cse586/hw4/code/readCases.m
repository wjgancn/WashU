clear all
close all

allFiles = dir('dat/107*.pts');

figure

for iI=1:length(allFiles)
  
  cPts = readPoints( strcat('dat/',allFiles(iI).name ) );
  drawFaceParts( -cPts, 'k-' );
  
end