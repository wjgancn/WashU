function pts = readPoints( fName )

fid = fopen( fName, 'r' );

% ignore the first line

fgetl(fid);
% get the number of points from the second

cl = fgetl(fid);

numPts = str2num( cl(11:end) );

fgetl(fid); % get rid of the next line

% now get the points

pts = zeros(numPts,2);

for iI=1:numPts
  cl = fgetl(fid);
  pc = sscanf(cl,'%f');
  
  pts(iI,:) = pc;
  
end
