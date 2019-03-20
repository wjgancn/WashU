function [] = drawFaceParts( pts, spec )

% function [] = drawFaceParts( pts, spec )
% 
% drawFaceParts( pts, 'k-' )
%
% draws the face with black lines

chin = [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14]+1;
leyebrow = [15 16 17 18 19 20]+1;
reyebrow = [21 22 23 24 25 26]+1;
leye = [27 28 29 30]+1;
reye = [32 33 34 35]+1;
nose = [37 38 39 40 41 42 43 44 45]+1;
nostrils = [46 47]+1;
lipOutline = [48 49 50 51 52 53 54 55 56 57 58 59]+1;
topLip = [48 60 61 62 54]+1;
bottomLip = [54 63 64 65 48]+1;
mouthMidLine = [51 64 66 61 57]+1;
chinMidline = [57 7]+1;
lMouthLine = [3 48]+1;
rMouthLine = [11 54]+1;

% now draw it

xCoors = pts(:,1);
yCoors = pts(:,2);

hold on

plot(xCoors( chin ), yCoors( chin ), spec);
plot(xCoors( leyebrow ), yCoors( leyebrow ), spec);
plot(xCoors( reyebrow ), yCoors( reyebrow ), spec);
plot(xCoors( leye ), yCoors( leye ), spec);
plot(xCoors( reye ), yCoors( reye ), spec);
plot(xCoors( nose ), yCoors( nose ), spec);
plot(xCoors( nostrils ), yCoors( nostrils ), spec);
plot(xCoors( lipOutline ), yCoors( lipOutline ), spec);
plot(xCoors( topLip ), yCoors( topLip ), spec);
plot(xCoors( bottomLip ), yCoors( bottomLip ), spec);
plot(xCoors( mouthMidLine ), yCoors( mouthMidLine ), spec);
plot(xCoors( chinMidline ), yCoors( chinMidline ), spec);
plot(xCoors( lMouthLine ), yCoors( lMouthLine ), spec);
plot(xCoors( rMouthLine ), yCoors( rMouthLine ), spec);
