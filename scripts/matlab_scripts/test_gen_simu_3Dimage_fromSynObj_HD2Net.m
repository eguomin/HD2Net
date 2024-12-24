% % *** default settings:
% clear all;
% close all;
tStart = tic;
addpath(genpath('..\PhaseDiversity\mFunctions\'));
addpath(genpath('C:\Users\admin\Downloads\NoiseLevelEstimate\'));

flagGPU = 1;
objType = 2; % 1: 0.8NA; 2: 1.1NA; 3: 0.71NA; 4: 1.2NA (confocal) 5: 1.0NA (2P)
lambda = 0.532; % um
pIn = 3:28; % 0: piston; 1:tilt X; 2: tilt Y;
zernNum = length(pIn);
% nType = 'poisson'; % 'none', 'gaussian','poisson'
nType = 'none';
% SNR = 20;
RI = 1.33;
psfChoice = 1; % 0: wide-field; 1: light-sheet; 2: confocal or 2P(normalized)

% % ****** customize aberrations here ****************
% random zernike coefficients
aValue = 0.04;
zernType = 'random1'; % 'defocus','astig','coma','trefoil','sphe','random1','random2'
spheValueBase = 0.15;
spheValueRange = 0.1;
defocusRange = 0.1;
astigRange = 0.1;


repNum = 5; % repeat number for each image

flagBg = 0;% 0.8NA: 1; 1.1NA: 0; 0.71NA: 0
bgValue = 95;
flagSavePSF = 0; % save PSF or not, 1: save; 0: not
LsFWHMz = 2; 
switch objType
    case 1
        NA = 0.8;
        pixelSize = 0.1625; % um
        zStepSize = 1.0; % um
        psfChoice = 1;
        LsFWHMz = 2.5; % light-sheet thickness, unit: um,
    case 2
        NA = 1.1;
        pixelSize = 0.130; % um
        zStepSize = 0.7686; % um
        % zStepSize = 0.130; % um
        psfChoice = 1;
        LsFWHMz = 2; % um
    case 3
        NA = 0.71;
        pixelSize =  0.227; % um
        zStepSize = 1.1701; % um
        psfChoice = 1;
        LsFWHMz = 2; % um
    case 4
        NA = 1.2;
        pixelSize =  0.27; % um
        zStepSize = 1.5; % um
        psfChoice = 2;
    case 5
        NA = 1.0;
        pixelSize =  0.12; % um
        zStepSize = 0.5; % um
        psfChoice = 2;
        lambda = 0.960; % um
end

amValues = [ 2 ]; % unit: rad
amTotalNum = length(amValues);

noiseHigh = 200;
noiseLow = 0.1;

fileFolderIn = 'F:\Data\SynObj\';
fileFolderOut = 'F:\Data\SynObj\Train_NoiseGen200_0p1\';
fileNameBase = 'img_';
imgNumStart = 1;
imgNumEnd = 20;
imgNumOutBase = 0; % modify for each pos folder

imgNumVali = 16;
imgNumValiInterval = 10;

% output folders
fileFolderOutInput = [fileFolderOut, 'GT\'];
fileFolderOutNF = [fileFolderOut, 'NF\'];
fileFolderOutAbe = [fileFolderOut, 'Aberrated\'];
% fileFolderOutInputMP = [fileFolderOut, 'GT_ZProj\'];
fileFolderOutAbeMP = [fileFolderOut, 'Aberrated_ZProj\'];
fileFolderOutPSF = [fileFolderOut, 'PSF\'];

fileFolderOutVali = [fileFolderOut, 'Validation\'];
fileFolderOutValiInput = [fileFolderOutVali, 'GT\'];
fileFolderOutValiAbe = [fileFolderOutVali, 'Aberrated\'];
% fileFolderOutValiInputMP = [fileFolderOutVali, 'GT_ZProj\'];
fileFolderOutValiAbeMP = [fileFolderOutVali, 'Aberrated_ZProj\'];

if isequal(exist(fileFolderOut, 'dir'),7)
    disp(['output folder:' fileFolderOut]);
else
    mkdir(fileFolderOut);
    disp(['output folder created:' fileFolderOut]);
end
mkdir(fileFolderOutInput);
mkdir(fileFolderOutNF);
% mkdir(fileFolderOutInputMP);
mkdir(fileFolderOutAbe);
% mkdir(fileFolderOutAbeMP);
mkdir(fileFolderOutPSF);

mkdir(fileFolderOutVali);
mkdir(fileFolderOutValiAbe);
% mkdir(fileFolderOutValiInput);
% mkdir(fileFolderOutValiAbeMP);
% mkdir(fileFolderOutValiInputMP);

disp('********************************************************');
disp('Start processing ... ... ');
flagSavePSFtrigger = 1;
for i = imgNumStart:imgNumEnd
    cTime1 = toc(tStart);
    disp(['Image #: ', num2str(i)]);
    iOut = i + imgNumOutBase;
    
    % generate phase diversity images;
    fileImgSample = [fileFolderIn, fileNameBase, num2str(i), '.tif'];
    disp('... Generating simulated images...');
    img0 = single(ReadTifStack(fileImgSample));
    [Sx, Sy, Sz] = size(img0);
    % subtract background
    if(flagBg==1)
        img0 = max(img0-bgValue, 0);
    end
    
    for j = 0:repNum-1
        coeffs = gen_zern_coeffs(pIn,aValue,zernType); % uniform aberration
        randValue = 2*rand - 1; % [-1, 1]
        spheCoeff = sign(randValue) * spheValueBase + spheValueRange * randValue; % spherical aberration
        coeffs(pIn==12) = spheCoeff;
        randValue = 2*rand - 1; % [-1, 1]
        defocusCoeff = defocusRange*randValue; % defocus aberration
        coeffs(pIn==4) = defocusCoeff;
        randValue = 2*rand - 1; % [-1, 1]
        astigCoeff = astigRange*randValue; % astigmatism aberration
        coeffs(pIn==3) = astigCoeff;
        randValue = 2*rand - 1; % [-1, 1]
        astigCoeff = astigRange*randValue; % astigmatism aberration
        coeffs(pIn==5) = astigCoeff;
        coeffs_GT = gen_zern_coeffs(pIn,aValue,'zero'); % zero aberration
        [~, ~, staPara, ~] = coeffs2wavefront(pIn,coeffs,Sx,...
                pixelSize, lambda, NA, 0);
        for k = amValues
            % % rescale wavefront amplitude
            amRad = k;
            coeffs_temp = coeffs*amRad/staPara.rmsPhase;

            [imgAbe, PSF_aberrated] = gen_simu_3Dimage(img0, pIn, coeffs_temp, pixelSize, ...
                lambda, NA, zStepSize, RI, nType, psfChoice, LsFWHMz);
            [imgGT, PSF_GT] = gen_simu_3Dimage(img0, pIn, coeffs_GT, pixelSize, ...
                lambda, NA, zStepSize, RI, nType, psfChoice, LsFWHMz);
            
            if(flagSavePSFtrigger==1)
                WriteTifStack(PSF_aberrated,[fileFolderOutPSF, 'PSF_aberrated_', num2str(iOut), '_', num2str(j), '_Am', num2str(k), '.tif'],16);
            end
            iNoise = rand*(noiseHigh-noiseLow) + noiseLow;
            WriteTifStack(imgGT,[fileFolderOutInput, fileNameBase, num2str(iOut),'_', num2str(j), '_Am', num2str(k), '_Noise', num2str(iNoise), '.tif'],16);
            WriteTifStack(imgAbe,[fileFolderOutNF, fileNameBase, num2str(iOut),'_', num2str(j), '_Am', num2str(k), '_Noise', num2str(iNoise), '.tif'],16);
            imgAbeNoise = iNoise * imgAbe;
            for iz = 1:Sz
                imgAbeNoise(:,:,iz) = addpoissonnoise(imgAbeNoise(:,:,iz));
            end
            WriteTifStack(imgAbeNoise,[fileFolderOutAbe, fileNameBase, num2str(iOut),'_', num2str(j), '_Am', num2str(k), '_Noise', num2str(iNoise), '.tif'],16);
        end
        
    end
    if(flagSavePSF==0) % turn off trigger next time point
        flagSavePSFtrigger = 0;
    end
    cTime2 = toc(tStart);
    disp(['... time cost: ', num2str(cTime2-cTime1)]);
end
cTime = toc(tStart);
disp(['Processing completed!!! Total time cost:', num2str(cTime), ' s']);
