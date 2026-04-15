
% for iii in $(seq -f '%03g' 1 100); do vglrun itksnap -g toy_nifti_dataset/image_$iii.nii -s  toy_nifti_dataset/label_$iii.nii ;done
% Toy dataset generator: 3D cylinders in NIfTI format
% Creates 100 image-label pairs (64x64x64)

clear; clc;

% Parameters
numSamples = 100;
volSize = [64, 64, 64];
outputDir = fullfile(pwd, 'toy_nifti_dataset');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Coordinate grid
[x, y, z] = ndgrid(1:volSize(1), 1:volSize(2), 1:volSize(3));
coords = [x(:), y(:), z(:)];

for i = 1:numSamples
    fprintf('Generating sample %d/%d\n', i, numSamples);

    % --- Random cylinder parameters ---
    center = volSize .* rand(1,3);              % random center
    dirVec = randn(1,3);                        % random orientation
    dirVec = dirVec / norm(dirVec);             % normalize

    lengthCyl = randi([20, 50]);                % cylinder length
    radius = randi([1, 5]);                    % cylinder radius

    % Project all points onto cylinder axis
    relCoords = coords - center;
    projLength = relCoords * dirVec';           % projection along axis

    % Closest point on axis
    projPoint = projLength .* dirVec;

    % Distance from axis
    distFromAxis = sqrt(sum((relCoords - projPoint).^2, 2));

    % Cylinder mask condition
    maskVec = (abs(projLength) <= lengthCyl/2) & (distFromAxis <= radius);

    % Reshape to 3D
    mask = reshape(maskVec, volSize);

    % --- Generate image with noise ---
    signalIntensity = 1.0;
    image = signalIntensity * mask;

    % Add Gaussian noise (random SNR)
    snr = rand()*10 + 5; % SNR between 5 and 15
    noiseStd = signalIntensity / snr;
    noise = noiseStd * randn(volSize);

    image = image + noise;

    % Normalize image to [0,1]
    image = image - min(image(:));
    image = image / max(image(:));

    % --- Save as NIfTI ---
    imgFilename = fullfile(outputDir, sprintf('image_%03d.nii', i));
    labelFilename = fullfile(outputDir, sprintf('label_%03d.nii', i));

    niftiwrite(single(image), imgFilename);
    niftiwrite(uint8(mask), labelFilename);
end

fprintf('Dataset generation complete. Saved to: %s\n', outputDir);
