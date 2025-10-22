function relieff_feature_selection(rateStr,classifier,window,dataFolder,outFolder,Xfname,yfname)
    % Convert inputs to string
    rateStr    = string(rateStr);
    classifier = string(classifier);
    window     = string(window);

    % Validate input parameters
    if strlength(rateStr)==0 || strlength(classifier)==0 || strlength(window)==0
        error('Input parameters cannot be empty.');
    end

    % Define the folder where the files are stored
    % dataFolder = fullfile('4_FEATS_COMBINED', rateStr + '_Hz_sampling', classifier, 'EXP');
    % outFolder  = fullfile('5_FEATS_SELECTION', rateStr + '_Hz_sampling', classifier);


    % Build full paths
    baseName = rateStr + classifier;

    Xfile = fullfile(dataFolder, Xfname);
    yfile = fullfile(dataFolder, yfname);

    % Import data
    X = single(round(readmatrix(Xfile, 'NumHeaderLines', 1), 3));
    y = single(round(readmatrix(yfile, 'NumHeaderLines', 1), 0));

    % ReliefF feature selection
    [idx, weights] = relieff(X, y, 10);
    
    % Round weights to 6 decimals
    weights = double(weights(:))';
    weights = round(weights, 6);
    
    % Replace NaN with 0
    weights(isnan(weights)) = 0;
    
    % Combine idx and weights into one matrix
    results = zeros(length(idx),2);
    for i = 1:length(idx)
        results(i,:) = [idx(i), weights(idx(i))];
    end

    % Define output file
    resultsFile = fullfile(outFolder, ...
        'Matlab_relieff_feature_indices_weights_' + baseName + window + '.csv');
    
    % Save with two columns (idx, weight)
    writematrix(results, resultsFile);
    
    % Save with header
    fid = fopen(resultsFile, 'w');
    fprintf(fid, 'Feature_Index,ReliefF_Weight\n');  % headers
    fprintf(fid, '%d,%.6f\n', [results(:,1) results(:,2)].'); 
    fclose(fid);
end
