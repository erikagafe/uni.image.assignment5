    
function ApplyGaussMixEM
    
    % read image with 3 channels!
    [file, path, image, img_size] = read_image('Select input file for segmentation');
 
    % Number of dimensions
    n_dims = img_size(3);
    
    %--------------------------------------------------------------------------
    % number of desired components (clusters)
    % vary this parameter to find an appropriate value for the input
    % image (Task )!!!
    n_comp = 10;
    %--------------------------------------------------------------------------
    
%     Describe the problems and observations that you made regarding the segmentation result?
%     Gaussian Mixture Model is running on RGB, colors are created by the combination of red, blue and green. 
%     Human eyes recognize the difference of color scales as being the same color, however; the computer cluster the colors depending their values on RGB.  
%     There might be a red color with high Blue tones that might give a purple hue compared to a bright red. Using the Rubik´s Cube picture, 
%     the best n cluster was 10, it clusters most of the colors correct. There were some discrepancies but it was due to lightning and high contrasts on the image. 
    
    
    
    % reshaping of vectors for input of EM
    trainVect = reshape(image, [img_size(1)*img_size(2),n_dims] );
    
    % sample the vectors to reduce amount of data
    
    desired_number = 1000;
    step = img_size(1)*img_size(2) / desired_number; 
    indices = round(1:step:img_size(1)*img_size(2));
    trainVect = trainVect(indices,:);
    
    % filter out edge points (lead to covariance-matrices which are not invertible)
    s_t = sum(trainVect,2);
    test = find(s_t ~= 3.0);
    trainVect = trainVect(test, :);
    
    % GaussMixModel mittels EM lernen...
    model = LearnGaussMixModel(trainVect, n_comp);

    % classify pixels
    ClassImg = ClassifyImage(model, image);

    % visualize result
    figure(2); 
    subplot(1,2,1), imshow(image), title('Original Image');
    subplot(1,2,2), imshow(ClassImg,[]), colormap(jet), title('Classification Result');
end



%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% PASTE HERE YOUR IMPLEMENTED FUNCTION CalcLnVectorProb (TASK B.a)
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

function LnVectorProb = CalcLnVectorProb(model, trainVect)

% IMPLEMENT THIS FUNCTION (TASK A.a)
 
     LnVectorProb = []; 
    
     
     sizeW = size(model.weight,1);
     sizeV = size(trainVect,1);
     
     for i = 1:sizeW 
         clusters =[];
         cov = squeeze(model.covar(i,:,:));

         for j = 1:sizeV  

             logProb = log(model.weight(i))-0.5*(log(det(cov))+ (ctranspose(ctranspose(trainVect(j,:))- ctranspose(model.mean(i,:))))*inv(cov)*(ctranspose(trainVect(j,:))-(ctranspose(model.mean(i,:))))); 
              
             clusters = [clusters,logProb]; 
              
         end 
         LnVectorProb = [LnVectorProb;clusters];         
     end 
 

end


%--------------------------------------------------------------------------
function  ClassImg = ClassifyImage(model, image)
    
    % image dimensions
    s = size(image);
   
    % reshaping of feature vectors
    FeatureVects = reshape(image, [s(1)*s(2),s(3)]);
    
    % probability of all vectors for all clusters (classes)
    LnVectorProb = CalcLnVectorProb(model, FeatureVects);  
    
    % get the maximum value --> this is the corresponding class membership
    [max_values, max_pos]  = max(LnVectorProb,[],1); 
    
    % reshape vector to result array
    ClassImg = uint8(reshape(max_pos, s(1:2)));
end

%--------------------------------------------------------------------------
function [file, path, image, s] = read_image(text)

    % open a dialogue to pick afile
    [file, path] = uigetfile('*.*', text);
 
    % read image and convert to double [0,...,1]
    image = mat2gray( imread([path,file]) );
    
    % determine size/dimensions of image
    s = size(image);
end