
function [out]=HWQ1()
    clc; home;
    close all hidden
    
    Img = imread('inputEx5.jpg');
    I = (uint8(mean(Img, 3)));
    imshow(Img)


end


