i0=imread("bicicleta.jpg");
i1 = impyramid(i0, 'reduce');

size(i0)

I=double(rgb2gray(i0));
I=I/max(max(I));  % image should be in [0 1]
[M,N,C] = size(I)

figure, imshow(i0);title('\fontsize{20}original');
figure, imshow('\fontsize{20}1 reducción');