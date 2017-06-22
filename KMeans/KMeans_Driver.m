range = [1,2,3,4,5,8,10,20,50];
%%
load royce_hall_small
figure
imshow(I(:,:,:),[])
data=reshape(I,[96*144 3]);
%%
%1 and 2 K-Means and Segemnted Images
Image = data;
j=1;
figure
for K=range    
    [labels,centroids] = KMeans(data,K);
    for i=1:96*144
        Image(i,:) = centroids(labels(i),:); 
    end
    im = reshape(Image,[96 144 3]);    
    subplot(3,3,j);imshow(im(:,:,:))
    title(['K = ' num2str(K)])
    j=j+1;
end
% For K = 1 we simply color the image one color. For each K we add an other
% color that segments the images differently. For K=2 we see sometimes it
% seperates the sky and everything else, and other times it makes
% the building more distinct and connects it to he ground. For K=4 see distinction 
% between the trees, royce, the ground, and the sky meaning that we are getting
% what I think are all the major structures with only a few clusters to work with
% At K=5 we add a little more detail on the ground since we are capturing more 
% colors now and also see a difference in the sky which is usually sperated by
% by different shades of blue depending on the runs. From this point on as K
% increases we see the sky contain more and more shades meaning more
% clusters are being allocated to this area, as well as more detail such as
% grass being green and the shadows being to start coming in. Basically the
% more clusters we add the closer we get to recreating the original image.
% The different clusters seem to capture different sections of the image
% reagrdless of how many we use. This also corresponds to the color amount
% we use. 
% For which K is the best, I think that the range lies between 4-7 since
% the main structures in the picture to seperate  are the ground,vegetation
% (trees and grass),Royce, the sky, and the shadowed regions. The segments 
% utilizing just these values seems to seperate well, anything above 8 runs into
% problems with shading the sky in too many colors which is necessary for
% segmentation of the image unless maybe there were clouds. Anyways in the
% range specified we get clear portions of the different structures in the
% image.
%%
%3 Scatter plot
K=4;
[labels,centroids] = KMeans(data,K);
    for i=1:96*144
        Image(i,:) = centroids(labels(i),:); 
    end
figure
im = reshape(Image,[96 144 3]); 
imshow(im(:,:,:))
figure
hold on
scatter3(data(:,1), data(:,2), data(:,3),'.')
scatter3(centroids(:,1),centroids(:,2),centroids(:,3),100,'s','filled')
hold off
xlabel('R');
ylabel('G');
zlabel('B');
title('3-D Scatter Plot of Color Channels')
% Referring back the image of K=4 we see that the image has 4 centroids and
% therefore also 4 colors to segement the image. We see that one cluster
% refers to the darker portions of the image such as trees and shadows, 
% we can see since it is closest to the value of 0 for all axes and the trees
% in the image are a color that resembles a dark green, and the shadows are 
% dark.
% An other cluster is representative of Royce which is a medium hue of
% brown it seems. Its a lower value of blue but is higher than other points
% for red except the floor whihc is light tan the cluster lies at the 
% middle of thi region. The centroid at the top of the closest
% to 1,1,1 for all axes is ground since it is the lightest color almost
% similar to white in the recreated image. The centroid exist in the middle of this cluster and so 
% helps to differentiate this material. There is also some brown on the
% ground that resembles the color of the building since they are both brick
% that is associated with the centroid classifying Royce. The last centroid 
% high on the blue axis refers to the sky which is covered in this hue and 
% is separated from the rest of the data. 
%%
%4 Add Noise
u=var(data);
e = u.*randn(96*144,3);
data_noise = data + e;
im = reshape(data_noise,[96 144 3]);
figure
imshow(im(:,:,:),[])
Image = data_noise;
%%
% 2 again
range = [1,2,3,4,5,8,10,20,50];
figure
j = 1;
for K=range    
    [labels,centroids] = KMeans(data_noise,K);
    for i=1:96*144
        Image(i,:) = centroids(labels(i),:); 
    end
    im = reshape(Image,[96 144 3]); 
    subplot(3,3,j);imshow(im(:,:,:),[])
    title(['K = ' num2str(K)])
    j=j+1;
end
%%
%3 again Scatter plot
K=4;
[labels,centroids] = KMeans(data_noise,K);
    for i=1:96*144
        Image(i,:) = centroids(labels(i),:); 
    end
im = reshape(Image,[96 144 3]); 
figure
hold on
scatter3(data_noise(:,1), data_noise(:,2), data_noise(:,3),'.')
scatter3(centroids(:,1),centroids(:,2),centroids(:,3),200,'s','filled')
hold off
xlabel('R');
ylabel('G');
zlabel('B');
title('3-D Scatter Plot of Color Channels w/ noise')
% K=1 is actually the same and can be disregarded. When recreating the image
% we see the data now contains more pixels and is not as clear as it used to 
% be. It is a grittier version of the previous picture now. 
% Redoing number 2 of the Lab we see that the recreating
% segmented images are following a similar pattern to the last one in terms
% of the segemented parts seen. The images though now have pixelated
% portions where the noise has affected the assignment. At K =2 we see this
% is really evident since before the addition of noise the image was a clear
% segmentation between two parts of the image (top and bottom) 
% but now it is not as clean and there seems to be assignments mixed around
% the entire image. The addition of more clusters follows a similar fashion
% with each retaining more noise of the noisy image we are running kmeans
% on. The structures are not clearly segregated at any K due to the noise
% and so always seem a little pixelated.
% The addition of the nosie has also affected the scatterplot of the data
% as expected since it now looks similar to gaussian distrubutions. The
% centroid characterizing the sky is not in the center of the
% cirlce, but a little of to the side making it closer to the other points
% outside of the circle that characterizes the sky. The data this time
% around seems more centered than before since no values are at the ends of
% the spectrums such as 1,1,1 or 0,0,0. The centroids as before classify
% the same things, but with the case K=4 two centroids may be used to
% classify the sky and end up close to each other. With the gaussian noise 
% it is easier to see the clusters but since the points are closer
% assignment isnt as clean as before. Note centroids still classify same
% things here and lie in porportional region to original image.
%%
%6
num = 96*144;
RMSE = zeros(1,100);
E = zeros(100, num);
k=1;
figure
for i=1:100
    [labels,centroids]=kmeans(data,i);
    for j=1:96*144
        Image(j,:) = centroids(labels(j),:); 
    end
    E(i,:) = 1/(sqrt(3))*sqrt(sum((data-Image).^2,2));
    RMSE(i) = sum(E(i,:),2);
    if any(range == i)
        im = reshape(E(i,:),[96 144 1]);
        subplot(3,3,k);imshow(im(:,:,:),[])
        title(['K = ' num2str(i)])
        k=k+1;
    end
end
figure
plot(RMSE)
title('Cost Function with K')
xlabel('K')
ylabel('RMSE (sum of the vector)')

% For this step of the lab we print the RMSE of the and explore what the
% images look like after. Darker colors are closer to zero and therefore
% after K=10 we see no latge change in the error meeaning we are getting
% close to recreaing the original image. For K=1 (just for reference) we see
% almost everything is a mid shasde of gray and showing error. 
%  K=2 classifies the sky since well since now it is a darker shade. Since it
% classified the shaded portions equivalent to the centroid assigned to the
% sky cluster we notice error here since these shadow regions shouldnt be
% blue. Edges here are well defined and segregate the different portions.
% Not until K=4 do we see a value give a low error (darker color) for most
% of the image leavig me to believe that 4 is a good value to compress the
% image. K=5 and K=8 have darker shades but they seem to just better
% categorize specific portions of the Royce image which we arent looking for in
% terms of image segmentation.
% I also printed the sums of the RMSE for all values and we see a
% decreasing function. Around K=4,5,6,7,8 we have a bend of the curve which might
% imply this is a good value for segregating the image. These values are
% the error over the image as whole not by pixel

%%
%7 RMSE for 
RMSE = zeros(1,100);
for i=1:100
%     [labels,centroids]=kmeans(data,i);
%     for j=1:96*144
%         Image(j,:) = centroids(labels(j),:); 
%     end
%     E = 1/(sqrt(3))*sqrt(sum((data-Image).^2,2));
    RMSE(i) = mean(E(i,:));
end

figure
plot(RMSE, '.-')
xlabel('K Value')
ylabel('Error')
title('Error at K Clusters')

% As before we see that the RMSE is decreasing as K increases. This is the 
% mean error per pixel and so we see how the image is affected at each
% pixel rather than over the entire image. The error starts at around .3
% per pixel and decreases from there. As K->infinity I would expect the
% error to go to zero since we are creating the image perfectly for each
% value (essentially assigning the color of each pixel to itself since we
% have enough centroids to do so). K means can be used for compression
% since we can get a somewhat accurate recreation of an image with only a few
% colors meaning we reduce the amount of bits we must use, as compared if we
% were to use all possible color values in the image and recreate it.
% Slightly different shades of color are almost unnoticable when displaying
% a clustered image. The role of K dictates the amount of colors we will
% use to recreate the image. The more K the colors we have to paint the
% picture. So low error with low K means we are classifying portions of the
% image really well and therefore can compress the image utilizing this
% value of K rather than all 24 bits per pixel which save a lot of space.
%%
%8
num = 96*144;
comp_ratio = zeros(1,100);
comp_ratio_ciel = zeros(1,100);
for k=1:100
    comp_ratio(k) = (24*k + num*log2(k))/(24*num);
    comp_ratio_ciel(k) = ceil((24*k + num*log2(k)))/(24*num);
end
figure
subplot(1,2,1);scatter(RMSE,comp_ratio,'.');
xlabel('RMSE')
ylabel('Compression Ratio')
title('RMSE vs. Compression Ratio')
subplot(1,2,2);scatter(RMSE,comp_ratio_ciel,'.');
xlabel('RMSE')
ylabel('Compression Ratio w/ ceil')
title('RMSE vs. Compression Ratio (w/ ceiling)')
figure
plot(comp_ratio, '.-')
xlabel('K')
ylabel('Compression Ratio')
title('Compression Ratio at K')

% The cureve is strictly increasing which is intuitive since as we increase
% K we are getting closer to using the same amount of colors as used in the
% original image which also mean we are getting closer to using the same
% amount of bits as used in the original image. To get the compression
% between 10-15% we should choose a value of K that exists between [5-10]
% 10 is at the top so maybe just 5-9, 4 is slightly outside of the .1
% compression mark.
%%
%9
[labels,centroids]=kmeans(data,50);
figure
hold on
scatter3(data(:,1), data(:,2), data(:,3),'.')
scatter3(centroids(:,1),centroids(:,2),centroids(:,3),60,'s','filled')
hold off
xlabel('R');
ylabel('G');
zlabel('B');
title('3-D Scatter Plot of Color Channels')

% In signal processing we compress a continuous function into a few points
% that can help us understand the general trend of the function. This helps
% with storage since saveing a few points is much more effective then every
% point possible and make is possible to put on a computer. We are
% essentially doing the same thing with the amount of K we choose. The
% centroids classify a portion of the image into K different componets.
% Which means we are compressing the image to be just a few different
% colors similar to how the continuous curve is compressed to a few points.
% The difference here lies in the fact that we can still store images
% regardless but doing this saves space for rather than saving the entire 
% image. The reason its vetor valued functions is because each of our data 
% points are plotted with an RGB value which a 3x1 vector. So our input that
% has many various RGB values generally is compressed to only store a few
% Classified by the centroids. By looking at the Scatter plot we see each
% centroid is classifying a different color of the image reducing it from
% the amount we have originally. This is analogous to the signal
% represented in digital form.


    
