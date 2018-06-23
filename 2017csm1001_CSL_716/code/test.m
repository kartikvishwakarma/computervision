%{
old_path = path;
dir = addpath('/home/kartik/Desktop/exp_flower/2flowers/jpg/');
path(old_path, dir);
%label_1 = imageDatastore('/home/kartik/Desktop/exp_flower/2flowers/jpg/0/');
%label_2 = imageDatastore('/home/kartik/Desktop/exp_flower/2flowers/jpg/1/');
%unlabel_5 = imageDatastore('/home/kartik/Desktop/exp_flower/2flowers/jpg/5/');
%unlabel_15 = imageDatastore('/home/kartik/Desktop/exp_flower/2flowers/jpg/15/');
%%
name1 = fun('/home/kartik/Desktop/exp_flower/2flowers/jpg/0/', '/home/kartik/Desktop/exp_flower/2flowers/jpg/5/');
%name2 = fun(label_2, unlabel_15);
%file1 = fopen('label.txt', 'w');

%save('test.txt', 'name1', '-ascii');

%disp(name2);
%}

function test(path_l, path_u,id)
    disp('executing matlab....')
    label = imageDatastore(path_l);
    unlabel = imageDatastore(path_u);
    img_index = indexImages(label)
    list = []
    for i = 1:length(unlabel.Files)
        %disp(i);
        I = readimage(unlabel, i);

        [img_id,score] = retrieveImages(I, img_index);
        best_match = img_id(1);
        l_name = label.Files(best_match);
        u_name = unlabel.Files(i);
        list = cat(1,list, [l_name, u_name, score(1)]);
        %disp(list);
        %best_img = imread(img_index.ImageLocation{best_match});
        %figure
        %imshowpair(I, best_img);
        %pause(2);
        %close()
        %img = imresize(I, [224 224]);
        %list_of_img = cat(4, list_of_img, img);
    end
    fname = sprintf('label_%d.mat', id);
    save(fname)
end



%%

%img_index = indexImages(path);

