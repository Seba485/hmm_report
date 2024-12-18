clear all
close all
clc

root_folder = "/home/sebastiano/Desktop/Data";
subjects = ["c7", "d6", "g2", "h7", "h5", "h6"];

target_file = 'HMM_report.mat';

no_T = []; %for the across subjects accuracy
T_1 = [];
T_2 = [];

for n = 1:length(subjects)

    subject_folder = root_folder+'/'+subjects(n);
    
    content = dir(subject_folder);
    
    confusion.no_T = zeros(3,4,1);
    confusion.T_1 = zeros(3,4,1);
    confusion.T_2 = zeros(3,4,1);
    
    overall_acc.no_T = [];
    overall_acc.T_1 = [];
    overall_acc.T_2 = [];
    
    file_found = false;
    n_aq = 0;

    % collecting data from the acquisitions
    for k = 3:length(content) %skip . and ..
        if isfolder(subject_folder+'/'+content(k).name)
            try
                load(subject_folder+'/'+content(k).name+'/'+target_file)
                file_found = true;
            catch 
                %pass
            end
        else
            if strcmp(content(k).name,target_file)
                load(subject_folder+'/'+content(k).name)
                file_found = true;
            end
        end
        
        if file_found
            n_aq = n_aq + 1;
            file_found = false;
            
            try
                confusion.no_T(:,:,n_aq) = accuracy.no_T.confusion{end}{:,:};
                confusion.T_1(:,:,n_aq) = accuracy.T_1.confusion{end}{:,:};
                confusion.T_2(:,:,n_aq) = accuracy.T_2.confusion{end}{:,:};
            catch
                %pass
            end
        
            overall_acc.no_T = [overall_acc.no_T accuracy.no_T.overall(1:end-1)];
            overall_acc.T_1 = [overall_acc.T_1 accuracy.T_1.overall(1:end-1)];
            overall_acc.T_2 = [overall_acc.T_2 accuracy.T_2.overall(1:end-1)];
        end
    end

    no_T_confusion = sum(confusion.no_T,3);
    no_T_confusion = no_T_confusion./sum(no_T_confusion(1,:));
    
    T_1_confusion = sum(confusion.T_1,3);
    T_1_confusion = T_1_confusion./sum(T_1_confusion(1,:));
    
    T_2_confusion = sum(confusion.T_2,3);
    T_2_confusion = T_2_confusion./sum(T_2_confusion(1,:));
    
    % collecting data for all the subgects
    if subjects(n)~="h6" %excluding h6 because it wasnt in controll
        no_T = [no_T overall_acc.no_T];
        T_1 = [T_1 overall_acc.T_1];
        T_2 = [T_2 overall_acc.T_2];
    end
    
    % box plot
    data_mat = [overall_acc.no_T', overall_acc.T_1', overall_acc.T_2'];
    
    if size(data_mat,1)>1
        chart = "box";
        dev_vect = std(data_mat);
        mean_vect = mean(data_mat);
    else
        chart = "";
        dev_vect = zeros(size(data_mat));
        mean_vect = data_mat;
    end
    
    base = [1,2,3]; 
    
    figure()
    
    if chart == "box"
    
        boxplot(data_mat, 'Labels',{'T_{off}', 'T_1', 'T_2'},'Widths', 0.3, 'OutlierSize', 6)
        hold on
        plot(base,mean_vect,'Color',[0.5 0.5 0.5], 'LineWidth',2,'LineStyle',':')
        plot(base,mean_vect,'o','MarkerFaceColor','k','MarkerSize',7)
        
    else
        plot(base,mean_vect,'o','MarkerFaceColor','b','MarkerSize',7)
        hold on
        plot(base,mean_vect,'Color',[0.5 0.5 0.5], 'LineWidth',2,'LineStyle',':')
        plot([base(1),base(1)], [mean_vect(1)-dev_vect(1), mean_vect(1)+dev_vect(1)],'Color','k' ,'Marker','_','LineWidth',2)
        plot([base(2),base(2)], [mean_vect(2)-dev_vect(2), mean_vect(2)+dev_vect(2)],'Color','k' ,'Marker','_','LineWidth',2)
        plot([base(3),base(3)], [mean_vect(3)-dev_vect(3), mean_vect(3)+dev_vect(3)],'Color','k' ,'Marker','_','LineWidth',2)
    
    end
    text(base(1),25,[num2str(mean_vect(1),'%.2f') ' \pm ' num2str(dev_vect(1),'%.2f') ],'FontSize',16,'HorizontalAlignment', 'center')
    text(base(2),25,[num2str(mean_vect(2),'%.2f') ' \pm ' num2str(dev_vect(2),'%.2f') ],'FontSize',16,'HorizontalAlignment', 'center')
    text(base(3),25,[num2str(mean_vect(3),'%.2f') ' \pm ' num2str(dev_vect(3),'%.2f') ],'FontSize',16,'HorizontalAlignment', 'center')
    hold off
    
    xlim([0.7 3.3])
    xticks(base)
    xticklabels({'$T_{off}$', '$T_1$', '$T_2$'})
    set(gca, 'TickLabelInterpreter', 'latex')
    
    ylabel('Accuracy [%]')
    xlabel('Modality')
    ylim([0 100])
    ax = gca;
    ax.YGrid = 'on';
    
    title(upper(subjects(n)) + " Overall Accuracy")
    
    set(gca, 'FontSize',15,'LineWidth',2)
    
    % confusion matrix
    
    figure()
    
    sgtitle(upper(subjects(n))+" Confusion Matrix")
    colormap parula
    
    subplot(131)
    
        imagesc(no_T_confusion);
        colorbar off;
        clim([0, 1]);
        
        xlabel('Predicted Class');
        ylabel('True Class');
        title('T_{off}');
        
        [nRows, nCols] = size(no_T_confusion);
        for i = 1:nRows
            for j = 1:nCols
                value = no_T_confusion(i, j);
                % Adjust text color based on background intensity
                if value > 0.5
                    textColor = 'black';  % Light background, so use dark text
                else
                    textColor = 'white';  % Dark background, so use light text
                end
    
                text(j, i, num2str(round(no_T_confusion(i, j),2)), ...
                     'Color', textColor, 'FontSize', 18, ...
                     'HorizontalAlignment', 'center', ...
                     'VerticalAlignment', 'middle');
            end
        end
        
        set(gca, 'XTick', 1:nCols, 'YTick', 1:nRows);
        set(gca, 'XAxisLocation', 'top');
        
        set(gca, 'XTickLabel', {'BF', 'Rest', 'BH', 'Miss'}, ...
             'YTickLabel', {'BF', 'Rest', 'BH'});
        set(gca, 'FontSize',18,'LineWidth',2)
    
        
        axis tight; 
        axis square; 
    
    subplot(132)
    
        imagesc(T_1_confusion);
        colorbar off; 
        clim([0, 1]);
        
        xlabel('Predicted Class');
        ylabel('True Class');
        title('T_1');
        
        [nRows, nCols] = size(T_1_confusion);
        for i = 1:nRows
            for j = 1:nCols
                value = T_1_confusion(i, j);
                % Adjust text color based on background intensity
                if value > 0.5
                    textColor = 'black';  % Light background, so use dark text
                else
                    textColor = 'white';  % Dark background, so use light text
                end
    
                text(j, i, num2str(round(T_1_confusion(i, j),2)), ...
                     'Color', textColor, 'FontSize', 18, ...
                     'HorizontalAlignment', 'center', ...
                     'VerticalAlignment', 'middle');
            end
        end
        
        set(gca, 'XTick', 1:nCols, 'YTick', 1:nRows);
        set(gca, 'XAxisLocation', 'top');
        
        
        set(gca, 'XTickLabel', {'BF', 'Rest', 'BH', 'Miss'}, ...
             'YTickLabel', {'BF', 'Rest', 'BH'});
        set(gca, 'FontSize',18,'LineWidth',2)
    
        
        axis tight; 
        axis square; 
    
    subplot(133)
    
        imagesc(T_2_confusion);
        clim([0, 1]);

        cb = colorbar; 
        set(cb, 'Position', [0.92 0.287 0.02 0.461])
        
        
        xlabel('Predicted Class');
        ylabel('True Class');
        title('T_2');
        
        
        [nRows, nCols] = size(T_2_confusion);
        for i = 1:nRows
            for j = 1:nCols
                value = T_2_confusion(i, j);
                % Adjust text color based on background intensity
                if value > 0.5
                    textColor = 'black';  % Light background, so use dark text
                else
                    textColor = 'white';  % Dark background, so use light text
                end
                
                text(j, i, num2str(round(T_2_confusion(i, j),2)), ...
                     'Color', textColor, 'FontSize', 18, ...
                     'HorizontalAlignment', 'center', ...
                     'VerticalAlignment', 'middle');
            end
        end
        
        set(gca, 'XTick', 1:nCols, 'YTick', 1:nRows);
        set(gca, 'XAxisLocation', 'top');
        
        set(gca, 'XTickLabel', {'BF', 'Rest', 'BH', 'Miss'}, ...
             'YTickLabel', {'BF', 'Rest', 'BH'});
        set(gca, 'FontSize',18,'LineWidth',2)
    
        
        axis tight;
        axis square; 
end

% mean among the subjects
subject='Across Subjects';
chart = "box";

data_mat = [no_T', T_1', T_2'];

dev_vect = std(data_mat);
mean_vect = mean(data_mat);


base = [1,2,3]; 


figure()
if chart == "box"

    h = boxplot(data_mat, 'Labels',{'T_{off}', 'T_1', 'T_2'},'Widths', 0.3, 'OutlierSize', 6);
    set(h,'LineWidth',2)
    hold on
    plot(base,mean_vect,'Color',[0.5 0.5 0.5], 'LineWidth',2.5,'LineStyle',':')
    plot(base,mean_vect,'o','Color','k','MarkerFaceColor','k','MarkerSize',7)
    
else
    plot(base,mean_vect,'o','MarkerFaceColor','b','MarkerSize',7)
    hold on
    plot(base,mean_vect,'Color',[0.5 0.5 0.5], 'LineWidth',2,'LineStyle',':')
    plot([base(1),base(1)], [mean_vect(1)-dev_vect(1), mean_vect(1)+dev_vect(1)],'Color','k' ,'Marker','_','LineWidth',2)
    plot([base(2),base(2)], [mean_vect(2)-dev_vect(2), mean_vect(2)+dev_vect(2)],'Color','k' ,'Marker','_','LineWidth',2)
    plot([base(3),base(3)], [mean_vect(3)-dev_vect(3), mean_vect(3)+dev_vect(3)],'Color','k' ,'Marker','_','LineWidth',2)
   
    

end
text(base(1),25,[num2str(mean_vect(1),'%.2f') ' \pm ' num2str(dev_vect(1),'%.2f') ],'FontSize',20,'HorizontalAlignment', 'center','FontWeight','bold')
text(base(2),25,[num2str(mean_vect(2),'%.2f') ' \pm ' num2str(dev_vect(2),'%.2f') ],'FontSize',20,'HorizontalAlignment', 'center','FontWeight','bold')
text(base(3),25,[num2str(mean_vect(3),'%.2f') ' \pm ' num2str(dev_vect(3),'%.2f') ],'FontSize',20,'HorizontalAlignment', 'center','FontWeight','bold')
hold off

xlim([0.7 3.3])
xticklabels({'\bf \it T_{off}', '\bf \it T_1', '\bf \it T_2'})
set(gca, 'TickLabelInterpreter', 'tex')

ylabel('Accuracy [%]')
xlabel('Modality')
ylim([0 100])
ax = gca;
ax.YGrid = 'on';

title(subject + " Overall Accuracy")

set(gca, 'FontSize',25,'LineWidth',2)

% statistical test between the accuracy

disp('T_{off} - T_1')
[h, p, ~, ~] = ttest(no_T, T_1,'Alpha',0.05,'Tail','both');

% Display results
disp(['Hypothesis test result: ', num2str(h)]);
if h==1
    disp('Null hypotesis rejected --> the distributions are different')
else
    disp('Null hypotesis accepted --> the distributions are the same')
end
disp(['P-value: ', num2str(p)]);

disp('T_{off} - T_2')
[h, p, ~, ~] = ttest(no_T, T_2,'Alpha',0.05,'Tail','both');

% Display results
disp(['Hypothesis test result: ', num2str(h)]);
if h==1
    disp('Null hypotesis rejected --> the distributions are different')
else
    disp('Null hypotesis accepted --> the distributions are the same')
end
disp(['P-value: ', num2str(p)]);

disp('T_1 - T_2')
[h, p, ~, ~] = ttest(T_1, T_2,'Alpha',0.05,'Tail','both');

% Display results
disp(['Hypothesis test result: ', num2str(h)]);
if h==1
    disp('Null hypotesis rejected --> the distributions are different')
else
    disp('Null hypotesis accepted --> the distributions are the same')
end
disp(['P-value: ', num2str(p)]);


%% Accuracy media per tutti i soggettti

C7 = [51.66 58.33 68.61]';
D6 = [56.01 54.22 75.41]';
G2 = [50 47.92 69.59]';
H5 = [25 63.33 63.33]';
H6 = [33.33 33.33 33.33]';
H7 = [51.67 61.67 81.67]';

data = [C7 D6 G2 H5 H6 H7];

base = [1 2 3];

figure()

plot(base, data(:,1), 'LineStyle','--','LineWidth',2,'Color',[0, 0.4470, 0.7410],'Marker','o','MarkerSize',6,'MarkerFaceColor',[0, 0.4470, 0.7410]);
hold on
plot(base, data(:,2), 'LineStyle','--','LineWidth',2,'Color',[0.8500, 0.3250, 0.0980],'Marker','o','MarkerSize',6,'MarkerFaceColor',[0.8500, 0.3250, 0.0980]);
plot(base, data(:,3), 'LineStyle','--','LineWidth',2,'Color',[0.3010, 0.7450, 0.9330],'Marker','o','MarkerSize',6,'MarkerFaceColor',[0.3010, 0.7450, 0.9330]);
plot(base, data(:,4), 'LineStyle','--','LineWidth',2,'Color',[0.4940, 0.1840, 0.5560],'Marker','o','MarkerSize',6,'MarkerFaceColor',[0.4940, 0.1840, 0.5560]);
plot(base, data(:,5), 'LineStyle','--','LineWidth',2,'Color',[0.4660, 0.6740, 0.1880],'Marker','o','MarkerSize',6,'MarkerFaceColor',[0.4660, 0.6740, 0.1880]);
plot(base, data(:,6), 'LineStyle','--','LineWidth',2,'Color',[0.6350, 0.0780, 0.1840],'Marker','o','MarkerSize',6,'MarkerFaceColor',[0.6350, 0.0780, 0.1840]);
hold off

xlim([0.7, 3.3])
ylim([0, 100])
xticks(base)
xticklabels({'T_{off}', 'T_1', 'T_2'})
title('Overall Accuracy')
legend('C7', 'D6', 'G2', 'H5', 'H6', 'H7')
ax = gca;
ax.YGrid = 'on';
set(ax, 'FontSize',15,'LineWidth',2)

%% binary accuracy
g2 = [72.5 85 80];
d6 = [95 100 100];
c7 = [90 75 87.5];
h7 = [95];

disp(num2str(mean(c7))+"+-"+num2str(std(c7)))
disp(num2str(mean(d6))+"+-"+num2str(std(d6)))
disp(num2str(mean(g2))+"+-"+num2str(std(g2)))

%%
idx = 6;

%c7
no_T_mat(:,:,1)=[10 6 2 2;
                3 3 0 14;
                0 0 19 1];
%d6
no_T_mat(:,:,2) = [19 1 0 0;
                    9 1 0 10;
                    0 3 17 0];
%g2
no_T_mat(:,:,3) = [17 3 0 0;
                    1 9 1 9;
                    0 11 5 4];
%h5
no_T_mat(:,:,4) = [10 9 0 1;
                    13 5 0 2;
                    6 14 0 0];
%h6
no_T_mat(:,:,5)=[0 10 0 0;
                0 10 0 0;
                0 10 0 0];
%H7
no_T_mat(:,:,6)= [15 5 0 0;
                    8 8 2 2;
                    0 12 8 0];

no_T_confusion = no_T_mat(:,:,idx);

figure()

subplot(131)
imagesc(no_T_confusion);
colorbar off;
clim([0, 20]);

xlabel('Predicted Class');
ylabel('True Class');
title('T_{off}');

[nRows, nCols] = size(no_T_confusion);
for i = 1:nRows
    for j = 1:nCols
        value = no_T_confusion(i, j);
        % Adjust text color based on background intensity
        if value > 0.5
            textColor = 'black';  % Light background, so use dark text
        else
            textColor = 'white';  % Dark background, so use light text
        end

        text(j, i, num2str(round(no_T_confusion(i, j),2)), ...
             'Color', textColor, 'FontSize', 18, ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'middle');
    end
end

set(gca, 'XTick', 1:nCols, 'YTick', 1:nRows);
set(gca, 'XAxisLocation', 'top');

set(gca, 'XTickLabel', {'BF', 'Rest', 'BH', 'Miss'}, ...
     'YTickLabel', {'BF', 'Rest', 'BH'});
set(gca, 'FontSize',18,'LineWidth',2)


axis tight; 
axis square; 

