clear all
close all

subject_vect = ["C7", "G2", "D6"];

for k = 1:length(subject_vect)
 subject = subject_vect(k);

switch subject
    case 'C7'
        data.no_t = [53.33 53.33 41.66 58.33];
        data.t_1 = [58.33 55 70 50];
        data.t_2 = [86.67 73.33 63.33 51.11];
    case 'D6'
        data.no_t = [52.38 60 61.66 50];
        data.t_1 = [61.9 56.66 48.33 50];
        data.t_2 = [83.33 73.33 76.67 68.33];
    case 'G2'
        data.no_t = [46.67 51.67 48.33 53.33];
        data.t_1 = [51.67 51.67 48.33 40];
        data.t_2 = [66.67 70 66.67 75];
end

dev.no_t = std(data.no_t);
dev.t_1 = std(data.t_1);
dev.t_2 = std(data.t_2);

dev_vect = [dev.no_t, dev.t_1, dev.t_2];

m.no_t = mean(data.no_t);
m.t_1 = mean(data.t_1);
m.t_2 = mean(data.t_2);

mean_vect = [m.no_t, m.t_1, m.t_2];


base = [1,2,3]; 

[min_mean, min_mean_idx] = min(mean_vect);
min_limit = min_mean-dev_vect(min_mean_idx);
[max_mean, max_mean_idx] = max(mean_vect);
max_limit = max_mean+dev_vect(max_mean_idx);

figure(k)
plot(base,mean_vect,'o','MarkerFaceColor','b','MarkerSize',7)
hold on
plot(base,mean_vect,'Color',[0.5 0.5 0.5], 'LineWidth',2,'LineStyle',':')
plot([base(1),base(1)], [mean_vect(1)-dev_vect(1), mean_vect(1)+dev_vect(1)],'Color','k' ,'Marker','_','LineWidth',2)
text(base(1),35,[num2str(mean_vect(1),'%.2f') ' \pm ' num2str(dev_vect(1),'%.2f') ],'FontSize',13,'HorizontalAlignment', 'center')
plot([base(2),base(2)], [mean_vect(2)-dev_vect(2), mean_vect(2)+dev_vect(2)],'Color','k' ,'Marker','_','LineWidth',2)
text(base(2),35,[num2str(mean_vect(2),'%.2f') ' \pm ' num2str(dev_vect(2),'%.2f') ],'FontSize',13,'HorizontalAlignment', 'center')
plot([base(3),base(3)], [mean_vect(3)-dev_vect(3), mean_vect(3)+dev_vect(3)],'Color','k' ,'Marker','_','LineWidth',2)
text(base(3),35,[num2str(mean_vect(3),'%.2f') ' \pm ' num2str(dev_vect(3),'%.2f') ],'FontSize',13,'HorizontalAlignment', 'center')
hold off

xlim([0.7 3.3])
ylim([0 100])
%ylim([min_limit*0.9, max_limit*1.1])
ylabel('Accuracy [%]')
xlabel('Modality')
xticks(base)
xticklabels({'T_{off}', 'T_1', 'T_2'})
ax = gca;
ax.YGrid = 'on';

title(subject + " overall accuracy")

set(gca, 'FontSize',15,'LineWidth',2)

end

%% mean among the subjects


% data.no_t = [53.33 53.33 41.66 58.33];
% data.t_1 = [58.33 55 70 50];
% data.t_2 = [86.67 73.33 63.33 51.11];

%d6
% data.no_t = [52.38 60 61.66 50];
% data.t_1 = [61.9 56.66 48.33 50];
% data.t_2 = [83.33 73.33 76.67 68.33];

%g2
data.no_t = [46.67 51.67 48.33 53.33 53.33 53.33 41.66 58.33 52.38 60 61.66 50];
data.t_1 = [51.67 51.67 48.33 40 58.33 55 70 50 61.9 56.66 48.33 50];
data.t_2 = [66.67 70 66.67 75 86.67 73.33 63.33 51.11 83.33 73.33 76.67 68.33];

subject='Across subjects';

dev.no_t = std(data.no_t);
dev.t_1 = std(data.t_1);
dev.t_2 = std(data.t_2);

dev_vect = [dev.no_t, dev.t_1, dev.t_2];

m.no_t = mean(data.no_t);
m.t_1 = mean(data.t_1);
m.t_2 = mean(data.t_2);

mean_vect = [m.no_t, m.t_1, m.t_2];


base = [1,2,3]; 

[min_mean, min_mean_idx] = min(mean_vect);
min_limit = min_mean-dev_vect(min_mean_idx);
[max_mean, max_mean_idx] = max(mean_vect);
max_limit = max_mean+dev_vect(max_mean_idx);

figure(length(subject_vect)+1)
plot(base,mean_vect,'o','MarkerFaceColor','b','MarkerSize',7)
hold on
plot(base,mean_vect,'Color',[0.5 0.5 0.5], 'LineWidth',2,'LineStyle',':')
plot([base(1),base(1)], [mean_vect(1)-dev_vect(1), mean_vect(1)+dev_vect(1)],'Color','k' ,'Marker','_','LineWidth',2)
text(base(1),35,[num2str(mean_vect(1),'%.2f') ' \pm ' num2str(dev_vect(1),'%.2f') ],'FontSize',13,'HorizontalAlignment', 'center')
plot([base(2),base(2)], [mean_vect(2)-dev_vect(2), mean_vect(2)+dev_vect(2)],'Color','k' ,'Marker','_','LineWidth',2)
text(base(2),35,[num2str(mean_vect(2),'%.2f') ' \pm ' num2str(dev_vect(2),'%.2f') ],'FontSize',13,'HorizontalAlignment', 'center')
plot([base(3),base(3)], [mean_vect(3)-dev_vect(3), mean_vect(3)+dev_vect(3)],'Color','k' ,'Marker','_','LineWidth',2)
text(base(3),35,[num2str(mean_vect(3),'%.2f') ' \pm ' num2str(dev_vect(3),'%.2f') ],'FontSize',13,'HorizontalAlignment', 'center')
hold off

xlim([0.7 3.3])
ylim([0 100])
%ylim([min_limit*0.9, max_limit*1.1])
ylabel('Accuracy [%]')
xlabel('Modality')
xticks(base)
xticklabels({'T_{off}', 'T_1', 'T_2'})
ax = gca;
ax.YGrid = 'on';

title([subject ' overall accuracy'])

set(gca, 'FontSize',15,'LineWidth',2)

%%
subject_vect = ["C7", "G2", "D6"];

for k = 1:length(subject_vect)
 subject = subject_vect(k);

switch subject
    case 'C7'
        %C7
        no_t_mat = [0.350	0.475	0.038	0.138
            0.088	0.375	0.063	0.475
            0.000	0.150	0.825	0.025];
			    			            
        t_1_mat = [0.575	0.304	0.071	0.050
            0.118	0.458	0.091	0.333
            0.046	0.188	0.742	0.025];
			            
         t_2_mat = [0.517	0.325	0.096	0.063
            0.067	0.788	0.063	0.083
            0.013	0.192	0.754	0.042];
    case 'D6'
        no_t_mat = [0.933	0.067	0.000	0.000
        0.467	0.150	0.000	0.383
        0.000	0.367	0.633	0.000];
	        
        t_1_mat = [0.917	0.067	0.000	0.017
        0.483	0.100	0.000	0.417
        0.000	0.383	0.533	0.083];
    
        t_2_mat = [0.882	0.118	0.000	0.000
        0.350	0.633	0.000	0.017
        0.000	0.300	0.683	0.017];
    case 'G2'
        no_t_mat = [0.604	0.290	0.052	0.054
        0.125	0.613	0.050	0.213
        0.000	0.663	0.263	0.075];

        t_1_mat = [0.663	0.288	0.038	0.013
        0.150	0.488	0.013	0.350
        0.000	0.625	0.288	0.088];
       
        t_2_mat = [0.800	0.125	0.075	0.000
        0.113	0.850	0.013	0.025
        0.000	0.535	0.440	0.025];
end

% Create a figure
figure (k+4);
sgtitle("Confusion matrix "+ subject)

subplot(131)
% Create a colored confusion matrix
imagesc(no_t_mat);
colormap(jet); % Choose a colormap, you can use 'hot', 'cool', etc.
colorbar; % Optional: show a colorbar
set(colorbar, 'Limits', [0, 1]);

% Set the axis labels
xlabel('Predicted Class');
ylabel('True Class');
title('T_{off}');

% Add text in the center of each square
[nRows, nCols] = size(no_t_mat);
for i = 1:nRows
    for j = 1:nCols
        % Calculate the position to place the text
        text(j, i, num2str(no_t_mat(i, j)), ...
             'Color', 'white', 'FontSize', 14, ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'middle');
    end
end

% Adjust axis ticks
set(gca, 'XTick', 1:nCols, 'YTick', 1:nRows);
set(gca, 'XAxisLocation', 'top');

% Optionally, set tick labels
set(gca, 'XTickLabel', {'771', 'Rest', '773', 'Miss'}, ...
         'YTickLabel', {'771', 'Rest', '773'});
set(gca, 'FontSize',15,'LineWidth',2)

% Set axis limits
axis tight; % Adjust limits to fit the data
axis square; % Make the axes equal in length

subplot(132)
% Create a colored confusion matrix
imagesc(t_1_mat);
colormap(jet); % Choose a colormap, you can use 'hot', 'cool', etc.
colorbar; % Optional: show a colorbar
set(colorbar, 'Limits', [0, 1]);

% Set the axis labels
xlabel('Predicted Class');
ylabel('True Class');
title('T_1');

% Add text in the center of each square
[nRows, nCols] = size(no_t_mat);
for i = 1:nRows
    for j = 1:nCols
        % Calculate the position to place the text
        text(j, i, num2str(t_1_mat(i, j)), ...
             'Color', 'white', 'FontSize', 14, ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'middle');
    end
end

% Adjust axis ticks
set(gca, 'XTick', 1:nCols, 'YTick', 1:nRows);
set(gca, 'XAxisLocation', 'top');

% Optionally, set tick labels
set(gca, 'XTickLabel', {'771', 'Rest', '773', 'Miss'}, ...
         'YTickLabel', {'771', 'Rest', '773'});
set(gca, 'FontSize',15,'LineWidth',2)
% Set axis limits
axis tight; % Adjust limits to fit the data
axis square; % Make the axes equal in length

subplot(133)
% Create a colored confusion matrix
imagesc(t_2_mat);
colormap("jet"); % Choose a colormap, you can use 'hot', 'cool', etc.
colorbar; % Optional: show a colorbar
set(colorbar, 'Limits', [0, 1]);

% Set the axis labels
xlabel('Predicted Class');
ylabel('True Class');
title('T_2');

% Add text in the center of each square
[nRows, nCols] = size(no_t_mat);
for i = 1:nRows
    for j = 1:nCols
        % Calculate the position to place the text
        text(j, i, num2str(t_2_mat(i, j)), ...
             'Color', 'white', 'FontSize', 14, ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'middle');
    end
end

% Adjust axis ticks
set(gca, 'XTick', 1:nCols, 'YTick', 1:nRows);
set(gca, 'XAxisLocation', 'top');

% Optionally, set tick labels
set(gca, 'XTickLabel', {'771', 'Rest', '773', 'Miss'}, ...
         'YTickLabel', {'771', 'Rest', '773'});
set(gca, 'FontSize',15,'LineWidth',2)
% Set axis limits
axis tight; % Adjust limits to fit the data
axis square; % Make the axes equal in length

end

