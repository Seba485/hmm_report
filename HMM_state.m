function [] = HMM_state()

    base = [0:0.01:1]';
    % A =
    % B = 
    % A_1 = 
    % B_1 = 
    %param = [A, B, A_1, B_1];
    task1.hmm_state = hmm_state(base,'task_1');%,'param',param);
    task2.hmm_state = hmm_state(base,'task_2');
    rest.hmm_state = hmm_state(base,'rest');

    y_lim = [0 12];
    figure()
    sgtitle('HMM state')
    subplot(131)
    plot(base,task1.hmm_state,'b-','LineWidth',2)
    ylim(y_lim)
    grid on
    title('task_1')

    subplot(132)
    plot(base,task2.hmm_state,'r-','LineWidth',2)
    ylim(y_lim)
    grid on
    title('task_2')

    subplot(133)
    plot(base,rest.hmm_state,'g-','LineWidth',2)
    ylim(y_lim)
    grid on
    title('rest')
    
end