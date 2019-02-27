% 1. Euler' s Method
fprintf("\nProblem 1 \n")
df = @(y, t)(2 - 2*y - exp(-4*t)); % Problem Formulation.

y_truth = @(t)(1 + 0.5*exp(-4*t) - 0.5*exp(-2*t)); % Closed-Form solution.
y_estimated = @(y, t, h)(y + h*df(y, t)); % Euler' s Method solution.

% (b)sub-problem
y_estimated_cur = 1; t = 0; % Init value of y and t.
h = 1; % Step in Euler' s Method.
fprintf("(b)sub-problem \n");
for i = 1 : 6
    t = t + h;
    y_estimated_last = y_estimated_cur;
    y_estimated_cur = y_estimated(y_estimated_cur, t, h);
    fprintf("n: %d y_estimated_last: %.6f t_n: %.2f f(t_n, y_n): %.6f h: %d dalta_y: %.6f y_estimated_cur: %.6f y_truth: %.6f \n", ...
    i-1, y_estimated_last, t-h, df(y_estimated_last, t-h), 1, y_estimated_cur-y_estimated_last, y_estimated_cur, y_truth(t));
end

% (c)sub-problem
fprintf("(c)sub-problem: See Ouput Figure \n");
h = [0.1, 0.05, 0.01, 0.005, 0.001]; % Step in Euler' s Method.
iter_times = [10, 20, 100, 200, 1000]; % How many step can algorithm compute value of y(1).

for iter = 1 : size(h, 2)
    y_estimated_cur = 1; t = 0;% Init value of y.
    clear y_estimated_all  y_truth_all;
    
    for i = 1 : iter_times(iter)
        t = t + h(iter);
        y_estimated_cur = y_estimated(y_estimated_cur, t, h(iter));
        y_truth_cur = y_truth(t);
        
        y_estimated_all(i) = y_estimated_cur; %#ok<*SAGROW>
        y_truth_all(i) = y_truth_cur;
    end
    
    subplot(1, 5, iter);
    plot(y_estimated_all, 'LineWidth', 2);
    hold on;
    plot(y_truth_all, 'LineWidth', 2);
    title("h=" + string(h(iter)));
    legend("Euler' s Method", "Truth");
    
end
