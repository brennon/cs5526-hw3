% Load data from CSV file.
% loadsaheart;
saheart = [0 3 -1; 1 4 -1; 2 5 -1; 3 6 -1; 1 1 1; 2 2 1; 3 3 1; 4 4 1; 5 5 1];

X = saheart(:,1:end-1);
y = saheart(:,end);

% Data points with a -1 in the final column are negative examples. Those
% with a +1 in the final column are positive examples.
negative_indices = y == -1;

% Build matrix A.

P = X(~negative_indices,:);
N = X(negative_indices,:);

n = size(X,2);
m_pos = size(P,1);
m_neg = size(N,1);

P = [-P  -eye(m_pos, m_pos) zeros(m_pos, m_neg) -ones(m_pos, 1)];
N = [ N zeros(m_neg, m_pos)  -eye(m_neg, m_neg)  ones(m_neg, 1)];

A = [P; N];

% Add constraints to y's and z's.
for i=n+1:n+m_pos+m_neg
    A = [A; zeros(1,size(A,2))];
    A(end,i) = -1;
end

% Build b vector
b = -ones(size(A,1),1);
for i=m_pos+m_neg+1:size(A,1)
    b(i) = 0;
end

% Build objective function.
y_pos_mult = 1/m_pos;
y_neg_mult = 1/m_neg;
f = zeros(size(X,2) + m_pos + m_neg + 1, 1);
f(n+1:n+m_pos) = y_pos_mult;
f(n+m_pos+1:n+m_pos+m_neg) = y_neg_mult;

% Run linear program.
[x, fval, exitflag, output] = linprog(f, A, b);

% Seed random number generator.
% rng(1,'twister');

% Generate cross-validation indices.
% folds = 10;
% indices = crossvalind('Kfold', size(saheart,1), folds);

% Test_mse = [];
% Train_mse = [];
% for i=1:folds
%     X_Train = saheart(~(indices == i),1:end-1);
%     y_Train = saheart(~(indices == i),end);

%     X_Test = saheart(indices == i,1:end-1);
%     y_Test = saheart(indices == i,end);

%     % Negated identity matrix
%     I = eye(m,m);
% 
    % A in A'x <= b (appending columns for intercept and delta)
%     A = [X_Train ones(size(X_Train,1),2)];
    
%     % Flip the sign of the b column when y == 0
%     negative_indices = y_Train == 0;
%     A(negative_indices,end-1) = -1;

%     % b in A'x <= b
%     b = ones(size(A,1),1);
    
    % Set b to -1 when y == 0
%     b(negative_indices) = -1;

%     % Objective function is delta
%     f = zeros(size(X_Train,2) + 2, 1);
%     f(end) = -1;

%     % Run optimization
%     [x, fval, exitflag, output] = linprog(f, A, b);
% 
%     % Extract coefficients
%     coefficients = x(1:n);
%     intercept = x(end);
% 
%     % Check model on training set.
%     Train_predicted = [X_Train ones(size(X_Train, 1), 1)] * [coefficients; intercept];
%     Train_mse = [Train_mse; mean((Train_predicted - y_Train).^2)];
% 
%     % Check model on testing set.
%     Test_predicted = [X_Test ones(size(X_Test, 1), 1)] * [coefficients; intercept];
%     Test_mse = [Test_mse; mean((Test_predicted - y_Test).^2)];
% end

% fprintf('Cross-validation Results\n');
% fprintf('Fold\tTrain MSE\tTest MSE\n');
% for i=1:length(Train_mse)
%     fprintf('%d\t%.4f\t\t%.4f\n', i, Train_mse(i), Test_mse(i));
% end
% fprintf('Means:\t%.4f\t\t%.4f\n', mean(Train_mse), mean(Test_mse));
