% Load data from CSV file.
loadprostate;
prostate = prostate(:,2:end);

% Seed random number generator.
rng(1,'twister');

% Generate cross-validation indices.
folds = 10;
indices = crossvalind('Kfold', size(prostate,1), folds);

Test_mse = [];
Train_mse = [];
for i=1:folds
    X_Train = prostate(~(indices == i),2:end-1);
    y_Train = prostate(~(indices == i),end);

    X_Test = prostate(indices == i,2:end-1);
    y_Test = prostate(indices == i,end);

    m = size(X_Train,1);
    n = size(X_Train,2);

    % Negated identity matrix
    I = eye(m,m);

    % A in A'x <= b
    A = [X_Train -I ones(m,1); -X_Train -I -ones(m,1)];

    % b in A'x <= b
    b = [y_Train; -y_Train];

    % Objective function is the sum of all e_i's.
    f = zeros(m + n + 1, 1);
    f(n+1:n+m,1) = 1;

    % Run optimization
    [x, fval, exitflag, output] = linprog(f, A, b);

    % Extract coefficients
    coefficients = x(1:n);
    intercept = x(end);

    % Check model on training set.
    Train_predicted = [X_Train ones(size(X_Train, 1), 1)] * [coefficients; intercept];
    Train_mse = [Train_mse; mean((Train_predicted - y_Train).^2)];

    % Check model on testing set.
    Test_predicted = [X_Test ones(size(X_Test, 1), 1)] * [coefficients; intercept];
    Test_mse = [Test_mse; mean((Test_predicted - y_Test).^2)];
end

fprintf('Cross-validation Results\n');
fprintf('Fold\tTrain MSE\tTest MSE\n');
for i=1:length(Train_mse)
    fprintf('%d\t%.4f\t\t%.4f\n', i, Train_mse(i), Test_mse(i));
end
fprintf('Means:\t%.4f\t\t%.4f\n', mean(Train_mse), mean(Test_mse));
