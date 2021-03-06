% Load data from CSV file.
loadsaheart;

X_Train = saheart(:,1:end-1);
y_Train = saheart(:,end);

% Seed random number generator.
rng(1,'twister');

% Generate cross-validation indices.
folds = 10;
indices = crossvalind('Kfold', size(saheart,1), folds);

Train_accuracy = [];
Test_accuracy = [];
for i=1:folds
    X_Train = saheart(~(indices == i),1:end-1);
    y_Train = saheart(~(indices == i),end);

    X_Test = saheart(indices == i,1:end-1);
    y_Test = saheart(indices == i,end);

    % Data points with a -1 in the final column are negative examples. Those
    % with a +1 in the final column are positive examples.
    negative_indices = y_Train == -1;

    % Build matrix A.    
    P = X_Train(~negative_indices,:);
    N = X_Train(negative_indices,:);

    n = size(X_Train,2);
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
    f = zeros(size(X_Train,2) + m_pos + m_neg + 1, 1);
    f(n+1:n+m_pos) = y_pos_mult;
    f(n+m_pos+1:n+m_pos+m_neg) = y_neg_mult;

    % Run linear program.
    [x, fval, exitflag, output] = linprog(f, A, b);
    
    % Extract coefficients
    w = x(1:n);
    b = x(end);

    % Check model on training set.
    Train_predicted = [X_Train ones(size(X_Train, 1), 1)] * [w; b];
    Train_accuracy = [Train_accuracy; sum(sign(Train_predicted) == sign(y_Train)) / length(Train_predicted)];

    % Check model on testing set.
    Test_predicted = [X_Test ones(size(X_Test, 1), 1)] * [w; b];
    Test_accuracy = [Test_accuracy; sum(sign(Test_predicted) == sign(y_Test)) / length(Test_predicted)];
end

fprintf('Cross-validation Results\n');
fprintf('Fold\tTrain Accuracy\tTest Accuracy\n');
for i=1:length(Train_accuracy)
    fprintf('%d\t%.4f%%\t%.4f%%\n', i, Train_accuracy(i) * 100, Test_accuracy(i) * 100);
end
fprintf('Means:\t%.4f%%\t%.4f%%\n', mean(Train_accuracy) * 100, mean(Test_accuracy) * 100);
