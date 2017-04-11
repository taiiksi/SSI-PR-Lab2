

%--------------------------%
%                          %
%  II. Logistic regression %
%                          %
%--------------------------%



% E = 69.31 != 0.6931 ?
% E = E / N added
% grad = grad/N added;
% grad = grad + (y_Phi - y(n)) * Phi(:,n); and not  grad = grad + (y_Phi - y(n)) * y_Phi;



%---------------------%
%  Visualize the data %
%---------------------%


data1 =  importdata('lab2data1.txt');
X = data1(:,1:2);
y = data1(:,3);

X_Accepted = zeros(sum(y),2);
X_Refused = zeros(size(X,1) - sum(y),2);


index_X_Accepted = 1;
index_Y_Accepted = 1;

for i = 1:size(X,1)
   if y(i) == 1
       X_Accepted(index_X_Accepted,:) = X(i,:);
       index_X_Accepted = index_X_Accepted + 1;
   else
       X_Refused(index_Y_Accepted,:) = X(i,:);
       index_Y_Accepted = index_Y_Accepted + 1;
   end
end

figure(1);
plot(X_Accepted(:,1), X_Accepted(:,2), '+');
hold on;
plot(X_Refused(:,1), X_Refused(:,2), 'o');
title('Our data, + for students accepted, o for students refused');
xlabel('First mark');
ylabel('Second mark');





%-----------------------------------%
%  Check the sigmoid_function       %
%-----------------------------------%


abs_sigma = 1:0.1:10;
check_sigma = sigmoid_function(abs_sigma);

figure(2)
plot(abs_sigma, check_sigma);
title('Check the sigmoid function')
xlabel('Input of the sigmoid function');
ylabel('Output of the sigmoid function');





%---------------------------------------------------------------------%
%  Compute the minium error to get w such as: y = w0 + w1 *x1 + w2*x2 
%                          (Linear regression)
%---------------------------------------------------------------------%

w_init = [0; 0; 0];
[check_E, check_cost] = costFunction(X,y,w_init);

options = optimset('GradObj', 'on', 'MaxIter', 400);
[w_linear, cost] = fminunc( @(w)(costFunction(X,y,w)), w_init, options ); 





%--------------------------------------------------------%
%  Plot the line obtained with w = transpose([w0 w1 w2]) %
%--------------------------------------------------------%


P_array = zeros(100);
Save_coordinate = [];

k = 1;
for i = 1:100
    for j = 1:100
        P_array(i,j) = w_linear(1) + w_linear(2) * i + w_linear(3) * j;
        if ( ( 0.4 < P_array(i,j) ) && ( P_array(i,j) < 0.6 ) )
            Save_coordinate(k,:) = [i j];
            k = k + 1;
        end
    end
end



X1 = Save_coordinate(:,1);
X2 = Save_coordinate(:,2);


figure(3);
plot(X_Accepted(:,1), X_Accepted(:,2), '+');
hold on;
plot(X_Refused(:,1), X_Refused(:,2), 'o');

hold on;
plot(X1, X2, '*');

title('Linear Regression');
xlabel('First mark');
ylabel('Second mark');




%---------------------------------------------------------------------%
%          Same with Polynomial regression of order 3
%       Compute the minium error to get w (Optional work)
%---------------------------------------------------------------------%


w_init = [0; 0; 0; 0; 0; 0; 0];

options = optimset('GradObj', 'on', 'MaxIter', 400);
[w_poly, cost] = fminunc( @(w)(costFunction(X,y,w)), w_init, options ); 



%------------------------------------------------------%
%  Plot the polynome  (Optional work)                  %
%------------------------------------------------------%


P_array = zeros(100);
Save_coordinate = [];

k = 1;
for i = 1:100
    for j = 1:100
        P_array(i,j) = w_poly(1) + w_poly(2) * i + w_poly(3) * j + w_poly(4) * i^2 + w_poly(5) * j^2 + w_poly(6) * i^3 + w_poly(7) * j^3;
        if ( ( 0.4 < P_array(i,j) ) && ( P_array(i,j) < 0.6 ) )
            Save_coordinate(k,:) = [i j];
            k = k + 1;
        end
    end
end





X1 = Save_coordinate(:,1);
X2 = Save_coordinate(:,2);


figure(4);
plot(X_Accepted(:,1), X_Accepted(:,2), '+');
hold on;
plot(X_Refused(:,1), X_Refused(:,2), 'o');

hold on;
plot(X1, X2, '*');

title('Polynomial Regression with order = 3, without regularization');
xlabel('First mark');
ylabel('Second mark');



%------------------------------------------------------%
%  Predict if an student is admitted or not            %
%------------------------------------------------------%

y_predicted = predict(X, w_linear);



%------------------------------------------------------%
%  Compute the accuracy of the prediction              %
%------------------------------------------------------%




    N = size(X,1);
    Error_prediction = sum(sqrt((y - y_predicted).^2))/N;
    Accuracy_prediction = 1 - Error_prediction;



