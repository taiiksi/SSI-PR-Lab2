function phi_X = transformFeatures(X)

%-------------------------------------------------------------%
% calculation of Phi to transform X into polynomial terms     %
%-------------------------------------------------------------%


% Phi_X = [1                          1              ...     1 
%          X_test1(1)                 X_test1(2)     ...     X_test1(118)
%          X_test2(1)                 X_test2(2)     ...     X_test1(118)
    
%          X_test1^2(1)               X_test1^2(2)   ...     X_test1^2(118)
%          X_test1(1)* X_test2(1)     ...
%          X_test2^2(1)         

%          X_test1^3(1)
%          X_test1(1)*X_test2^2(1)
%          X_test2^3(1)

%          ...

%          X_test1^6(1)
%          X_test1*X_test2^5(1)
%          X_test2^6(1)];
       

order = 6;

X_transpose = transpose(X);
phi_X = ones(order*3, size(X,1));
phi_X(2,:) = X_transpose(1,:);
phi_X(3,:) = X_transpose(2,:);
 

k = 4; % Starting with phi_X(3,:) by phi_X(3,:)= X_test1^2;
for i = 1 : (order-1) 
    phi_X(k,:) = X_transpose(1,:).^(i+1);
    phi_X(k + 1,:) = X_transpose(1,:) .* X_transpose(2,:).^i;
    phi_X(k + 1 + 1,:) = X_transpose(2,:).^(i+1);
    k = k + 3;
end



