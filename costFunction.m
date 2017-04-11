function [E, grad] = costFunction(X, y, w)


%--------------------------%
% calculation of  Phi      %
%--------------------------%



if mod(size(w,1) ,2) == 0
    disp('Error: you must choose W with 2 * n + 1 elements')
else
    
    
Order = (size(w,1) - 1)/2;

Transpose_Phi = ones(size(X,1), Order * 2 + 1);


k = 1;
for i = 1:Order
    Transpose_Phi(:,k+1:k+2) = X.^(i);
    k = k + 2;
end



Phi = transpose(Transpose_Phi);

           

% transpose([ y0 ]) = ([w0 w1 w2 w3 w4 w5 w6 ...]) *  [ 1         1        ...  1        ]
%          ([ y1 ])                                   [ X1(1)     X1(2)    ...  X1(100)  ]
%          ([... ])                                   [ X2(1)     X2(2)    ...  X2(100)  ]
%          ([y100])                                   [ X1(1)^2   X1(2)^2  ...  X1(100)^2]
%                                                     [ X2(1)^2   X2(2)^2  ...  X2(100)^2]
%                                                     [ X1(1)^3   X1(2)^3  ...  X1(100)^3]
%                                                     [ X1(1)^3   X1(2)^3  ...  X1(100)^3]
%                                                     [ ...       ...      ...  ...      ]                            

% transpose[y] = transpose(w) * Phi





%--------------------------%
% calculation of E         %
%--------------------------%       


N = size(X,1);   


E = 0;


for n = 1:N   
    y_Phi = sigmoid_function( transpose(w) * Phi(:,n));
    E = E - (  y(n) * log(y_Phi) + (1 - y(n)) * log(1 - y_Phi) );
end
    %   added:
    E = E / N;
   





%--------------------------%
% calculation of grad      %
%--------------------------%


grad = 0;
 
for n = 1:N
    y_Phi = sigmoid_function( transpose(w) * Phi(:,n));
    grad = grad + (y_Phi - y(n)) * Phi(:,n);
end
    % added
    grad = grad/N;
end



            
 end