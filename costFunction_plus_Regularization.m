function [E, grad] = costFunction_plus_Regularization(X, Phi, y, w, Lambda)



Order = 6;

           

%--------------------------%
% calculation of E         %
%--------------------------%       


N = size(X,1);   


E = 0;

% We add the term: Lambda/2 * sum ( w(k)^2 ), k = 1 to Order.
% M = Order and M = Dimension + 1;
% y = x^2 + x^ 1 -> Dimension = 2.
% w(Order + 1) = last term of w. 
% But we don't take w(0) = 1 in the sum.
% So we get:  Lambda/2 * sum ( w(k)^2 ), k = 2 to Order.

for n = 1:N   
    y_Phi = sigmoid_function( transpose(w) * Phi(:,n));
    term = sum((w(2 : Order + 1)).^2);
    E = E - (  y(n) * log(y_Phi) + (1 - y(n)) * log(1 - y_Phi) ) + Lambda/2 * term;
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



            
