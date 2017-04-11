function[y] =  predict(X,w)

N = size(X,1);
X1 = X(:,1);
X2 = X(:,2);

y = zeros(size(X1));

for n = 1:N
       y(n) = w(1) + w(2) * X1(n) + w(3) * X2(n);
       if y(n) > 0.5
           y(n) = 1;
       else
           y(n) = 0;
       end
end

   
end