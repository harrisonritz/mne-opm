function out = sensors(D, ~, grad)
if nargin < 3
  out = D.grad;
else
  D.grad = grad;
  out = D;
end
end
