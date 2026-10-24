function s = std(x, varargin)
% Column-wise sample standard deviation, accepting MATLAB's 'omitnan' flag,
% which Octave 8's std does not (spm_ipharm uses it).
omit = any(strcmp(varargin, 'omitnan'));
if isvector(x)
  x = x(:);
end
ok = ~isnan(x) | ~omit;
x(~ok) = 0;
n = sum(ok, 1);
mu = sum(x, 1) ./ n;
s = sqrt(sum(((x - mu) .* ok) .^ 2, 1) ./ (n - 1));
end
