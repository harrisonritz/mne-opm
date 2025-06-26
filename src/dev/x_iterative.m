function X = x_iterative(S,PHI,Lin,Lout,ni)

% Code for iterative implementation of the Signal Space Separation method.
% Developed by Samu Taulu and described in 'An iterative implementation of
% the signal space separation method for magnetoencephalography systems
% with low channel counts' by Niall Holmes (niall.holmes@nottingham.ac.uk),
% Richard Bowtell, Matthew Brooked and Samu Taulu.

% The method is based on the assumption that the SSS vectors represent the
% MEG data in a hierarchical manner where the low-order components always
% explain a larger amount of signal energy than the high-order components.

% inputs
% S - column normalised SSS basis
% PHI - MEG data in format (Nchans,timepoints) 
% Lin - number of inner harmonics used in S
% Lout - number of outer harmonics used in S
% ni - number of iterations

% outputs 
% X - SSS weights found via iterative method, estimate signal as S*X

ni_default = 5; % Default number of iterations if not specified
if nargin < 5
   ni = ni_default;
end

nsamp = size(PHI,2); % Number of time samples
dim_m = (Lin+1)^2-1; % Dimension of the internal SSS basis

% Extract the column vectors corresponding to each l-value of the internal basis
for n = 1:Lin
   dim1 = (n-1+1)^2;
   dim2 = (n+1)^2-1;
   dimv(n,:) = [dim1 dim2]; % indices to extract each order
end
X = zeros(size(S,2),nsamp); % initial zero weights vector

for n = 1:Lin
   indices{n} = [dimv(n,1):dimv(n,2) dim_m+1:dim_m+(Lout+1)^2-1]; % indices for Lin-specific components and all Lout components
   pS{n} = pinv(S(:,indices{n})); % Pre-computed pseudoinverse matrices for individual l-values
end

if Lin >= Lout % check dimensions okay
    for i = 1:ni % for each iteration
        for j = 1:Lin % for each inner order 
            inds = indices{j}; % find relevant indices
            X(inds,:) = zeros(length(inds),nsamp); % extract relevant weights
            XN = pS{j}*(PHI-S*X); % Update the l-specific multipole moments, subtracting the contribution of other l-components
            X(inds,:) = XN; % update weights
        end
    end
else
   error('Lin should be at least as large as Lout!');
end
