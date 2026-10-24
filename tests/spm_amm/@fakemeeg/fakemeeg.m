function D = fakemeeg(data, fsample, labels, chantype, bad, grad)
% Minimal stand-in for SPM's meeg object: just what spm_opm_amm uses.
s.data = data;
s.fsample = fsample;
s.labels = labels;
s.chantype = chantype;
s.bad = bad;
s.grad = grad;
D = class(s, 'fakemeeg');
end
