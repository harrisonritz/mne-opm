function run_spm_amm(infile, outfile)
% Run SPM's spm_opm_amm, spm_ipharm and spm_epharm on the inputs in INFILE.
if exist('OCTAVE_VERSION', 'builtin')
    addpath(fullfile(fileparts(mfilename('fullpath')), 'octave'));
end
in = load(infile);
labels = cellstr(in.labels);
n = numel(labels);
grad.label = labels;
grad.chanpos = in.pos;
grad.chanori = in.ori;
grad.coilori = in.ori;
grad.tra = eye(n);
grad.balance.current = 'none';
D = fakemeeg(in.data, in.fsample, labels, repmat({'MEG'}, n, 1), in.bad(:)', grad);
S = struct('D', D, 'li', in.li, 'le', in.le, 'corrLim', in.corrLim, 'window', in.window);
mfD = spm_opm_amm(S);
out.data = mfD(:, :, 1);
g = sensors(mfD, 'MEG');
out.tra = g.tra;
out.balance = g.balance.current;
out.internal = spm_ipharm(in.v, in.n, in.a, in.b, in.li);
out.external = spm_epharm(in.v, in.n, in.a, in.b, in.le);
save(outfile, '-struct', 'out', '-v7');
end
