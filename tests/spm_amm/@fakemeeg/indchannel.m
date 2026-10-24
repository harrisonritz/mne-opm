function ind = indchannel(D, labels)
[~, ind] = ismember(labels, D.labels);
ind = ind(:);
end
