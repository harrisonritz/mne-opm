function ind = indchantype(D, type)
ind = find(strcmp(D.chantype, type));
ind = ind(:)';
end
