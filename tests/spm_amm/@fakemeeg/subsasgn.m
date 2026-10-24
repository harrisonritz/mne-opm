function D = subsasgn(D, S, val)
D.data(S(1).subs{:}) = val;
end
