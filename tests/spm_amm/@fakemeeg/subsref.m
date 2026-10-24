function varargout = subsref(D, S)
switch S(1).type
  case '()'
    varargout = {D.data(S(1).subs{:})};
  case '.'
    switch S(1).subs
      case 'fsample'
        varargout = {D.fsample};
      case 'save'
        varargout = {};
      otherwise  % method call with dot syntax, e.g. D.sensors('MEG')
        args = {};
        if numel(S) > 1
          args = S(2).subs;
        end
        varargout = {feval(S(1).subs, D, args{:})};
    end
end
end
