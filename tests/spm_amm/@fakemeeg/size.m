function varargout = size(D, varargin)
[varargout{1:max(nargout, 1)}] = size(D.data, varargin{:});
end
