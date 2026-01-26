function binary_path = searchBinary(name, searchpaths, error_if_not_found)
% function binary_path = searchBinary(name, searchpaths, error_if_not_found)
%
% Search for a binary in a list of paths
%
% arguments:
%   name:               name of the binary to search for
%   searchpaths:        cell array or string of paths to search
%   error_if_not_found: if true (default), error if binary not found
%
% returns:
%   binary_path: full path to the binary, or empty string if not found
%
% openEMS matlab interface
% -----------------------

if nargin < 3
    error_if_not_found = 1;
end

if ischar(searchpaths)
    searchpaths = {searchpaths};
end

binary_path = '';

% First check if the binary is in the system PATH
[status, result] = system(['which ' name ' 2>/dev/null']);
if status == 0
    binary_path = strtrim(result);
    return;
end

% Search in provided paths
for n = 1:numel(searchpaths)
    full_path = fullfile(searchpaths{n}, name);
    if exist(full_path, 'file')
        binary_path = full_path;
        return;
    end
end

if isempty(binary_path) && error_if_not_found
    error(['searchBinary: binary "' name '" not found!']);
end
