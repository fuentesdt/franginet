function idx = findModel(models, archModes)
% Return row indices in models whose archMode is in archModes (cell of strings).
% Returns empty if none of the requested archs were trained.
    if ischar(archModes), archModes = {archModes}; end
    idx = find(cellfun(@(s) ismember(s.archMode, archModes), models(:,2)));
end
