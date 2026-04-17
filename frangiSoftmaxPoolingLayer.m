% frangiSoftmaxPoolingLayer — REMOVED
%
% Channel pooling is now handled inside learnableFrangiLayer.predict via a
% pixelwise max across channels (dim 4).  This file is retained only to
% produce a clear error if stale references remain.

error('frangiSoftmaxPoolingLayer has been removed. ' ...
      'learnableFrangiLayer now outputs the pixelwise max across its channels directly.');
