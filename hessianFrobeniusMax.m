function S_max = hessianFrobeniusMax(vol, sigma)
% HESSIANFROBENIUSMAX  Maximum scale-normalised Hessian Frobenius norm.
%
%   S_max = hessianFrobeniusMax(vol, sigma)
%
%   Returns the maximum over the volume of
%     S = sqrt( Lxx² + Lyy² + Lzz² + 2*(Lxy² + Lxz² + Lyz²) )
%   where each second derivative is computed by convolution with the
%   corresponding scale-normalised Gaussian Hessian kernel at scale sigma.
%
%   Used to initialise logC in learnableFrangiLayer:
%     logCInit = log(0.5 * hessianFrobeniusMax(vol, sigma))

    ks = max(5, 2*ceil(3*sigma)+1);
    r  = floor(ks/2);
    [x, y, z] = meshgrid(-r:r, -r:r, -r:r);
    G   = exp(-(x.^2 + y.^2 + z.^2) / (2*sigma^2)) / (2*pi*sigma^2)^(3/2);
    sc  = sigma^2;   % scale normalisation

    Lxx = sc * imfilter(vol, G .* (x.^2/sigma^4 - 1/sigma^2), 'replicate');
    Lyy = sc * imfilter(vol, G .* (y.^2/sigma^4 - 1/sigma^2), 'replicate');
    Lzz = sc * imfilter(vol, G .* (z.^2/sigma^4 - 1/sigma^2), 'replicate');
    Lxy = sc * imfilter(vol, G .* (x.*y/sigma^4),              'replicate');
    Lxz = sc * imfilter(vol, G .* (x.*z/sigma^4),              'replicate');
    Lyz = sc * imfilter(vol, G .* (y.*z/sigma^4),              'replicate');

    S2    = Lxx.^2 + Lyy.^2 + Lzz.^2 + 2*(Lxy.^2 + Lxz.^2 + Lyz.^2);
    S_max = sqrt(double(max(S2(:))));
end
