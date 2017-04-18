function [ localGradient ] = HiddenLayerLocalGradient(y, sumLayer)
    localGradient = (y * (1 - y) * sumLayer);
end

